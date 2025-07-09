import json
from pathlib import Path
from jinja2 import Template
from langgraph.graph import StateGraph, END
from loguru import logger

from app.dependencies import openai_client
from app.models.schemas import ConversationState
from app.services.rag import query_collection

# --- Prompt Loading ---

PROMPTS_DIR = Path(__file__).parent.parent / "../prompts"


def load_prompt_template(filename: str) -> Template:
    """Loads a prompt from the prompts directory and returns a Jinja2 Template."""
    try:
        with open(PROMPTS_DIR / filename, "r", encoding="utf-8") as f:
            return Template(f.read())
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {filename}")
        raise
    except Exception as e:
        logger.error(f"Error loading prompt {filename}: {e}")
        raise


ANALYZE_ANSWERS_TEMPLATE = load_prompt_template("analyze_answers.txt")
ANALYZE_QUERY_TEMPLATE = load_prompt_template("analyze_query.txt")
FOLLOW_UP_QUESTION_TEMPLATE = load_prompt_template("follow_up_question.txt")
RECOMMENDATION_TEMPLATE = load_prompt_template("recommendation.txt")


# ----------- Pipeline Node Functions -----------


def ask_follow_up_questions(state: ConversationState) -> ConversationState:
    """Asks a direct follow-up question based on the current conversation state."""
    system_prompt = FOLLOW_UP_QUESTION_TEMPLATE.render()
    logger.info(f"Node: ask_follow_up_questions with prompt {system_prompt[:40]}")
    messages = [{"role": "system", "content": system_prompt}, *state.conversation]
    follow_up = ask_ai(messages)
    state.follow_up_question = follow_up
    return state


def analyze_answers(state: ConversationState) -> ConversationState:
    """Analyzes user answers to determine if enough information is present for recommendations."""
    logger.info("Node: analyze_answers")
    full_conversation = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in state.conversation]
    )
    system_prompt = ANALYZE_ANSWERS_TEMPLATE.render()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"conversation: \n {full_conversation}"},
    ]
    response = ask_ai(messages)
    try:
        final_response = json.loads(response.strip("```json").strip())
        logger.info(f'analyzed ans {final_response}')
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON from analyze_answers: {response}")
        final_response = {"ready_for_recommendation": False, "optimized_query": ""}

    state.ready_for_recommendation = final_response.get("ready_for_recommendation", False)
    state.recommendation_query = final_response.get("optimized_query", state.query) # Use original query as fallback
    state.is_follow_up = final_response.get("is_follow_up", False)

    return state


def analyze_query(state: ConversationState) -> ConversationState:
    """Analyzes the initial user query to decide the next step."""
    logger.info("Node: analyze_query")
    results = query_collection("skincare_combined", state.query, n_results=5) 
    doc_metadata_pairs = list(zip(results["documents"][0], results["metadatas"][0]))
    
    top_docs = list(zip(*doc_metadata_pairs)) if doc_metadata_pairs else ([], [])

    system_prompt = ANALYZE_QUERY_TEMPLATE.render(retrieved_content=top_docs)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state.query},
    ]
    response = ask_ai(messages)
    try:
        final_response = json.loads(response.strip("```json").strip())
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON from analyze_query: {response}")
        final_response = {"is_follow_up": False, "answer": ""}

    state.is_follow_up = final_response.get("is_follow_up", False)
    state.follow_up_question = final_response.get("answer", "")
    if not state.is_follow_up:
        # If no follow-up is needed, we are ready for recommendation with the original query
        state.ready_for_recommendation = True
        state.recommendation_query = state.query
        
    return state


def retrieve_documents(state: ConversationState) -> ConversationState:
    """Retrieves documents from the vector store based on the recommendation query."""
    logger.info("Node: retrieve_documents")
    # Using recommendation_query directly as per todo.txt
    results = query_collection("skincare", state.recommendation_query, n_results=10)
    state.retrieved_documents = list(zip(results["documents"][0], results["metadatas"][0]))
    return state


def recommend_products(state: ConversationState) -> ConversationState:
    """Generates recommendations based on retrieved documents."""
    logger.info("Node: recommend_products")
    
    top_pairs = state.retrieved_documents[:5]
    
    top_docs, top_metadata = zip(*top_pairs) if top_pairs else ([], [])
    state.citations = top_docs

    # Clean up metadata for the prompt
    clean_metadata = []
    for meta in top_metadata:
        meta_copy = meta.copy()
        meta_copy.pop("id", None)
        meta_copy.pop("margin", None)
        meta_copy.pop("product_id", None)
        clean_metadata.append(meta_copy)

    recommendation_prompt = RECOMMENDATION_TEMPLATE.render(product_data=clean_metadata)
    messages = [
        {"role": "system", "content": recommendation_prompt},
        {"role": "user", "content": state.recommendation_query},
    ]
    state.recommendation = ask_ai(messages)
    state.is_follow_up = "False"
    return state


# ----------- Helper Functions -----------


def ask_ai(messages: list) -> str:
    """Sends messages to the OpenAI client and returns the response."""
    logger.debug(f"Sending {len(messages)} messages to OpenAI.")
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=500,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def build_graph():
    """Builds and compiles the conversation pipeline graph."""
    logger.info("Building the StateGraph pipeline.")
    graph = StateGraph(state_schema=ConversationState)

    graph.add_node("analyze_query", analyze_query)
    graph.add_node("ask_questions", ask_follow_up_questions)
    graph.add_node("analyze_answers", analyze_answers)
    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("recommend_products", recommend_products)

    graph.set_entry_point("analyze_query")

    # def should_ask_questions(state: ConversationState):
    #     if state.is_follow_up:
    #         return "ask_questions"
    #     # If no follow up, go straight to retrieve docs
    #     return "retrieve_documents"
    def should_ask_questions(state: ConversationState):
        if state.ready_for_recommendation or state.follow_up_count >= 2:
            return "retrieve_documents"
        if state.is_follow_up:
            state.follow_up_count += 1  # Increment follow-up count
            return "ask_questions"
        return "retrieve_documents"


    graph.add_conditional_edges(
        "analyze_query",
        should_ask_questions,
        {"ask_questions": "ask_questions", "retrieve_documents": "retrieve_documents"}
    )
    
    graph.add_edge("ask_questions", "analyze_answers")

    def decide_after_answers(state: ConversationState):
        if state.ready_for_recommendation or state.follow_up_count >= 2:
            return "retrieve_documents"
        # Not enough info, end the conversation for now.
        # The user can continue the conversation, which will trigger a new run.
        return END
        
    graph.add_conditional_edges(
        "analyze_answers",
        decide_after_answers,
        {"retrieve_documents": "retrieve_documents", END: END}
    )
    
    graph.add_edge("retrieve_documents", "recommend_products")
    graph.add_edge("recommend_products", END)

    logger.info("StateGraph pipeline compiled successfully.")
    return graph.compile()
