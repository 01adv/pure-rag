from fastapi import APIRouter, HTTPException
from app.models.schemas import SearchRequest, ConversationState
from app.services.session import get_or_create_session, update_session
from app.utils.pipeline import build_graph
from loguru import logger

router = APIRouter(prefix="/api", tags=["search"])


@router.post("/search")
async def search(data: SearchRequest):
    """
    Handles search requests, uses Redis for session management, and invokes the graph pipeline.
    """
    try:
        # 1. Get or create the session and conversation history from Redis
        session_id, conversation_history = get_or_create_session(data.session_id)
        logger.info(f"Received search query: '{data.query}' | session_id: {session_id}")

        # 2. Append the user's new query to the history
        conversation_history.append({"role": "user", "content": data.query})

        # 3. Build and invoke the conversational pipeline
        product_pipeline = build_graph()
        logger.debug("Invoking product pipeline with current conversation state.")
        result = product_pipeline.invoke(
            ConversationState(conversation=conversation_history, query=data.query)
        )

        # 4. Append the assistant's response to the history
        follow_up = result.get("follow_up_question")
        if follow_up:
            conversation_history.append({"role": "assistant", "content": follow_up})
            logger.debug(f"Appended assistant response to cache: {follow_up}")

        # 5. Save the updated conversation back to Redis
        update_session(session_id, conversation_history)

        # Add session_id to the response
        result["session_id"] = session_id

        logger.success("Search query processed successfully.")
        return result

    except Exception as e:
        logger.error(f"Error processing search query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
