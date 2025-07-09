from app.dependencies import openai_client
from loguru import logger


# Function to generate embedding for a given text using OpenAI API
def generate_embedding(text: str) -> list:
    """
    Generate an embedding vector for the provided text using OpenAI's API.

    Args:
        text (str): The input text to generate embedding for.

    Returns:
        list: The embedding vector as a list of floats.

    Raises:
        Exception: If the embedding generation fails.
    """
    try:
        logger.debug("Generating embedding for text: '{}'", text)
        # Call OpenAI API to generate embedding
        response = openai_client.embeddings.create(
            input=text, model="text-embedding-ada-002"
        )
        logger.info("Embedding generated successfully.")
        return response.data[0].embedding
    except Exception as e:
        logger.error("Failed to generate embedding: {}", str(e), exc_info=True)
        raise
