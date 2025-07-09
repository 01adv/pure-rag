import json
from uuid import uuid4
from loguru import logger
from app.dependencies import redis_client

SESSION_EXPIRATION_SECONDS = 3600  # 1 hour


def get_or_create_session(session_id: str | None) -> tuple[str, list]:
    """
    Retrieves an existing session history from Redis or creates a new one.

    Args:
        session_id (str | None): The session ID from the request.

    Returns:
        tuple[str, list]: A tuple containing the session ID and the conversation history.
    """
    if session_id is None:
        session_id = str(uuid4())
        logger.info(f"No session ID provided. Created new session: {session_id}")
        return session_id, []

    try:
        session_data = redis_client.get(session_id)
        if session_data:
            logger.info(f"Retrieved existing session: {session_id}")
            conversation_history = json.loads(session_data)
            return session_id, conversation_history
        else:
            logger.info(f"No data for session {session_id}. Starting new conversation.")
            return session_id, []
    except Exception as e:
        logger.error(f"Error retrieving session {session_id} from Redis: {e}")
        # Fallback to a new session to avoid crashing
        return session_id, []


def update_session(session_id: str, conversation_history: list):
    """
    Updates a session in Redis with the latest conversation history.

    Args:
        session_id (str): The session ID.
        conversation_history (list): The full conversation history to save.
    """
    try:
        session_data = json.dumps(conversation_history)
        redis_client.set(session_id, session_data, ex=SESSION_EXPIRATION_SECONDS)
        logger.info(f"Successfully updated session: {session_id}")
    except Exception as e:
        logger.error(f"Error updating session {session_id} in Redis: {e}")

