from pydantic_settings import BaseSettings
from loguru import logger


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    OPENAI_API_KEY: str
    REDIS_URL: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


try:
    settings = Settings()
    logger.info("Application settings loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load application settings: {e}")

