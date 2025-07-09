from openai import OpenAI
import chromadb
from chromadb.config import Settings as ChromaSettings
# from upstash_redis import Redis, UpstashError
from upstash_redis import Redis
from loguru import logger
from .config import settings

# -----------------------------------------------------------
# Initialize OpenAI client
# -----------------------------------------------------------
try:
    openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    logger.info("OpenAI client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")

# -----------------------------------------------------------
# Initialize ChromaDB client
# -----------------------------------------------------------
try:
    chroma_client = chromadb.PersistentClient(
        path="./chroma_db", settings=ChromaSettings(anonymized_telemetry=False)
    )
    logger.info("ChromaDB PersistentClient initialized at './chroma_db'.")
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB PersistentClient: {e}")

# -----------------------------------------------------------
# Initialize Redis client
# -----------------------------------------------------------
# try:
#     redis_client = Redis.from_url(settings.REDIS_URL)
#     redis_client.ping()
#     logger.info("Redis client initialized and connected successfully.")
# # except error as e:
# #     logger.error(f"Failed to connect to Upstash Redis: {e}")
# except Exception as e:
#     logger.error(f"Failed to initialize Redis client: {e}")


try:
    # redis_client = Redis(url=settings.REDIS_URL)
    redis_client = Redis(url="https://big-wasp-38128.upstash.io", token="AZTwAAIjcDEyYTdlOGM2ZmQ2NmI0NzYyODg0NTNiMDc2MDQ3OTYxM3AxMA")
    redis_client.ping()
    logger.info("Redis client initialized and connected successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Redis client: {e}")