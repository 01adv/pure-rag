import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from .routers import search
from scripts.ingest_data import main as ingest_main
# Create FastAPI app
app = FastAPI(
    title="Conversational Search API - V2",
    description="An improved backend for conversational search and recommendations using Redis, decoupled ingestion, and externalized prompts.",
    version="2.0.0",
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # IMPORTANT: Restrict this in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """
#     Application lifespan context manager.
#     Initializes cache and ingests skincare data at startup.
#     Cleans up resources at shutdown.
#     """
#     logger.info("Initializing application lifespan.")
#     # Initialize in-memory cache for the app
#     # app.state.cache = TTLCache(maxsize=1000, ttl=3000)

#     # Ingest skincare data at startup
#     try:
#         ingest_main()
#         logger.success("Skincare data ingested successfully.")
#     except Exception as e:
#         logger.error(f"Error during skincare data ingestion: {e}")

#     yield  # Application runs here

#     logger.info("Application shutdown completed.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing application lifespan.")
    try:
        # Run ingestion in a separate thread to avoid blocking
        await asyncio.to_thread(ingest_main)
        logger.success("Skincare data ingested successfully.")
    except Exception as e:
        logger.error(f"Error during skincare data ingestion: {e}")
        # Optionally raise to prevent server startup if ingestion is critical
        # raise e

    yield
    logger.info("Application shutdown completed.")

# Include API routers
app.include_router(search.router)

@app.post("/ingest-data")
async def ingest_data_endpoint():
    ingest_main()  # Use ingest_main instead of ingest_data
    return {"message": "Data ingestion completed"}


@app.get("/", tags=["Status"])
async def read_root():
    """
    Root endpoint to check API status.
    """
    logger.info("Root endpoint accessed.")
    return {"status": "ok hello", "version": app.version}

