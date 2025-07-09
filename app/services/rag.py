from app.services.embedding import generate_embedding
from loguru import logger
from app.dependencies import chroma_client


def add_document(collection, text: str, metadata: dict):
    """
    Adds a document to the specified collection with its embedding and metadata.

    Args:
        collection: The ChromaDB collection object.
        text (str): The document text to embed and store.
        metadata (dict): Metadata associated with the document (must include 'id').
    """
    try:
        logger.debug("Generating embedding for new document.")
        embedding = generate_embedding(text)
        logger.debug(f"Adding document with ID: {metadata.get('id')} to collection.")
        collection.add(
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[text],
            ids=[metadata["id"]],
        )
        logger.info(f"Added document with ID: {metadata['id']}")
    except Exception:
        logger.exception(f"Failed to add document with ID: {metadata.get('id')}")
        raise


def query_collection(collection_name: str, query_text: str, n_results: int = 5):
    """
    Queries the specified collection for documents similar to the query text.

    Args:
        collection_name (str): The name of the collection to query.
        query_text (str): The text to query against the collection.
        n_results (int): Number of top results to retrieve.

    Returns:
        dict: Query results from the collection.
    """
    try:
        logger.debug(f"Retrieving '{collection_name}' collection from Chroma client.")
        collection_instance = chroma_client.get_collection(name=collection_name)
        logger.debug("Generating embedding for query text.")
        query_embedding = generate_embedding(query_text)
        logger.debug(f"Querying collection for top {n_results} results.")
        results = collection_instance.query(
            query_embeddings=[query_embedding], n_results=n_results
        )
        logger.info(
            f"Retrieved {len(results['documents'][0])} documents for query: '{query_text}'"
        )
        return results
    except Exception:
        logger.exception(f"Failed to query collection '{collection_name}' for text: '{query_text}'")
        raise
