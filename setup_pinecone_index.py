#!/usr/bin/env python3
"""Script to recreate Pinecone index with correct dimensions for Gemini embeddings."""

from pinecone import Pinecone
from financial_analyzer.config import settings
from financial_analyzer.utils.logger import logger

def recreate_pinecone_index():
    """Recreate Pinecone index with correct dimensions for Gemini embeddings."""
    
    config = settings.get_pinecone_config()
    pc = Pinecone(api_key=config["api_key"])
    
    index_name = config["index_name"]
    
    # Check if index exists
    logger.info(f"Checking if index '{index_name}' exists...")
    
    try:
        # List all indexes
        indexes = pc.list_indexes()
        index_names = [idx.name for idx in indexes]
        
        if index_name in index_names:
            logger.info(f"Found existing index '{index_name}'. Deleting...")
            pc.delete_index(index_name)
            logger.info(f"Index '{index_name}' deleted successfully.")
        
        # Create new index with correct dimensions (3072 for Gemini embeddings)
        logger.info(f"Creating new index '{index_name}' with dimension=3072...")
        from pinecone import ServerlessSpec
        
        pc.create_index(
            name=index_name,
            dimension=3072,  # Gemini embedding-2-preview produces 3072 dimensions
            metric=config.get("metric", "cosine"),
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        
        logger.info(f"Index '{index_name}' created successfully with dimension=3072!")
        logger.info("You can now upload PDFs in the Streamlit app.")
        
    except Exception as e:
        logger.error(f"Error recreating index: {str(e)}")
        raise

if __name__ == "__main__":
    recreate_pinecone_index()
