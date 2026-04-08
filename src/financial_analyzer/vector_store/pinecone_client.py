"""Pinecone vector store client."""

from typing import List, Dict, Optional, Tuple
import os
from datetime import datetime

from pinecone import Pinecone, ServerlessSpec

from financial_analyzer.config import settings
from financial_analyzer.utils.logger import logger
from financial_analyzer.utils.errors import PineconeAPIError


class PineconeClient:
    """Manages Pinecone vector store operations."""

    def __init__(self):
        """Initialize Pinecone client."""
        self.logger = logger
        self.config = settings.get_pinecone_config()
        self.pc = None
        self.index = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize Pinecone client and index."""
        try:
            self.logger.info("Initializing Pinecone client...")

            # Initialize Pinecone with API key and environment
            self.pc = Pinecone(api_key=self.config["api_key"])

            self.logger.info(f"Connecting to index: {self.config['index_name']}")

            # Connect to existing index (use lowercase index() method)
            self.index = self.pc.index(self.config["index_name"])

            # Test connection
            index_stats = self.index.describe_index_stats()
            self.logger.info(
                f"Connected to Pinecone index '{self.config['index_name']}' "
                f"with {index_stats.total_vector_count} vectors"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise PineconeAPIError(f"Pinecone initialization failed: {str(e)}") from e

    def create_index(
        self,
        index_name: str,
        dimension: int = 3072,  # Gemini embedding dimension
        metric: str = "cosine",
    ) -> None:
        """
        Create a new Pinecone index.

        Args:
            index_name: Name of the index
            dimension: Vector dimension (default: 3072 for Gemini embeddings)
            metric: Similarity metric ('cosine', 'euclidean', 'dotproduct')

        Raises:
            PineconeAPIError: If index creation fails
        """
        try:
            self.logger.info(f"Creating index: {index_name} (dimension={dimension})")

            # Create serverless index (free tier)
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

            self.logger.info(f"Index created successfully: {index_name}")

        except Exception as e:
            if "already exists" in str(e):
                self.logger.warning(f"Index already exists: {index_name}")
            else:
                self.logger.error(f"Failed to create index: {str(e)}")
                raise PineconeAPIError(f"Failed to create index: {str(e)}") from e

    def delete_index(self, index_name: str) -> None:
        """
        Delete a Pinecone index.

        Args:
            index_name: Name of the index

        Raises:
            PineconeAPIError: If deletion fails
        """
        try:
            self.logger.info(f"Deleting index: {index_name}")
            self.pc.delete_index(index_name)
            self.logger.info(f"Index deleted: {index_name}")

        except Exception as e:
            self.logger.error(f"Failed to delete index: {str(e)}")
            raise PineconeAPIError(f"Failed to delete index: {str(e)}") from e

    def upsert_vectors(
        self,
        vectors: List[Tuple[str, List[float], Dict]],
        namespace: Optional[str] = None,
    ) -> None:
        """
        Upsert vectors to Pinecone.

        Args:
            vectors: List of (id, embedding, metadata) tuples
            namespace: Optional namespace for partitioning

        Raises:
            PineconeAPIError: If upsert fails
        """
        try:
            if not vectors:
                self.logger.warning("No vectors to upsert")
                return

            namespace = namespace or self.config["namespace"]

            # Batch upsert for efficiency
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i : i + batch_size]
                self.index.upsert(vectors=batch, namespace=namespace)

            self.logger.info(
                f"Upserted {len(vectors)} vectors to namespace '{namespace}'"
            )

        except Exception as e:
            self.logger.error(f"Failed to upsert vectors: {str(e)}")
            raise PineconeAPIError(f"Failed to upsert vectors: {str(e)}") from e

    def query_vectors(
        self,
        embedding: List[float],
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Query Pinecone for similar vectors.

        Args:
            embedding: Query embedding
            top_k: Number of results to return
            namespace: Optional namespace to search
            filter: Optional metadata filter

        Returns:
            List of result dictionaries with id, score, and metadata

        Raises:
            PineconeAPIError: If query fails
        """
        try:
            namespace = namespace or self.config["namespace"]

            results = self.index.query(
                vector=embedding,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
                filter=filter,
            )

            # Format results
            formatted_results = []
            for match in results.matches:
                formatted_results.append({
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata or {},
                })

            self.logger.debug(f"Query returned {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            self.logger.error(f"Failed to query vectors: {str(e)}")
            raise PineconeAPIError(f"Failed to query vectors: {str(e)}") from e

    def delete_vectors(
        self,
        vector_ids: List[str],
        namespace: Optional[str] = None,
    ) -> None:
        """
        Delete vectors from Pinecone.

        Args:
            vector_ids: List of vector IDs to delete
            namespace: Optional namespace

        Raises:
            PineconeAPIError: If deletion fails
        """
        try:
            namespace = namespace or self.config["namespace"]

            self.index.delete(ids=vector_ids, namespace=namespace)

            self.logger.info(f"Deleted {len(vector_ids)} vectors from namespace '{namespace}'")

        except Exception as e:
            self.logger.error(f"Failed to delete vectors: {str(e)}")
            raise PineconeAPIError(f"Failed to delete vectors: {str(e)}") from e

    def clear_namespace(self, namespace: Optional[str] = None) -> None:
        """
        Clear all vectors from a namespace.

        Args:
            namespace: Namespace to clear

        Raises:
            PineconeAPIError: If clear fails
        """
        try:
            namespace = namespace or self.config["namespace"]

            self.index.delete(delete_all=True, namespace=namespace)

            self.logger.info(f"Cleared namespace: {namespace}")

        except Exception as e:
            self.logger.error(f"Failed to clear namespace: {str(e)}")
            raise PineconeAPIError(f"Failed to clear namespace: {str(e)}") from e

    def get_index_stats(self) -> Dict:
        """
        Get index statistics.

        Returns:
            Dictionary with index stats

        Raises:
            PineconeAPIError: If stats retrieval fails
        """
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "namespaces": {
                    ns: {
                        "vector_count": info.vector_count,
                    }
                    for ns, info in stats.namespaces.items()
                },
            }

        except Exception as e:
            self.logger.error(f"Failed to get index stats: {str(e)}")
            raise PineconeAPIError(f"Failed to get index stats: {str(e)}") from e


def initialize_index() -> None:
    """
    Initialize Pinecone index (standalone function).

    Creates index if it doesn't exist, otherwise confirms connection.
    """
    try:
        logger.info("Initializing Pinecone index...")
        pc = Pinecone(api_key=settings.pinecone_api_key)

        index_name = settings.pinecone_index_name

        # Check if index exists
        try:
            pc.describe_index(index_name)
            logger.info(f"Index already exists: {index_name}")
        except:
            # Create index
            logger.info(f"Creating new index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=3072,  # Gemini embedding dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            logger.info(f"Index created: {index_name}")

    except Exception as e:
        logger.error(f"Failed to initialize index: {str(e)}")
        raise PineconeAPIError(f"Failed to initialize index: {str(e)}") from e
