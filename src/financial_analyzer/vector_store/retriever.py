"""Document retrieval from Pinecone."""

from typing import List, Dict, Optional

from financial_analyzer.utils.logger import logger
from financial_analyzer.utils.errors import RetrievalError
from financial_analyzer.vector_store.pinecone_client import PineconeClient
from financial_analyzer.config import settings


class Retriever:
    """Retrieves relevant documents from Pinecone."""

    def __init__(self, pinecone_client: Optional[PineconeClient] = None):
        """
        Initialize retriever.

        Args:
            pinecone_client: Pinecone client instance (creates new if not provided)
        """
        self.client = pinecone_client or PineconeClient()
        self.logger = logger
        self.max_results = settings.max_retrieved_chunks

    def retrieve(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        namespace: Optional[str] = None,
        filter: Optional[Dict] = None,
        threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Retrieve relevant documents.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return (default: from settings)
            namespace: Optional namespace to search
            filter: Optional metadata filter (e.g., document_id)
            threshold: Minimum similarity score (default: from settings)

        Returns:
            List of retrieved documents with metadata

        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            top_k = top_k or self.max_results
            threshold = threshold or settings.retrieval_threshold

            # Query Pinecone
            results = self.client.query_vectors(
                embedding=query_embedding,
                top_k=top_k,
                namespace=namespace,
                filter=filter,
            )

            # Filter by score threshold
            if threshold > 0:
                results = [r for r in results if r["score"] >= threshold]

            self.logger.info(
                f"Retrieved {len(results)} documents "
                f"(top_k={top_k}, threshold={threshold})"
            )

            return results

        except Exception as e:
            self.logger.error(f"Retrieval failed: {str(e)}")
            raise RetrievalError(f"Failed to retrieve documents: {str(e)}") from e

    def retrieve_with_source(
        self,
        query_embedding: List[float],
        source: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Retrieve documents from a specific source.

        Args:
            query_embedding: Query embedding vector
            source: Document source/filename to filter by
            top_k: Number of results

        Returns:
            List of retrieved documents
        """
        filter_dict = None
        if source:
            filter_dict = {"source": {"$eq": source}}

        return self.retrieve(
            query_embedding=query_embedding,
            top_k=top_k,
            filter=filter_dict,
        )

    def get_document_stats(self, namespace: Optional[str] = None) -> Dict:
        """
        Get statistics about indexed documents.

        Args:
            namespace: Optional namespace to get stats for

        Returns:
            Dictionary with document statistics
        """
        try:
            stats = self.client.get_index_stats()
            return stats
        except Exception as e:
            self.logger.error(f"Failed to get stats: {str(e)}")
            raise RetrievalError(f"Failed to get document stats: {str(e)}") from e

    def delete_by_source(
        self,
        source: str,
        namespace: Optional[str] = None,
    ) -> int:
        """
        Delete all chunks from a specific source document.

        Note: Pinecone doesn't support bulk delete by metadata filter directly.
        This would require storing vector IDs by source or using a workaround.

        Args:
            source: Document source/filename
            namespace: Optional namespace

        Returns:
            Number of vectors deleted (approximate)
        """
        # This is a limitation of Pinecone free tier
        # In production, store document ID mappings
        self.logger.warning(
            "Bulk delete by source not directly supported in Pinecone. "
            "Consider maintaining a separate ID registry."
        )
        return 0
