"""Gemini API embeddings wrapper."""

from typing import List, Optional

import google.generativeai as genai

from financial_analyzer.config import settings
from financial_analyzer.utils.logger import logger
from financial_analyzer.utils.errors import EmbeddingError


class GeminiEmbedder:
    """Generates embeddings using Gemini API."""

    # Supported task types for Gemini embeddings
    TASK_TYPES = {
        "retrieval_document": "RETRIEVAL_DOCUMENT",
        "retrieval_query": "RETRIEVAL_QUERY",
        "question_answering": "QUESTION_ANSWERING",
        "fact_verification": "FACT_VERIFICATION",
        "semantic_similarity": "SEMANTIC_SIMILARITY",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        task_type: str = "question_answering",
    ):
        """
        Initialize Gemini embedder.

        Args:
            api_key: Gemini API key (default: from settings)
            model: Model to use (default: from settings)
            task_type: Task type for embedding optimization
        """
        self.api_key = api_key or settings.google_api_key
        self.model = model or settings.embedding_model
        self.task_type = self.TASK_TYPES.get(task_type, "RETRIEVAL_DOCUMENT")
        self.logger = logger

        # Configure Gemini API
        genai.configure(api_key=self.api_key)

    def embed_text(
        self,
        text: str,
        task_type: Optional[str] = None,
    ) -> List[float]:
        """
        Embed a single text string.

        Args:
            text: Text to embed
            task_type: Optional override for task type

        Returns:
            Embedding vector (list of floats)

        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            if not text or len(text.strip()) == 0:
                raise ValueError("Cannot embed empty text")

            task = task_type or self.task_type

            self.logger.debug(f"Embedding text ({len(text)} chars, task={task})")

            response = genai.embed_content(
                model=self.model,
                content=text,
                task_type=task,
                title="Financial document chunk",
            )

            embedding = response["embedding"]

            self.logger.debug(f"Generated embedding with {len(embedding)} dimensions")
            return embedding

        except Exception as e:
            self.logger.error(f"Failed to embed text: {str(e)}")
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}") from e

    def embed_batch(
        self,
        texts: List[str],
        task_type: Optional[str] = None,
        batch_size: int = 100,
    ) -> List[List[float]]:
        """
        Embed multiple texts efficiently.

        Args:
            texts: List of texts to embed
            task_type: Optional override for task type
            batch_size: Batch size for API calls

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            if not texts:
                self.logger.warning("No texts to embed")
                return []

            embeddings = []
            task = task_type or self.task_type

            self.logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}")

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                try:
                    response = genai.embed_content(
                        model=self.model,
                        content=batch,
                        task_type=task,
                        title="Financial document chunks",
                    )

                    # Handle batch response
                    if isinstance(response.get("embedding"), list):
                        if isinstance(response["embedding"][0], float):
                            # Single embedding returned
                            embeddings.append(response["embedding"])
                        else:
                            # Multiple embeddings returned
                            embeddings.extend(response["embedding"])
                    else:
                        embeddings.append(response["embedding"])

                except Exception as batch_error:
                    self.logger.warning(f"Batch {i // batch_size} failed, falling back to individual: {batch_error}")
                    # Fall back to individual embeddings
                    for text in batch:
                        try:
                            emb = self.embed_text(text, task_type)
                            embeddings.append(emb)
                        except Exception as e:
                            self.logger.error(f"Failed to embed individual text: {str(e)}")
                            raise

            self.logger.info(f"Successfully embedded {len(embeddings)} texts")
            return embeddings

        except Exception as e:
            self.logger.error(f"Batch embedding failed: {str(e)}")
            raise EmbeddingError(f"Failed to generate batch embeddings: {str(e)}") from e

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query string (asymmetric - optimized for queries).

        Args:
            query: Query text to embed

        Returns:
            Embedding vector

        Raises:
            EmbeddingError: If embedding fails
        """
        return self.embed_text(query, task_type=self.TASK_TYPES["retrieval_query"])

    def embed_document(self, document: str) -> List[float]:
        """
        Embed a document string (asymmetric - optimized for documents).

        Args:
            document: Document text to embed

        Returns:
            Embedding vector

        Raises:
            EmbeddingError: If embedding fails
        """
        return self.embed_text(document, task_type=self.TASK_TYPES["retrieval_document"])

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings from this model.

        Returns:
            Dimension size (typically 3072 for Gemini)
        """
        # Gemini embedding-001 produces 768-dimensional embeddings by default
        # But can be configured. For now, return standard dimension
        return 768  # Default for Gemini embedding-001
