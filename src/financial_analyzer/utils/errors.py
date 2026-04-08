"""Custom exception classes for Financial Analyzer."""


class FinancialAnalyzerError(Exception):
    """Base exception for Financial Analyzer."""

    pass


class ConfigError(FinancialAnalyzerError):
    """Raised when configuration is invalid."""

    pass


class APIError(FinancialAnalyzerError):
    """Raised when API calls fail."""

    pass


class GeminiAPIError(APIError):
    """Raised when Gemini API calls fail."""

    pass


class PineconeAPIError(APIError):
    """Raised when Pinecone API calls fail."""

    pass


class VectorStoreError(FinancialAnalyzerError):
    """Raised when vector store operations fail."""

    pass


class DocumentProcessingError(FinancialAnalyzerError):
    """Raised when document processing fails."""

    pass


class EmbeddingError(FinancialAnalyzerError):
    """Raised when embedding generation fails."""

    pass


class RetrievalError(FinancialAnalyzerError):
    """Raised when document retrieval fails."""

    pass


class ChunkingError(DocumentProcessingError):
    """Raised when document chunking fails."""

    pass


class PDFLoadingError(DocumentProcessingError):
    """Raised when PDF loading fails."""

    pass


class RAGChainError(FinancialAnalyzerError):
    """Raised when RAG chain operations fail."""

    pass
