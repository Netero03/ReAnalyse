"""Configuration management for Financial Analyzer."""

import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseSettings, Field, validator

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Gemini API Configuration
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    embedding_model: str = Field(default="models/embedding-001", env="EMBEDDING_MODEL")
    model_name: str = Field(default="gemini-1.5-flash", env="MODEL_NAME")

    # Pinecone Configuration
    pinecone_api_key: str = Field(..., env="PINECONE_API_KEY")
    pinecone_environment: str = Field(..., env="PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field(
        default="financial-documents", env="PINECONE_INDEX_NAME"
    )
    pinecone_namespace: str = Field(default="default", env="PINECONE_NAMESPACE")
    pinecone_metric: str = Field(default="cosine", env="PINECONE_METRIC")

    # Chunking Configuration
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, env="CHUNK_OVERLAP")

    # Retrieval Configuration
    max_retrieved_chunks: int = Field(default=5, env="MAX_RETRIEVED_CHUNKS")
    retrieval_threshold: float = Field(default=0.5, env="RETRIEVAL_THRESHOLD")

    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Streamlit Configuration
    streamlit_server_headless: bool = Field(default=True, env="STREAMLIT_SERVER_HEADLESS")
    streamlit_server_port: int = Field(default=8501, env="STREAMLIT_SERVER_PORT")

    # Application Configuration
    debug_mode: bool = Field(default=False, env="DEBUG_MODE")
    max_file_size_mb: int = Field(default=200, env="MAX_FILE_SIZE_MB")
    allowed_file_types: tuple = Field(default=("pdf",), env="ALLOWED_FILE_TYPES")

    @validator("google_api_key", pre=True, always=True)
    def validate_google_api_key(cls, v: Optional[str]) -> str:
        """Validate that Gemini API key is provided."""
        if not v or v == "your_gemini_api_key_here":
            raise ValueError(
                "GOOGLE_API_KEY not configured. Set it in .env or environment variables."
            )
        return v

    @validator("pinecone_api_key", pre=True, always=True)
    def validate_pinecone_api_key(cls, v: Optional[str]) -> str:
        """Validate that Pinecone API key is provided."""
        if not v or v == "your_pinecone_api_key_here":
            raise ValueError(
                "PINECONE_API_KEY not configured. Set it in .env or environment variables."
            )
        return v

    @validator("pinecone_environment", pre=True, always=True)
    def validate_pinecone_environment(cls, v: Optional[str]) -> str:
        """Validate that Pinecone environment is provided."""
        if not v or v == "your_pinecone_environment_here":
            raise ValueError(
                "PINECONE_ENVIRONMENT not configured. Set it in .env or environment variables."
            )
        return v

    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of {valid_levels}")
        return v.upper()

    @validator("chunk_size")
    def validate_chunk_size(cls, v: int) -> int:
        """Validate chunk size."""
        if v < 100:
            raise ValueError("Chunk size must be at least 100 characters")
        if v > 5000:
            raise ValueError("Chunk size must not exceed 5000 characters")
        return v

    @validator("chunk_overlap")
    def validate_chunk_overlap(cls, v: int, values) -> int:
        """Validate chunk overlap."""
        if v < 0:
            raise ValueError("Chunk overlap cannot be negative")
        if "chunk_size" in values and v >= values["chunk_size"]:
            raise ValueError("Chunk overlap must be less than chunk size")
        return v

    @validator("max_retrieved_chunks")
    def validate_max_retrieved_chunks(cls, v: int) -> int:
        """Validate max retrieved chunks."""
        if v < 1 or v > 20:
            raise ValueError("Max retrieved chunks must be between 1 and 20")
        return v

    class Config:
        """Pydantic config."""

        env_file = ".env"
        case_sensitive = False

    def get_gemini_config(self) -> dict:
        """Get Gemini API configuration."""
        return {
            "api_key": self.google_api_key,
            "model": self.model_name,
            "embedding_model": self.embedding_model,
        }

    def get_pinecone_config(self) -> dict:
        """Get Pinecone configuration."""
        return {
            "api_key": self.pinecone_api_key,
            "environment": self.pinecone_environment,
            "index_name": self.pinecone_index_name,
            "namespace": self.pinecone_namespace,
            "metric": self.pinecone_metric,
        }


# Global settings instance
try:
    settings = Settings()
except Exception as e:
    raise RuntimeError(f"Failed to load configuration: {str(e)}") from e


def get_settings() -> Settings:
    """Get application settings."""
    return settings
