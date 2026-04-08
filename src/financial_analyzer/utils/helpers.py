"""Helper utilities for Financial Analyzer."""

import os
import hashlib
from typing import List, Optional
from datetime import datetime
from pathlib import Path

import json


def calculate_file_hash(file_path: Path) -> str:
    """
    Calculate SHA256 hash of a file.

    Args:
        file_path: Path to file

    Returns:
        Hex digest of file hash
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def format_bytes(bytes_size: int) -> str:
    """
    Format bytes to human-readable string.

    Args:
        bytes_size: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to remove invalid characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")
    return filename


def generate_document_id(source: str, timestamp: Optional[str] = None) -> str:
    """
    Generate unique document ID.

    Args:
        source: Document source/filename
        timestamp: Optional timestamp (default: current time)

    Returns:
        Generated document ID
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat()

    sanitized = sanitize_filename(source)
    return f"{sanitized}_{timestamp}"


def chunk_text_by_words(text: str, chunk_size: int = 500) -> List[str]:
    """
    Split text into chunks by word count.

    Args:
        text: Text to split
        chunk_size: Target words per chunk

    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def chunk_text_by_paragraphs(text: str, max_size: int = 5000) -> List[str]:
    """
    Split text into chunks by paragraphs, respecting size limit.

    Args:
        text: Text to split
        max_size: Maximum characters per chunk

    Returns:
        List of text chunks
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) <= max_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def estimate_tokens(text: str, model: str = "gemini") -> int:
    """
    Estimate token count for text.

    Args:
        text: Text to estimate
        model: Model type (affects token estimation)

    Returns:
        Estimated token count
    """
    # Rough estimation: Gemini ~4 characters per token
    if model == "gemini":
        return len(text) // 4
    else:
        # Default conservative estimate
        return len(text) // 3


def estimate_cost(token_count: int, rate_per_million: float = 0.05) -> float:
    """
    Estimate API cost for tokens.

    Args:
        token_count: Number of tokens
        rate_per_million: Cost per million tokens

    Returns:
        Estimated cost in dollars
    """
    return (token_count / 1_000_000) * rate_per_million


def parse_csv_to_dict(csv_content: str) -> List[dict]:
    """
    Parse CSV content to list of dictionaries.

    Args:
        csv_content: CSV content as string

    Returns:
        List of row dictionaries
    """
    import csv
    import io

    rows = []
    reader = csv.DictReader(io.StringIO(csv_content))
    for row in reader:
        rows.append(row)
    return rows


def validate_api_key(api_key: str, min_length: int = 20) -> bool:
    """
    Validate API key format.

    Args:
        api_key: API key to validate
        min_length: Minimum key length

    Returns:
        True if valid, False otherwise
    """
    return bool(api_key and len(api_key) >= min_length)


def create_metadata_dict(**kwargs) -> dict:
    """
    Create standardized metadata dictionary.

    Args:
        **kwargs: Metadata key-value pairs

    Returns:
        Metadata dictionary with standard fields
    """
    metadata = {
        "created_at": datetime.now().isoformat(),
        **kwargs,
    }
    return metadata


def merge_metadata(*metadata_dicts: dict) -> dict:
    """
    Merge multiple metadata dictionaries.

    Args:
        *metadata_dicts: Metadata dictionaries to merge

    Returns:
        Merged metadata dictionary
    """
    result = {}
    for metadata in metadata_dicts:
        if isinstance(metadata, dict):
            result.update(metadata)
    return result
