"""Text chunking for large documents."""

from typing import List, Optional
import re

from financial_analyzer.utils.logger import logger
from financial_analyzer.utils.errors import ChunkingError


class Chunker:
    """Splits documents into overlapping chunks."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        separator: str = "\n\n",
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Number of characters per chunk
            chunk_overlap: Number of overlapping characters between chunks
            separator: Text separator for semantic boundaries
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.logger = logger

    def chunk_text(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> List[dict]:
        """
        Split text into overlapping chunks with metadata.

        Args:
            text: Text to chunk
            metadata: Document metadata to attach to each chunk

        Returns:
            List of chunk dictionaries with content and metadata

        Raises:
            ChunkingError: If chunking fails
        """
        try:
            if not text or len(text.strip()) == 0:
                self.logger.warning("Empty text provided for chunking")
                return []

            metadata = metadata or {}
            chunks = []

            # Split by primary separator first
            paragraphs = text.split(self.separator)

            # Recursively split paragraphs if they exceed chunk size
            split_paragraphs = []
            for para in paragraphs:
                if len(para) > self.chunk_size:
                    # Further split by sentences
                    sentences = self._split_into_sentences(para)
                    split_paragraphs.extend(sentences)
                else:
                    split_paragraphs.append(para)

            # Combine paragraphs into chunks
            current_chunk = ""
            for para in split_paragraphs:
                if len(current_chunk) + len(para) <= self.chunk_size:
                    # Add paragraph to current chunk
                    if current_chunk:
                        current_chunk += self.separator
                    current_chunk += para
                else:
                    # Save current chunk and start new one
                    if current_chunk:
                        chunks.append(current_chunk)
                    # Start new chunk with overlap
                    current_chunk = para

            # Don't forget the last chunk
            if current_chunk:
                chunks.append(current_chunk)

            # Add overlap between chunks
            overlapped_chunks = self._add_overlap(chunks)

            # Convert to list of dicts with metadata
            result = []
            for chunk_idx, chunk_content in enumerate(overlapped_chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_index": chunk_idx,
                    "total_chunks": len(overlapped_chunks),
                    "chunk_size": len(chunk_content),
                })
                result.append({
                    "content": chunk_content,
                    "metadata": chunk_metadata,
                })

            self.logger.info(
                f"Split text into {len(result)} chunks "
                f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
            )
            return result

        except Exception as e:
            self.logger.error(f"Error chunking text: {str(e)}")
            raise ChunkingError(f"Failed to chunk text: {str(e)}") from e

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Simple sentence splitting by common punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """
        Add overlap between chunks.

        Args:
            chunks: List of chunks

        Returns:
            List of chunks with overlap
        """
        if len(chunks) <= 1:
            return chunks

        overlapped = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk as is
                overlapped.append(chunk)
            else:
                # Add overlap from previous chunk
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
                combined = overlap_text + self.separator + chunk
                overlapped.append(combined)

        return overlapped

    def estimate_tokens(self, text: str) -> int:
        """
        Rough estimate of token count (1 token ≈ 4 characters).

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        return len(text) // 4


class SemanticChunker(Chunker):
    """Enhanced chunker with semantic awareness (section-based)."""

    SECTION_MARKERS = [
        r"^#{1,6}\s+",  # Markdown headers
        r"^[A-Z][A-Z\s]{3,}$",  # All caps titles
        r"^\d+\.\s+[A-Z]",  # Numbered sections
    ]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ):
        """Initialize semantic chunker."""
        super().__init__(chunk_size, chunk_overlap, separator="\n\n")
        self.section_markers = self.SECTION_MARKERS

    def find_sections(self, text: str) -> List[tuple]:
        """
        Find section boundaries in text.

        Args:
            text: Text to analyze

        Returns:
            List of (start_pos, end_pos, section_title) tuples
        """
        sections = []
        lines = text.split("\n")
        current_section = 0

        for line_num, line in enumerate(lines):
            for marker in self.section_markers:
                if re.match(marker, line):
                    sections.append((current_section, line_num, line))
                    current_section = line_num

        if current_section < len(lines):
            sections.append((current_section, len(lines), ""))

        return sections

    def chunk_by_sections(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> List[dict]:
        """
        Chunk text while respecting section boundaries.

        Args:
            text: Text to chunk
            metadata: Document metadata

        Returns:
            List of chunk dictionaries
        """
        # Use parent chunking for now
        # Can be enhanced to detect and preserve sections
        return self.chunk_text(text, metadata)
