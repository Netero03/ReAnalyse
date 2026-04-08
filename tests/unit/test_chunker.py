"""Unit tests for text chunking."""

import pytest
from financial_analyzer.ingest.chunker import Chunker, SemanticChunker
from financial_analyzer.utils.errors import ChunkingError


class TestChunker:
    """Tests for Chunker class."""

    def test_chunk_text_basic(self, sample_text):
        """Test basic text chunking."""
        chunker = Chunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_text(sample_text)

        assert len(chunks) > 0
        assert all("content" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)

    def test_chunk_text_respects_size(self, sample_text):
        """Test that chunks respect size limits."""
        chunker = Chunker(chunk_size=300, chunk_overlap=30)
        chunks = chunker.chunk_text(sample_text)

        # Most chunks should be under the limit (allowing some overflow for complete words)
        for chunk in chunks[:-1]:  # Exclude last chunk
            assert len(chunk["content"]) <= 400  # Allow 33% overflow for complete words

    def test_chunk_text_with_metadata(self, sample_text):
        """Test chunking with metadata preservation."""
        metadata = {"source": "test.pdf", "date": "2024-01-01"}
        chunker = Chunker()
        chunks = chunker.chunk_text(sample_text, metadata)

        for chunk in chunks:
            assert chunk["metadata"]["source"] == "test.pdf"
            assert chunk["metadata"]["date"] == "2024-01-01"

    def test_chunk_text_empty_string(self):
        """Test chunking empty string."""
        chunker = Chunker()
        chunks = chunker.chunk_text("")

        assert len(chunks) == 0

    def test_chunk_text_adds_chunk_index(self, sample_text):
        """Test that chunks are properly indexed."""
        chunker = Chunker()
        chunks = chunker.chunk_text(sample_text)

        for i, chunk in enumerate(chunks):
            assert chunk["metadata"]["chunk_index"] == i
            assert chunk["metadata"]["total_chunks"] == len(chunks)

    def test_chunker_invalid_overlap(self):
        """Test that invalid overlap raises error."""
        with pytest.raises(ValueError):
            Chunker(chunk_size=100, chunk_overlap=150)

    def test_estimate_tokens(self):
        """Test token estimation."""
        chunker = Chunker()
        text = "a" * 4000  # 4000 characters

        tokens = chunker.estimate_tokens(text)
        assert tokens == 1000  # 4000 / 4 = 1000


class TestSemanticChunker:
    """Tests for SemanticChunker class."""

    def test_semantic_chunker_initialization(self):
        """Test SemanticChunker initialization."""
        chunker = SemanticChunker()
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 100

    def test_find_sections(self):
        """Test section detection."""
        text = """
        # Introduction
        Some content here.
        
        ## Methodology
        More content.
        
        ### Results
        Final content.
        """

        chunker = SemanticChunker()
        sections = chunker.find_sections(text)

        assert len(sections) > 0

    def test_chunk_by_sections(self, sample_text):
        """Test chunking by sections."""
        chunker = SemanticChunker()
        chunks = chunker.chunk_by_sections(sample_text)

        assert len(chunks) > 0
        assert all("content" in chunk for chunk in chunks)
