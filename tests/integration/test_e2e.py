"""Integration test for full RAG pipeline."""

import pytest


class TestRAGPipeline:
    """Integration tests for RAG pipeline."""

    def test_end_to_end_flow(self, sample_text):
        """
        Test complete flow: text → chunks → embeddings → retrieval → answer.
        
        This is a skeleton test. In real usage, mock Gemini and Pinecone API calls.
        """
        from financial_analyzer.ingest.chunker import Chunker
        from financial_analyzer.utils.helpers import estimate_tokens

        # Step 1: Chunk text
        chunker = Chunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_text(sample_text, metadata={"source": "test.pdf"})

        assert len(chunks) > 0
        assert all("content" in c for c in chunks)

        # Step 2: Extract content and estimate tokens
        chunk_contents = [c["content"] for c in chunks]
        total_tokens = sum(estimate_tokens(content) for content in chunk_contents)

        assert total_tokens > 0

        # Step 3: Verify metadata preservation
        for chunk in chunks:
            assert chunk["metadata"]["source"] == "test.pdf"
            assert "chunk_index" in chunk["metadata"]

    def test_query_preprocessing(self):
        """Test query preprocessing."""
        from financial_analyzer.rag.prompts import format_chat_history

        chat_history = [
            {"role": "user", "content": "What is the revenue?"},
            {"role": "assistant", "content": "The revenue is $100M"},
            {"role": "user", "content": "What about profit?"},
        ]

        formatted = format_chat_history(chat_history)
        assert "revenue" in formatted.lower()
        assert "profit" in formatted.lower()
