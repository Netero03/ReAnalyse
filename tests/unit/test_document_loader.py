"""Unit tests for document loading."""

import pytest
from io import BytesIO
from unittest.mock import Mock, patch

from financial_analyzer.ingest.document_loader import DocumentLoader, Document
from financial_analyzer.utils.errors import PDFLoadingError


class TestDocumentClass:
    """Tests for Document class."""

    def test_document_initialization(self):
        """Test creating a Document."""
        doc = Document(
            content="Sample content",
            source="test.pdf",
            page_number=1,
            chunk_index=0,
        )

        assert doc.content == "Sample content"
        assert doc.source == "test.pdf"
        assert doc.page_number == 1
        assert doc.chunk_index == 0
        assert doc.upload_date is not None

    def test_document_with_metadata(self):
        """Test Document with custom metadata."""
        metadata = {"key": "value"}
        doc = Document(
            content="Content",
            source="test.pdf",
            page_number=1,
            chunk_index=0,
            metadata=metadata,
        )

        assert doc.metadata["key"] == "value"


class TestDocumentLoader:
    """Tests for DocumentLoader class."""

    def test_loader_initialization(self):
        """Test DocumentLoader initialization."""
        loader = DocumentLoader()
        assert loader.logger is not None

    @patch("financial_analyzer.ingest.document_loader.PdfReader")
    def test_load_pdf_from_bytes(self, mock_pdf_reader):
        """Test loading PDF from bytes."""
        # Mock PDF reader
        mock_reader = Mock()
        mock_reader.pages = [Mock(extract_text=Mock(return_value="Page 1 content"))]
        mock_pdf_reader.return_value = mock_reader

        loader = DocumentLoader()
        text, metadata = loader.load_pdf_from_bytes(b"fake pdf", "test.pdf")

        assert "Page 1" in text
        assert metadata["source"] == "test.pdf"
        assert metadata["num_pages"] == 1

    def test_load_pdf_from_bytes_empty(self):
        """Test loading empty PDF."""
        loader = DocumentLoader()

        with pytest.raises(PDFLoadingError):
            loader.load_pdf_from_bytes(b"", "empty.pdf")

    def test_load_multiple_pdfs_empty_list(self):
        """Test loading empty list of PDFs."""
        loader = DocumentLoader()
        results = loader.load_multiple_pdfs([])

        assert len(results) == 0
