"""Document loading from PDF files."""

import io
from typing import List, Optional, Tuple
from pathlib import Path
from datetime import datetime

from pypdf import PdfReader

from financial_analyzer.utils.logger import logger
from financial_analyzer.utils.errors import PDFLoadingError


class Document:
    """Represents a document chunk with metadata."""

    def __init__(
        self,
        content: str,
        source: str,
        page_number: int,
        chunk_index: int,
        metadata: Optional[dict] = None,
    ):
        """
        Initialize a document.

        Args:
            content: Text content of the chunk
            source: Source file name
            page_number: Page number in the PDF
            chunk_index: Index among all chunks from this document
            metadata: Additional metadata
        """
        self.content = content
        self.source = source
        self.page_number = page_number
        self.chunk_index = chunk_index
        self.metadata = metadata or {}
        self.upload_date = datetime.now().isoformat()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Document(source={self.source}, page={self.page_number}, "
            f"chunk={self.chunk_index}, len={len(self.content)})"
        )


class DocumentLoader:
    """Loads and extracts text from PDF files."""

    def __init__(self):
        """Initialize the document loader."""
        self.logger = logger

    def load_pdf_from_file(self, file_path: Path) -> Tuple[str, dict]:
        """
        Load PDF from file path.

        Args:
            file_path: Path to PDF file

        Returns:
            Tuple of (text_content, metadata)

        Raises:
            PDFLoadingError: If PDF cannot be loaded or parsed
        """
        try:
            self.logger.info(f"Loading PDF from: {file_path}")

            if not file_path.exists():
                raise FileNotFoundError(f"PDF file not found: {file_path}")

            if not file_path.suffix.lower() == ".pdf":
                raise ValueError(f"File is not a PDF: {file_path}")

            pdf_reader = PdfReader(file_path)
            num_pages = len(pdf_reader.pages)

            self.logger.info(f"PDF has {num_pages} pages")

            # Extract text from all pages
            text_content = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    text_content += f"\n--- Page {page_num + 1} ---\n{page_text}"
                except Exception as e:
                    self.logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                    text_content += f"\n--- Page {page_num + 1} ---\n[Failed to extract text]"

            metadata = {
                "source": file_path.name,
                "file_path": str(file_path),
                "num_pages": num_pages,
                "file_size": file_path.stat().st_size,
                "created_date": datetime.now().isoformat(),
            }

            self.logger.info(f"Successfully loaded PDF: {file_path.name} ({num_pages} pages)")
            return text_content, metadata

        except Exception as e:
            self.logger.error(f"Error loading PDF: {str(e)}")
            raise PDFLoadingError(f"Failed to load PDF from {file_path}: {str(e)}") from e

    def load_pdf_from_bytes(self, pdf_bytes: bytes, filename: str) -> Tuple[str, dict]:
        """
        Load PDF from bytes (useful for file uploads in web apps).

        Args:
            pdf_bytes: PDF content as bytes
            filename: Name of the PDF file

        Returns:
            Tuple of (text_content, metadata)

        Raises:
            PDFLoadingError: If PDF cannot be loaded or parsed
        """
        try:
            self.logger.info(f"Loading PDF from bytes: {filename}")

            pdf_file = io.BytesIO(pdf_bytes)
            pdf_reader = PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)

            self.logger.info(f"PDF has {num_pages} pages")

            # Extract text from all pages
            text_content = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    text_content += f"\n--- Page {page_num + 1} ---\n{page_text}"
                except Exception as e:
                    self.logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                    text_content += f"\n--- Page {page_num + 1} ---\n[Failed to extract text]"

            metadata = {
                "source": filename,
                "num_pages": num_pages,
                "file_size": len(pdf_bytes),
                "created_date": datetime.now().isoformat(),
            }

            self.logger.info(f"Successfully loaded PDF: {filename} ({num_pages} pages)")
            return text_content, metadata

        except Exception as e:
            self.logger.error(f"Error loading PDF from bytes: {str(e)}")
            raise PDFLoadingError(f"Failed to load PDF {filename}: {str(e)}") from e

    def load_multiple_pdfs(self, file_paths: List[Path]) -> List[Tuple[str, dict]]:
        """
        Load multiple PDFs.

        Args:
            file_paths: List of paths to PDF files

        Returns:
            List of tuples (text_content, metadata)
        """
        results = []
        for file_path in file_paths:
            try:
                text, metadata = self.load_pdf_from_file(file_path)
                results.append((text, metadata))
            except PDFLoadingError as e:
                self.logger.error(f"Skipping file {file_path}: {str(e)}")
                continue

        self.logger.info(f"Successfully loaded {len(results)}/{len(file_paths)} PDFs")
        return results
