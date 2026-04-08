"""Streamlit web interface for Financial Document Analyzer."""

import streamlit as st
from datetime import datetime
import io
from pathlib import Path

from financial_analyzer.ingest.document_loader import DocumentLoader
from financial_analyzer.ingest.chunker import Chunker
from financial_analyzer.embeddings.gemini_embedder import GeminiEmbedder
from financial_analyzer.rag.chains import RAGChain
from financial_analyzer.config import settings
from financial_analyzer.utils.logger import logger


# Page configuration
st.set_page_config(
    page_title="ReAnalyse - Financial Document Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
    }
    .message-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
    }
    .assistant-message {
        background-color: #f0f4f8;
        border-left: 4px solid #1f77b4;
    }
    .source-box {
        background-color: #fff9e6;
        border-left: 4px solid #fbc02d;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = {}

    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None

    if "vector_store_ready" not in st.session_state:
        st.session_state.vector_store_ready = False


def load_and_index_pdf(uploaded_file, progress_container):
    """
    Load PDF and index it in Pinecone.

    Args:
        uploaded_file: Streamlit uploaded file object
        progress_container: Container for progress updates
    """
    try:
        # Load PDF
        with progress_container.status("Processing PDF...", expanded=True) as status:
            status.write("📄 Extracting text from PDF...")

            # Load PDF from uploaded bytes
            loader = DocumentLoader()
            pdf_bytes = uploaded_file.read()
            text_content, metadata = loader.load_pdf_from_bytes(pdf_bytes, uploaded_file.name)

            status.write(f"✓ Extracted text ({len(text_content)} characters)")

            # Chunk text
            status.write("✂️ Splitting into chunks...")
            chunker = Chunker(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            chunks = chunker.chunk_text(text_content, metadata)
            status.write(f"✓ Created {len(chunks)} chunks")

            # Generate embeddings
            status.write("🧠 Generating embeddings...")
            embedder = GeminiEmbedder(
                api_key=settings.google_api_key,
                task_type="retrieval_document",
            )

            chunk_contents = [chunk["content"] for chunk in chunks]
            embeddings = embedder.embed_batch(chunk_contents, batch_size=10)
            status.write(f"✓ Generated {len(embeddings)} embeddings")

            # Store processed PDF info (actual Pinecone indexing happens during RAG queries)
            status.write("📌 Storing document metadata...")
            processed_pdfs = st.session_state.get("processed_pdfs", {})
            processed_pdfs[uploaded_file.name] = {
                "chunks": chunks,
                "embeddings": embeddings,
                "processed_at": str(datetime.now())
            }
            st.session_state.processed_pdfs = processed_pdfs
            status.write(f"✓ Stored metadata for {len(chunks)} chunks")
            status.update(label=f"✅ {uploaded_file.name} processed successfully!", state="complete")

            return True

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        with progress_container:
            st.error(f"❌ Error processing PDF: {str(e)}")
        return False


def display_chat_message(message: dict):
    """Display a chat message in the UI."""
    role = message["role"]
    content = message["content"]

    if role == "user":
        st.markdown(
            f'<div class="message-box user-message"><b>You:</b> {content}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="message-box assistant-message"><b>Assistant:</b> {content}</div>',
            unsafe_allow_html=True,
        )

        # Display sources if available
        if "sources" in message:
            with st.expander("📚 Source Documents"):
                for source in message["sources"]:
                    st.markdown(
                        f'<div class="source-box">'
                        f'<b>From: {source["source"]}</b><br>'
                        f'{source["content"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


def main():
    """Main Streamlit application."""
    initialize_session_state()

    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📊 ReAnalyse")
        st.markdown("*Financial Document Analyzer - Ask questions about your PDFs*")
    with col2:
        # Statistics
        stats_container = st.container()

    # Sidebar for uploads
    with st.sidebar:
        st.header("📤 Upload Documents")

        uploaded_files = st.file_uploader(
            "Select PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader",
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.uploaded_files:
                    progress_container = st.container()
                    success = load_and_index_pdf(uploaded_file, progress_container)

                    if success:
                        st.session_state.uploaded_files[uploaded_file.name] = {
                            "size": uploaded_file.size,
                            "uploaded_at": datetime.now().isoformat(),
                        }

        # Display uploaded files
        if st.session_state.uploaded_files:
            st.subheader("✅ Uploaded Files")
            for filename, info in st.session_state.uploaded_files.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"• {filename}")
                with col2:
                    if st.button("🗑️", key=f"delete_{filename}", help="Delete"):
                        del st.session_state.uploaded_files[filename]
                        st.rerun()

        # Configuration
        with st.expander("⚙️ Settings"):
            settings.chunk_size = st.slider(
                "Chunk Size",
                min_value=100,
                max_value=2000,
                value=settings.chunk_size,
                step=100,
            )
            settings.max_retrieved_chunks = st.slider(
                "Top K Results",
                min_value=1,
                max_value=10,
                value=settings.max_retrieved_chunks,
            )

        # Help
        with st.expander("❓ Help"):
            st.markdown("""
            ### How to use:

            1. **Upload PDFs**: Select one or more PDF files to analyze
            2. **Wait for Processing**: The app will extract text and create embeddings
            3. **Ask Questions**: Type your question in the chat
            4. **View Results**: See the answer and source documents

            ### Tips:
            - Ask specific questions for better results
            - Questions can reference content like "revenue", "risks", "financial metrics"
            - View source documents by expanding "Source Documents"
            - Clear chat to start fresh conversation
            """)

    # Main chat interface
    st.header("💬 Chat")

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            display_chat_message(message)

    # Input area
    st.divider()

    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input(
            "Ask a question about your documents...",
            placeholder="E.g., What was the total revenue in 2023?",
            key="chat_input",
        )

    with col2:
        # Buttons
        col_send, col_clear = st.columns(2)
        with col_send:
            send_button = st.button("📤 Send", use_container_width=True)
        with col_clear:
            clear_button = st.button("🔄 Clear", use_container_width=True)

    # Process user input
    if send_button and user_input:
        if not st.session_state.uploaded_files:
            st.error("❌ Please upload at least one PDF document first.")
        else:
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input,
            })

            # Initialize RAG chain if needed
            if st.session_state.rag_chain is None:
                with st.spinner("🔄 Initializing AI model..."):
                    try:
                        st.session_state.rag_chain = RAGChain()
                    except Exception as e:
                        st.error(f"❌ Failed to initialize RAG chain: {str(e)}")
                        st.stop()

            # Get response
            with st.spinner("🤔 Thinking..."):
                try:
                    response = st.session_state.rag_chain.query(
                        user_input,
                        chat_history=st.session_state.chat_history[:-1],  # Exclude current input
                    )

                    # Add assistant message to history
                    assistant_message = {
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response.get("sources", []),
                    }
                    st.session_state.chat_history.append(assistant_message)

                    st.rerun()

                except Exception as e:
                    logger.error(f"Error processing query: {str(e)}")
                    st.error(f"❌ Error processing query: {str(e)}")

    if clear_button:
        st.session_state.chat_history = []
        st.rerun()

    # Footer
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption(f"📁 Documents: {len(st.session_state.uploaded_files)}")
    with col2:
        st.caption(f"💬 Messages: {len(st.session_state.chat_history)}")
    with col3:
        st.caption("🚀 Powered by Gemini AI + Pinecone")


if __name__ == "__main__":
    main()
