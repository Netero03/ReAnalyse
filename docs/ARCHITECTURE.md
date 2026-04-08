"""Architecture documentation."""

# Financial Document Analyzer - Architecture Guide

## System Overview

ReAnalyse is a **Retrieval-Augmented Generation (RAG)** chatbot for financial document analysis. Users upload PDFs (annual reports, 10-K filings, etc.) and ask AI-powered questions about them.

### High-Level Flow

```
User PDF Upload
    ↓
[Document Loader] → Extract text from PDF
    ↓
[Text Chunker] → Split into overlapping segments
    ↓
[Gemini Embedder] → Generate vector embeddings
    ↓
[Pinecone] → Store vectors with metadata
    ↓
User Question
    ↓
[Gemini Embedder] → Embed query
    ↓
[Pinecone Retriever] → Semantic search
    ↓
[LangChain RAG Chain] → Combine retrieved chunks + question
    ↓
[Gemini LLM] → Generate contextual answer
    ↓
Response to User
```

## Component Architecture

### 1. Document Ingestion (`src/financial_analyzer/ingest/`)

#### `document_loader.py`
- **Responsibility**: Extract text from PDF files
- **Key Classes**:
  - `Document`: Represents a chunk with metadata
  - `DocumentLoader`: Loads PDFs (from file path or bytes)
- **Features**:
  - Multi-page PDF support
  - Metadata extraction (filename, upload date, page count)
  - Memory-efficient streaming for large files
  - Error handling for corrupted PDFs

#### `chunker.py`
- **Responsibility**: Split documents into overlapping chunks
- **Key Classes**:
  - `Chunker`: Semantic-aware text splitting
  - `SemanticChunker`: Section-respecting chunking
- **Features**:
  - Configurable chunk size and overlap
  - Preserves semantic boundaries (sections, paragraphs)
  - Metadata attached to each chunk (source, page, chunk index)
  - Token estimation

### 2. Vector Embeddings (`src/financial_analyzer/embeddings/`)

#### `gemini_embedder.py`
- **Responsibility**: Generate embeddings using Gemini API
- **Key Class**: `GeminiEmbedder`
- **Features**:
  - Asymmetric embeddings (query vs. document)
  - Task-specific optimization (`RETRIEVAL_QUERY`, `RETRIEVAL_DOCUMENT`, etc.)
  - Batch processing for efficiency
  - Error recovery and logging

### 3. Vector Store (`src/financial_analyzer/vector_store/`)

#### `pinecone_client.py`
- **Responsibility**: Manage Pinecone vector database
- **Key Class**: `PineconeClient`
- **Features**:
  - Index creation and initialization
  - Upsert vectors with metadata filters
  - Semantic search queries
  - Index statistics and management
  - Namespace partitioning

#### `retriever.py`
- **Responsibility**: Retrieve relevant documents from Pinecone
- **Key Class**: `Retriever`
- **Features**:
  - `retrieve()`: Semantic search with score filtering
  - `retrieve_with_source()`: Filter by document source
  - Metadata-based filtering
  - Result formatting for downstream use

### 4. RAG Chain (`src/financial_analyzer/rag/`)

#### `prompts.py`
- **Responsibility**: Define prompt templates for financial Q&A
- **Components**:
  - `SYSTEM_PROMPT`: Financial analyst context and guidelines
  - `QA_PROMPT_TEMPLATE`: Question-answering format
  - Few-shot examples
  - Chat history formatting
- **Features**:
  - Domain-specific instructions
  - Source attribution prompts
  - Conversation context handling

#### `chains.py`
- **Responsibility**: Orchestrate LangChain RAG pipeline
- **Key Class**: `RAGChain`
- **Features**:
  - LLM initialization (ChatGoogleGenerativeAI)
  - Embedding setup (GoogleGenerativeAIEmbeddings)
  - Pinecone vector store wrapping
  - RetrievalQA chain construction
  - Optional conversation memory
  - Query processing and response generation

### 5. Web Interface (`src/financial_analyzer/ui/`)

#### `streamlit_app.py`
- **Responsibility**: User-facing web application
- **Features**:
  - PDF upload widget
  - Real-time processing feedback
  - Chat interface with history
  - Source document display
  - Configuration panel
  - Session state management

### 6. Configuration (`src/financial_analyzer/`)

#### `config.py`
- **Responsibility**: Centralized configuration management
- **Key Class**: `Settings`
- **Features**:
  - Environment variable loading (`.env`)
  - Pydantic validation
  - Default values
  - Gemini and Pinecone configuration
  - Runtime validation

### 7. Utilities (`src/financial_analyzer/utils/`)

#### `logger.py`
- Structured logging with color support
- File and console handlers
- Configurable log levels

#### `errors.py`
- Custom exception hierarchy
- Specific error types for each component

#### `helpers.py`
- File hashing and validation
- Token estimation
- Text utilities

## Data Flow Examples

### Document Upload and Indexing

```
1. User uploads PDF via Streamlit UI
2. streamlit_app.py → DocumentLoader.load_pdf_from_bytes()
3. Extract: text_content + metadata (filename, date, pages)
4. Pass to Chunker → split into ~10 chunks (1000 chars each, 100 overlap)
5. Each chunk: {"content": "...", "metadata": {...}}
6. Chunk contents → GeminiEmbedder.embed_batch() → 768-dim vectors
7. Prepare upsert payload: [
   ("doc_name_chunk_0", [0.1, 0.2, ...], {"source": "report.pdf", "page": 1}),
   ...
]
8. PineconeClient.upsert_vectors() → Stored in Pinecone
```

### Question Answering

```
1. User types: "What was the revenue?"
2. RAGChain.query(question, chat_history=[])
3. GeminiEmbedder.embed_query() → query_embedding
4. Retriever.retrieve(embedding, top_k=5)
5. Pinecone searches semantic space → returns 5 best matches + metadata
6. Format retrieved chunks as context
7. Construct prompt:
   - System: Financial analyst Instructions
   - Retrieved context: Top 5 chunks
   - Question: "What was the revenue?"
8. Pass to ChatGoogleGenerativeAI.invoke()
9. LLM generates response with source attribution
10. Return answer + source metadata to UI
```

## Key Design Decisions

### 1. **Asymmetric Embeddings**
- Use different embeddings for queries (`RETRIEVAL_QUERY`)vs. documents (`RETRIEVAL_DOCUMENT`)
- Improves semantic relevance for Q&A scenarios
- Configurable via Gemini task_type parameter

### 2. **Chunking Strategy**
- Chunk size: 1000 characters (configurable)
- Overlap: 100 characters (preserve context between chunks)
- Semantic boundaries respected (paragraphs, sections)
- Rationale: Balance between context preservation and vector store efficiency

### 3. **Metadata Preservation**
- Each chunk carries source document info, page number, chunk index
- Enables source attribution in responses
- Allows filtering by document type or date

### 4. **Conversation Memory**
- Uses LangChain's `ConversationBufferMemory`
- Optional for MVP; can disable for stateless mode
- Allows multi-turn conversations with context

### 5. **Streamlit for UI**
- No separate backend/frontend split
- Rapid development and deployment
- st.session_state for state management
- Direct deployment to Streamlit Cloud

## Security & Limitations

### Security
- API keys via environment variables/Streamlit secrets
- No API keys in logs (careful with debug mode)
- User uploads stored only in session (not persisted)
- Pinecone servers handle encryption at rest

### Limitations
- Gemini multimodal: 6-page limit per request (handled via chunking)
- Pinecone free tier: ~100k vectors max
- Streamlit: single user per session (no global state)
- No persistent chat history (cleared on restart)

## Scaling Considerations

### Current Constraints (MVP)
- Single-user, in-memory state
- No persistent database
- Gemini embeddings via API (rate limits)
- Pinecone free tier capacity

### Future Enhancements
- **Multi-user**: Add user authentication + session isolation
- **Persistent History**: SQLite for chat/document storage
- **Batch Processing**: Use Gemini Batch API for cost optimization (50% savings)
- **Caching**: Cache frequently asked questions
- **Advanced Retrieval**: Hybrid search (dense + sparse/keyword)
- **Document Management**: S3 for PDF storage, versioning
- **Analytics**: Track query patterns, usage metrics

## Testing Strategy

### Unit Tests (`tests/unit/`)
- `test_chunker.py`: Chunk sizing, metadata, overlap
- `test_document_loader.py`: PDF parsing, error handling
- Other component tests as needed

### Integration Tests (`tests/integration/`)
- End-to-end PDF → question → answer flow
- API integration tests (mock Gemini/Pinecone if needed)
- Chat context preservation

### Manual Testing
- Upload sample financial documents
- Verify answer accuracy against documents
- Check source citation correctness

## Deployment

### Local Development
```bash
pip install -r requirements-dev.txt
streamlit run src/financial_analyzer/ui/streamlit_app.py
```

### Streamlit Cloud
1. Push repo to GitHub
2. Deploy via share.streamlit.io
3. Set secrets: GOOGLE_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME
4. Live at public URL

### Docker
Build image and deploy to cloud (AWS ECS, GCP Cloud Run, etc.)

## Monitoring & Debugging

### Logging Levels
- `DEBUG`: Detailed embedding/retrieval info
- `INFO`: Component initialization, query summaries
- `WARNING`: Skipped PDFs, partial failures
- `ERROR`: API failures, unrecoverable errors

### Common Issues
1. **"No API key"**: Ensure .env with GOOGLE_API_KEY and PINECONE_API_KEY
2. **Slow queries**: Check Pinecone index size; consider pruning old vectors
3. **High costs**: Monitor token usage; consider Batch API
4. **PDF parsing fails**: Check for corrupted/encrypted PDFs

## References

- LangChain: https://python.langchain.com
- Gemini API: https://ai.google.dev
- Pinecone: https://docs.pinecone.io
- Streamlit: https://docs.streamlit.io
