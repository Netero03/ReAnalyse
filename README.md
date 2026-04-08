# ReAnalyse - Financial Document Analyzer

A Retrieval-Augmented Generation (RAG) chatbot for analyzing financial documents. Upload PDF annual reports, financial statements, or filings and ask questions about them using AI.

## 🚀 Features

- **Multi-PDF Upload**: Upload multiple financial documents (annual reports, 10-Ks, earnings statements, etc.)
- **AI-Powered Q&A**: Ask natural language questions about your documents
- **Semantic Search**: Uses vector embeddings to find relevant document sections
- **Transparent Retrieval**: See which document chunks were used to answer your question
- **Multi-turn Chat**: Maintain conversation context across multiple questions
- **Financial-Focused**: Prompt templates optimized for financial document analysis

## 🛠️ Tech Stack

- **LangChain**: Orchestration framework for RAG pipelines
- **Gemini API**: Embeddings (multimodal) + LLM for generation
- **Pinecone**: Serverless vector database for semantic search
- **Streamlit**: Web interface for document upload and chat
- **PyPDF**: PDF text extraction

## 📋 Prerequisites

- Python 3.9+
- Gemini API key (free tier available at [Google AI Studio](https://makersuite.google.com/app/apikey))
- Pinecone API key (free tier available at [Pinecone](https://www.pinecone.io))

## ⚙️ Setup

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/ReAnalyse.git
cd ReAnalyse
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env` with:
- `GOOGLE_API_KEY`: Your Gemini API key
- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_ENVIRONMENT`: Your Pinecone environment (e.g., `gcp-starter`)
- `PINECONE_INDEX_NAME`: Name for your Pinecone index (e.g., `financial-documents`)

### 3. Initialize Pinecone Index

Before running the app, create your Pinecone index:

```bash
python -c "from src.financial_analyzer.vector_store.pinecone_client import initialize_index; initialize_index()"
```

### 4. Run Locally

```bash
streamlit run src/financial_analyzer/ui/streamlit_app.py
```

The app will open at `http://localhost:8501`

## 📖 Usage

1. **Upload PDFs**: Use the sidebar file uploader to select one or more PDF files
2. **Wait for Processing**: The app will:
   - Extract text from PDFs
   - Split into overlapping chunks
   - Generate embeddings using Gemini API
   - Index chunks in Pinecone
3. **Ask Questions**: Type your questions in the chat input
   - Example: "What was the total revenue in 2023?"
   - Example: "What are the main risks mentioned?"
4. **Review Results**: Each response shows the source document chunks

## 🏗️ Project Structure

```
ReAnalyse/
├── src/financial_analyzer/
│   ├── ingest/               # PDF loading and chunking
│   ├── embeddings/           # Gemini API embedding wrapper
│   ├── vector_store/         # Pinecone client and retrieval
│   ├── rag/                  # LangChain chains and prompts
│   ├── ui/                   # Streamlit application
│   ├── utils/                # Logging and error handling
│   └── config.py             # Configuration management
├── tests/                    # Unit and integration tests
├── docs/                     # Architecture documentation
└── pyproject.toml            # Python project configuration
```

## 🧪 Testing

Run unit tests:

```bash
pytest tests/unit/
```

Run integration tests:

```bash
pytest tests/integration/
```

Run all tests with coverage:

```bash
pytest --cov=src tests/
```

## 🚀 Deployment

### Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy → New app → Select your repo and `src/financial_analyzer/ui/streamlit_app.py`
4. Set secrets in Streamlit Cloud settings:
   - `GOOGLE_API_KEY`
   - `PINECONE_API_KEY`
   - `PINECONE_ENVIRONMENT`
   - `PINECONE_INDEX_NAME`

### Docker

```bash
docker build -t financial-analyzer .
docker run -p 8501:8501 -e GOOGLE_API_KEY=... -e PINECONE_API_KEY=... financial-analyzer
```

## 📊 Architecture

```
User Upload (PDF)
    ↓
[Document Loader] → Extract text & metadata
    ↓
[Text Splitter] → Semantic chunks (500 chars, 10% overlap)
    ↓
[Gemini Embedder] → Generate embeddings (3072 dims)
    ↓
[Pinecone] → Index and store vectors
    ↓
User Question
    ↓
[Gemini Embedder] → Embed query
    ↓
[Pinecone Search] → Retrieve top 5 relevant chunks
    ↓
[LangChain RAG Chain] → Combine context + question
    ↓
[Gemini LLM] → Generate answer
    ↓
Response to User
```

## 🔧 Configuration

Key settings in `.env`:

- `CHUNK_SIZE`: Characters per chunk (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 100)
- `MAX_RETRIEVED_CHUNKS`: Number of relevant chunks to retrieve (default: 5)
- `MODEL_NAME`: Gemini model for generation (default: `gemini-1.5-flash`)
- `LOG_LEVEL`: Logging verbosity (default: INFO)

## 🐛 Troubleshooting

### "No API key provided"
Ensure `.env` file exists and contains `GOOGLE_API_KEY` and `PINECONE_API_KEY`

### "Index not found"
Run the initialization script:
```bash
python -c "from src.financial_analyzer.vector_store.pinecone_client import initialize_index; initialize_index()"
```

### Slow embeddings
Consider using the Batch API for large PDF collections (50% cost savings)

### High vector costs
Start with Pinecone free tier; monitor token usage with Gemini API

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 👨‍💻 Author

Created as a portfolio project showcasing RAG applications with modern AI APIs.

## 📚 Resources

- [LangChain Documentation](https://python.langchain.com)
- [Gemini API Docs](https://ai.google.dev)
- [Pinecone Documentation](https://docs.pinecone.io)
- [Streamlit Documentation](https://docs.streamlit.io)
