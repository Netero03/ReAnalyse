"""Quick Start Guide for ReAnalyse."""

# QuickStart - Get Running in 5 Minutes

## One-Time Setup (5 min)

### 1. Get API Keys

**Gemini API Key** (Free):
1. Go to https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key

**Pinecone API Key** (Free tier):
1. Go to https://www.pinecone.io
2. Sign up (free account)
3. Create project and index
4. Copy API key and environment name

### 2. Clone & Install

```bash
git clone https://github.com/yourusername/ReAnalyse.git
cd ReAnalyse

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
```

Edit `.env` and add your keys:
```
GOOGLE_API_KEY=your_gemini_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=gcp-starter  # or your environment
PINECONE_INDEX_NAME=financial-documents
```

### 4. Initialize Pinecone Index

```bash
python -c "from src.financial_analyzer.vector_store.pinecone_client import initialize_index; initialize_index()"
```

You should see: ✅ `Index created: financial-documents`

## Run the App

```bash
streamlit run src/financial_analyzer/ui/streamlit_app.py
```

Browser opens at `http://localhost:8501`

## First Use

### Step 1: Upload a PDF
- Sidebar → "Select PDF documents"
- Choose a financial document (10-K, annual report, etc.)
- Wait for "✅ processed successfully"

### Step 2: Ask a Question
- Type in chat: "What was the revenue?"
- Click "Send"
- Wait for response (~5-10 sec first time)

### Step 3: View Results
- See answer in chat
- Click "Source Documents" to see which PDF sections were used

## Examples to Try

```
"What is the company's total revenue?"
"What are the main risks mentioned?"
"How did gross profit change year-over-year?"
"List all business segments and their revenue."
"What are the cash flow from operations?"
```

## Troubleshooting

### Error: "No API key provided"
- Check `.env` file exists
- Verify `GOOGLE_API_KEY` and `PINECONE_API_KEY` are filled
- Make sure no extra spaces around `=`

### Error: "Index not found"
- Run the initialization again:
```bash
python -c "from src.financial_analyzer.vector_store.pinecone_client import initialize_index; initialize_index()"
```

### App crashes on PDF upload
- Check PDF is valid (try opening in Adobe Reader)
- Check file size < 200MB
- See logs for full error

### LLM response is slow
- First query initializes model (can take 10+ sec)
- Subsequent queries are faster
- Check internet connection

## Next Steps

1. **Deploy Online**
   - See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for Streamlit Cloud
   - Takes ~5 minutes to go live

2. **Improve Accuracy**
   - Adjust chunk size in Settings panel
   - Try different questions
   - Read [ARCHITECTURE.md](docs/ARCHITECTURE.md)

3. **Add More Features**
   - Multi-user support
   - Persistent chat history
   - Document management UI

## Common Settings to Tweak

In the app sidebar:

- **Chunk Size**: 500-2000 chars (lower = more specific retrieval)
- **Top K Results**: 3-10 (more = slower but better context)

## Cost Control

- **Free tier covers**: ~50-100 documents of typical length
- **Gemini API**: ~$0.05 per 1M tokens embeddings
- **Pinecone**: Free tier = 100k vectors
- **Estimate**: 100 queries = ~$0.05 total

## File Structure

```
ReAnalyse/
├── src/financial_analyzer/
│   ├── ui/streamlit_app.py  ← Run this!
│   ├── ingest/              ← PDF loading & chunking
│   ├── embeddings/          ← Gemini embeddings
│   ├── vector_store/        ← Pinecone storage
│   ├── rag/                 ← LLM & prompts
│   └── config.py            ← Settings
├── .env.example             ← Copy to .env
├── README.md                ← Full documentation
└── docs/
    ├── ARCHITECTURE.md      ← How it works
    └── DEPLOYMENT.md        ← Go live
```

## Quick Reference

| Task | Command |
|------|---------|
| Start app | `streamlit run src/financial_analyzer/ui/streamlit_app.py` |
| Run tests | `pytest` |
| Format code | `black src/` |
| Check types | `mypy src/` |
| Deploy | See DEPLOYMENT.md |

## Support

- 📖 Full docs: [README.md](README.md)
- 🏗️ Architecture: [ARCHITECTURE.md](docs/ARCHITECTURE.md)
- 🚀 Deploy: [DEPLOYMENT.md](docs/DEPLOYMENT.md)
- 🐛 Issues: GitHub Issues

**Happy analyzing! 📊**
