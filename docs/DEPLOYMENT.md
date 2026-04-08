"""Deployment Guide for ReAnalyse."""

# Deployment Guide - Financial Document Analyzer

## Option 1: Streamlit Cloud (Recommended - Easiest)

### Prerequisites
- GitHub account
- Gemini API key
- Pinecone account with API key

### Steps

#### 1. Push Code to GitHub

```bash
git remote add origin https://github.com/yourusername/ReAnalyse.git
git branch -M main
git push -u origin main
```

#### 2. Create Streamlit Cloud Account

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "Sign in with GitHub"
3. Authorize Streamlit

#### 3. Deploy Application

1. Click "New app"
2. Select:
   - **Repository**: yourusername/ReAnalyse
   - **Branch**: main
   - **Main file path**: src/financial_analyzer/ui/streamlit_app.py
3. Click "Deploy"

#### 4. Configure Secrets

After deployment:

1. Click the menu (3 lines) → Settings
2. Go to "Secrets" section
3. Add the following (copy from your `.env`):

```toml
GOOGLE_API_KEY = "your_gemini_api_key"
PINECONE_API_KEY = "your_pinecone_api_key"
PINECONE_ENVIRONMENT = "gcp-starter"
PINECONE_INDEX_NAME = "financial-documents"
LOG_LEVEL = "INFO"
CHUNK_SIZE = "1000"
CHUNK_OVERLAP = "100"
MAX_RETRIEVED_CHUNKS = "5"
MODEL_NAME = "gemini-1.5-flash"
EMBEDDING_MODEL = "models/embedding-001"
```

#### 5. Access Your App

After secrets are saved, your app will redeploy automatically. Access it at:
```
https://yourusername-reanalyse.streamlit.app
```

### Monitoring

**In Streamlit Cloud Dashboard:**
- View app logs: Click app → Manage app → View logs
- Check resource usage (CPU, memory)
- Monitor API call patterns

### Troubleshooting Streamlit Cloud

| Issue | Solution |
|-------|----------|
| "ModuleNotFoundError" | Ensure all imports use absolute paths from `src/` |
| "API key not found" | Check secrets spelling matches `GOOGLE_API_KEY` exactly |
| App crashes on startup | Check logs for missing dependencies; add to `requirements.txt` |
| Slow uploads | Large PDFs may timeout; consider splitting or upgrading Streamlit tier |

---

## Option 2: Docker Deployment

### Build Docker Image

```bash
docker build -t financial-analyzer:latest .
```

### Run Locally

```bash
docker run -p 8501:8501 \
  -e GOOGLE_API_KEY=your_key \
  -e PINECONE_API_KEY=your_key \
  -e PINECONE_ENVIRONMENT=gcp-starter \
  -e PINECONE_INDEX_NAME=financial-documents \
  financial-analyzer:latest
```

Access at `http://localhost:8501`

### Deploy to Cloud

#### AWS ECS (Elastic Container Service)

```bash
# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com

docker tag financial-analyzer:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/financial-analyzer:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/financial-analyzer:latest

# Create ECS task and service
# (Use AWS Console or CLI)
```

#### Google Cloud Run

```bash
# Enable API
gcloud services enable run.googleapis.com

# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/financial-analyzer
gcloud run deploy financial-analyzer \
  --image gcr.io/PROJECT_ID/financial-analyzer \
  --platform managed \
  --region us-central1 \
  --set-env-vars GOOGLE_API_KEY=xxx,PINECONE_API_KEY=yyy,...
```

#### Azure Container Instances

```bash
# Push to ACR
az acr build --registry myregistry --image financial-analyzer:latest .

# Deploy
az container create --resource-group mygroup \
  --name financial-analyzer \
  --image myregistry.azurecr.io/financial-analyzer:latest \
  --environment-variables GOOGLE_API_KEY=xxx PINECONE_API_KEY=yyy
```

---

## Option 3: Traditional Server Deployment

### Prerequisites
- Ubuntu 20.04+ or similar Linux
- Python 3.9+
- Systemd

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/ReAnalyse.git
cd ReAnalyse

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
nano .env

# Test locally
streamlit run src/financial_analyzer/ui/streamlit_app.py
```

### Create Systemd Service

Create `/etc/systemd/system/financial-analyzer.service`:

```ini
[Unit]
Description=Financial Document Analyzer
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/financial-analyzer
Environment="PATH=/opt/financial-analyzer/venv/bin"
ExecStart=/opt/financial-analyzer/venv/bin/streamlit run \
  src/financial_analyzer/ui/streamlit_app.py \
  --server.port=8501 \
  --server.address=0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Start Service

```bash
sudo systemctl daemon-reload
sudo systemctl enable financial-analyzer
sudo systemctl start financial-analyzer

# Check status
sudo systemctl status financial-analyzer

# View logs
sudo journalctl -u financial-analyzer -f
```

### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name analyzer.example.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;
    }
}
```

---

## Option 4: Development Setup (For Contributors)

### Clone and Install

```bash
git clone https://github.com/yourusername/ReAnalyse.git
cd ReAnalyse

# Create virtual environment
python -m venv venv

# Activate venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dev dependencies
pip install -r requirements-dev.txt

# Create .env from example
cp .env.example .env
# Add your API keys to .env
```

### Run Tests

```bash
pytest tests/unit/
pytest tests/integration/
pytest --cov=src tests/
```

### Run Locally

```bash
streamlit run src/financial_analyzer/ui/streamlit_app.py
```

### Code Quality

```bash
# Format code
black src tests
isort src tests

# Lint
flake8 src tests --max-line-length=100

# Type check
mypy src --ignore-missing-imports
```

---

## Cost Optimization

### Reduce Gemini API Costs

1. **Use Task-Specific Embeddings**
   - Set correct `task_type` in GeminiEmbedder
   - Different task types have optimized pricing

2. **Batch Processing**
   - Use Gemini Batch API for 50% cost savings
   - Implement in `gemini_embedder.py`

3. **Cache Frequently Asked Questions**
   - Pre-compute answers for common queries
   - Store in Redis or local cache

4. **Limit Chunk Count**
   - Reduce `MAX_RETRIEVED_CHUNKS` in `.env`
   - Smaller context = fewer tokens

### Reduce Pinecone Costs

1. **Start with Free Tier**
   - Supports ~100k vectors
   - Adequate for 50-100 documents

2. **Index Maintenance**
   - Delete old/unused documents periodically
   - Use namespaces to organize data

3. **Query Optimization**
   - Filter by metadata to reduce search scope
   - Use appropriate `top_k` (5-10 typically sufficient)

### Monitoring Costs

```bash
# Gemini API Dashboard
https://aistudio.google.com/app/usage

# Pinecone Dashboard
https://app.pinecone.io
```

---

## Post-Deployment Checklist

- [ ] App accessible from public URL
- [ ] Can upload PDFs successfully
- [ ] Chat generates responses from documents
- [ ] Source documents displayed correctly
- [ ] API keys not exposed in logs
- [ ] Error messages are user-friendly
- [ ] Response time < 5 seconds typical
- [ ] App handles concurrent users gracefully
- [ ] Secrets configured properly
- [ ] README updated with deployment URL
- [ ] Monitor cost trends for first week

---

## Rollback Procedure

### Streamlit Cloud
1. Go to app settings
2. Click "Revert to previous version"
3. Select earlier deployment

### Docker
```bash
# Redeploy previous image
docker pull myregistry/financial-analyzer:v1.0
docker run -p 8501:8501 myregistry/financial-analyzer:v1.0
```

### Systemd Service
```bash
# Revert code
git revert HEAD~1
git push

# Reload service
sudo systemctl restart financial-analyzer
```

---

## Support & Troubleshooting

### Common Deployment Issues

1. **"ImportError: No module named 'financial_analyzer'"**
   - Solution: Ensure `PYTHONPATH` includes project root
   - In Streamlit: check main file path

2. **API Key Issues**
   - Verify key format and expiration
   - Check Streamlit secrets spelling

3. **Slow Performance**
   - Monitor Pinecone query latency
   - Check internet connection speed
   - Consider upgrading Streamlit tier

4. **High Memory Usage**
   - Reduce `MAX_RETRIEVED_CHUNKS`
   - Clear old chat history periodically

### Get Help

- GitHub Issues: https://github.com/yourusername/ReAnalyse/issues
- LangChain Docs: https://python.langchain.com
- Gemini Support: https://ai.google.dev/support
- Pinecone Support: https://support.pinecone.io
