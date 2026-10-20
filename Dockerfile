# PharmaAssist API. Ollama and the tracker run on the host; see docker-compose.yml.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# CPU-only PyTorch, the same version as the development venv (2.14.1+cpu),
# installed first so sentence-transformers does not pull the CUDA build.
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements-runtime.txt .
RUN pip install -r requirements-runtime.txt

COPY app ./app
COPY scripts/ingest_kb.py ./scripts/ingest_kb.py
COPY docs/knowledge_base ./docs/knowledge_base

# Download the embedding model and build the Chroma store at build time,
# so the container needs no network access to Hugging Face at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
 && python scripts/ingest_kb.py --db chroma_db/noctx --no-context

RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8001/health', timeout=4).status == 200 else 1)"

CMD ["python", "-m", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8001"]