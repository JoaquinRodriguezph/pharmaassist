# PharmaAssist

A chat assistant for pharmacy staff, built as a portfolio project. It answers questions from the pharmacy's
own documents (policies, SOPs, product leaflets) with section-level citations, looks up live stock, expiry,
restock and sales data through read-only calls to a separate inventory tracker, and refuses personal medical
advice, out-of-scope requests and prompt injection.

**All data is fictional.** The documents, products, branches and sales are invented. Nothing here is medical
advice.

## How it works

```
question ──► input guard ──► (blocked) scope refusal
                │
                ▼
        pre-retrieval: top 5 sections (MiniLM embeddings, ChromaDB)
                │
                ▼
        tool-calling loop (Ollama llama3.2 or Gemini)
          ├─ search_knowledge_base ──► ChromaDB
          └─ get_branches / get_low_stock / get_expiring_batches /
             get_restock / get_top_sellers / get_monthly_sales ──► tracker API (GET only)
                │
                ▼
        citation check + one repair turn ──► output guards ──► answer with [file.md#Section] citations
```

- **RAG:** structure-aware chunking of Markdown sections, `all-MiniLM-L6-v2` embeddings, ChromaDB with
  idempotent fingerprinted ingestion and an embedder-mismatch guard. A FAISS flat index was compared on the same
  vectors (`docs/vector_store_comparison.md`).
- **Tools:** six read-only tracker endpoints behind an allow-list, with argument validation, branch-name and
  branch-id translation, and a client-side product filter. The client has no write methods.
- **Citations:** answers cite `[file.md#Section]`. Every citation is checked against the retrieved sections;
  a citation to a document that does not exist replaces the answer with "The provided sources do not answer
  this question."
- **Guards in code, not only in the prompt:** prompt-injection, data-change and off-topic requests are refused
  before any model call; personal medical questions get a fixed refusal plus only cited leaflet facts; leaked
  prompt or tool text, and answers built on failed tool calls, are replaced.
- **Backends:** `LLMClient` interface with Ollama and Gemini clients, quota-aware retries, and tool calling on
  both.

## Results

Measured on a 25-question test set; details and limits in `docs/agent_eval.md` and `docs/model_comparison.md`.

| | llama3.2 Q4_K_M (default) | llama3.2 Q8_0 | Gemini (6 questions) |
|---|---|---|---|
| Passed | 19 of 25 | 19 of 25 | 4 of 6 |
| Tool questions | 6 of 6 | 6 of 6 | 1 of 1 |
| Refusal and injection questions | 6 of 6 | 6 of 6 | 1 of 1 |
| Median latency per question | 2.8 s | 3.3 s | 26.0 s |

**These are tuned numbers:** the guards were developed against the same 25 questions, so they overstate
quality on new questions. The remaining failures are mostly incomplete answers that miss a required phrase.

## Running it

Requirements: Python 3.12, [Ollama](https://ollama.com) with `llama3.2`, and the Pharmacy Inventory & Sales
Tracker (separate repository) running on port 8000.

```
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
ollama pull llama3.2
py scripts\ingest_kb.py --db chroma_db\noctx --no-context
py -m uvicorn app.api.main:app --port 8001
```

Open http://localhost:8001 for the chat page, or `POST /chat` with `{"message": "...", "history": []}`.

Configuration is read from a `.env` file (git-ignored, never committed): `LLM_BACKEND` (`ollama` or
`gemini`), `OLLAMA_MODEL`, `OLLAMA_HOST`, `OLLAMA_NUM_CTX`, `GEMINI_API_KEY`, `GEMINI_MODEL`,
`TRACKER_BASE_URL`, `RAG_DB_PATH`, `RAG_WITH_CONTEXT`.

### Docker

The API runs in a container; Ollama and the tracker stay on the host. The image builds the vector store and
contains the embedding model, so it needs no Hugging Face access at runtime.

```
docker compose up --build -d
```

The tracker must listen on `0.0.0.0:8000`, and Ollama must accept connections from the container
(`OLLAMA_HOST=0.0.0.0` on the host if it does not).

### Tests and evaluation

```
py -m pytest -q                           # 685 tests, no model or tracker needed
py scripts\eval_retrieval.py              # retrieval hit@k and MRR
py scripts\eval_agent.py                  # the 25-question agent evaluation (Ollama, tracker running)
py scripts\compare_runs.py <log> <log>    # compare evaluation runs
```

## Project layout

```
app/llm/      LLM clients (Ollama, Gemini), retries, tool-calling sessions
app/rag/      chunking, embeddings, Chroma store, FAISS index, citations, prompts, metrics
app/tools/    tracker client, tool registry and validation, executor
app/agent.py  the tool-calling agent and its guards
app/guards.py input and output patterns
app/api/      FastAPI service and chat page
docs/         knowledge base and design and evaluation notes
eval/         test sets and run logs
scripts/      ingestion, evaluation and comparison scripts
```

## Limits

- 25 test questions, used for both tuning and measurement; no held-out set yet.
- `must_contain` checks are substring matches; citation checks show a source was retrieved, not that it
  supports the sentence.
- Guards are regular expressions, tested on the test set and a few hand-written cases only.
- Single user: requests are served one at a time. No authentication.
- The knowledge base has four short documents.

## How this was built

I built this project with an AI assistant (Claude). For the setup, retrieval and RAG steps, I wrote the core
logic and the assistant reviewed it. For the tool-calling agent, the guards, the API, the chat page and the
Docker setup, most of the code was written by the assistant to my specification. I ran every test and
evaluation, read the failures, and decided what to change. Every number in this repository comes from a run
on my machine.