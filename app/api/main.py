"""PharmaAssist HTTP API.

    py -m uvicorn app.api.main:app --port 8001

Port 8001, because the tracker uses 8000. The agent is built once at startup (model client,
embedder, tracker client, knowledge-base ids). Requests are served one at a time: a lock around
the agent keeps a single local GPU model from being asked for several answers at once.

Environment: LLM_BACKEND and the backend's variables (see app/llm), TRACKER_BASE_URL,
RAG_DB_PATH (default chroma_db/noctx), OLLAMA_NUM_CTX (default 8192).
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from app.api.schemas import ChatRequest, ChatResponse, HealthResponse
from app.llm.base import LLMConnectionError, LLMError, LLMModelNotFoundError, LLMRateLimitError

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = REPO_ROOT / "chroma_db" / "noctx"
KB_DIR = REPO_ROOT / "docs" / "knowledge_base"


def build_agent():
    """The real agent, with every dependency read from the environment."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    from app.agent import Agent
    from app.llm.factory import create_llm_client
    from app.rag.citations import kb_section_ids
    from app.rag.embedding import SentenceTransformerEmbedder
    from app.rag.store import retrieve
    from app.tools.tracker_client import TrackerClient

    db_path = Path(os.environ.get("RAG_DB_PATH", str(DEFAULT_DB)))
    embedder = SentenceTransformerEmbedder()
    return Agent(
        llm_client=create_llm_client(),
        tracker=TrackerClient(),
        kb_search=lambda query: retrieve(db_path, embedder, query, k=5),
        known_section_ids=kb_section_ids(KB_DIR),
        prefetch=True,
        context_window=int(os.environ.get("OLLAMA_NUM_CTX", "8192")),
    )


def create_app(agent_factory: Callable = build_agent) -> FastAPI:
    state: dict = {}
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["agent"] = agent_factory()
        yield
        state.clear()

    app = FastAPI(title="PharmaAssist", version="0.1.0", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok" if "agent" in state else "starting",
                              backend=os.environ.get("LLM_BACKEND", "unknown"))

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        agent = state.get("agent")
        if agent is None:
            raise HTTPException(status_code=503, detail="The assistant is still starting.")
        history = [{"role": m.role, "content": m.content} for m in request.history]
        try:
            with lock:
                result = agent.run(request.message, history)
        except LLMRateLimitError as error:
            raise HTTPException(status_code=429, detail=str(error)) from error
        except (LLMConnectionError, LLMModelNotFoundError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except LLMError as error:
            raise HTTPException(status_code=502, detail=f"The language model failed: {error}") from error
        return ChatResponse.from_result(result)

    return app


app = create_app()