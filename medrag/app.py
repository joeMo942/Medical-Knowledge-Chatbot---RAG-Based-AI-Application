"""FastAPI app: POST /chat, GET /health and the static chat page at /."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from .answering import ChatResponse, answer_question
from .config import EMBEDDING_PLACEHOLDER, Settings
from .ingest import kb_dir, load_knowledge_base
from .language import detect_language
from .llm import LLMClient, build_llm
from .models import Section
from .retrieval import LexicalRetriever
from .sessions import SessionStore

log = logging.getLogger("medrag")
STATIC = Path(__file__).parent / "static"


def _error(status: int, message: str, session_id: str | None = None) -> JSONResponse:
    headers = {"X-Session-Id": session_id} if session_id else None
    return JSONResponse({"error": message}, status_code=status, headers=headers)


def create_app(
    sections: list[Section] | tuple[Section, ...],
    llm: LLMClient,
    settings: Settings | None = None,
    embedding_name: str = EMBEDDING_PLACEHOLDER,
) -> FastAPI:
    settings = settings or Settings()
    retriever = LexicalRetriever(sections)
    store = SessionStore()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        warm = getattr(llm, "warm", None)
        task = asyncio.create_task(warm()) if warm else None
        yield
        if task:
            task.cancel()

    app = FastAPI(title="Drug Information Assistant", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "llm": llm.model, "embedding": embedding_name}

    @app.post("/chat")
    async def chat(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return _error(400, "request body must be JSON")
        message = body.get("message") if isinstance(body, dict) else None
        if not isinstance(message, str) or not message.strip():
            return _error(400, "'message' is required and must be non-empty text")
        session_id = store.resolve(body.get("session_id") or request.headers.get("x-session-id"))
        try:
            result: ChatResponse = await asyncio.wait_for(
                answer_question(message, detect_language(message), retriever, llm, settings.top_k),
                timeout=settings.budget_s,
            )
        except asyncio.TimeoutError:
            log.error("chat timed out after %.1fs", settings.budget_s)
            return _error(500, "the request timed out", session_id)
        except Exception as exc:
            log.exception("chat failed")
            return _error(500, f"internal error: {type(exc).__name__}", session_id)
        store.add_turn(session_id, message, result.answer)
        return JSONResponse(result.model_dump(), headers={"X-Session-Id": session_id})

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app


def create_default_app() -> FastAPI:
    """Entry point for uvicorn (`--factory`): KB from MEDRAG_KB_DIR, models from env."""
    logging.basicConfig(level=logging.INFO)
    settings = Settings.from_env()
    docs = load_knowledge_base(kb_dir())
    sections = [s for d in docs for s in d.sections]
    log.info("loaded %d Sections from %d KB Documents", len(sections), len(docs))
    return create_app(sections, build_llm(settings), settings)
