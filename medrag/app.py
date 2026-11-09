"""FastAPI app: POST /chat, GET /health and the static chat page at /."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .answering import ChatResponse
from .config import EMBEDDING_PLACEHOLDER, Settings
from .gate import Busy, LLMGate
from .ingest import kb_dir, kb_files_hash, load_knowledge_base
from .language import detect_language
from .llm import LLMClient, build_llm
from .models import Section
from .pipeline import respond
from .retrieval import LexicalRetriever, Retriever
from .support import SupportCheck
from .routing import HISTORY_TURNS
from .sessions import SessionStore, Turn
from .usage import USAGE_HEADER, start_meter

log = logging.getLogger("medrag")
STATIC = Path(__file__).parent / "static"
POLL_S = 0.2  # how often a running request checks whether the client is still there


class BadRequest(ValueError):
    pass


def _error(status: int, message: str, session_id: str | None = None,
           extra_headers: dict[str, str] | None = None) -> JSONResponse:
    headers = {"X-Session-Id": session_id} if session_id else {}
    headers.update(extra_headers or {})
    return JSONResponse({"error": message}, status_code=status, headers=headers or None)


async def _read_json(request: Request, max_chars: int) -> dict:
    """Body read with a hard size cap (so a huge upload cannot stall the server)."""
    # Worst legal case: every character outside the BMP (e.g. an emoji), sent
    # ASCII-escaped as a surrogate pair "\\ud83d\\ude00" = 12 bytes; plus room for
    # session_id and JSON punctuation. 2000 chars -> ~28 KB, still tiny.
    limit = max_chars * 12 + 4096
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise BadRequest("request body too large")
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise BadRequest("request body too large")
        chunks.append(chunk)
    try:
        body = json.loads(b"".join(chunks))
    except ValueError:
        raise BadRequest("request body must be JSON")
    if not isinstance(body, dict):
        raise BadRequest("request body must be a JSON object")
    return body


def _validate_message(body: dict, max_chars: int) -> str:
    message = body.get("message")
    if not isinstance(message, str) or not message.strip():
        raise BadRequest("'message' is required and must be non-empty text")
    if len(message) > max_chars:
        raise BadRequest(f"'message' is too long (maximum {max_chars} characters)")
    try:
        message.encode("utf-8")  # a lone surrogate cannot be encoded
    except UnicodeEncodeError:
        raise BadRequest("'message' contains invalid characters")
    return message


async def _run_unless_disconnected(request: Request, coro) -> ChatResponse | None:
    """Run the pipeline; cancel it (and return None) if the client goes away."""
    task = asyncio.ensure_future(coro)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=POLL_S)
            if done:
                return task.result()
            if await request.is_disconnected():
                task.cancel()
                return None
    finally:
        if not task.done():
            task.cancel()


def create_app(
    sections: list[Section] | tuple[Section, ...],
    llm: LLMClient,
    settings: Settings | None = None,
    embedding_name: str = EMBEDDING_PLACEHOLDER,
    retriever: Retriever | None = None,
) -> FastAPI:
    """`retriever` is anything with `search(query, k)`; without one the Sections are searched by BM25 alone.
    A retriever that has to load (a model, an index) also has `ready`, `load()` and `model_id`: /health is 503
    and /chat refuses until `ready`, and `load()` runs in a worker thread at startup."""
    settings = settings or Settings()
    retriever = retriever or LexicalRetriever(sections)
    support = SupportCheck(sections)
    store = SessionStore(settings.sessions_db)
    gate = LLMGate(llm, settings.llm_concurrency, settings.llm_est_s, settings.decode_tok_s,
                   settings.llm_margin_s, settings.min_output_tokens, settings.max_output_tokens)

    warmed = {"done": getattr(gate, "warm", None) is None}  # nothing to warm: ready at once
    loader = getattr(retriever, "load", None)

    def search_ready() -> bool:
        return bool(getattr(retriever, "ready", True))

    async def load_retriever() -> None:
        try:
            await asyncio.to_thread(loader)
        except Exception:
            log.exception("the search index could not be loaded; /health stays 503")

    async def warm_up() -> None:
        try:
            await gate.warm()
        finally:
            warmed["done"] = True  # warm() is best effort: a failure must not keep /health red for ever

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        tasks = [asyncio.create_task(warm_up())] if not warmed["done"] else []
        if loader and not search_ready():
            tasks.append(asyncio.create_task(load_retriever()))
        yield
        for task in tasks:
            task.cancel()

    app = FastAPI(title="Drug Information Assistant", lifespan=lifespan)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error(exc.status_code, str(exc.detail))

    @app.exception_handler(Exception)
    async def unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return _error(500, f"internal error: {type(exc).__name__}")

    @app.get("/health")
    async def health() -> JSONResponse:
        ready = getattr(gate, "ready", None)
        # Not "ok" until the model is loaded: the first request must not pay for the load (it would
        # run into the routing timeout and the request budget).
        ok = warmed["done"] and search_ready() and ((await ready()) if ready else True)
        body = {"status": "ok" if ok else "error", "llm": gate.model,
                "embedding": getattr(retriever, "model_id", None) or embedding_name}
        if getattr(retriever, "revision", ""):
            body["embedding_revision"] = retriever.revision
        return JSONResponse(body, status_code=200 if ok else 503)

    async def pipeline(message: str, deadline: float, history: list[Turn]) -> ChatResponse:
        async def search(text: str) -> list[Section]:
            t0 = time.monotonic()
            found = await asyncio.to_thread(retriever.search, text, settings.top_k)
            log.info("search: %d Sections in %.0f ms (embedding %.0f ms)", len(found), (time.monotonic() - t0) * 1000,
                     getattr(retriever, "last_embed_ms", 0.0))
            return found

        return await respond(message, detect_language(message), search, gate, deadline, support, history)

    async def load_history(session_id: str) -> list[Turn]:
        """The last few turns, or none: a store problem never blocks a reply."""
        try:
            return await asyncio.to_thread(store.turns, session_id, HISTORY_TURNS)
        except Exception:
            log.exception("could not read the Session history; answering without it")
            return []

    async def save_turn(session_id: str, message: str, result: ChatResponse) -> None:
        try:
            await asyncio.to_thread(store.add_turn, session_id, message, result.answer, result.status)
        except Exception:
            log.exception("could not store the turn; the reply is sent anyway")

    @app.post("/chat")
    async def chat(request: Request) -> JSONResponse:
        try:
            body = await _read_json(request, settings.max_message_chars)
            message = _validate_message(body, settings.max_message_chars)
        except BadRequest as exc:
            return _error(400, str(exc))
        if not search_ready():
            return _error(503, "the search index is still loading, try again shortly")
        session_id = store.resolve(body.get("session_id") or request.headers.get("x-session-id"))
        deadline = time.monotonic() + settings.budget_s
        history = await load_history(session_id)
        meter = start_meter()  # before the pipeline task exists, so the gate adds to this meter
        usage = lambda: {USAGE_HEADER: meter.header_value()} if settings.usage_header else None  # noqa: E731
        try:
            result = await asyncio.wait_for(
                _run_unless_disconnected(request, pipeline(message, deadline, history)),
                timeout=settings.budget_s,
            )
        except asyncio.TimeoutError:
            log.error("chat timed out after %.1fs", settings.budget_s)
            return _error(500, "the request timed out", session_id, usage())
        except Busy as exc:
            log.warning("chat rejected, LLM busy: %s", exc)
            return _error(500, "the server is busy, try again shortly", session_id, usage())
        except Exception as exc:
            log.exception("chat failed")
            return _error(500, f"internal error: {type(exc).__name__}", session_id, usage())
        if result is None:  # client disconnected; nobody is listening
            return _error(499, "client closed the request", session_id, usage())
        await save_turn(session_id, message, result)
        return JSONResponse(result.model_dump(), headers={"X-Session-Id": session_id, **(usage() or {})})

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app


def build_retriever(sections, settings: Settings, files_hash: str = "") -> Retriever:
    """BM25 alone, or the hybrid retriever (loaded later, in a worker thread: see create_app)."""
    if settings.retriever == "bm25":
        return LexicalRetriever(sections)
    from .embedding import BGEM3Embedder  # torch is imported only now, never by the tests or CI
    from .hybrid import HybridRetriever
    from .modelstore import MODEL_ID, REVISION, ensure_model

    def make_embedder():
        directory = ensure_model(settings.embed_path, settings.embed_seed or None)
        return BGEM3Embedder(directory, threads=settings.embed_threads or None)

    return HybridRetriever(sections, make_embedder, MODEL_ID, REVISION, cache_dir=settings.index_dir or None,
                           files_hash=files_hash, min_dense=settings.min_dense)


def create_default_app() -> FastAPI:
    """Entry point for uvicorn (`--factory`): KB from MEDRAG_KB_DIR, models from env."""
    logging.basicConfig(level=logging.INFO)
    settings = Settings.from_env()
    docs = load_knowledge_base(kb_dir())
    sections = [s for d in docs for s in d.sections]
    log.info("loaded %d Sections from %d KB Documents", len(sections), len(docs))
    return create_app(sections, build_llm(settings), settings, retriever=build_retriever(sections, settings, kb_files_hash()))
