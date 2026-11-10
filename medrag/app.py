"""FastAPI app: POST /chat, GET /health and the static chat page at /."""
from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException as StarletteHTTPException

from .admin import admin_guard
from .analytics import ERROR, AnalyticsLog, Window
from .answering import ChatResponse
from .config import EMBEDDING_PLACEHOLDER, Settings
from .gate import Busy, LLMGate
from .handoff import MAX_BODY_BYTES, HandoffLog, InvalidHandoff, NotEscalated, validate as validate_handoff
from .ingest import kb_dir, kb_files_hash, load_knowledge_base
from .kbmanager import Busy as KBBusy, KBManager, KBSnapshot, NotFound, NotReady, open_store
from .kbstore import MAX_UPLOAD_BYTES, InvalidDocument, KBStore, check_kb_id
from .language import detect_language
from .llm import LLMClient, build_llm
from .models import KBDocument, Section
from .pipeline import respond
from .retrieval import LexicalRetriever, Retriever
from .routing import HISTORY_TURNS
from .sessions import SessionStore, Turn
from .usage import USAGE_HEADER, start_meter

log = logging.getLogger("medrag")
STATIC = Path(__file__).parent / "static"
POLL_S = 0.2  # how often a running request checks whether the client is still there


class BadRequest(ValueError):
    pass


class Loading(RuntimeError):
    """Search was needed before the embedding model and the index were loaded."""


def _error(status: int, message: str, session_id: str | None = None,
           extra_headers: dict[str, str] | None = None, background: BackgroundTask | None = None) -> JSONResponse:
    headers = {"X-Session-Id": session_id} if session_id else {}
    headers.update(extra_headers or {})
    return JSONResponse({"error": message}, status_code=status, headers=headers or None, background=background)


def _parse_time(name: str, value: str | None) -> float | None:
    """An ISO 8601 date or date-time as epoch seconds; no zone means UTC. A bad value is a 400."""
    if value is None or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(400, f"'{name}' must be an ISO 8601 date or date-time")
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp()


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
    except (ValueError, RecursionError):  # deeply nested JSON raises RecursionError on Python 3.12
        raise BadRequest("request body must be JSON")
    if not isinstance(body, dict):
        raise BadRequest("request body must be a JSON object")
    return body


class TooLarge(ValueError):
    pass


async def _read_upload(request: Request) -> dict:
    """The JSON body of a document upload (the file as base64), read with a cap of the largest file plus its overhead."""
    limit = MAX_UPLOAD_BYTES * 4 // 3 + 64 * 1024
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise TooLarge
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise TooLarge
        chunks.append(chunk)
    try:
        body = json.loads(b"".join(chunks))
    except (ValueError, RecursionError):
        raise InvalidDocument("request body must be JSON")
    if not isinstance(body, dict):
        raise InvalidDocument("request body must be a JSON object")
    return body


def _decode_file(body: dict) -> tuple[bytes, str]:
    raw = body.get("content_base64")
    if not isinstance(raw, str) or not raw:
        raise InvalidDocument("'content_base64' is required: the .docx file, base64-encoded")
    try:
        content = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        raise InvalidDocument("'content_base64' is not valid base64")
    if len(content) > MAX_UPLOAD_BYTES:
        raise TooLarge
    name = body.get("filename", "")
    if not isinstance(name, str):
        raise InvalidDocument("'filename' must be text")
    return content, name.strip()[:200]  # shown in the list only; never used as a path


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
    kb_store: KBStore | None = None,
    documents: tuple[KBDocument, ...] | None = None,
) -> FastAPI:
    """`retriever` is anything with `search(query, k)`; without one the Sections are searched by BM25 alone.
    A retriever that has to load (a model, an index) also has `ready`, `load()` and `model_id`: /health is 503
    and a /chat that needs search answers 500 until `ready` (escalations do not need it), and `load()` runs in a
    worker thread at startup.

    `kb_store` (see kbstore.py) turns on the admin document endpoints; `documents` are then the store's ingested KB
    Documents (read from the store when not given) and `sections` / `retriever` must have been built from them.
    The KB itself lives in a KBSnapshot that each /chat request reads once (kbmanager.py)."""
    settings = settings or Settings()
    if kb_store is not None and documents is None:
        documents = kb_store.load_documents()
        sections = [s for d in documents for s in d.sections]
    retriever = retriever or LexicalRetriever(sections)
    manager = KBManager(KBSnapshot.build(tuple(documents or ()), retriever, tuple(sections)), kb_store)
    store = SessionStore(settings.sessions_db)
    analytics = AnalyticsLog(store)
    handoffs = HandoffLog(store)
    admin = Depends(admin_guard(settings.admin_token))
    gate = LLMGate(llm, settings.llm_concurrency, settings.llm_est_s, settings.decode_tok_s,
                   settings.llm_margin_s, settings.min_output_tokens, settings.max_output_tokens)

    warmed = {"done": getattr(gate, "warm", None) is None}  # nothing to warm: ready at once
    loader = getattr(retriever, "load", None)

    def search_ready() -> bool:
        return manager.snapshot.ready

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
        if loader and not getattr(retriever, "ready", True):
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
        snap = manager.snapshot
        body = {"status": "ok" if ok else "error", "llm": gate.model,
                "embedding": getattr(snap.retriever, "model_id", None) or embedding_name}
        if manager.enabled:  # only with the writable KB store: the plain /health body is unchanged
            body["kb_documents"] = len({s.kb_id for s in snap.sections})
        if getattr(snap.retriever, "revision", ""):
            body["embedding_revision"] = snap.retriever.revision
        return JSONResponse(body, status_code=200 if ok else 503)

    async def pipeline(message: str, deadline: float, history: list[Turn]) -> ChatResponse:
        snap = manager.snapshot  # read once: this request answers from one KB even if an admin change lands meanwhile

        async def search(text: str) -> list[Section]:
            if not snap.ready:  # escalation (rules + routing) has already run: only a request that needs search waits
                raise Loading("the search index is still loading")
            t0 = time.monotonic()
            found = await asyncio.to_thread(snap.retriever.search, text, settings.top_k)
            log.info("search: %d Sections in %.0f ms (embedding %.0f ms)", len(found), (time.monotonic() - t0) * 1000,
                     getattr(snap.retriever, "last_embed_ms", 0.0))
            return found

        return await respond(message, detect_language(message), search, gate, deadline, snap.support, history, snap.names)

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

    def write_log(session_id: str, message: str, status: str, language: str | None, kb_ids: list[str],
                  latency_s: float, calls: list) -> None:
        """Runs after the response is sent, in a worker thread. A logging problem is reported and swallowed:
        it must never make a /chat request fail."""
        try:
            analytics.record(session_id, message, status, language, kb_ids, latency_s, calls)
        except Exception:
            log.exception("could not write the conversation log; the reply was sent anyway")

    @app.get("/admin/analytics", dependencies=[admin])
    async def admin_analytics(since: str | None = None, until: str | None = None, bucket: str = "day",
                              limit: int = 10) -> JSONResponse:
        if bucket not in ("hour", "day"):
            raise HTTPException(400, "'bucket' must be hour or day")
        if not 1 <= limit <= 1000:
            raise HTTPException(400, "'limit' must be between 1 and 1000")
        window = Window(_parse_time("since", since), _parse_time("until", until))
        report = await asyncio.to_thread(analytics.report, window, bucket, limit)
        report["handoffs"] = await asyncio.to_thread(handoffs.listing, window, limit)
        return JSONResponse(report)

    @app.get("/admin/handoffs", dependencies=[admin])
    async def admin_handoffs(since: str | None = None, until: str | None = None, limit: int = 10) -> JSONResponse:
        if not 1 <= limit <= 1000:
            raise HTTPException(400, "'limit' must be between 1 and 1000")
        window = Window(_parse_time("since", since), _parse_time("until", until))
        return JSONResponse(await asyncio.to_thread(handoffs.listing, window, limit))

    @app.post("/handoff")
    async def handoff(request: Request) -> JSONResponse:
        session_id = request.headers.get("x-session-id")
        if not session_id or store.resolve(session_id) != session_id:
            return _error(400, "the X-Session-Id header is required (the Session of the escalated question)")
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            return _error(413, "request body too large", session_id)
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_BODY_BYTES:
                return _error(413, "request body too large", session_id)
            chunks.append(chunk)
        try:
            body = json.loads(b"".join(chunks))
        except (ValueError, RecursionError):
            return _error(400, "request body must be JSON", session_id)
        if not isinstance(body, dict):
            return _error(400, "request body must be a JSON object", session_id)
        try:
            summary, contact = validate_handoff(body)
            stored = await asyncio.to_thread(handoffs.add, session_id, summary, contact)
        except InvalidHandoff as exc:
            return _error(400, str(exc), session_id)
        except NotEscalated as exc:
            return _error(409, str(exc), session_id)
        except Exception:
            log.exception("could not store the hand-off")
            return _error(500, "the hand-off could not be saved, please try again", session_id)
        return JSONResponse(stored, status_code=201, headers={"X-Session-Id": session_id})

    def need_store() -> None:
        if not manager.enabled:
            raise HTTPException(503, "KB store not configured: set MEDRAG_KB_STORE")

    @app.get("/admin/documents", dependencies=[admin])
    async def admin_documents() -> JSONResponse:
        need_store()
        return JSONResponse({"documents": await asyncio.to_thread(manager.listing)})

    @app.post("/admin/documents", dependencies=[admin])
    async def admin_upload(request: Request) -> JSONResponse:
        need_store()
        try:
            body = await _read_upload(request)
            content, file_name = _decode_file(body)
            summary = await asyncio.to_thread(manager.upload, body.get("kb_id"), body.get("diseases"), content, file_name)
        except TooLarge:
            return _error(413, f"the file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        except InvalidDocument as exc:
            return _error(400, str(exc))
        except KBBusy as exc:
            return _error(409, str(exc))
        except NotReady as exc:
            return _error(503, str(exc))
        return JSONResponse(summary)

    @app.delete("/admin/documents/{kb_id}", dependencies=[admin])
    async def admin_delete(kb_id: str) -> JSONResponse:
        need_store()
        try:
            check_kb_id(kb_id)
            await asyncio.to_thread(manager.delete, kb_id)
        except InvalidDocument as exc:
            return _error(400, str(exc))
        except NotFound:
            return _error(404, f"no KB Document {kb_id}")
        except KBBusy as exc:
            return _error(409, str(exc))
        except NotReady as exc:
            return _error(503, str(exc))
        return JSONResponse({"deleted": kb_id})

    @app.get("/admin", include_in_schema=False)
    async def admin_page() -> FileResponse:
        return FileResponse(STATIC / "admin.html")

    @app.post("/chat")
    async def chat(request: Request) -> JSONResponse:
        started = time.monotonic()
        try:
            body = await _read_json(request, settings.max_message_chars)
            message = _validate_message(body, settings.max_message_chars)
        except BadRequest as exc:
            return _error(400, str(exc))
        session_id = store.resolve(body.get("session_id") or request.headers.get("x-session-id"))
        deadline = time.monotonic() + settings.budget_s
        history = await load_history(session_id)
        meter = start_meter()  # before the pipeline task exists, so the gate adds to this meter
        usage = lambda: {USAGE_HEADER: meter.header_value()} if settings.usage_header else None  # noqa: E731

        def logged(status: str, result: ChatResponse | None = None) -> BackgroundTask:
            """The log write, to run once the response is decided: the latency is fixed here, the write comes after it."""
            kb_ids = list(dict.fromkeys(c.doc_id for c in result.citations)) if result else []
            return BackgroundTask(write_log, session_id, message, status, result.language if result else None, kb_ids,
                                  time.monotonic() - started, list(meter.calls))
        try:
            result = await asyncio.wait_for(
                _run_unless_disconnected(request, pipeline(message, deadline, history)),
                timeout=settings.budget_s,
            )
        except asyncio.TimeoutError:
            log.error("chat timed out after %.1fs", settings.budget_s)
            return _error(500, "the request timed out", session_id, usage(), logged(ERROR))
        except Loading:
            return _error(500, "the search index is still loading, try again shortly", session_id, usage(), logged(ERROR))  # 500: the contract has no 503
        except Busy as exc:
            log.warning("chat rejected, LLM busy: %s", exc)
            return _error(500, "the server is busy, try again shortly", session_id, usage(), logged(ERROR))
        except Exception as exc:
            log.exception("chat failed")
            return _error(500, f"internal error: {type(exc).__name__}", session_id, usage(), logged(ERROR))
        if result is None:  # client disconnected; nobody is listening
            return _error(499, "client closed the request", session_id, usage(), logged(ERROR))
        await save_turn(session_id, message, result)
        return JSONResponse(result.model_dump(), headers={"X-Session-Id": session_id, **(usage() or {})},
                            background=logged(result.status, result))

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
    """Entry point for uvicorn (`--factory`): KB from MEDRAG_KB_DIR (or the writable store MEDRAG_KB_STORE), models from env."""
    logging.basicConfig(level=logging.INFO)
    settings = Settings.from_env()
    store = None
    if settings.kb_store:
        store, docs = open_store(settings.kb_store, kb_dir())
        files_hash = ""  # the index key already covers every Section's text
    else:
        docs = load_knowledge_base(kb_dir())
        files_hash = kb_files_hash()
    sections = [s for d in docs for s in d.sections]
    log.info("loaded %d Sections from %d KB Documents", len(sections), len(docs))
    return create_app(sections, build_llm(settings), settings, retriever=build_retriever(sections, settings, files_hash),
                      kb_store=store, documents=tuple(docs))
