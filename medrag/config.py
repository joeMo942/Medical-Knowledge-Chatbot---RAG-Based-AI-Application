"""Runtime settings, read from environment variables only (no secrets in the repo)."""
from __future__ import annotations

import os
from dataclasses import dataclass

# What retrieval actually uses right now; /health reports this string verbatim.
# Replaced by the real model name when hybrid BGE-M3 search lands (#6).
EMBEDDING_PLACEHOLDER = "none (lexical placeholder until #6)"


@dataclass(frozen=True)
class Settings:
    backend: str = "ollama"  # "ollama" or "openai" (any OpenAI-compatible endpoint)
    ollama_url: str = "http://ollama:11434"
    ollama_model: str = "qwen3.5:4b"
    ollama_num_ctx: int = 8192
    openai_base_url: str = ""
    openai_model: str = ""
    openai_api_key: str = ""
    top_k: int = 4
    # Total time budget for the whole /chat pipeline; the contract allows 30 s.
    budget_s: float = 25.0
    max_message_chars: int = 2000
    llm_concurrency: int = 1  # Ollama serves one request at a time
    llm_est_s: float = 5.0  # initial guess of one LLM call, for the fail-fast queue check

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ.get
        d = cls()
        return cls(
            backend=(e("MEDRAG_LLM_BACKEND") or d.backend).lower(),
            ollama_url=e("MEDRAG_OLLAMA_URL") or d.ollama_url,
            ollama_model=e("MEDRAG_LLM_MODEL") or d.ollama_model,
            ollama_num_ctx=int(e("MEDRAG_OLLAMA_NUM_CTX") or d.ollama_num_ctx),
            openai_base_url=e("MEDRAG_OPENAI_BASE_URL") or d.openai_base_url,
            openai_model=e("MEDRAG_OPENAI_MODEL") or d.openai_model,
            openai_api_key=e("MEDRAG_OPENAI_API_KEY") or d.openai_api_key,
            top_k=int(e("MEDRAG_TOP_K") or d.top_k),
            budget_s=float(e("MEDRAG_BUDGET_S") or d.budget_s),
            max_message_chars=int(e("MEDRAG_MAX_MESSAGE_CHARS") or d.max_message_chars),
            llm_concurrency=int(e("MEDRAG_LLM_CONCURRENCY") or d.llm_concurrency),
            llm_est_s=float(e("MEDRAG_LLM_EST_S") or d.llm_est_s),
        )
