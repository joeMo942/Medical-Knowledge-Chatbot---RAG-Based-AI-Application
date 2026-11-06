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
    # Floor of the gate's duration estimate for one LLM call (the estimate is the
    # slowest recent call, never below this); used by the fail-fast queue check.
    llm_est_s: float = 5.0
    # Each call's output cap (Ollama num_predict / OpenAI max_tokens) is sized to the time
    # the request has left: (remaining - llm_margin_s) x decode_tok_s, clamped to
    # [min_output_tokens, max_output_tokens]. Measured on the RTX 5070 Laptop: 52-63 tok/s
    # decode, prompt evaluation <= 1 s. A reply cut off at its cap becomes out_of_scope.
    max_output_tokens: int = 1200
    min_output_tokens: int = 150
    decode_tok_s: float = 53.0
    llm_margin_s: float = 2.5

    def __post_init__(self) -> None:
        if self.budget_s <= self.llm_est_s:
            # Every request would be refused as "busy" before reaching the model.
            raise ValueError(f"MEDRAG_BUDGET_S ({self.budget_s}) must be above MEDRAG_LLM_EST_S ({self.llm_est_s})")

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
            max_output_tokens=int(e("MEDRAG_MAX_OUTPUT_TOKENS") or d.max_output_tokens),
            min_output_tokens=int(e("MEDRAG_MIN_OUTPUT_TOKENS") or d.min_output_tokens),
            decode_tok_s=float(e("MEDRAG_DECODE_TOK_S") or d.decode_tok_s),
            llm_margin_s=float(e("MEDRAG_LLM_MARGIN_S") or d.llm_margin_s),
        )
