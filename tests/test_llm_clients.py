"""Request shape and result parsing of both LLM clients, against a mocked HTTP transport."""
import asyncio
import json

import httpx
import pytest

from medrag.answering import ANSWER_SCHEMA
from medrag.llm import KEEP_ALIVE, LLMError, OllamaClient, OpenAICompatClient

REPLY = {"answer": "x", "status": "answered", "citations": []}


def run(coro):
    return asyncio.run(coro)


def test_ollama_request_shape_and_result():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"], seen["body"] = request.url.path, json.loads(request.content)
        return httpx.Response(200, json={
            "message": {"content": json.dumps(REPLY)},
            "prompt_eval_count": 123, "eval_count": 45,
            "total_duration": 4_000_000_000, "prompt_eval_duration": 1_500_000_000,
            "eval_duration": 2_000_000_000})

    c = OllamaClient("http://ollama:11434", "qwen3.5:4b", num_ctx=4321, timeout=77,
                     transport=httpx.MockTransport(handler))
    res = run(c.complete_json("sys", "user", ANSWER_SCHEMA))
    b = seen["body"]
    assert seen["path"] == "/api/chat"
    assert b["model"] == "qwen3.5:4b" and b["think"] is False and b["stream"] is False
    assert b["format"] == ANSWER_SCHEMA and b["keep_alive"] == KEEP_ALIVE
    assert b["options"]["num_ctx"] == 4321 and b["options"]["temperature"] == 0
    assert [m["role"] for m in b["messages"]] == ["system", "user"]
    assert c._http.timeout.read == 77 and c.timeout == 77
    assert res.data == REPLY
    assert (res.prompt_tokens, res.completion_tokens) == (123, 45)
    assert (res.total_s, res.prompt_eval_s, res.eval_s) == (4.0, 1.5, 2.0)


def test_ollama_warmup_uses_the_same_context_and_keep_alive():
    seen = {}

    def handler(request):
        seen["path"], seen["body"] = request.url.path, json.loads(request.content)
        return httpx.Response(200, json={})

    c = OllamaClient("http://o", "m:1", num_ctx=4321, transport=httpx.MockTransport(handler))
    run(c.warm())
    assert seen["path"] == "/api/generate"
    assert seen["body"]["options"]["num_ctx"] == 4321 and seen["body"]["keep_alive"] == KEEP_ALIVE


def test_ollama_bad_json_and_http_errors_raise_llm_error():
    def bad(request):
        return httpx.Response(200, json={"message": {"content": "not json"}})

    def down(request):
        return httpx.Response(500, text="boom")

    for handler in (bad, down):
        c = OllamaClient("http://o", "m:1", transport=httpx.MockTransport(handler))
        with pytest.raises(LLMError):
            run(c.complete_json("s", "u", ANSWER_SCHEMA))


def test_ollama_ready_checks_the_model_is_present():
    def tags(models):
        return lambda request: httpx.Response(200, json={"models": [{"name": n} for n in models]})

    ok = OllamaClient("http://o", "qwen3.5:4b", transport=httpx.MockTransport(tags(["qwen3.5:4b"])))
    missing = OllamaClient("http://o", "qwen3.5:4b", transport=httpx.MockTransport(tags(["other:1b"])))
    assert run(ok.ready()) and not run(missing.ready())

    def refuse(request):
        raise httpx.ConnectError("down")

    assert not run(OllamaClient("http://o", "m:1", transport=httpx.MockTransport(refuse)).ready())


def test_openai_request_shape_and_result():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"], seen["auth"] = request.url.path, request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps(REPLY)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

    c = OpenAICompatClient("https://api.example.com/v1/", "gpt-x", "sk-test", timeout=33,
                           transport=httpx.MockTransport(handler))
    res = run(c.complete_json("sys", "user", ANSWER_SCHEMA))
    b = seen["body"]
    assert seen["path"] == "/v1/chat/completions" and seen["auth"] == "Bearer sk-test"
    assert b["model"] == "gpt-x" and b["temperature"] == 0
    js = b["response_format"]["json_schema"]
    assert js["strict"] is True and js["schema"] == ANSWER_SCHEMA
    assert c._http.timeout.read == 33
    assert res.data == REPLY and (res.prompt_tokens, res.completion_tokens) == (10, 5)
    assert res.total_s is not None and res.prompt_eval_s is None


def test_answer_schema_is_strict_mode_compatible():
    def check(node):
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
            for child in node["properties"].values():
                check(child)
        if node.get("type") == "array":
            check(node["items"])

    check(ANSWER_SCHEMA)
