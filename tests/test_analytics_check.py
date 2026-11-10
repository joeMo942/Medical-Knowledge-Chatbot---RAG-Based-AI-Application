"""evaluation/analytics_check.py: the run-vs-analytics comparison (pure functions, no server)."""
from evaluation.analytics_check import compare, iso_utc

REPORT = {
    "summary": {"turns": 3, "unanswered": 2, "by_status": {"answered": 1, "escalated": 1, "out_of_scope": 1}},
    "top_questions": [{"count": 2}, {"count": 1}],
    "latency_ms": {"p50": 1000.0, "p95": 3000.0},
    "tokens_by_call_type": {
        "by_kind": [{"kind": "answer", "calls": 2, "prompt_tokens": 30, "completion_tokens": 6, "mean_latency_ms": 900.0}],
        "llm_total": {"calls": 3, "prompt_tokens": 40, "completion_tokens": 8, "total_tokens": 48}},
    "tokens_per_turn": {"turns": 3, "mean_prompt_tokens": 14.0, "mean_completion_tokens": 3.0, "mean_total_tokens": 17.0},
}


def rows(usage=True):
    u = lambda p, c, n: {"prompt_tokens": p, "completion_tokens": c, "llm_calls": n} if usage else None  # noqa: E731
    return [
        {"http_status": 200, "actual_status": "answered", "latency_s": 3.0, "usage": u(20, 4, 2)},
        {"http_status": 200, "actual_status": "out_of_scope", "latency_s": 1.0, "usage": u(20, 4, 1)},
        {"http_status": 200, "actual_status": "escalated", "latency_s": 0.1, "usage": u(0, 0, 0)},
    ]


def test_matching_numbers_pass_and_latency_is_only_reported():
    lines, ok = compare(rows(), REPORT)
    assert ok
    assert any(line.startswith("info latency p50") for line in lines)


def test_a_difference_in_counts_or_tokens_is_flagged():
    bad = {**REPORT, "summary": {**REPORT["summary"], "unanswered": 1}}
    lines, ok = compare(rows(), bad)
    assert not ok and any(line.startswith("DIFF unanswered") for line in lines)
    r = rows()
    r[0]["usage"]["prompt_tokens"] = 21
    assert not compare(r, REPORT)[1]


def test_non_200_turns_count_as_error_and_missing_usage_skips_the_token_check():
    r = rows(usage=False)
    r[2] = {"http_status": 500, "actual_status": None, "latency_s": 2.0, "usage": None}
    report = {**REPORT, "summary": {"turns": 3, "unanswered": 1, "by_status": {"answered": 1, "error": 1, "out_of_scope": 1}}}
    lines, ok = compare(r, report)
    assert ok and any(line.startswith("skip tokens") for line in lines)


def test_iso_utc_is_a_utc_timestamp_the_endpoint_parses():
    assert iso_utc(0) == "1970-01-01T00:00:00.000000Z"
