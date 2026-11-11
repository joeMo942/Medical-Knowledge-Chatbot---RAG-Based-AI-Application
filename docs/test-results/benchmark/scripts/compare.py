"""Side-by-side table of two benchmark arms: python compare.py <results dir> <labelA> <labelB>"""
import json
import statistics
import sys
from pathlib import Path

D, A, B = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
SETS = ["main", "issue5", "issue7", "issue8", "issue32"]


def load(label, s):
    return json.loads((D / f"2026-11-11-{label}-{s}.json").read_text(encoding="utf-8"))


def frac(x):
    return f"{x['ok']}/{x['n']}"


def pctl(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))]


print("| set | model | status | language | citations (expected doc) | snippets | relevance | contract | non-200 | p50 / p95 s | prompt / completion tok per turn |")
print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
allrows = {A: [], B: []}
for s in SETS:
    for lab in (A, B):
        d = load(lab, s); m = d["summary"]
        allrows[lab] += d["rows"]
        lat = [r["latency_s"] for r in d["rows"] if r["latency_s"] is not None]
        t = m["tokens"]
        print(f"| {s} | {lab} | {frac(m['status']['overall'])} | {frac(m['language']['overall'])} | "
              f"{frac(m['citation']['expected_doc_hit'])} | {frac(m['snippet']['citations_valid'])} | {frac(m['relevance']['overall'])} | "
              f"{m['contract']['violations']} | {m['http']['non_200']} | {pctl(lat, .5):.2f} / {pctl(lat, .95):.2f} | "
              f"{t['prompt_mean']:.0f} / {t['completion_mean']:.0f} |")
print()
for lab, rows in allrows.items():
    lat = [r["latency_s"] for r in rows if r["latency_s"] is not None]
    ans = [r["latency_s"] for r in rows if r["latency_s"] is not None and r["actual_status"] == "answered"]
    ok = sum(bool(r["status_ok"]) for r in rows)
    print(f"{lab}: all turns {ok}/{len(rows)} status; latency p50 {pctl(lat, .5):.2f} p95 {pctl(lat, .95):.2f} max {max(lat):.2f}; "
          f"answered turns p50 {pctl(ans, .5):.2f} p95 {pctl(ans, .95):.2f}; non-200 {sum(r['http_status'] != 200 for r in rows)}")
print()
key = lambda r: (r["case_id"], r["turn"])
a = {key(r): r for r in allrows[A]}; b = {key(r): r for r in allrows[B]}
print("Turns where the two disagree on status correctness:")
for k in sorted(a.keys() & b.keys()):
    if bool(a[k]["status_ok"]) != bool(b[k]["status_ok"]):
        print(f"- {k[0]} t{k[1]} (expected {a[k]['expected_status']}): {A}={a[k]['actual_status']} {B}={b[k]['actual_status']} | {a[k]['message'][:70]}")
print("\nRelevance disagreements:")
for k in sorted(a.keys() & b.keys()):
    if a[k].get("relevance_ok") is not None and b[k].get("relevance_ok") is not None and a[k]["relevance_ok"] != b[k]["relevance_ok"]:
        print(f"- {k[0]} t{k[1]}: {A}={a[k]['relevance_ok']} {B}={b[k]['relevance_ok']}")
