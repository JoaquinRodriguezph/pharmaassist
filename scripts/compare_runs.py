"""Compare eval_agent.py logs side by side.

    py scripts/compare_runs.py eval/runs/agent-q4.jsonl eval/runs/agent-q8.jsonl --labels q4 q8

For each log: questions, passed (overall and per route), median and total latency, median
steps, largest single prompt, and how often each guard fired. Then a per-question table of
pass/fail for the questions present in every log, so a subset run (Gemini) is compared with
the same questions from the full runs.
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROUTES = ("rag", "rag_unanswerable", "tool", "both", "rag_then_refer", "refuse")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summary(rows: list[dict]) -> dict:
    latencies = [r["latency_seconds"] for r in rows if r.get("latency_seconds") is not None]
    prompts = [r["max_prompt_tokens"] for r in rows if r.get("max_prompt_tokens") is not None]
    return {
        "questions": len(rows),
        "passed": sum(r["passed"] for r in rows),
        "routes": {route: (sum(r["passed"] for r in rows if r["route"] == route),
                           sum(1 for r in rows if r["route"] == route)) for route in ROUTES},
        "median_latency": statistics.median(latencies) if latencies else None,
        "total_latency": sum(latencies) if latencies else None,
        "median_steps": statistics.median([r["steps"] for r in rows]) if rows else None,
        "max_prompt": max(prompts) if prompts else None,
        "guards": Counter((r.get("blocked_by") or "").split(":")[0] for r in rows if r.get("blocked_by")),
        "model": next((r["model"] for r in rows if r.get("model")), None),
    }


def fmt(value, spec="") -> str:
    return "-" if value is None else format(value, spec)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+")
    args = parser.parse_args(argv)
    labels = args.labels or [p.stem for p in args.logs]
    if len(labels) != len(args.logs):
        print("ERROR: give one label per log")
        return 1

    runs = {label: load(path) for label, path in zip(labels, args.logs)}
    summaries = {label: summary(rows) for label, rows in runs.items()}
    width = max(12, *(len(l) for l in labels)) + 2

    def line(name, values):
        print(f"{name:<28}" + "".join(f"{v:>{width}}" for v in values))

    line("", labels)
    line("model", [str(summaries[l]["model"])[-width + 1:] for l in labels])
    line("passed", [f"{summaries[l]['passed']} of {summaries[l]['questions']}" for l in labels])
    for route in ROUTES:
        line(f"  {route}", [f"{p} of {n}" if n else "-" for p, n in (summaries[l]["routes"][route] for l in labels)])
    line("median latency (s)", [fmt(summaries[l]["median_latency"], ".1f") for l in labels])
    line("total latency (s)", [fmt(summaries[l]["total_latency"], ".0f") for l in labels])
    line("median steps", [fmt(summaries[l]["median_steps"]) for l in labels])
    line("largest prompt (tokens)", [fmt(summaries[l]["max_prompt"]) for l in labels])
    guard_names = sorted(set().union(*(summaries[l]["guards"] for l in labels)))
    for guard in guard_names:
        line(f"  guard {guard}"[:27], [str(summaries[l]["guards"].get(guard, 0)) for l in labels])

    common = [r["id"] for r in runs[labels[0]] if all(any(x["id"] == r["id"] for x in runs[l]) for l in labels)]
    print(f"\nper question ({len(common)} in every log)")
    line("id / route", labels)
    by_id = {l: {r["id"]: r for r in runs[l]} for l in labels}
    for qid in common:
        route = by_id[labels[0]][qid]["route"]
        line(f"{qid} {route}", ["Y" if by_id[l][qid]["passed"] else "n" for l in labels])
    if len(common) < max(len(rows) for rows in runs.values()):
        subset = {l: sum(by_id[l][q]["passed"] for q in common) for l in labels}
        line("passed on these", [f"{subset[l]} of {len(common)}" for l in labels])
    return 0


if __name__ == "__main__":
    sys.exit(main())