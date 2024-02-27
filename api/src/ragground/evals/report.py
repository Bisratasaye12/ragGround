"""Report files (JSON for machines, Markdown for humans and PR comments) and the regression gate."""

import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Metrics where higher is better, checked by `compare`. Safety metrics have hard limits instead.
RETRIEVAL_GATED = ["recall_at_5", "mrr", "ndcg_at_10"]
ANSWER_GATED = [
    "abstain_accuracy_on_unanswerable",
    "answer_rate_on_answerable",
    "faithfulness",
    "correctness",
    "citation_rate",
]
MUST_BE_ZERO = ["acl_leak_rate", "injection_violation_rate"]


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or None
    except OSError:
        return None


def write_report(out_dir: Path, name: str, payload: dict[str, Any], markdown: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"created_at": datetime.now(UTC).isoformat(), "git_sha": git_sha(), **payload}
    json_path = out_dir / f"{name}.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    (out_dir / f"{name}.md").write_text(markdown)
    return json_path


def _fmt(x: object) -> str:
    if isinstance(x, float):
        return "n/a" if math.isnan(x) else f"{x:.3f}"
    return str(x)


def retrieval_markdown(title: str, config: dict[str, Any], runs: list[dict[str, Any]]) -> str:
    lines = [
        f"## {title}",
        "",
        "| " + " | ".join(f"`{k}`={v}" for k, v in config.items()) + " |",
        "|" + "---|" * len(config),
        "",
        "| mode | Recall@5 | Recall@10 | MRR | nDCG@10 | Hit@1 | ACL leak | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        o = r["summary"]["overall"]
        lines.append(
            f"| {r['mode']} | {_fmt(o['recall_at_5'])} | {_fmt(o['recall_at_10'])} | "
            f"{_fmt(o['mrr'])} | {_fmt(o['ndcg_at_10'])} | {_fmt(o['hit_at_1'])} | "
            f"{_fmt(r['summary']['acl_leak_rate'])} | {r['summary']['latency_p50_ms']} | "
            f"{r['summary']['latency_p95_ms']} |"
        )
    lines += ["", f"Answerable cases scored: {runs[0]['summary']['overall']['n'] if runs else 0}."]
    return "\n".join(lines) + "\n"


def answers_markdown(title: str, config: dict[str, Any], summary: dict[str, Any]) -> str:
    rows = [
        ("Abstains on unanswerable / no-access", summary["abstain_accuracy_on_unanswerable"]),
        ("Answers answerable questions", summary["answer_rate_on_answerable"]),
        (f"Faithfulness (LLM judge, n={summary['faithfulness_n']})", summary["faithfulness"]),
        (f"Correctness (LLM judge, n={summary['correctness_n']})", summary["correctness"]),
        ("Answers with ≥1 valid citation", summary["citation_rate"]),
        ("Answers with invalid citations", summary["invalid_citation_rate"]),
        ("ACL leak rate (must be 0)", summary["acl_leak_rate"]),
        ("Prompt-injection violations (must be 0)", summary["injection_violation_rate"]),
        ("Errors", summary["errors"]),
        ("Latency p50 / p95 (ms)", f"{summary['latency_p50_ms']} / {summary['latency_p95_ms']}"),
    ]
    head = " · ".join(f"`{k}`={v}" for k, v in config.items())
    body = "\n".join(f"| {k} | {_fmt(v)} |" for k, v in rows)
    return (
        f"## {title}\n\n{head}\n\n| Metric | Value |\n|---|---|\n{body}\n\nCases: {summary['n']}\n"
    )


def compare(baseline: dict[str, Any], current: dict[str, Any], tolerance: float) -> list[str]:
    """Regressions of `current` vs `baseline`. Works for retrieval and answer reports."""
    problems: list[str] = []

    def check(scope: str, base: dict[str, Any], cur: dict[str, Any], gated: list[str]) -> None:
        for m in MUST_BE_ZERO:
            if m in cur and isinstance(cur[m], float) and not math.isnan(cur[m]) and cur[m] > 0:
                problems.append(f"{scope}: {m} = {cur[m]} (must be 0)")
        for m in gated:
            b, c = base.get(m), cur.get(m)
            if not (isinstance(b, float) and isinstance(c, float)):
                continue
            if math.isnan(b) or math.isnan(c):
                continue
            if c < b - tolerance:
                problems.append(f"{scope}: {m} dropped {b:.3f} -> {c:.3f} (tolerance {tolerance})")

    if "retrieval" in current:
        base_modes = {r["mode"]: r["summary"] for r in baseline.get("retrieval", [])}
        for run in current["retrieval"]:
            if run["mode"] in base_modes:
                cur = run["summary"]["overall"] | {"acl_leak_rate": run["summary"]["acl_leak_rate"]}
                check(
                    f"retrieval/{run['mode']}",
                    base_modes[run["mode"]]["overall"],
                    cur,
                    RETRIEVAL_GATED,
                )
    if "answers" in current:
        check(
            "answers",
            baseline.get("answers", {}).get("summary", {}),
            current["answers"]["summary"],
            ANSWER_GATED,
        )
    return problems
