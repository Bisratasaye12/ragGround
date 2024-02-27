"""Run the golden set through retrieval and/or full answering and aggregate the metrics."""

import dataclasses
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ragground.evals.corpus import EvalOrg, relpath_of
from ragground.evals.golden import GoldenCase
from ragground.evals.judge import Judge
from ragground.evals.metrics import leaked, mean, percentile, score_retrieval
from ragground.generation.answer import AnswerResult, ChatConfig, DoneEvent, answer_question
from ragground.providers.llm import LLM
from ragground.retrieval.search import Retriever, SearchMode

ANSWERABLE = {"factual", "factual_restricted", "multi_hop", "injection"}
Progress = Callable[[str], None]


@dataclass
class RetrievalRun:
    mode: SearchMode
    per_case: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


async def run_retrieval(
    cases: list[GoldenCase],
    org: EvalOrg,
    retriever: Retriever,
    modes: list[SearchMode],
    k: int = 10,
    progress: Progress | None = None,
) -> list[RetrievalRun]:
    runs = []
    for mode in modes:
        run = RetrievalRun(mode=mode)
        latencies: list[float] = []
        for case in cases:
            user = org.users[case.as_user]
            result = await retriever.search(user, case.question, mode, k=k)
            latencies.append(result.timings_ms["total"])
            uris = [h.source_uri for h in result.hits]
            row: dict[str, Any] = {
                "id": case.id,
                "category": case.category,
                "latency_ms": result.timings_ms["total"],
                "top": [f"{relpath_of(h.source_uri)} :: {h.heading_path}" for h in result.hits[:3]],
                "leak": leaked(uris, case.forbidden_files_prefix),
            }
            if case.category in ANSWERABLE:
                row.update(dataclasses.asdict(score_retrieval(case, result.hits)))
            run.per_case.append(row)
        run.summary = summarise_retrieval(run.per_case, latencies)
        if progress:
            s = run.summary["overall"]
            progress(
                f"  {mode:14s} recall@5={s['recall_at_5']:.3f} mrr={s['mrr']:.3f} "
                f"p50={run.summary['latency_p50_ms']}ms"
            )
        runs.append(run)
    return runs


def summarise_retrieval(rows: list[dict[str, Any]], latencies: list[float]) -> dict[str, Any]:
    metric_keys = ["hit_at_1", "hit_at_5", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_10"]
    scored = [r for r in rows if "mrr" in r]
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in scored:
        by_cat[r["category"]].append(r)
    acl_rows = [r for r in rows if r["category"] == "acl"]
    return {
        "overall": {m: mean([r[m] for r in scored]) for m in metric_keys} | {"n": len(scored)},
        "by_category": {
            cat: {m: mean([r[m] for r in rs]) for m in metric_keys} | {"n": len(rs)}
            for cat, rs in sorted(by_cat.items())
        },
        "acl_leak_rate": mean([float(r["leak"]) for r in acl_rows]) if acl_rows else 0.0,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
    }


async def run_answers(
    cases: list[GoldenCase],
    org: EvalOrg,
    retriever: Retriever,
    llm: LLM,
    cfg: ChatConfig,
    judge: Judge | None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for i, case in enumerate(cases, start=1):
        t0 = time.perf_counter()
        result: AnswerResult | None = None
        async for event in answer_question(
            org.users[case.as_user], case.question, retriever, llm, cfg
        ):
            if isinstance(event, DoneEvent):
                result = event.result
        assert result is not None
        row: dict[str, Any] = {
            "id": case.id,
            "category": case.category,
            "question": case.question,
            "answer": result.answer,
            "abstained": result.abstained,
            "abstain_reason": result.abstain_reason,
            "abstain_correct": result.abstained == case.expect_abstain,
            "cited": result.cited,
            "invalid_citations": result.invalid_citations,
            "uncited": result.uncited,
            "error": result.error,
            "sources": [
                {
                    "n": s.n,
                    "file": relpath_of(s.source_uri),
                    "section": s.heading_path,
                    "rerank": s.rerank_score,
                }
                for s in result.sources
            ],
            "leak": leaked([s.source_uri for s in result.sources], case.forbidden_files_prefix),
            "injection_violation": any(
                b.lower() in result.answer.lower() for b in case.must_not_contain
            ),
            "latency_ms": result.timings_ms.get("total"),
            "output_tokens": result.output_tokens,
        }
        if judge is not None and not result.abstained and not result.error and result.answer:
            faith = await judge.faithfulness(result.answer, [s.text for s in result.sources])
            row["faithfulness"] = faith.score
            row["faithfulness_detail"] = faith.detail
            if case.expected_answer:
                corr = await judge.correctness(case.question, case.expected_answer, result.answer)
                row["correctness"] = corr.score
                row["correctness_detail"] = corr.detail
        rows.append(row)
        if progress:
            progress(
                f"  [{i}/{len(cases)}] {case.id}: "
                f"{'abstained' if result.abstained else 'answered'} "
                f"({'ok' if row['abstain_correct'] else 'WRONG'}) "
                f"faith={row.get('faithfulness')} corr={row.get('correctness')} "
                f"{(time.perf_counter() - t0):.0f}s"
            )
    return {"per_case": rows, "summary": summarise_answers(rows)}


def summarise_answers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def share(pred: Callable[[dict[str, Any]], bool], subset: list[dict[str, Any]]) -> float:
        return mean([float(pred(r)) for r in subset]) if subset else float("nan")

    should_abstain = [r for r in rows if r["category"] in {"unanswerable", "acl"}]
    should_answer = [r for r in rows if r["category"] in ANSWERABLE]
    answered = [r for r in rows if not r["abstained"] and not r["error"]]
    faith = [r["faithfulness"] for r in rows if r.get("faithfulness") is not None]
    corr = [r["correctness"] for r in rows if r.get("correctness") is not None]
    injection = [r for r in rows if r["category"] == "injection"]
    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms")]
    return {
        "n": len(rows),
        "abstain_accuracy_on_unanswerable": share(lambda r: r["abstained"], should_abstain),
        "answer_rate_on_answerable": share(lambda r: not r["abstained"], should_answer),
        "faithfulness": mean(faith),
        "faithfulness_n": len(faith),
        "correctness": mean(corr),
        "correctness_n": len(corr),
        "citation_rate": share(lambda r: bool(r["cited"]), answered),
        "invalid_citation_rate": share(lambda r: bool(r["invalid_citations"]), answered),
        "injection_violation_rate": share(lambda r: r["injection_violation"], injection),
        "acl_leak_rate": share(lambda r: r["leak"], [r for r in rows if r["category"] == "acl"]),
        "errors": sum(1 for r in rows if r["error"]),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
    }
