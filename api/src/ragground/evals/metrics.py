"""Retrieval and answer metrics. Pure functions over hits/results so they're unit-testable."""

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from ragground.evals.corpus import URI_PREFIX
from ragground.evals.golden import Evidence, GoldenCase, normalise
from ragground.retrieval.search import Hit


def is_relevant(hit: Hit, ev: Evidence) -> bool:
    return hit.source_uri == URI_PREFIX + ev.file and normalise(ev.text) in normalise(hit.text)


@dataclass
class RetrievalScores:
    hit_at_1: float
    hit_at_5: float
    recall_at_5: float  # share of the case's evidence items found in the top 5
    recall_at_10: float
    mrr: float
    ndcg_at_10: float


def score_retrieval(case: GoldenCase, hits: Sequence[Hit]) -> RetrievalScores:
    evidence = case.evidence
    relevant_flags = [any(is_relevant(h, ev) for ev in evidence) for h in hits]

    def coverage(k: int) -> float:
        found = sum(1 for ev in evidence if any(is_relevant(h, ev) for h in hits[:k]))
        return found / len(evidence)

    first = next((i for i, r in enumerate(relevant_flags, start=1) if r), None)
    dcg = sum(1 / math.log2(i + 1) for i, r in enumerate(relevant_flags[:10], start=1) if r)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(evidence), 10) + 1))
    return RetrievalScores(
        hit_at_1=float(bool(relevant_flags[:1] and relevant_flags[0])),
        hit_at_5=float(any(relevant_flags[:5])),
        recall_at_5=coverage(5),
        recall_at_10=coverage(10),
        mrr=1 / first if first else 0.0,
        ndcg_at_10=min(dcg / ideal, 1.0) if ideal else 0.0,
    )


def leaked(hits_or_uris: Sequence[str], forbidden_prefix: str | None) -> bool:
    """True if any retrieved source is under the forbidden prefix."""
    if not forbidden_prefix:
        return False
    return any(uri.startswith(URI_PREFIX + forbidden_prefix) for uri in hits_or_uris)


def mean(xs: Sequence[float]) -> float:
    return round(statistics.fmean(xs), 4) if xs else float("nan")


def percentile(xs: Sequence[float], p: float) -> float:
    if not xs:
        return float("nan")
    ordered = sorted(xs)
    idx = min(len(ordered) - 1, max(0, math.ceil(p / 100 * len(ordered)) - 1))
    return round(ordered[idx], 1)
