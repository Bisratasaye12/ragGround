"""Golden-set schema, loading and label validation.

Relevance is labelled with *evidence phrases* tied to a file, not chunk ids: a retrieved chunk is
relevant when it comes from that file and contains the phrase (after normalisation). Labels
therefore survive re-chunking, which is what lets the experiment matrix compare chunkers.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Category = Literal["factual", "factual_restricted", "multi_hop", "unanswerable", "acl", "injection"]
EvalUser = Literal["admin", "member", "outsider"]


@dataclass(frozen=True)
class Evidence:
    file: str  # path relative to the corpus root, e.g. "tutorial/cors.md"
    text: str


@dataclass
class GoldenCase:
    id: str
    category: Category
    as_user: EvalUser
    question: str
    expected_answer: str | None
    evidence: list[Evidence]
    expect_abstain: bool
    must_not_contain: list[str] = field(default_factory=list)
    forbidden_files_prefix: str | None = None


_MARKUP = re.compile(r"[*`_]")
_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>]*)?/?>")
_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Lower-case, drop HTML tags and Markdown emphasis/code markers, collapse whitespace,
    unify quotes. Applied to both evidence phrases and chunk text before matching."""
    text = _TAG.sub("", text)
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return _WS.sub(" ", _MARKUP.sub("", text)).strip().lower()


def load_golden(path: Path) -> list[GoldenCase]:
    cases = []
    for n, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        try:
            cases.append(
                GoldenCase(
                    id=raw["id"],
                    category=raw["category"],
                    as_user=raw["as_user"],
                    question=raw["question"],
                    expected_answer=raw.get("expected_answer"),
                    evidence=[Evidence(**e) for e in raw.get("evidence", [])],
                    expect_abstain=raw["expect_abstain"],
                    must_not_contain=raw.get("must_not_contain", []),
                    forbidden_files_prefix=raw.get("forbidden_files_prefix"),
                )
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{path}:{n}: invalid case: {exc}") from exc
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids in golden set")
    return cases


def validate_evidence(cases: list[GoldenCase], corpus_root: Path) -> list[str]:
    """Every evidence phrase must occur in its file. Returns human-readable problems."""
    problems = []
    cache: dict[str, str] = {}
    for case in cases:
        if not case.expect_abstain and not case.evidence:
            problems.append(f"{case.id}: answerable case without evidence")
        for ev in case.evidence:
            if ev.file not in cache:
                path = corpus_root / ev.file
                if not path.exists():
                    problems.append(f"{case.id}: missing file {ev.file}")
                    cache[ev.file] = ""
                    continue
                cache[ev.file] = normalise(path.read_text())
            if normalise(ev.text) not in cache[ev.file]:
                problems.append(f"{case.id}: phrase not found in {ev.file}: {ev.text!r}")
    return problems
