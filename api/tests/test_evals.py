import uuid
from pathlib import Path

import pytest

from ragground.evals.golden import Evidence, GoldenCase, load_golden, normalise
from ragground.evals.judge import Judge
from ragground.evals.metrics import is_relevant, leaked, score_retrieval
from ragground.evals.report import compare
from ragground.providers.llm import ScriptedLLM
from ragground.retrieval.search import Hit

GOLDEN = Path(__file__).resolve().parents[2] / "evals" / "golden" / "golden.jsonl"


def _hit(file: str, text: str) -> Hit:
    return Hit(uuid.uuid4(), uuid.uuid4(), "t", f"eval://{file}", "", text, 0)


def _case(*evidence: tuple[str, str]) -> GoldenCase:
    return GoldenCase(
        id="c",
        category="factual",
        as_user="outsider",
        question="q",
        expected_answer="a",
        evidence=[Evidence(f, t) for f, t in evidence],
        expect_abstain=False,
    )


def test_normalise_ignores_markup_html_and_quotes() -> None:
    raw = 'Install <a href="x">`python-multipart`</a> **first**, it\u2019s "needed".'
    assert normalise(raw) == 'install python-multipart first, it\'s "needed".'


def test_relevance_requires_matching_file_and_phrase() -> None:
    ev = Evidence("tutorial/cors.md", "An origin is the combination of protocol")
    assert is_relevant(
        _hit("tutorial/cors.md", "An **origin** is the combination of protocol, ..."), ev
    )
    assert not is_relevant(
        _hit("tutorial/other.md", "An origin is the combination of protocol"), ev
    )
    assert not is_relevant(_hit("tutorial/cors.md", "unrelated"), ev)


def test_retrieval_scores() -> None:
    case = _case(("a.md", "alpha fact"), ("b.md", "beta fact"))
    hits = [
        _hit("x.md", "noise"),
        _hit("a.md", "the alpha fact here"),
        _hit("y.md", "noise"),
        _hit("z.md", "noise"),
        _hit("w.md", "noise"),
        _hit("b.md", "beta fact"),
    ]
    s = score_retrieval(case, hits)
    assert s.hit_at_1 == 0.0
    assert s.hit_at_5 == 1.0
    assert s.mrr == 0.5  # first relevant at rank 2
    assert s.recall_at_5 == 0.5  # only alpha within top 5
    assert s.recall_at_10 == 1.0
    assert 0 < s.ndcg_at_10 < 1


def test_leak_detection() -> None:
    assert leaked(["eval://tutorial/security/x.md"], "tutorial/security/")
    assert not leaked(["eval://tutorial/cors.md"], "tutorial/security/")
    assert not leaked(["eval://tutorial/security/x.md"], None)


async def test_judge_parses_json_even_with_chatter() -> None:
    llm = ScriptedLLM(
        'Sure! {"claims": [{"claim": "a", "supported": true}, '
        '{"claim": "b", "supported": false}]} Hope that helps.'
    )
    j = await Judge(llm).faithfulness("answer", ["source"])
    assert j.score == 0.5
    assert j.detail == {"unsupported": ["b"]}


async def test_judge_returns_none_on_garbage() -> None:
    assert (await Judge(ScriptedLLM("no json here")).faithfulness("a", ["s"])).score is None
    corr = await Judge(ScriptedLLM('{"score": 9}')).correctness("q", "ref", "a")
    assert corr.score is None  # out of the 1-5 range


async def test_judge_correctness_rescales() -> None:
    corr = await Judge(ScriptedLLM('{"score": 4, "reason": "minor gap"}')).correctness(
        "q", "r", "a"
    )
    assert corr.score == 0.75


def _retrieval_report(recall: float, leak: float) -> dict[str, object]:
    return {
        "retrieval": [
            {
                "mode": "hybrid",
                "summary": {
                    "overall": {"recall_at_5": recall, "mrr": 0.8, "ndcg_at_10": 0.8},
                    "acl_leak_rate": leak,
                },
            }
        ]
    }


def test_compare_flags_regressions_and_leaks() -> None:
    base = _retrieval_report(0.90, 0.0)
    assert compare(base, _retrieval_report(0.89, 0.0), tolerance=0.02) == []
    drop = compare(base, _retrieval_report(0.80, 0.0), tolerance=0.02)
    assert len(drop) == 1 and "recall_at_5" in drop[0]
    leak = compare(base, _retrieval_report(0.95, 0.125), tolerance=0.02)
    assert any("acl_leak_rate" in p for p in leak)


def test_golden_set_is_well_formed() -> None:
    cases = load_golden(GOLDEN)
    assert len(cases) >= 100
    for c in cases:
        assert c.expect_abstain == (c.category in {"unanswerable", "acl"}), c.id
        if not c.expect_abstain:
            assert c.evidence and c.expected_answer, c.id
        if c.category == "acl":
            assert c.as_user == "outsider" and c.forbidden_files_prefix, c.id


@pytest.mark.parametrize("bad", ['{"id": "x"}', "not json"])
def test_load_golden_rejects_bad_rows(tmp_path: Path, bad: str) -> None:
    f = tmp_path / "g.jsonl"
    f.write_text(bad + "\n")
    with pytest.raises(ValueError):
        load_golden(f)
