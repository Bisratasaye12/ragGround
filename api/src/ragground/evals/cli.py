"""`ragground-eval` command line.

    uv run ragground-eval validate
    uv run ragground-eval retrieval --chunker heading --modes fulltext,vector,hybrid,hybrid_rerank
    uv run ragground-eval answer --limit 10 --judge
    uv run ragground-eval compare evals/baseline.json evals/reports/<run>.json

Uses the database/models from the normal settings (RAGGROUND_* env / api/.env).
"""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from ragground.config import get_settings
from ragground.db import make_engine, make_sessionmaker
from ragground.evals.corpus import EvalOrg, load_corpus, prepare_eval_org
from ragground.evals.golden import GoldenCase, load_golden, validate_evidence
from ragground.evals.judge import Judge
from ragground.evals.report import answers_markdown, compare, retrieval_markdown, write_report
from ragground.evals.runner import run_answers, run_retrieval
from ragground.generation.answer import ChatConfig
from ragground.ingestion.chunkers import ChunkerName, get_chunker
from ragground.ingestion.pipeline import IngestDeps
from ragground.ingestion.storage import LocalStorage
from ragground.log_config import configure_logging
from ragground.providers.embeddings import make_embedder
from ragground.providers.llm import make_llm
from ragground.providers.reranker import make_reranker
from ragground.retrieval.search import Retriever, SearchConfig, SearchMode

REPO = Path(__file__).resolve().parents[4]
EVALS = REPO / "evals"
ALL_MODES: list[SearchMode] = ["fulltext", "vector", "hybrid", "hybrid_rerank"]


def _say(msg: str) -> None:
    print(msg, flush=True)


def _select(cases: list[GoldenCase], args: argparse.Namespace) -> list[GoldenCase]:
    if args.categories:
        wanted = set(args.categories.split(","))
        cases = [c for c in cases if c.category in wanted]
    if args.ids:
        wanted = set(args.ids.split(","))
        cases = [c for c in cases if c.id in wanted]
    if args.limit:
        # Stratified: keep category proportions when sampling a small smoke subset.
        by_cat: dict[str, list[GoldenCase]] = {}
        for c in cases:
            by_cat.setdefault(c.category, []).append(c)
        picked: list[GoldenCase] = []
        while len(picked) < min(args.limit, len(cases)):
            for group in by_cat.values():
                if group and len(picked) < args.limit:
                    picked.append(group.pop(0))
        cases = picked
    return cases


async def _setup(args: argparse.Namespace) -> tuple[EvalOrg, IngestDeps, Retriever]:
    settings = get_settings()
    engine = make_engine(settings.database_url)
    sm = make_sessionmaker(engine)
    embedder = make_embedder(settings.embedding_provider, settings.embedding_model)
    chunker_name = cast(ChunkerName, args.chunker)
    deps = IngestDeps(
        sessionmaker=sm,
        storage=LocalStorage(settings.storage_dir),
        embedder=embedder,
        chunker=get_chunker(chunker_name, args.max_size, args.size_unit),
    )
    size = f"{args.max_size}" if args.size_unit == "words" else f"{args.max_size}tok"
    org_name = f"ragground-eval/{args.chunker}-{size}/{embedder.model_name}"
    _say(f"Preparing eval org {org_name!r} …")
    org = await prepare_eval_org(deps, load_corpus(EVALS / "corpus"), org_name, args.reingest)
    _say(f"  {'reused' if org.reused else 'ingested'}: {org.documents} docs, {org.chunks} chunks")
    reranker = make_reranker(
        settings.reranker_provider, args.reranker_model or settings.reranker_model
    )
    search_cfg = SearchConfig(
        rerank_top_n=args.rerank_top_n or settings.rerank_top_n,
        rerank_max_words=args.rerank_max_words or settings.rerank_max_words,
        fulltext_operator=args.fulltext_operator or settings.fulltext_operator,
    )
    return org, deps, Retriever(sm, embedder, reranker, search_cfg)


def _config(args: argparse.Namespace, org: EvalOrg) -> dict[str, Any]:
    settings = get_settings()
    return {
        "chunker": args.chunker,
        "max_size": args.max_size,
        "size_unit": args.size_unit,
        "embedding": settings.embedding_model,
        "reranker": (args.reranker_model or settings.reranker_model)
        if settings.reranker_provider != "none"
        else "none",
        "rerank_top_n": args.rerank_top_n or settings.rerank_top_n,
        "rerank_max_words": args.rerank_max_words or settings.rerank_max_words,
        "fulltext_operator": args.fulltext_operator or settings.fulltext_operator,
        "chunks": org.chunks,
    }


def _name(args: argparse.Namespace, kind: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return args.out or f"{stamp}-{kind}-{args.chunker}{args.max_size}{args.size_unit[0]}"


async def cmd_retrieval(args: argparse.Namespace) -> int:
    cases = _select(load_golden(EVALS / "golden" / "golden.jsonl"), args)
    org, _, retriever = await _setup(args)
    modes = cast(list[SearchMode], args.modes.split(",")) if args.modes else ALL_MODES
    _say(f"Retrieval eval: {len(cases)} cases x {len(modes)} modes")
    runs = await run_retrieval(cases, org, retriever, modes, k=10, progress=_say)
    config = _config(args, org)
    retrieval = [{"mode": r.mode, "summary": r.summary, "per_case": r.per_case} for r in runs]
    payload = {"kind": "retrieval", "config": config, "retrieval": retrieval}
    md = retrieval_markdown("Retrieval eval", config, retrieval)
    path = write_report(EVALS / "reports", _name(args, "retrieval"), payload, md)
    _say(md)
    _say(f"Report: {path}")
    return 0


async def cmd_answer(args: argparse.Namespace) -> int:
    settings = get_settings()
    cases = _select(load_golden(EVALS / "golden" / "golden.jsonl"), args)
    org, _, retriever = await _setup(args)
    llm = make_llm(
        settings.llm_provider, settings.llm_model, settings.ollama_base_url, settings.llm_num_ctx
    )
    if llm is None:
        _say("No LLM configured (RAGGROUND_LLM_PROVIDER)")
        return 2
    judge_llm = (
        make_llm(
            settings.llm_provider,
            args.judge_model or settings.llm_model,
            settings.ollama_base_url,
            settings.llm_num_ctx,
        )
        if args.judge
        else None
    )
    cfg = ChatConfig(
        mode=settings.search_default_mode,
        k=settings.chat_k,
        abstain_below_rerank_score=settings.abstain_below_rerank_score,
        max_context_words=settings.chat_max_context_words,
        max_tokens=settings.chat_max_tokens,
    )
    _say(
        f"Answer eval: {len(cases)} cases with {llm.provider}:{llm.model}"
        f"{' + judge ' + judge_llm.model if judge_llm else ''}"
    )
    result = await run_answers(
        cases, org, retriever, llm, cfg, Judge(judge_llm) if judge_llm else None, progress=_say
    )
    config = _config(args, org) | {
        "llm": f"{llm.provider}:{llm.model}",
        "judge": judge_llm.model if judge_llm else "none",
        "abstain_below": cfg.abstain_below_rerank_score,
    }
    payload = {"kind": "answers", "config": config, "answers": result}
    md = answers_markdown("Answer eval", config, result["summary"])
    path = write_report(EVALS / "reports", _name(args, "answers"), payload, md)
    _say(md)
    _say(f"Report: {path}")
    return 0


def cmd_validate(_: argparse.Namespace) -> int:
    cases = load_golden(EVALS / "golden" / "golden.jsonl")
    corpus = EVALS / "corpus"
    # Golden paths are corpus-relative: tutorial/… under fastapi/, injected/… at the root.
    problems = validate_evidence(
        [c for c in cases if all(e.file.startswith("tutorial/") for e in c.evidence)],
        corpus / "fastapi",
    ) + validate_evidence(
        [c for c in cases if any(e.file.startswith("injected/") for e in c.evidence)], corpus
    )
    for p in problems:
        _say(f"  ✗ {p}")
    _say(f"{len(cases)} cases, {len(problems)} problems")
    return 1 if problems else 0


def cmd_compare(args: argparse.Namespace) -> int:
    baseline = json.loads(Path(args.baseline).read_text())
    current = json.loads(Path(args.current).read_text())
    problems = compare(baseline, current, args.tolerance)
    for p in problems:
        _say(f"  ✗ {p}")
    _say("No regressions." if not problems else f"{len(problems)} regression(s).")
    return 1 if problems else 0


def main() -> None:
    configure_logging("WARNING")
    parser = argparse.ArgumentParser(prog="ragground-eval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--chunker", choices=["heading", "fixed"], default="heading")
        p.add_argument("--max-size", type=int, default=350, help="chunk size limit")
        p.add_argument("--size-unit", choices=["words", "tokens"], default="tokens")
        p.add_argument("--reingest", action="store_true", help="rebuild the eval org")
        p.add_argument("--categories", help="comma-separated golden categories")
        p.add_argument("--ids", help="comma-separated case ids")
        p.add_argument("--limit", type=int, help="stratified subset size")
        p.add_argument("--out", help="report name (default: timestamped)")
        p.add_argument("--reranker-model", help="override the reranker model")
        p.add_argument("--rerank-top-n", type=int, help="candidates passed to the reranker")
        p.add_argument("--rerank-max-words", type=int, help="truncate passages for reranking")
        p.add_argument("--fulltext-operator", choices=["and", "or"])

    p_ret = sub.add_parser("retrieval", help="retrieval metrics for each search mode")
    common(p_ret)
    p_ret.add_argument("--modes", help=f"comma-separated, default {','.join(ALL_MODES)}")
    p_ans = sub.add_parser("answer", help="end-to-end answers (+ optional LLM judge)")
    common(p_ans)
    p_ans.add_argument("--judge", action="store_true", help="score with the LLM judge")
    p_ans.add_argument("--judge-model", help="judge model (default: same as answer model)")
    sub.add_parser("validate", help="check golden-set evidence phrases exist in the corpus")
    p_cmp = sub.add_parser("compare", help="fail if a report regresses vs a baseline")
    p_cmp.add_argument("baseline")
    p_cmp.add_argument("current")
    p_cmp.add_argument("--tolerance", type=float, default=0.02)

    args = parser.parse_args()
    if args.cmd == "validate":
        code = cmd_validate(args)
    elif args.cmd == "compare":
        code = cmd_compare(args)
    elif args.cmd == "retrieval":
        code = asyncio.run(cmd_retrieval(args))
    else:
        code = asyncio.run(cmd_answer(args))
    sys.exit(code)


if __name__ == "__main__":
    main()
