"""LLM-as-judge for answer quality.

Two narrow rubrics (narrow rubrics are more reliable than one "rate this answer" prompt):
- faithfulness: share of the answer's factual claims supported by the provided sources;
- correctness: agreement with the reference answer, 1-5, rescaled to 0-1.

The judge's raw output is kept in reports so disagreements can be audited, and its reliability
is measured against human labels (`ragground-eval calibrate`) before its numbers are trusted.
"""

import json
import re
from dataclasses import dataclass

from ragground.providers.llm import LLM, Message, complete

FAITHFULNESS_SYSTEM = """You are a strict fact-checker. You receive SOURCES and an ANSWER.
Split the ANSWER into its individual factual claims (ignore citation markers like [1]).
For each claim decide if the SOURCES directly support it. Claims that add details, advice or
generalisations not stated in the SOURCES are NOT supported.
Reply with JSON only, no prose, in this exact shape:
{"claims": [{"claim": "...", "supported": true}, ...]}"""

CORRECTNESS_SYSTEM = """You grade an ANSWER to a QUESTION against a REFERENCE answer.
Score 1-5: 5 = same meaning as the reference with nothing wrong; 4 = correct but missing a minor
detail; 3 = partly correct; 2 = mostly wrong or missing the key fact; 1 = wrong or contradicts
the reference. Extra correct detail is fine. Reply with JSON only:
{"score": <1-5>, "reason": "<one short sentence>"}"""


@dataclass
class Judgement:
    score: float | None  # 0..1, None if the judge output couldn't be parsed
    raw: str
    detail: object = None


_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


def _parse_json(text: str) -> object | None:
    match = _JSON_OBJECT.search(text)
    if not match:
        return None
    try:
        parsed: object = json.loads(match.group(0))
        return parsed
    except json.JSONDecodeError:
        return None


class Judge:
    def __init__(self, llm: LLM, max_tokens: int = 600) -> None:
        self.llm = llm
        self.max_tokens = max_tokens

    async def faithfulness(self, answer: str, sources: list[str]) -> Judgement:
        src = "\n\n".join(f"[{i}] {s}" for i, s in enumerate(sources, start=1))
        msg = f"SOURCES:\n{src}\n\nANSWER:\n{answer}"
        raw, _ = await complete(
            self.llm, FAITHFULNESS_SYSTEM, [Message("user", msg)], max_tokens=self.max_tokens
        )
        data = _parse_json(raw)
        claims = data.get("claims") if isinstance(data, dict) else None
        if not isinstance(claims, list) or not claims:
            return Judgement(None, raw)
        supported = sum(1 for c in claims if isinstance(c, dict) and c.get("supported") is True)
        unsupported = [
            c.get("claim") for c in claims if isinstance(c, dict) and not c.get("supported")
        ]
        return Judgement(round(supported / len(claims), 4), raw, {"unsupported": unsupported})

    async def correctness(self, question: str, reference: str, answer: str) -> Judgement:
        msg = f"QUESTION:\n{question}\n\nREFERENCE:\n{reference}\n\nANSWER:\n{answer}"
        raw, _ = await complete(
            self.llm, CORRECTNESS_SYSTEM, [Message("user", msg)], max_tokens=200
        )
        data = _parse_json(raw)
        score = data.get("score") if isinstance(data, dict) else None
        if not isinstance(score, int | float) or not 1 <= score <= 5:
            return Judgement(None, raw)
        reason = data.get("reason") if isinstance(data, dict) else None
        return Judgement(round((float(score) - 1) / 4, 4), raw, {"reason": reason})
