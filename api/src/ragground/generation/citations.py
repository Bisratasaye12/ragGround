"""Parse and validate `[n]` citations in a model answer."""

import re
from dataclasses import dataclass

from ragground.generation.prompts import ABSTAIN_ANSWER

# Matches [1], [1, 3], [1,2,3]; ignores things like [link] or [x].
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class CitationCheck:
    answer: str  # answer with invalid citation markers removed
    cited: list[int]  # valid source numbers, in first-use order
    invalid: list[int]  # numbers that don't correspond to any provided source
    has_uncited_text: bool  # answer has content but no valid citation at all


def check_citations(answer: str, num_sources: int) -> CitationCheck:
    cited: list[int] = []
    invalid: list[int] = []

    def fix(match: re.Match[str]) -> str:
        nums = [int(x) for x in match.group(1).split(",")]
        keep = [n for n in nums if 1 <= n <= num_sources]
        for n in nums:
            target = cited if n in keep else invalid
            if n not in target:
                target.append(n)
        return "".join(f"[{n}]" for n in keep)

    cleaned = _CITATION.sub(fix, answer)
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned).strip()
    return CitationCheck(
        answer=cleaned,
        cited=cited,
        invalid=invalid,
        has_uncited_text=bool(cleaned) and not cited and not is_abstention(cleaned),
    )


def is_abstention(answer: str) -> bool:
    norm = re.sub(r"[^a-z ]", "", answer.lower().replace("\u2019", "'").replace("'", ""))
    target = re.sub(r"[^a-z ]", "", ABSTAIN_ANSWER.lower().replace("'", ""))
    return norm.strip().startswith(target[:30])  # tolerate small trailing variations
