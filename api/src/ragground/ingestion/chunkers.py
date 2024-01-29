"""Chunking strategies.

Chunk size is measured with a pluggable length function:
- `words` (whitespace tokens): cheap and model-agnostic, but a poor proxy for technical docs.
  Measured on the FastAPI corpus: 2.44 model tokens per word (code is token-dense), so a
  300-word chunk can be ~730 tokens and 11% of chunks overflowed the 512-token input limit of
  the embedding model and reranker, which silently truncate.
- `tokens`: the embedding model's own tokenizer, so chunks provably fit the model.

Strategies:
- `fixed`: sliding window over the whole text, ignoring structure (the baseline).
- `heading`: never crosses a section boundary; packs whole paragraphs up to the limit, splitting
  an oversized paragraph by sentences (then by words). Chunks carry their heading path, which
  is prepended to the text that gets embedded ("contextual" chunks).
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Protocol

from ragground.ingestion.parsers import ParsedDocument

ChunkerName = Literal["fixed", "heading"]
SizeUnit = Literal["words", "tokens"]
LengthFn = Callable[[str], int]
HEADING_SEP = " > "


def word_length(text: str) -> int:
    return len(text.split())


def token_length_fn(tokenizer_repo: str = "BAAI/bge-small-en-v1.5") -> LengthFn:
    """Length in model tokens (without special tokens), via the model's tokenizer.json."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(hf_hub_download(tokenizer_repo, "tokenizer.json"))

    @lru_cache(maxsize=65536)
    def length(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False).ids)

    return length


@dataclass
class Chunk:
    ord: int
    text: str
    heading_path: list[str]

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def embedding_text(self) -> str:
        """What gets embedded: headings give the chunk context it lacks on its own."""
        if not self.heading_path:
            return self.text
        return f"{HEADING_SEP.join(self.heading_path)}\n\n{self.text}"


class Chunker(Protocol):
    def chunk(self, doc: ParsedDocument) -> list[Chunk]: ...


def _pack_words(words: list[str], max_len: int, length: LengthFn) -> list[str]:
    """Greedily pack words into pieces whose summed per-word length stays within max_len.
    Summing per-word lengths slightly over-counts tokens (never under-counts), which is the
    safe direction for a hard model limit."""
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for w in words:
        n = max(length(w), 1)
        if current and size + n > max_len:
            pieces.append(" ".join(current))
            current, size = [], 0
        current.append(w)
        size += n
    if current:
        pieces.append(" ".join(current))
    return pieces


class FixedChunker:
    def __init__(self, size: int = 300, overlap: int = 50, length: LengthFn = word_length) -> None:
        if not 0 <= overlap < size:
            raise ValueError("need 0 <= overlap < size")
        self.size, self.overlap, self.length = size, overlap, length

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        words = doc.full_text.split()
        costs = [max(self.length(w), 1) for w in words]
        chunks: list[Chunk] = []
        start = 0
        while start < len(words):
            end, size = start, 0
            while end < len(words) and (end == start or size + costs[end] <= self.size):
                size += costs[end]
                end += 1
            chunks.append(Chunk(len(chunks), " ".join(words[start:end]), []))
            if end >= len(words):
                break
            # Step back so consecutive windows share ~`overlap` units of context.
            back, kept = end, 0
            while back > start + 1 and kept + costs[back - 1] <= self.overlap:
                back -= 1
                kept += costs[back]
            start = back
        return chunks


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def _split_long(paragraph: str, max_len: int, length: LengthFn) -> list[str]:
    """Split one paragraph that's over the limit: by sentences, then hard-wrap by words."""
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in _SENTENCE_END.split(paragraph):
        n = length(sentence)
        if n > max_len:  # a monster sentence (or a code block / table)
            if current:
                pieces.append(" ".join(current))
                current, size = [], 0
            pieces.extend(_pack_words(sentence.split(), max_len, length))
            continue
        if current and size + n > max_len:
            pieces.append(" ".join(current))
            current, size = [], 0
        current.append(sentence)
        size += n
    if current:
        pieces.append(" ".join(current))
    return pieces


class HeadingChunker:
    def __init__(
        self, max_len: int = 300, min_len: int = 40, length: LengthFn = word_length
    ) -> None:
        self.max_len, self.min_len, self.length = max_len, min_len, length

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in doc.sections:
            path = list(section.heading_path)
            buf: list[str] = []
            buf_len = 0

            def flush(path: list[str] = path) -> None:
                nonlocal buf, buf_len
                if buf:
                    chunks.append(Chunk(len(chunks), "\n\n".join(buf), list(path)))
                buf, buf_len = [], 0

            for para in (p.strip() for p in section.text.split("\n\n")):
                if not para:
                    continue
                n = self.length(para)
                if n > self.max_len:
                    flush()
                    for piece in _split_long(para, self.max_len, self.length):
                        chunks.append(Chunk(len(chunks), piece, list(path)))
                    continue
                if buf_len + n > self.max_len:
                    flush()
                buf.append(para)
                buf_len += n
            flush()
        return _merge_tiny(chunks, self.min_len, self.max_len, self.length)


def _merge_tiny(chunks: list[Chunk], min_len: int, max_len: int, length: LengthFn) -> list[Chunk]:
    """Fold a too-small chunk into the previous one when both share a heading path and fit."""
    merged: list[Chunk] = []
    for c in chunks:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and length(c.text) < min_len
            and prev.heading_path == c.heading_path
            and length(prev.text) + length(c.text) <= max_len
        ):
            prev.text = f"{prev.text}\n\n{c.text}"
            continue
        merged.append(Chunk(len(merged), c.text, c.heading_path))
    return merged


def get_chunker(
    name: ChunkerName, max_len: int = 300, unit: SizeUnit = "words", tokenizer_repo: str = ""
) -> Chunker:
    if unit == "tokens":
        length = token_length_fn(tokenizer_repo or "BAAI/bge-small-en-v1.5")
        min_len = max(max_len // 8, 10)
    else:
        length, min_len = word_length, 40  # original word-based defaults
    if name == "fixed":
        return FixedChunker(size=max_len, overlap=max_len // 6, length=length)
    return HeadingChunker(max_len=max_len, min_len=min_len, length=length)
