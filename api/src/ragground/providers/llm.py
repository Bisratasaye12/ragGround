"""LLM providers behind one streaming interface.

`stream()` yields text deltas and finishes with a `Usage` event. Ollama is the default (local,
free). Hosted adapters (Anthropic, OpenAI-compatible) implement the same protocol.
"""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol

import httpx


@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: str


@dataclass
class TextDelta:
    text: str


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int


LLMEvent = TextDelta | Usage


class LLMError(Exception):
    pass


class LLM(Protocol):
    provider: str
    model: str

    def stream(
        self,
        system: str,
        messages: list[Message],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> AsyncIterator[LLMEvent]: ...


async def complete(
    llm: LLM, system: str, messages: list[Message], **kwargs: float | int
) -> tuple[str, Usage]:
    """Non-streaming convenience wrapper (used by the eval judge)."""
    parts: list[str] = []
    usage = Usage(0, 0)
    async for event in llm.stream(system, messages, **kwargs):  # type: ignore[arg-type]
        if isinstance(event, TextDelta):
            parts.append(event.text)
        else:
            usage = event
    return "".join(parts), usage


class OllamaLLM:
    provider = "ollama"

    def __init__(
        self, base_url: str, model: str, num_ctx: int = 8192, timeout_s: float = 900.0
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        # CPU inference is slow: generous read timeout, short connect timeout.
        self._timeout = httpx.Timeout(timeout_s, connect=5.0)

    async def stream(
        self,
        system: str,
        messages: list[Message],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> AsyncIterator[LLMEvent]:
        payload = {
            "model": self.model,
            "stream": True,
            "messages": [{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,  # Ollama defaults to 4096 and silently truncates
            },
        }
        try:
            async with (
                httpx.AsyncClient(timeout=self._timeout) as client,
                client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp,
            ):
                if resp.status_code != 200:
                    body = (await resp.aread()).decode(errors="replace")[:500]
                    raise LLMError(f"ollama returned {resp.status_code}: {body}")
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if "error" in chunk:
                        raise LLMError(f"ollama error: {chunk['error']}")
                    text = chunk.get("message", {}).get("content", "")
                    if text:
                        yield TextDelta(text)
                    if chunk.get("done"):
                        yield Usage(
                            input_tokens=int(chunk.get("prompt_eval_count", 0)),
                            output_tokens=int(chunk.get("eval_count", 0)),
                        )
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama unreachable at {self.base_url}: {exc}") from exc


class ScriptedLLM:
    """Test double: streams a fixed reply word by word and records what it was asked."""

    provider = "scripted"

    def __init__(self, reply: str = "", model: str = "scripted-test") -> None:
        self.reply = reply
        self.model = model
        self.calls: list[tuple[str, list[Message]]] = []

    async def stream(
        self,
        system: str,
        messages: list[Message],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> AsyncIterator[LLMEvent]:
        self.calls.append((system, messages))
        words = self.reply.split(" ")
        for i, w in enumerate(words):
            yield TextDelta(w if i == 0 else f" {w}")
        yield Usage(
            input_tokens=len((system + messages[-1].content).split()), output_tokens=len(words)
        )


# USD per million tokens (input, output). Local models cost nothing per token.
PRICING: dict[str, tuple[float, float]] = {}


def cost_usd(provider: str, model: str, usage: Usage) -> float:
    if provider in {"ollama", "scripted"}:
        return 0.0
    price_in, price_out = PRICING.get(model, (0.0, 0.0))
    return (usage.input_tokens * price_in + usage.output_tokens * price_out) / 1_000_000


def make_llm(provider: str, model: str, ollama_base_url: str, num_ctx: int = 8192) -> LLM | None:
    if provider == "ollama":
        return OllamaLLM(ollama_base_url, model, num_ctx=num_ctx)
    if provider == "none":
        return None
    raise ValueError(f"unknown LLM provider {provider!r} (hosted adapters land with API keys)")
