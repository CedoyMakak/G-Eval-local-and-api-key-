from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class CompletionResult:
    text: str
    provider: str
    model: str
    logprobs: list[float] | None = None


class JudgeProvider(Protocol):
    name: str
    model: str

    async def complete(self, prompt: str, *, temperature: float = 0.0) -> CompletionResult:
        ...
