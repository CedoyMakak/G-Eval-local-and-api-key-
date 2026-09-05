from __future__ import annotations

import httpx
from openai import AsyncOpenAI

from app.providers.base import CompletionResult


class OpenAIProvider:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str | None = None,
        name: str = "openai",
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("API-ключ провайдера не задан")
        self.name = name
        self.model = model
        kwargs: dict = {
            "api_key": api_key,
            "http_client": httpx.AsyncClient(trust_env=False, timeout=60.0),
        }
        if base_url:
            kwargs["base_url"] = base_url
        if extra_headers:
            kwargs["default_headers"] = extra_headers
        self._client = AsyncOpenAI(**kwargs)

    async def complete(self, prompt: str, *, temperature: float = 0.0) -> CompletionResult:
        messages = [
            {
                "role": "system",
                "content": "Ты строгий независимый судья качества ответов. Отвечай только по инструкции.",
            },
            {"role": "user", "content": prompt},
        ]
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=messages,
                logprobs=True,
                top_logprobs=5,
            )
        except Exception:
            response = await self._client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=messages,
            )
        choice = response.choices[0]
        text = choice.message.content or ""
        logprobs = _extract_score_logprobs(choice)
        return CompletionResult(
            text=text,
            provider=self.name,
            model=self.model,
            logprobs=logprobs,
        )


def _extract_score_logprobs(choice) -> list[float] | None:
    content = getattr(getattr(choice, "logprobs", None), "content", None)
    if not content:
        return None

    scores: list[float] = []
    for token_info in content:
        token = (getattr(token_info, "token", "") or "").strip()
        if token.isdigit() and token in {"1", "2", "3", "4", "5"}:
            scores.append(float(token_info.logprob))
    return scores or None
