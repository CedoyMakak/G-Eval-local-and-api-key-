from __future__ import annotations

import math

import httpx
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI

from app.providers.base import CompletionResult

_CLIENTS: dict[tuple, AsyncOpenAI] = {}


def _shared_client(
    *,
    api_key: str,
    model: str,
    base_url: str | None,
    name: str,
    extra_headers: dict[str, str] | None,
) -> AsyncOpenAI:
    key = (
        name,
        model,
        base_url or "",
        api_key[:16],
        tuple(sorted((extra_headers or {}).items())),
    )
    client = _CLIENTS.get(key)
    if client is not None:
        return client
    kwargs: dict = {
        "api_key": api_key,
        "timeout": 60.0,
        "max_retries": 1,
        "http_client": httpx.AsyncClient(trust_env=False, timeout=60.0),
    }
    if base_url:
        kwargs["base_url"] = base_url
    if extra_headers:
        kwargs["default_headers"] = extra_headers
    client = AsyncOpenAI(**kwargs)
    _CLIENTS[key] = client
    return client


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
        self._client = _shared_client(
            api_key=api_key,
            model=model,
            base_url=base_url,
            name=name,
            extra_headers=extra_headers,
        )

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
        except (APIConnectionError, APITimeoutError, httpx.TimeoutException, httpx.ConnectError):
            raise
        except Exception:
            response = await self._client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=messages,
            )
        choice = response.choices[0]
        text = choice.message.content or ""
        logprobs, expected_units = _extract_score_logprobs(choice)
        return CompletionResult(
            text=text,
            provider=self.name,
            model=self.model,
            logprobs=logprobs,
            expected_units=expected_units,
        )


def _extract_score_logprobs(choice) -> tuple[list[float] | None, list[float] | None]:
    content = getattr(getattr(choice, "logprobs", None), "content", None)
    if not content:
        return None, None

    chosen: list[float] = []
    expected: list[float] = []
    for token_info in content:
        token = (getattr(token_info, "token", "") or "").strip()
        if not (token.isdigit() and token in {"1", "2", "3", "4", "5"}):
            continue
        chosen.append(float(token_info.logprob))
        unit = _expected_unit(token_info)
        if unit is not None:
            expected.append(unit)
    return (chosen or None), (expected or None)


def _expected_unit(token_info) -> float | None:
    alts = getattr(token_info, "top_logprobs", None) or []
    dist: dict[int, float] = {}
    for alt in alts:
        token = (getattr(alt, "token", "") or "").strip()
        if token.isdigit() and token in {"1", "2", "3", "4", "5"}:
            dist[int(token)] = math.exp(float(alt.logprob))
    if not dist:
        token = (getattr(token_info, "token", "") or "").strip()
        if token.isdigit() and token in {"1", "2", "3", "4", "5"}:
            return (int(token) - 1) / 4.0
        return None
    total = sum(dist.values())
    if total <= 0:
        return None
    expected = sum(score * weight for score, weight in dist.items()) / total
    return max(0.0, min(1.0, (expected - 1.0) / 4.0))
