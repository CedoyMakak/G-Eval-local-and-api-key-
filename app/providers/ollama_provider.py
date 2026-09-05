from __future__ import annotations

import httpx

from app.providers.base import CompletionResult


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def complete(self, prompt: str, *, temperature: float = 0.0) -> CompletionResult:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": temperature},
            "messages": [
                {
                    "role": "system",
                    "content": "Ты строгий независимый судья качества ответов. Отвечай только по инструкции.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        timeout = httpx.Timeout(60.0, connect=3.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

        text = data.get("message", {}).get("content") or data.get("response") or ""
        return CompletionResult(
            text=text,
            provider=self.name,
            model=self.model,
            logprobs=None,
        )
