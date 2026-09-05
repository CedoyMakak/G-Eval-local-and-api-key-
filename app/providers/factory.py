from __future__ import annotations

from app.config import Settings
from app.providers.base import JudgeProvider
from app.providers.ollama_provider import OllamaProvider
from app.providers.openai_provider import OpenAIProvider


def get_judge_provider(settings: Settings) -> JudgeProvider:
    provider = settings.judge_provider.strip().lower()
    if provider == "openai":
        return OpenAIProvider(
            settings.openai_api_key,
            settings.openai_model,
            base_url=settings.openai_base_url or None,
        )
    if provider == "openrouter":
        return OpenAIProvider(
            settings.openrouter_api_key or settings.openai_api_key,
            settings.openrouter_model,
            base_url=settings.openrouter_base_url,
            name="openrouter",
            extra_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "LLM Answer Quality Evaluation",
            },
        )
    if provider == "ollama":
        return OllamaProvider(settings.ollama_base_url, settings.ollama_model)
    raise ValueError(f"Неизвестный JUDGE_PROVIDER: {settings.judge_provider}")
