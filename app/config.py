from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    judge_provider: str = "ollama"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = ""
    openrouter_api_key: str = ""
    openrouter_model: str = "google/gemma-4-31b-it"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    weight_semantic: float = 0.0
    weight_lexical: float = 0.0
    weight_judge: float = 1.0
    disagreement_threshold: float = 0.25
    judge_temperature: float = 0.0
    judge_ensemble: str = "off"
    judge_ensemble_providers: str = "openrouter,ollama"
    judge_swap_check: bool = False
    position_bias_penalty: float = 0.08
    position_bias_threshold: float = 0.15

    @property
    def judge_model_name(self) -> str:
        provider = self.judge_provider.strip().lower()
        if provider == "openai":
            return self.openai_model
        if provider == "openrouter":
            return self.openrouter_model
        return self.ollama_model

    @property
    def ensemble_mode(self) -> str:
        mode = (self.judge_ensemble or "off").strip().lower()
        return mode if mode in {"mean", "min"} else "off"

    @property
    def ensemble_provider_names(self) -> list[str]:
        raw = self.judge_ensemble_providers or ""
        names = [part.strip().lower() for part in raw.split(",") if part.strip()]
        return names or [self.judge_provider.strip().lower()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()
