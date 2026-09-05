from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="LLM Answer Quality Evaluation",
    description=(
        "Компоненты системы автоматизированной оценки качества ответов "
        "больших языковых моделей: лексические метрики, семантика, "
        "LLM-as-a-Judge (G-Eval) и офлайн-валидация по ручным меткам."
    ),
    version="1.0.0",
)
app.include_router(router)
