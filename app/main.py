from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router

STATIC_DIR = Path(__file__).parent / "static"

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


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
