from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from app.config import Settings, get_settings, reload_settings
from app.evaluators.common import gated_judge_score, lexical_mean
from app.evaluators.pipeline import evaluate_answer, evaluate_pairwise
from app.evaluators.semantic import probe_semantic_method
from app.schemas import (
    BatchEvaluateRequest,
    BatchEvaluateResponse,
    EvaluateRequest,
    HealthResponse,
    ImportDocumentsRequest,
    ImportDocumentsResponse,
    JudgeSettingsUpdate,
    JudgeSettingsView,
    JudgeTestResponse,
    LabeledItem,
    LabelsInfoResponse,
    LabelsListResponse,
    PairwiseRequest,
    PairwiseResponse,
    QualityReport,
    SaveLabelResponse,
    ValidateRequest,
    ValidateResponse,
)
from app.providers.factory import get_judge_provider
from app.ingest import IngestError, parse_eval_documents
from app.security import public_error_message, require_local_client
from app.settings_store import update_env
from app.validation.biases import compute_biases
from app.validation.errors import analyze_errors
from app.validation.labels_store import LABELS_PATH, load_labels, save_label
from app.validation.metrics import correlate
from app.validation.weights import search_weights

router = APIRouter()
ROOT = Path(__file__).resolve().parents[2]
SAMPLES_PATH = ROOT / "data" / "sample_eval.json"


@router.get("/settings", response_model=JudgeSettingsView)
async def get_judge_settings(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> JudgeSettingsView:
    require_local_client(request)
    return _settings_view(settings)


@router.put("/settings", response_model=JudgeSettingsView)
async def put_judge_settings(request: Request, payload: JudgeSettingsUpdate) -> JudgeSettingsView:
    require_local_client(request)
    updates = {"JUDGE_PROVIDER": payload.provider}
    _put_if_text(updates, "OPENAI_MODEL", payload.openai_model)
    _put_if_text(updates, "OPENAI_BASE_URL", payload.openai_base_url)
    _put_if_secret(updates, "OPENAI_API_KEY", payload.openai_api_key)
    _put_if_text(updates, "OPENROUTER_MODEL", payload.openrouter_model)
    _put_if_text(updates, "OPENROUTER_BASE_URL", payload.openrouter_base_url)
    _put_if_secret(updates, "OPENROUTER_API_KEY", payload.openrouter_api_key)
    _put_if_text(updates, "OLLAMA_BASE_URL", payload.ollama_base_url)
    _put_if_text(updates, "OLLAMA_MODEL", payload.ollama_model)
    update_env(updates)
    return _settings_view(reload_settings())


@router.post("/settings/test", response_model=JudgeTestResponse)
async def test_judge(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> JudgeTestResponse:
    require_local_client(request)
    try:
        provider = get_judge_provider(settings)
        completion = await provider.complete("Ответь одним словом: OK", temperature=0.0)
        sample = (completion.text or "").strip().replace("\n", " ")[:180]
        return JudgeTestResponse(
            ok=True,
            provider=completion.provider,
            model=completion.model,
            message="Судья отвечает.",
            sample=sample,
        )
    except Exception as exc:  # noqa: BLE001
        return JudgeTestResponse(
            ok=False,
            provider=settings.judge_provider,
            model=settings.judge_model_name,
            message=public_error_message(exc),
        )


@router.get("/health", response_model=HealthResponse)
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        judge_provider=settings.judge_provider,
        judge_model=settings.judge_model_name,
        embedding_model=settings.embedding_model,
        semantic_method=probe_semantic_method(),
        ensemble=settings.ensemble_mode if settings.ensemble_mode != "off" else None,
    )


@router.post("/evaluate", response_model=QualityReport)
async def evaluate(payload: EvaluateRequest, settings: Settings = Depends(get_settings)) -> QualityReport:
    return await evaluate_answer(payload, settings)


@router.get("/samples", response_model=ImportDocumentsResponse)
async def sample_documents() -> ImportDocumentsResponse:
    if not SAMPLES_PATH.exists():
        raise HTTPException(status_code=404, detail="Нет data/sample_eval.json")
    try:
        items = parse_eval_documents(SAMPLES_PATH.read_text(encoding="utf-8"), SAMPLES_PATH.name)
    except IngestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ImportDocumentsResponse(items=[EvaluateRequest.model_validate(item) for item in items], n=len(items))


@router.post("/evaluate/import", response_model=ImportDocumentsResponse)
async def import_documents(payload: ImportDocumentsRequest) -> ImportDocumentsResponse:
    try:
        rows = parse_eval_documents(payload.text, payload.filename)
    except (IngestError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ImportDocumentsResponse(items=[EvaluateRequest.model_validate(item) for item in rows], n=len(rows))


@router.post("/evaluate/batch", response_model=BatchEvaluateResponse)
async def evaluate_batch(
    payload: BatchEvaluateRequest,
    settings: Settings = Depends(get_settings),
) -> BatchEvaluateResponse:
    skip_all = payload.skip_judge or all(item.skip_judge for item in payload.items)
    workers = 4 if skip_all else 1
    semaphore = asyncio.Semaphore(workers)

    async def _one(item: EvaluateRequest) -> QualityReport:
        async with semaphore:
            item.skip_judge = item.skip_judge or payload.skip_judge
            return await evaluate_answer(item, settings)

    results = list(await asyncio.gather(*[_one(item) for item in payload.items]))
    mean_overall = round(sum(row.overall for row in results) / len(results), 4) if results else None
    return BatchEvaluateResponse(results=results, n=len(results), mean_overall=mean_overall)


@router.post("/evaluate/pairwise", response_model=PairwiseResponse)
async def evaluate_pairwise_route(
    payload: PairwiseRequest,
    settings: Settings = Depends(get_settings),
) -> PairwiseResponse:
    return await evaluate_pairwise(
        question=payload.question,
        answer_a=payload.answer_a,
        answer_b=payload.answer_b,
        reference=payload.reference,
        context=payload.context,
        settings=settings,
    )


@router.get("/labels", response_model=LabelsInfoResponse)
async def labels_info() -> LabelsInfoResponse:
    return LabelsInfoResponse(total=len(load_labels()), path=str(LABELS_PATH))


@router.get("/labels/items", response_model=LabelsListResponse)
async def labels_items() -> LabelsListResponse:
    rows = load_labels()
    items = [LabeledItem.model_validate(row) for row in rows]
    return LabelsListResponse(total=len(items), items=items)


@router.post("/labels", response_model=SaveLabelResponse)
async def create_label(payload: LabeledItem) -> SaveLabelResponse:
    item, total = save_label(payload)
    return SaveLabelResponse(item=item, total=total)


@router.post("/validate", response_model=ValidateResponse)
async def validate(
    payload: ValidateRequest,
    settings: Settings = Depends(get_settings),
) -> ValidateResponse:
    reports: list[QualityReport] = []
    human: list[float] = []
    for item in payload.items:
        report = await evaluate_answer(
            EvaluateRequest(
                question=item.question,
                answer=item.answer,
                reference=item.reference,
                context=item.context,
                skip_judge=payload.skip_judge,
            ),
            settings,
        )
        reports.append(report)
        human.append(item.human_score)

    judge_scores = [gated_judge_score(report.dimensions) for report in reports]
    semantic_scores = [report.semantic.cosine for report in reports if report.semantic.cosine is not None]
    semantic_human = [
        score
        for report, score in zip(reports, human)
        if report.semantic.cosine is not None
    ]
    lexical_pairs = [
        (lexical_mean(report.lexical), score)
        for report, score in zip(reports, human)
        if lexical_mean(report.lexical) is not None
    ]
    weight_pairs = [
        (report.semantic.cosine, lexical_mean(report.lexical), gated_judge_score(report.dimensions), score)
        for report, score in zip(reports, human)
        if report.semantic.cosine is not None and lexical_mean(report.lexical) is not None
    ]
    recommended = None
    if len(weight_pairs) >= 8:
        recommended = search_weights(
            semantic=[row[0] for row in weight_pairs],
            lexical=[row[1] for row in weight_pairs],
            judge=[row[2] for row in weight_pairs],
            human=[row[3] for row in weight_pairs],
        )
    human_high, auto_high = analyze_errors(payload.items, reports)

    return ValidateResponse(
        overall_vs_human=correlate([r.overall for r in reports], human),
        judge_vs_human=correlate(judge_scores, human),
        semantic_vs_human=correlate(semantic_scores, semantic_human),
        lexical_vs_human=correlate(
            [p for p, _ in lexical_pairs],
            [h for _, h in lexical_pairs],
        ),
        biases=compute_biases(reports),
        predictions=reports,
        human_high_auto_low=human_high,
        human_low_auto_high=auto_high,
        recommended_weights=recommended,
        semantic_method=next((r.semantic.method for r in reports if r.semantic.method), probe_semantic_method()),
        fallback_rate=sum(1 for r in reports if r.judge.fallback) / max(len(reports), 1),
    )


def _settings_view(settings: Settings) -> JudgeSettingsView:
    return JudgeSettingsView(
        provider=settings.judge_provider,
        model=settings.judge_model_name,
        openai_model=settings.openai_model,
        openai_base_url=settings.openai_base_url,
        openai_key_set=bool(settings.openai_api_key),
        openrouter_model=settings.openrouter_model,
        openrouter_base_url=settings.openrouter_base_url,
        openrouter_key_set=bool(settings.openrouter_api_key),
        ollama_base_url=settings.ollama_base_url,
        ollama_model=settings.ollama_model,
    )


def _put_if_text(target: dict[str, str], key: str, value: str | None) -> None:
    if value is None:
        return
    target[key] = value.strip()


def _put_if_secret(target: dict[str, str], key: str, value: str | None) -> None:
    if value is None:
        return
    secret = value.strip()
    if not secret or "•" in secret:
        return
    target[key] = secret
