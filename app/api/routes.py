from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.evaluators.pipeline import evaluate_answer, evaluate_pairwise
from app.schemas import (
    BatchEvaluateRequest,
    BatchEvaluateResponse,
    EvaluateRequest,
    HealthResponse,
    PairwiseRequest,
    PairwiseResponse,
    QualityReport,
    ValidateRequest,
    ValidateResponse,
)
from app.validation.biases import compute_biases
from app.validation.metrics import correlate

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        judge_provider=settings.judge_provider,
        judge_model=settings.judge_model_name,
        embedding_model=settings.embedding_model,
    )


@router.post("/evaluate", response_model=QualityReport)
async def evaluate(payload: EvaluateRequest, settings: Settings = Depends(get_settings)) -> QualityReport:
    return await evaluate_answer(payload, settings)


@router.post("/evaluate/batch", response_model=BatchEvaluateResponse)
async def evaluate_batch(
    payload: BatchEvaluateRequest,
    settings: Settings = Depends(get_settings),
) -> BatchEvaluateResponse:
    results: list[QualityReport] = []
    for item in payload.items:
        item.skip_judge = item.skip_judge or payload.skip_judge
        results.append(await evaluate_answer(item, settings))
    return BatchEvaluateResponse(results=results)


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

    judge_scores = [_dimension_mean(report) for report in reports]
    semantic_scores = [report.semantic.cosine for report in reports if report.semantic.cosine is not None]
    semantic_human = [
        score
        for report, score in zip(reports, human)
        if report.semantic.cosine is not None
    ]
    lexical_scores = [
        ((report.lexical.rouge_l or 0.0) + (report.lexical.bleu or 0.0)) / 2
        if report.lexical.rouge_l is not None or report.lexical.bleu is not None
        else None
        for report in reports
    ]

    return ValidateResponse(
        overall_vs_human=correlate([r.overall for r in reports], human),
        judge_vs_human=correlate(judge_scores, human),
        semantic_vs_human=correlate(semantic_scores, semantic_human),
        lexical_vs_human=correlate(
            [s for s in lexical_scores if s is not None],
            [h for s, h in zip(lexical_scores, human) if s is not None],
        ),
        biases=compute_biases(reports),
        predictions=reports,
    )


def _dimension_mean(report: QualityReport) -> float:
    dims = report.dimensions
    values = [dims.correctness, dims.relevance, dims.completeness, dims.coherence]
    if dims.groundedness is not None:
        values.append(dims.groundedness)
    return sum(values) / len(values)
