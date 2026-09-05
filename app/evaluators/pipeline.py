from __future__ import annotations

from app.config import Settings, get_settings
from app.evaluators.aggregator import aggregate
from app.evaluators.heuristics import compute_heuristics
from app.evaluators.lexical import compute_lexical
from app.evaluators.llm_judge import judge_pairwise, judge_pointwise
from app.evaluators.semantic import compute_semantic
from app.providers.base import JudgeProvider
from app.providers.factory import get_judge_provider
from app.schemas import (
    EvaluateRequest,
    PairwiseJudgment,
    PairwiseResponse,
    QualityReport,
)


async def evaluate_answer(
    request: EvaluateRequest,
    settings: Settings | None = None,
) -> QualityReport:
    settings = settings or get_settings()
    lexical = compute_lexical(request.answer, request.reference)
    semantic = compute_semantic(request.answer, request.reference, settings.embedding_model)
    heuristics = compute_heuristics(request.answer, request.question, request.reference)

    provider = None if request.skip_judge else _safe_provider(settings)
    dimensions, judge = await judge_pointwise(
        provider=provider,
        settings=settings,
        question=request.question,
        answer=request.answer,
        reference=request.reference,
        context=request.context,
        semantic=semantic,
        skip_judge=request.skip_judge,
    )

    return aggregate(
        settings=settings,
        lexical=lexical,
        semantic=semantic,
        dimensions=dimensions,
        judge=judge,
        heuristics=heuristics,
        has_reference=bool(request.reference and request.reference.strip()),
        has_context=bool(request.context and request.context.strip()),
    )


def _invert_winner(winner: str) -> str:
    if winner == "A":
        return "B"
    if winner == "B":
        return "A"
    return "tie"


async def evaluate_pairwise(
    question: str,
    answer_a: str,
    answer_b: str,
    reference: str | None = None,
    context: str | None = None,
    settings: Settings | None = None,
) -> PairwiseResponse:
    settings = settings or get_settings()
    provider = _safe_provider(settings)

    first = await judge_pairwise(
        provider=provider,
        settings=settings,
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        reference=reference,
        context=context,
        order="A,B",
    )
    swapped_raw = await judge_pairwise(
        provider=provider,
        settings=settings,
        question=question,
        answer_a=answer_b,
        answer_b=answer_a,
        reference=reference,
        context=context,
        order="B,A",
    )
    swapped = PairwiseJudgment(
        winner=_invert_winner(swapped_raw.winner),
        rationale=swapped_raw.rationale,
        order=swapped_raw.order,
    )
    consistent = first.winner == swapped.winner
    preferred: str
    if consistent:
        preferred = first.winner
    else:
        preferred = "inconsistent"

    return PairwiseResponse(
        first_pass=first,
        swapped_pass=swapped,
        consistent=consistent,
        position_bias_detected=not consistent,
        preferred=preferred,  # type: ignore[arg-type]
    )


def _safe_provider(settings: Settings) -> JudgeProvider:
    try:
        return get_judge_provider(settings)
    except Exception:
        return _UnavailableProvider(settings.judge_provider, settings.judge_model_name)


class _UnavailableProvider:
    def __init__(self, name: str, model: str) -> None:
        self.name = name
        self.model = model

    async def complete(self, prompt: str, *, temperature: float = 0.0):
        raise RuntimeError(f"Провайдер {self.name} не сконфигурирован")
