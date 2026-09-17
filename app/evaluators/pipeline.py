from __future__ import annotations

from app.config import Settings, get_settings
from app.evaluators.aggregator import aggregate
from app.evaluators.common import dimension_mean, gated_judge_score
from app.evaluators.factcheck import compute_factcheck
from app.evaluators.heuristics import compute_heuristics
from app.evaluators.lexical import compute_lexical
from app.evaluators.llm_judge import judge_pairwise, judge_pointwise, merge_dimensions
from app.evaluators.semantic import compute_semantic
from app.providers.base import JudgeProvider
from app.providers.factory import get_judge_provider
from app.schemas import (
    Dimensions,
    EvaluateRequest,
    JudgeMember,
    JudgeResult,
    PairwiseJudgment,
    PairwiseResponse,
    QualityReport,
)


def compute_offline_layers(request: EvaluateRequest, settings: Settings):
    lexical = compute_lexical(request.answer, request.reference)
    semantic = compute_semantic(request.answer, request.reference, settings.embedding_model)
    heuristics = compute_heuristics(request.answer, request.question, request.reference)
    factcheck = compute_factcheck(request.answer, request.context)
    return lexical, semantic, heuristics, factcheck


async def evaluate_answer(
    request: EvaluateRequest,
    settings: Settings | None = None,
) -> QualityReport:
    settings = settings or get_settings()
    lexical, semantic, heuristics, factcheck = compute_offline_layers(request, settings)

    dimensions, judge = await _run_judge(
        request=request,
        settings=settings,
        semantic=semantic,
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
        factcheck=factcheck,
    )


async def _run_judge(
    *,
    request: EvaluateRequest,
    settings: Settings,
    semantic,
) -> tuple[Dimensions, JudgeResult]:
    if request.skip_judge:
        return await judge_pointwise(
            provider=None,
            settings=settings,
            question=request.question,
            answer=request.answer,
            reference=request.reference,
            context=request.context,
            semantic=semantic,
            skip_judge=True,
        )

    names = _judge_names(settings)
    results: list[tuple[Dimensions, JudgeResult]] = []
    for name in names:
        provider = _safe_provider(settings, name)
        pair = await _pointwise_with_optional_swap(
            provider=provider,
            settings=settings,
            request=request,
            semantic=semantic,
        )
        results.append(pair)

    if len(results) == 1:
        return results[0]

    live = [(dims, judge) for dims, judge in results if not judge.fallback]
    usable = live or results
    mode = settings.ensemble_mode if live else "mean"
    if mode == "off":
        mode = "mean"
    merged = merge_dimensions([dims for dims, _ in usable], mode)
    members: list[JudgeMember] = []
    for _, judge in results:
        members.extend(judge.members or [
            JudgeMember(
                provider=judge.provider,
                model=judge.model,
                fallback=judge.fallback,
                rationale=judge.rationale,
                score=round(gated_judge_score(judge.raw_scores or merged), 4),
            )
        ])
    fallback = all(judge.fallback for _, judge in results)
    position_bias = any(judge.position_bias_detected for _, judge in results)
    deltas = [judge.position_delta for _, judge in results if judge.position_delta is not None]
    rationale_parts = [f"{judge.provider}/{judge.model}: {judge.rationale}" for _, judge in usable]
    return merged, JudgeResult(
        provider="ensemble",
        model="+".join(f"{judge.provider}:{judge.model}" for _, judge in results),
        rationale=" | ".join(part for part in rationale_parts if part)[:1200],
        raw_scores=merged,
        used_logprobs=any(judge.used_logprobs for _, judge in results),
        fallback=fallback,
        ensemble=mode,
        members=members,
        position_bias_detected=position_bias,
        position_delta=max(deltas) if deltas else None,
    )


async def _pointwise_with_optional_swap(
    *,
    provider: JudgeProvider,
    settings: Settings,
    request: EvaluateRequest,
    semantic,
) -> tuple[Dimensions, JudgeResult]:
    first_dims, first_judge = await judge_pointwise(
        provider=provider,
        settings=settings,
        question=request.question,
        answer=request.answer,
        reference=request.reference,
        context=request.context,
        semantic=semantic,
        skip_judge=False,
        swap_blocks=False,
    )
    if first_judge.fallback or not settings.judge_swap_check:
        return first_dims, first_judge

    swap_dims, swap_judge = await judge_pointwise(
        provider=provider,
        settings=settings,
        question=request.question,
        answer=request.answer,
        reference=request.reference,
        context=request.context,
        semantic=semantic,
        skip_judge=False,
        swap_blocks=True,
    )
    if swap_judge.fallback:
        return first_dims, first_judge

    delta = abs(dimension_mean(first_dims) - dimension_mean(swap_dims))
    biased = delta >= settings.position_bias_threshold
    merged = merge_dimensions([first_dims, swap_dims], "min" if biased else "mean")
    first_judge.raw_scores = merged
    first_judge.position_bias_detected = biased
    first_judge.position_delta = round(delta, 4)
    if first_judge.members:
        first_judge.members[0].score = round(gated_judge_score(merged), 4)
    if biased:
        first_judge.rationale = (
            f"{first_judge.rationale} [position-swap Δ={delta:.2f}; взят min]"
        )
    return merged, first_judge


def _judge_names(settings: Settings) -> list[str]:
    if settings.ensemble_mode == "off":
        return [settings.judge_provider.strip().lower()]
    names: list[str] = []
    for name in settings.ensemble_provider_names:
        if name not in names:
            names.append(name)
    return names or [settings.judge_provider.strip().lower()]


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
    provider = _safe_provider(settings, settings.judge_provider)

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
        stability=1.0 if consistent else 0.0,
    )


def _safe_provider(settings: Settings, provider_name: str | None = None) -> JudgeProvider:
    name = (provider_name or settings.judge_provider).strip().lower()
    try:
        return get_judge_provider(settings, name)
    except Exception:
        return _UnavailableProvider(name, _model_for(settings, name))


def _model_for(settings: Settings, provider_name: str) -> str:
    if provider_name == "openai":
        return settings.openai_model
    if provider_name == "openrouter":
        return settings.openrouter_model
    return settings.ollama_model


class _UnavailableProvider:
    def __init__(self, name: str, model: str) -> None:
        self.name = name
        self.model = model

    async def complete(self, prompt: str, *, temperature: float = 0.0):
        raise RuntimeError(f"Провайдер {self.name} не сконфигурирован")
