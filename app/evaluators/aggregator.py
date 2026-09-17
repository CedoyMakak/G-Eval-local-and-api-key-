from __future__ import annotations

from app.config import Settings
from app.evaluators.common import dimension_mean, gated_judge_score, lexical_mean
from app.schemas import (
    Dimensions,
    FactCheckMetrics,
    JudgeResult,
    LexicalMetrics,
    QualityFlags,
    QualityReport,
    SemanticMetrics,
)


def aggregate(
    *,
    settings: Settings,
    lexical: LexicalMetrics,
    semantic: SemanticMetrics,
    dimensions: Dimensions,
    judge: JudgeResult,
    heuristics: dict[str, float],
    has_reference: bool,
    has_context: bool,
    factcheck: FactCheckMetrics | None = None,
) -> QualityReport:
    judge_score = gated_judge_score(dimensions)
    semantic_score = semantic.cosine
    lexical_score = lexical_mean(lexical)
    position_bias = bool(judge.position_bias_detected)
    correctness_capped = judge_score + 0.15 < dimension_mean(dimensions)

    if has_reference and semantic_score is not None:
        overall = _weighted_overall(
            settings,
            semantic_score=semantic_score,
            lexical_score=lexical_score,
            judge_score=judge_score,
        )
        disagreement = abs(semantic_score - judge_score)
        confidence = max(0.0, min(1.0, 1.0 - disagreement))
        if lexical_score is not None:
            confidence = max(
                0.0,
                min(1.0, 0.7 * confidence + 0.3 * (1.0 - abs(lexical_score - judge_score))),
            )
    else:
        overall = 0.85 * judge_score + 0.15 * heuristics.get("length_score", 0.0)
        disagreement = 0.0
        confidence = 0.55 if judge.fallback else 0.7

    if factcheck is not None and factcheck.score is not None and has_context:
        overall = 0.9 * overall + 0.1 * factcheck.score
        if factcheck.unsupported_numbers:
            overall = min(overall, 0.55)
            confidence = min(confidence, 0.55)

    ensemble_gap = _ensemble_gap(judge)
    if ensemble_gap is not None and ensemble_gap >= settings.disagreement_threshold:
        confidence = min(confidence, 0.55)

    if position_bias:
        overall *= max(0.0, 1.0 - settings.position_bias_penalty)
        confidence = min(confidence, 0.5)

    if judge.fallback:
        confidence = min(confidence, 0.5)
    too_short = heuristics.get("length_words", 0) < 1
    if too_short:
        overall = min(overall, 0.15)
        confidence = min(confidence, 0.3)

    flags = QualityFlags(
        no_reference=not has_reference,
        no_context=not has_context,
        high_disagreement=disagreement >= settings.disagreement_threshold,
        empty_or_too_short=too_short,
        judge_fallback=judge.fallback,
        position_bias_detected=position_bias,
        ensemble_disagreement=bool(
            ensemble_gap is not None and ensemble_gap >= settings.disagreement_threshold
        ),
        correctness_capped=correctness_capped,
    )

    return QualityReport(
        overall=round(max(0.0, min(1.0, overall)), 4),
        confidence=round(confidence, 4),
        dimensions=dimensions,
        lexical=lexical,
        semantic=semantic,
        judge=judge,
        flags=flags,
        heuristics=heuristics,
        factcheck=factcheck,
    )


def _weighted_overall(
    settings: Settings,
    *,
    semantic_score: float,
    lexical_score: float | None,
    judge_score: float,
) -> float:
    parts = [
        (settings.weight_semantic, semantic_score),
        (settings.weight_judge, judge_score),
    ]
    if lexical_score is not None and settings.weight_lexical > 0:
        parts.append((settings.weight_lexical, lexical_score))
    weight_sum = sum(weight for weight, _ in parts)
    if weight_sum <= 0:
        return judge_score
    return sum(weight * value for weight, value in parts) / weight_sum


def _ensemble_gap(judge: JudgeResult) -> float | None:
    scores = [member.score for member in judge.members if member.score is not None and not member.fallback]
    if len(scores) < 2:
        return None
    return abs(max(scores) - min(scores))
