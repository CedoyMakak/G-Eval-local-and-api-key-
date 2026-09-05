from __future__ import annotations

from app.config import Settings
from app.schemas import (
    Dimensions,
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
) -> QualityReport:
    judge_score = _dimension_mean(dimensions)
    semantic_score = semantic.cosine
    lexical_score = _lexical_mean(lexical)

    if has_reference and semantic_score is not None:
        overall = (
            settings.weight_semantic * semantic_score
            + settings.weight_judge * judge_score
        )
        disagreement = abs(semantic_score - judge_score)
        confidence = max(0.0, min(1.0, 1.0 - disagreement))
        if lexical_score is not None:
            confidence = max(0.0, min(1.0, 0.7 * confidence + 0.3 * (1.0 - abs(lexical_score - judge_score))))
    else:
        overall = 0.85 * judge_score + 0.15 * heuristics.get("length_score", 0.0)
        disagreement = 0.0
        confidence = 0.55 if judge.fallback else 0.7

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
    )


def _dimension_mean(dimensions: Dimensions) -> float:
    values = [
        dimensions.correctness,
        dimensions.relevance,
        dimensions.completeness,
        dimensions.coherence,
    ]
    if dimensions.groundedness is not None:
        values.append(dimensions.groundedness)
    return sum(values) / len(values)


def _lexical_mean(lexical: LexicalMetrics) -> float | None:
    scores = [value for value in (lexical.rouge_l, lexical.bleu) if value is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)
