from __future__ import annotations

from app.schemas import Dimensions, LexicalMetrics, QualityReport


def dimension_mean(dimensions: Dimensions) -> float:
    values = [
        dimensions.correctness,
        dimensions.relevance,
        dimensions.completeness,
        dimensions.coherence,
    ]
    if dimensions.groundedness is not None:
        values.append(dimensions.groundedness)
    return sum(values) / len(values)


def style_mean(dimensions: Dimensions) -> float:
    values = [dimensions.relevance, dimensions.completeness, dimensions.coherence]
    if dimensions.groundedness is not None:
        values.append(dimensions.groundedness)
    return sum(values) / len(values)


def gated_judge_score(dimensions: Dimensions) -> float:
    """Беглость не компенсирует фактическую ошибку: overall ≤ correctness · style."""
    return max(0.0, min(1.0, dimensions.correctness * style_mean(dimensions)))


def lexical_mean(lexical: LexicalMetrics) -> float | None:
    scores = [value for value in (lexical.rouge_l, lexical.bleu, lexical.chrf) if value is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)


def report_judge_mean(report: QualityReport) -> float:
    return gated_judge_score(report.dimensions)
