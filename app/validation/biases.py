from __future__ import annotations

from app.schemas import BiasReport, QualityReport
from app.validation.metrics import correlate


def compute_biases(reports: list[QualityReport]) -> BiasReport:
    lengths = [item.heuristics.get("length_words", 0.0) for item in reports]
    overall = [item.overall for item in reports]
    verbosity = correlate(lengths, overall)

    disagreements: list[float] = []
    for item in reports:
        if item.semantic.cosine is None:
            continue
        judge_mean = _judge_mean(item)
        disagreements.append(abs(item.semantic.cosine - judge_mean))

    high = 0.0
    mean_dis = None
    if disagreements:
        mean_dis = round(sum(disagreements) / len(disagreements), 4)
        high = round(sum(1 for value in disagreements if value >= 0.25) / len(disagreements), 4)

    return BiasReport(
        verbosity_pearson=verbosity.pearson,
        verbosity_spearman=verbosity.spearman,
        mean_disagreement=mean_dis,
        high_disagreement_rate=high,
    )


def _judge_mean(report: QualityReport) -> float:
    dims = report.dimensions
    values = [dims.correctness, dims.relevance, dims.completeness, dims.coherence]
    if dims.groundedness is not None:
        values.append(dims.groundedness)
    return sum(values) / len(values)
