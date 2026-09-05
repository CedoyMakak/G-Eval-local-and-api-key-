from __future__ import annotations

from app.schemas import CorrelationReport


def correlate(pred: list[float], gold: list[float]) -> CorrelationReport:
    paired = [(p, g) for p, g in zip(pred, gold) if p is not None and g is not None]
    n = len(paired)
    if n < 2:
        return CorrelationReport(n=n)

    xs = [p for p, _ in paired]
    ys = [g for _, g in paired]
    if _constant(xs) or _constant(ys):
        return CorrelationReport(n=n, pearson=None, spearman=None, kendall=None)

    from scipy.stats import kendalltau, pearsonr, spearmanr

    pearson = float(pearsonr(xs, ys).statistic)
    spearman = float(spearmanr(xs, ys).statistic)
    kendall = float(kendalltau(xs, ys).statistic)
    return CorrelationReport(
        n=n,
        pearson=_safe(pearson),
        spearman=_safe(spearman),
        kendall=_safe(kendall),
    )


def _constant(values: list[float]) -> bool:
    return max(values) - min(values) < 1e-12


def _safe(value: float) -> float | None:
    if value != value:  # NaN
        return None
    return round(value, 4)
