from __future__ import annotations

from app.schemas import WeightSearchResult
from app.validation.metrics import correlate


def search_weights(
    *,
    semantic: list[float],
    lexical: list[float],
    judge: list[float],
    human: list[float],
    step: float = 0.05,
) -> WeightSearchResult:
    n = min(len(semantic), len(lexical), len(judge), len(human))
    best: WeightSearchResult | None = None
    ticks = _grid(step)
    for weight_semantic in ticks:
        for weight_lexical in ticks:
            weight_judge = round(1.0 - weight_semantic - weight_lexical, 4)
            if weight_judge < -1e-9:
                continue
            if weight_judge < 0:
                weight_judge = 0.0
            preds = [
                weight_semantic * semantic[i] + weight_lexical * lexical[i] + weight_judge * judge[i]
                for i in range(n)
            ]
            report = correlate(preds, human[:n])
            spearman = report.spearman
            if spearman is None:
                continue
            candidate = WeightSearchResult(
                weight_semantic=round(weight_semantic, 4),
                weight_lexical=round(weight_lexical, 4),
                weight_judge=round(weight_judge, 4),
                spearman=spearman,
                pearson=report.pearson,
                n=n,
            )
            if best is None or spearman > best.spearman + 1e-9:
                best = candidate
            elif best and abs(spearman - best.spearman) < 1e-9 and weight_judge > best.weight_judge:
                best = candidate
    if best is None:
        return WeightSearchResult(
            weight_semantic=0.25,
            weight_lexical=0.0,
            weight_judge=0.75,
            spearman=0.0,
            pearson=None,
            n=n,
        )
    return best


def _grid(step: float) -> list[float]:
    values: list[float] = []
    current = 0.0
    while current <= 1.0 + 1e-9:
        values.append(round(current, 4))
        current += step
    return values
