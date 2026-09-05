from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.evaluators.pipeline import evaluate_answer
from app.schemas import EvaluateRequest, LabeledItem, QualityReport
from app.validation.biases import compute_biases
from app.validation.metrics import correlate


def load_labels(path: Path) -> list[LabeledItem]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [LabeledItem.model_validate(item) for item in raw]


async def run(labels_path: Path, report_path: Path, skip_judge: bool) -> None:
    settings = get_settings()
    items = load_labels(labels_path)
    reports: list[QualityReport] = []
    human = [item.human_score for item in items]

    for item in items:
        report = await evaluate_answer(
            EvaluateRequest(
                question=item.question,
                answer=item.answer,
                reference=item.reference,
                context=item.context,
                skip_judge=skip_judge,
            ),
            settings,
        )
        reports.append(report)

    judge_scores = [_dim_mean(report) for report in reports]
    semantic_pairs = [
        (report.semantic.cosine, score)
        for report, score in zip(reports, human)
        if report.semantic.cosine is not None
    ]
    lexical_pairs = [
        (_lex_mean(report), score)
        for report, score in zip(reports, human)
        if _lex_mean(report) is not None
    ]
    biases = compute_biases(reports)

    overall = correlate([r.overall for r in reports], human)
    judge = correlate(judge_scores, human)
    semantic = correlate([p for p, _ in semantic_pairs], [h for _, h in semantic_pairs])
    lexical = correlate([p for p, _ in lexical_pairs], [h for _, h in lexical_pairs])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render_markdown(
            n=len(items),
            skip_judge=skip_judge,
            provider=settings.judge_provider,
            model=settings.judge_model_name,
            overall=overall,
            judge=judge,
            semantic=semantic,
            lexical=lexical,
            biases=biases,
            fallback_rate=sum(1 for r in reports if r.judge.fallback) / len(reports),
        ),
        encoding="utf-8",
    )
    print(f"Отчёт записан в {report_path}")


def _dim_mean(report: QualityReport) -> float:
    dims = report.dimensions
    values = [dims.correctness, dims.relevance, dims.completeness, dims.coherence]
    if dims.groundedness is not None:
        values.append(dims.groundedness)
    return sum(values) / len(values)


def _lex_mean(report: QualityReport) -> float | None:
    scores = [v for v in (report.lexical.rouge_l, report.lexical.bleu) if v is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _render_markdown(**kwargs) -> str:
    return f"""# Отчёт валидации системы оценки ответов LLM

- Размер выборки: **{kwargs['n']}**
- Провайдер судьи: `{kwargs['provider']}` / `{kwargs['model']}`
- skip_judge: `{kwargs['skip_judge']}`
- Доля heuristic-fallback: **{kwargs['fallback_rate']:.1%}**

## Корреляция с ручными метками

| Слой | n | Pearson | Spearman | Kendall |
|---|---:|---:|---:|---:|
| overall | {kwargs['overall'].n} | {_fmt(kwargs['overall'].pearson)} | {_fmt(kwargs['overall'].spearman)} | {_fmt(kwargs['overall'].kendall)} |
| LLM-as-a-Judge | {kwargs['judge'].n} | {_fmt(kwargs['judge'].pearson)} | {_fmt(kwargs['judge'].spearman)} | {_fmt(kwargs['judge'].kendall)} |
| semantic cosine | {kwargs['semantic'].n} | {_fmt(kwargs['semantic'].pearson)} | {_fmt(kwargs['semantic'].spearman)} | {_fmt(kwargs['semantic'].kendall)} |
| lexical ROUGE/BLEU | {kwargs['lexical'].n} | {_fmt(kwargs['lexical'].pearson)} | {_fmt(kwargs['lexical'].spearman)} | {_fmt(kwargs['lexical'].kendall)} |

## Искажения

- Verbosity bias (Pearson длина vs overall): **{_fmt(kwargs['biases'].verbosity_pearson)}**
- Verbosity bias (Spearman): **{_fmt(kwargs['biases'].verbosity_spearman)}**
- Среднее расхождение semantic vs judge: **{_fmt(kwargs['biases'].mean_disagreement)}**
- Доля высокой disagreement (>= 0.25): **{_fmt(kwargs['biases'].high_disagreement_rate)}**

## Как читать

HITL-контур не участвует в online-оценке. Ручные метки нужны, чтобы проверить, насколько автоматический overall согласуется с человеком, и увидеть типичные искажения (многословие, расхождение слоёв).
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Офлайн-валидация системы оценки ответов LLM")
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "data" / "human_labels.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "validation.md",
    )
    parser.add_argument("--skip-judge", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.labels, args.output, args.skip_judge))


if __name__ == "__main__":
    main()
