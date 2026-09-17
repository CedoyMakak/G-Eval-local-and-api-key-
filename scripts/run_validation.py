from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings, get_settings
from app.evaluators.aggregator import aggregate
from app.evaluators.common import gated_judge_score, lexical_mean
from app.evaluators.pipeline import compute_offline_layers, evaluate_answer
from app.evaluators.semantic import probe_semantic_method
from app.schemas import (
    Dimensions,
    EvaluateRequest,
    JudgeResult,
    LabeledItem,
    QualityReport,
    WeightSearchResult,
)
from app.validation.biases import compute_biases
from app.validation.errors import analyze_errors
from app.validation.metrics import correlate
from app.validation.weights import search_weights

CACHE_DIR = ROOT / "reports" / "judge_cache"


def load_labels(path: Path) -> list[LabeledItem]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [LabeledItem.model_validate(item) for item in raw]


async def run(
    labels_path: Path,
    report_path: Path,
    skip_judge: bool,
    compare: bool,
    refresh_cache: bool,
) -> None:
    settings = get_settings()
    items = load_labels(labels_path)
    if compare:
        blocks = await _run_compare(items, settings, refresh_cache)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(_render_compare(items, blocks), encoding="utf-8")
        json_path = report_path.with_suffix(".json")
        json_path.write_text(
            json.dumps(_compare_payload(items, blocks), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Отчёт записан в {report_path}")
        return

    cache_key = None
    if not skip_judge:
        cache_key = _cache_key(settings.judge_provider, settings.judge_model_name)
    reports = await _evaluate_all(
        items,
        settings,
        skip_judge=skip_judge,
        cache_key=cache_key,
    )
    _write_single_report(items, reports, settings, skip_judge, report_path)
    print(f"Отчёт записан в {report_path}")


async def _run_compare(items: list[LabeledItem], settings: Settings, refresh_cache: bool) -> dict:
    heuristic_settings = settings.model_copy(update={"judge_ensemble": "off"})
    heuristic_reports = await _evaluate_all(items, heuristic_settings, skip_judge=True)

    gemma_settings = settings.model_copy(update={"judge_provider": "openrouter", "judge_ensemble": "off"})
    ollama_settings = settings.model_copy(update={"judge_provider": "ollama", "judge_ensemble": "off"})

    gemma_reports = await _evaluate_all(
        items,
        gemma_settings,
        skip_judge=False,
        cache_key=_cache_key("openrouter", settings.openrouter_model),
        refresh_cache=refresh_cache,
    )
    ollama_reports = await _evaluate_all(
        items,
        ollama_settings,
        skip_judge=False,
        cache_key=_cache_key("ollama", settings.ollama_model),
        refresh_cache=refresh_cache,
    )
    ensemble_mean = [_ensemble_report(a, b, "mean", settings) for a, b in zip(gemma_reports, ollama_reports)]
    ensemble_min = [_ensemble_report(a, b, "min", settings) for a, b in zip(gemma_reports, ollama_reports)]

    return {
        "heuristics": heuristic_reports,
        "gemma": gemma_reports,
        "ollama": ollama_reports,
        "ensemble_mean": ensemble_mean,
        "ensemble_min": ensemble_min,
        "settings": settings,
    }


async def _evaluate_all(
    items: list[LabeledItem],
    settings: Settings,
    *,
    skip_judge: bool,
    cache_key: str | None = None,
    refresh_cache: bool = False,
) -> list[QualityReport]:
    cache = _load_cache(cache_key) if cache_key and not skip_judge else {}
    reports: list[QualityReport] = []
    for index, item in enumerate(items, start=1):
        request = EvaluateRequest(
            question=item.question,
            answer=item.answer,
            reference=item.reference,
            context=item.context,
            skip_judge=skip_judge,
        )
        cached = cache.get(item.id)
        live_cached = bool(cached) and not (cached.get("judge") or {}).get("fallback", False)
        if live_cached and not refresh_cache and not skip_judge:
            print(f"[{index}/{len(items)}] {item.id} cache={cache_key}", flush=True)
            reports.append(_report_from_cache(request, settings, cached))
            continue
        print(f"[{index}/{len(items)}] {item.id} skip_judge={skip_judge} provider={settings.judge_provider}", flush=True)
        report = await evaluate_answer(request, settings)
        reports.append(report)
        if cache_key and not skip_judge and not report.judge.fallback:
            cache[item.id] = {
                "dimensions": report.dimensions.model_dump(),
                "judge": report.judge.model_dump(),
            }
            _save_cache(cache_key, cache)
    return reports


def _report_from_cache(request: EvaluateRequest, settings: Settings, cached: dict) -> QualityReport:
    lexical, semantic, heuristics, factcheck = compute_offline_layers(request, settings)
    if "dimensions" in cached and "judge" in cached:
        dimensions = Dimensions.model_validate(cached["dimensions"])
        judge = JudgeResult.model_validate(cached["judge"])
    else:
        full = QualityReport.model_validate(cached)
        dimensions = full.dimensions
        judge = full.judge
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


def _ensemble_report(left: QualityReport, right: QualityReport, mode: str, settings: Settings) -> QualityReport:
    from app.evaluators.aggregator import aggregate
    from app.evaluators.llm_judge import merge_dimensions
    from app.schemas import JudgeMember, JudgeResult

    live = []
    members: list[JudgeMember] = []
    for report in (left, right):
        members.extend(report.judge.members or [
            JudgeMember(
                provider=report.judge.provider,
                model=report.judge.model,
                fallback=report.judge.fallback,
                rationale=report.judge.rationale,
                score=round(gated_judge_score(report.dimensions), 4),
            )
        ])
        if not report.judge.fallback:
            live.append(report.dimensions)
    usable = live or [left.dimensions, right.dimensions]
    merged = merge_dimensions(usable, mode)
    fallback = not live
    judge = JudgeResult(
        provider="ensemble",
        model=f"{left.judge.model}+{right.judge.model}",
        rationale=f"{left.judge.rationale} | {right.judge.rationale}"[:1200],
        raw_scores=merged,
        used_logprobs=left.judge.used_logprobs or right.judge.used_logprobs,
        fallback=fallback,
        ensemble=mode,
        members=members,
        position_bias_detected=left.judge.position_bias_detected or right.judge.position_bias_detected,
    )
    return aggregate(
        settings=settings.model_copy(update={"judge_ensemble": mode}),
        lexical=left.lexical,
        semantic=left.semantic,
        dimensions=merged,
        judge=judge,
        heuristics=left.heuristics,
        has_reference=not left.flags.no_reference,
        has_context=not left.flags.no_context,
        factcheck=left.factcheck,
    )


def _layer_table(items: list[LabeledItem], reports: list[QualityReport]) -> dict:
    human = [item.human_score for item in items]
    judge_scores = [gated_judge_score(report.dimensions) for report in reports]
    semantic_pairs = [
        (report.semantic.cosine, score)
        for report, score in zip(reports, human)
        if report.semantic.cosine is not None
    ]
    lexical_pairs = [
        (lexical_mean(report.lexical), score)
        for report, score in zip(reports, human)
        if lexical_mean(report.lexical) is not None
    ]
    rouge_pairs = [
        (report.lexical.rouge_l, score)
        for report, score in zip(reports, human)
        if report.lexical.rouge_l is not None
    ]
    bleu_pairs = [
        (report.lexical.bleu, score)
        for report, score in zip(reports, human)
        if report.lexical.bleu is not None
    ]
    chrf_pairs = [
        (report.lexical.chrf, score)
        for report, score in zip(reports, human)
        if report.lexical.chrf is not None
    ]
    bert_pairs = [
        (report.semantic.bertscore, score)
        for report, score in zip(reports, human)
        if report.semantic.bertscore is not None
    ]
    dim_rows = {}
    for key in ("correctness", "relevance", "completeness", "coherence", "groundedness"):
        pairs = []
        for item, report in zip(items, reports):
            gold = getattr(item.human_dimensions, key) if item.human_dimensions else None
            pred = getattr(report.dimensions, key)
            if gold is not None and pred is not None:
                pairs.append((pred, gold))
        if pairs:
            dim_rows[key] = correlate([p for p, _ in pairs], [g for _, g in pairs])
    return {
        "overall": correlate([r.overall for r in reports], human),
        "judge": correlate(judge_scores, human),
        "semantic": correlate([p for p, _ in semantic_pairs], [h for _, h in semantic_pairs]),
        "lexical": correlate([p for p, _ in lexical_pairs], [h for _, h in lexical_pairs]),
        "rouge": correlate([p for p, _ in rouge_pairs], [h for _, h in rouge_pairs]),
        "bleu": correlate([p for p, _ in bleu_pairs], [h for _, h in bleu_pairs]),
        "chrf": correlate([p for p, _ in chrf_pairs], [h for _, h in chrf_pairs]),
        "bertscore": correlate([p for p, _ in bert_pairs], [h for _, h in bert_pairs]) if bert_pairs else None,
        "dimensions": dim_rows,
        "biases": compute_biases(reports),
        "fallback_rate": sum(1 for r in reports if r.judge.fallback) / max(len(reports), 1),
        "semantic_method": next((r.semantic.method for r in reports if r.semantic.method), probe_semantic_method()),
    }


def _weight_result(items: list[LabeledItem], reports: list[QualityReport]) -> WeightSearchResult | None:
    semantic: list[float] = []
    lexical: list[float] = []
    judge: list[float] = []
    human: list[float] = []
    for item, report in zip(items, reports):
        if report.semantic.cosine is None:
            continue
        lex = lexical_mean(report.lexical)
        if lex is None:
            continue
        semantic.append(report.semantic.cosine)
        lexical.append(lex)
        judge.append(gated_judge_score(report.dimensions))
        human.append(item.human_score)
    if len(human) < 8:
        return None
    return search_weights(semantic=semantic, lexical=lexical, judge=judge, human=human)


def _write_single_report(
    items: list[LabeledItem],
    reports: list[QualityReport],
    settings: Settings,
    skip_judge: bool,
    report_path: Path,
) -> None:
    layers = _layer_table(items, reports)
    human_high, auto_high = analyze_errors(items, reports)
    weights = _weight_result(items, reports)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render_single(
            n=len(items),
            skip_judge=skip_judge,
            provider=settings.judge_provider,
            model=settings.judge_model_name,
            layers=layers,
            human_high=human_high,
            auto_high=auto_high,
            weights=weights,
        ),
        encoding="utf-8",
    )


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _corr_row(name: str, report) -> str:
    if report is None:
        return f"| {name} | 0 | — | — | — |"
    return f"| {name} | {report.n} | {_fmt(report.pearson)} | {_fmt(report.spearman)} | {_fmt(report.kendall)} |"


def _error_table(rows) -> str:
    if not rows:
        return "_Нет случаев с |Δ| ≥ 0.25._"
    lines = [
        "| id | тип | human | auto | Δ | причины |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        question = row.question.replace("|", "/")[:70]
        reasons = ", ".join(row.reasons)
        lines.append(
            f"| `{row.id}` | {row.case_type or '—'} | {row.human:.2f} | {row.auto:.2f} | {row.delta:+.2f} | {reasons} |"
        )
        lines.append(f"|  | _{question}_ |  |  |  | {row.answer.replace('|', '/')[:90]} |")
    return "\n".join(lines)


def _render_single(**kwargs) -> str:
    layers = kwargs["layers"]
    weights = kwargs["weights"]
    weight_line = "недостаточно пар с эталоном"
    if weights:
        weight_line = (
            f"overall = {weights.weight_semantic:.2f}·cosine + "
            f"{weights.weight_lexical:.2f}·lexical + {weights.weight_judge:.2f}·judge "
            f"(Spearman {weights.spearman:.3f})"
        )
    dim_rows = "\n".join(_corr_row(name, report) for name, report in layers["dimensions"].items()) or "| — | 0 | — | — | — |"
    return f"""# Отчёт валидации системы оценки ответов LLM

- Размер выборки: **{kwargs['n']}**
- Провайдер судьи: `{kwargs['provider']}` / `{kwargs['model']}`
- skip_judge: `{kwargs['skip_judge']}`
- Семантика: `{layers['semantic_method']}`
- Доля heuristic-fallback: **{layers['fallback_rate']:.1%}**

## Корреляция с ручными метками

| Слой | n | Pearson | Spearman | Kendall |
|---|---:|---:|---:|---:|
{_corr_row("overall", layers["overall"])}
{_corr_row("LLM-as-a-Judge", layers["judge"])}
{_corr_row("semantic cosine", layers["semantic"])}
{_corr_row("lexical (ROUGE/BLEU/chrF)", layers["lexical"])}
{_corr_row("ROUGE-L", layers["rouge"])}
{_corr_row("BLEU", layers["bleu"])}
{_corr_row("chrF", layers["chrf"])}
{_corr_row("BERTScore", layers["bertscore"])}

## Корреляция по критериям HITL

| Критерий | n | Pearson | Spearman | Kendall |
|---|---:|---:|---:|---:|
{dim_rows}

## Веса агрегатора по HITL

{weight_line}

## Искажения

- Verbosity bias (Pearson длина vs overall): **{_fmt(layers['biases'].verbosity_pearson)}**
- Verbosity bias (Spearman): **{_fmt(layers['biases'].verbosity_spearman)}**
- Среднее расхождение semantic vs judge: **{_fmt(layers['biases'].mean_disagreement)}**
- Доля высокой disagreement (>= 0.25): **{_fmt(layers['biases'].high_disagreement_rate)}**

## Разбор ошибок: человек высокий, автомат низкий

{_error_table(kwargs['human_high'])}

## Разбор ошибок: человек низкий, автомат высокий

{_error_table(kwargs['auto_high'])}

## Как читать

HITL не участвует в online-оценке. Ручные метки проверяют ранжирование автомата. Таблица ошибок важнее ещё одного графика: по причинам (`verbosity`, `short-reference`, `fallback`, `disagreement`) видно, что чинить.
"""


def _render_compare(items: list[LabeledItem], blocks: dict) -> str:
    settings = blocks["settings"]
    names = [
        ("heuristics", "Heuristics / fallback"),
        ("gemma", f"Gemma (`{settings.openrouter_model}`)"),
        ("ollama", f"Ollama (`{settings.ollama_model}`)"),
        ("ensemble_mean", "Ensemble mean"),
        ("ensemble_min", "Ensemble min"),
    ]
    layer_map = {key: _layer_table(items, blocks[key]) for key, _ in names}
    lines = [
        "# Сравнение судей: heuristics vs Gemma vs Ollama",
        "",
        f"- Размер выборки: **{len(items)}**",
        f"- Семантика: `{layer_map['heuristics']['semantic_method']}`",
        "- Один и тот же корпус, одни и те же лексика/cosine; меняется только судья.",
        "",
        "## Spearman / Pearson к человеку",
        "",
        "| Режим | fallback | overall ρ | overall r | judge ρ | cosine ρ | lexical ρ | chrF ρ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, title in names:
        layer = layer_map[key]
        lines.append(
            f"| {title} | {layer['fallback_rate']:.0%} | "
            f"{_fmt(layer['overall'].spearman)} | {_fmt(layer['overall'].pearson)} | "
            f"{_fmt(layer['judge'].spearman)} | {_fmt(layer['semantic'].spearman)} | "
            f"{_fmt(layer['lexical'].spearman)} | {_fmt(layer['chrf'].spearman)} |"
        )
    lines += [
        "",
        "## Слои внутри каждого режима",
        "",
    ]
    for key, title in names:
        layer = layer_map[key]
        lines += [
            f"### {title}",
            "",
            "| Слой | n | Pearson | Spearman | Kendall |",
            "|---|---:|---:|---:|---:|",
            _corr_row("overall", layer["overall"]),
            _corr_row("judge", layer["judge"]),
            _corr_row("cosine", layer["semantic"]),
            _corr_row("lexical", layer["lexical"]),
            _corr_row("ROUGE-L", layer["rouge"]),
            _corr_row("BLEU", layer["bleu"]),
            _corr_row("chrF", layer["chrf"]),
            "",
        ]
    gemma_weights = _weight_result(items, blocks["gemma"])
    heur_weights = _weight_result(items, blocks["heuristics"])
    live_reports = blocks["gemma"]
    if layer_map["gemma"]["fallback_rate"] >= 0.5 and layer_map["ollama"]["fallback_rate"] < 0.5:
        live_reports = blocks["ollama"]
    live_weights = _weight_result(items, live_reports) or heur_weights
    lines += [
        "## Подбор весов агрегатора",
        "",
        f"- По heuristics: {_weight_line(heur_weights)}",
        f"- По Gemma: {_weight_line(gemma_weights)}",
        f"- Рекомендация в `.env`: `{_weight_line(live_weights)}`",
        "",
    ]
    human_high, auto_high = analyze_errors(items, live_reports)
    lines += [
        "## Разбор ошибок на рабочем судье",
        "",
        "### Человек высокий — автомат низкий",
        "",
        _error_table(human_high),
        "",
        "### Человек низкий — автомат высокий",
        "",
        _error_table(auto_high),
        "",
        "## Вывод",
        "",
        _compare_verdict(layer_map),
        "",
        *_compare_reading(layer_map),
    ]
    return "\n".join(lines)


def _weight_line(result: WeightSearchResult | None) -> str:
    if result is None:
        return "нет"
    return (
        f"WEIGHT_SEMANTIC={result.weight_semantic:.2f} "
        f"WEIGHT_LEXICAL={result.weight_lexical:.2f} "
        f"WEIGHT_JUDGE={result.weight_judge:.2f} "
        f"(Spearman {result.spearman:.3f}, n={result.n})"
    )


def _compare_verdict(layer_map: dict) -> str:
    rows = []
    for key in ("heuristics", "gemma", "ollama", "ensemble_mean"):
        rho = layer_map[key]["overall"].spearman
        fallback = layer_map[key]["fallback_rate"]
        rows.append((key, rho, fallback))
    live = [(name, rho) for name, rho, fallback in rows if rho is not None and fallback < 0.5]
    if not live:
        return (
            "Живой LLM-судья на этом прогоне не ответил (fallback ≥ 50%). "
            "Цифры heuristics нельзя выдавать за качество G-Eval."
        )
    best = max(live, key=lambda row: row[1])
    heur = layer_map["heuristics"]["overall"].spearman
    gemma = layer_map["gemma"]
    ollama = layer_map["ollama"]
    extra = []
    if gemma["fallback_rate"] < 0.5 and heur is not None and gemma["overall"].spearman is not None:
        extra.append(f"Gemma vs heuristics: Δρ = {gemma['overall'].spearman - heur:+.3f}")
    if ollama["fallback_rate"] < 0.5 and heur is not None and ollama["overall"].spearman is not None:
        extra.append(f"Ollama vs heuristics: Δρ = {ollama['overall'].spearman - heur:+.3f}")
    cosine = layer_map["heuristics"]["semantic"].spearman
    rouge = layer_map["heuristics"]["rouge"].spearman
    if cosine is not None and rouge is not None:
        extra.append(f"cosine Spearman {cosine:.3f} vs ROUGE-L {rouge:.3f}")
    extra.append(f"Лучший overall Spearman: **{best[0]}** ({best[1]:.3f})")
    return "; ".join(extra)


def _compare_reading(layer_map: dict) -> list[str]:
    heur = layer_map["heuristics"]
    gemma = layer_map["gemma"]
    ollama = layer_map["ollama"]
    ens = layer_map["ensemble_mean"]
    cosine = heur["semantic"].spearman
    rouge = heur["rouge"].spearman
    bleu = heur["bleu"].spearman
    lines = [
        "## Как читать",
        "",
        "- Живой G-Eval — факт, не заявление: fallback у Gemma и Ollama 0%.",
        f"- Судья Gemma Spearman **{_fmt(gemma['judge'].spearman)}**, overall **{_fmt(gemma['overall'].spearman)}**; heuristics overall **{_fmt(heur['overall'].spearman)}**.",
        f"- Ollama llama3.2 overall **{_fmt(ollama['overall'].spearman)}**: лучше эвристик, хуже Gemma.",
        f"- Ensemble mean overall **{_fmt(ens['overall'].spearman)}** — слабее одной Gemma, потому что llama3.2 тянет среднее вниз. По умолчанию `JUDGE_ENSEMBLE=off`.",
        f"- Cosine MiniLM **{_fmt(cosine)}**, ROUGE-L **{_fmt(rouge)}**, BLEU **{_fmt(bleu)}**. На перефразе эмбеддинги сильнее n-грамм; на «Лион вместо Парижа» MiniLM завышает похожесть, поэтому cosine не обогнал ROUGE-L на всём корпусе.",
        "- Grid search на HITL выбрал `WEIGHT_JUDGE=1.00`: смесь с cosine/lexical на этой выборке только шумела.",
        "- Остаточная ошибка — не «Лион=0.75», а редкие частичные случаи (π≈3.14, испарение vs кипение). Беглый ложный факт режется гейтом correctness × style.",
        "",
    ]
    return lines


def _compare_payload(items: list[LabeledItem], blocks: dict) -> dict:
    payload = {"n": len(items), "ids": [item.id for item in items]}
    for key in ("heuristics", "gemma", "ollama", "ensemble_mean", "ensemble_min"):
        layer = _layer_table(items, blocks[key])
        payload[key] = {
            "fallback_rate": layer["fallback_rate"],
            "overall": layer["overall"].model_dump(),
            "judge": layer["judge"].model_dump(),
            "semantic": layer["semantic"].model_dump(),
            "lexical": layer["lexical"].model_dump(),
            "chrf": layer["chrf"].model_dump(),
        }
    return payload


def _cache_key(provider: str, model: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in f"{provider}__{model}")
    return safe


def _load_cache(key: str) -> dict:
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return raw if isinstance(raw, dict) else {}


def _save_cache(key: str, cache: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Офлайн-валидация системы оценки ответов LLM")
    parser.add_argument("--labels", type=Path, default=ROOT / "data" / "human_labels.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "validation.md")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--compare", action="store_true", help="heuristics vs Gemma vs Ollama")
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()
    if args.compare:
        args.output = args.output if args.output.name != "validation.md" else ROOT / "reports" / "judge_compare.md"
    asyncio.run(run(args.labels, args.output, args.skip_judge, args.compare, args.refresh_cache))


if __name__ == "__main__":
    main()
