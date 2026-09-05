from __future__ import annotations

import json
import logging
import math
import re
from typing import Literal

from app.config import Settings
from app.evaluators.heuristics import compute_heuristics
from app.providers.base import CompletionResult, JudgeProvider
from app.schemas import Dimensions, JudgeResult, PairwiseJudgment, SemanticMetrics

logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


POINTWISE_PROMPT = """Ты — независимый судья качества ответа языковой модели.
Оцени ответ по рубрике от 1 до 5. Сначала кратко рассуждай, затем верни ТОЛЬКО JSON.

Критерии:
- correctness: фактическая корректность относительно вопроса{ref_hint}
- relevance: отвечает ли текст именно на заданный вопрос
- completeness: достаточно ли полного ответа, нет ли существенных пробелов
- coherence: связность, ясность и языковое качество
{ground_hint}

Шкала:
1 — неприемлемо
2 — слабо
3 — приемлемо с пробелами
4 — хорошо
5 — отлично

Вопрос:
{question}

{context_block}{reference_block}Ответ кандидата:
{answer}

Формат ответа:
{{
  "rationale": "краткое обоснование на русском",
  "correctness": <1-5>,
  "relevance": <1-5>,
  "completeness": <1-5>,
  "coherence": <1-5>{ground_json}
}}
"""


PAIRWISE_PROMPT = """Ты сравниваешь два ответа на один вопрос. Выбери лучший.
Сначала кратко рассуждай, затем верни ТОЛЬКО JSON.

Вопрос:
{question}

{context_block}{reference_block}Ответ A:
{answer_a}

Ответ B:
{answer_b}

Критерии: корректность, релевантность, полнота, связность. Не предпочитай ответ только из-за длины или позиции.

Формат:
{{
  "rationale": "краткое обоснование на русском",
  "winner": "A" | "B" | "tie"
}}
"""


async def judge_pointwise(
    *,
    provider: JudgeProvider | None,
    settings: Settings,
    question: str,
    answer: str,
    reference: str | None,
    context: str | None,
    semantic: SemanticMetrics,
    skip_judge: bool = False,
) -> tuple[Dimensions, JudgeResult]:
    if skip_judge or provider is None:
        return _fallback_dimensions(answer, question, reference, semantic, context), JudgeResult(
            provider="heuristic",
            model="rule-fallback",
            rationale="LLM-судья пропущен, оценка собрана из эвристик.",
            used_logprobs=False,
            fallback=True,
        )

    prompt = _build_pointwise_prompt(question, answer, reference, context)
    try:
        completion = await provider.complete(prompt, temperature=settings.judge_temperature)
        dimensions = _parse_pointwise(completion)
        rationale = _extract_rationale(completion.text)
        if completion.logprobs:
            dimensions = _blend_with_logprobs(dimensions, completion.logprobs)
        return dimensions, JudgeResult(
            provider=completion.provider,
            model=completion.model,
            rationale=rationale,
            raw_scores=dimensions,
            used_logprobs=bool(completion.logprobs),
            fallback=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM-судья недоступен, используется эвристический fallback: %s", exc)
        dimensions = _fallback_dimensions(answer, question, reference, semantic, context)
        return dimensions, JudgeResult(
            provider=provider.name,
            model=provider.model,
            rationale=f"Провайдер недоступен ({exc}). Использован эвристический fallback.",
            raw_scores=dimensions,
            used_logprobs=False,
            fallback=True,
        )


async def judge_pairwise(
    *,
    provider: JudgeProvider,
    settings: Settings,
    question: str,
    answer_a: str,
    answer_b: str,
    reference: str | None,
    context: str | None,
    order: str,
) -> PairwiseJudgment:
    prompt = PAIRWISE_PROMPT.format(
        question=question.strip(),
        answer_a=answer_a.strip(),
        answer_b=answer_b.strip(),
        context_block=_optional_block("Контекст", context),
        reference_block=_optional_block("Эталон (не копируй слепо)", reference),
    )
    try:
        completion = await provider.complete(prompt, temperature=settings.judge_temperature)
        winner = _parse_winner(completion.text)
        return PairwiseJudgment(
            winner=winner,
            rationale=_extract_rationale(completion.text),
            order=order,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Pairwise-судья недоступен, fallback по длине/эвристике: %s", exc)
        winner = _fallback_pairwise(question, answer_a, answer_b, reference)
        return PairwiseJudgment(
            winner=winner,
            rationale=f"Провайдер недоступен ({exc}). Сравнение по эвристикам.",
            order=order,
        )


def _build_pointwise_prompt(
    question: str,
    answer: str,
    reference: str | None,
    context: str | None,
) -> str:
    has_ref = bool(reference and reference.strip())
    has_ctx = bool(context and context.strip())
    return POINTWISE_PROMPT.format(
        question=question.strip(),
        answer=answer.strip(),
        ref_hint=" и эталона" if has_ref else "",
        ground_hint="- groundedness: опирается ли ответ на приведённый контекст без выдумок" if has_ctx else "",
        ground_json=',\n  "groundedness": <1-5>' if has_ctx else "",
        context_block=_optional_block("Контекст", context),
        reference_block=_optional_block("Эталон (используй как ориентир, не наказывай за перефраз)", reference),
    )


def _optional_block(title: str, value: str | None) -> str:
    if not value or not value.strip():
        return ""
    return f"{title}:\n{value.strip()}\n\n"


def _parse_pointwise(completion: CompletionResult) -> Dimensions:
    data = _extract_json(completion.text)
    return Dimensions(
        correctness=_to_unit(data.get("correctness")),
        relevance=_to_unit(data.get("relevance")),
        completeness=_to_unit(data.get("completeness")),
        coherence=_to_unit(data.get("coherence")),
        groundedness=_to_unit(data["groundedness"]) if "groundedness" in data else None,
    )


def _blend_with_logprobs(dimensions: Dimensions, logprobs: list[float]) -> Dimensions:
    if not logprobs:
        return dimensions
    weights = [math.exp(lp) for lp in logprobs]
    mean_conf = sum(weights) / len(weights)
    # чуть сжимаем оценки к среднему при низкой уверенности токенов
    shrink = max(0.0, min(1.0, mean_conf))
    midpoint = 0.6

    def blend(value: float) -> float:
        return round(shrink * value + (1 - shrink) * midpoint, 4)

    return Dimensions(
        correctness=blend(dimensions.correctness),
        relevance=blend(dimensions.relevance),
        completeness=blend(dimensions.completeness),
        coherence=blend(dimensions.coherence),
        groundedness=blend(dimensions.groundedness) if dimensions.groundedness is not None else None,
    )


def _parse_winner(text: str) -> Literal["A", "B", "tie"]:
    data = _extract_json(text)
    winner = str(data.get("winner", "tie")).strip().upper()
    if winner in {"A", "B", "TIE"}:
        return "tie" if winner == "TIE" else winner  # type: ignore[return-value]
    lowered = text.lower()
    if "победитель" in lowered and "ответ a" in lowered:
        return "A"
    if "победитель" in lowered and "ответ b" in lowered:
        return "B"
    return "tie"


def _extract_json(text: str) -> dict:
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError("В ответе судьи нет JSON")
    raw = match.group(0)
    return json.loads(raw)


def _extract_rationale(text: str) -> str:
    try:
        data = _extract_json(text)
        rationale = str(data.get("rationale", "")).strip()
        if rationale:
            return rationale
    except Exception:  # noqa: BLE001
        pass
    return text.strip()[:500]


def _to_unit(value: object) -> float:
    score = float(value)
    if score > 5:
        score = 5.0
    if score < 1:
        # допускаем уже нормализованный 0-1
        if 0.0 <= score <= 1.0:
            return round(score, 4)
        score = 1.0
    return round((score - 1.0) / 4.0, 4)


def _fallback_dimensions(
    answer: str,
    question: str,
    reference: str | None,
    semantic: SemanticMetrics,
    context: str | None,
) -> Dimensions:
    heur = compute_heuristics(answer, question, reference)
    base = semantic.cosine if semantic.cosine is not None else heur["length_score"]
    relevance = min(1.0, 0.45 * heur["question_overlap"] + 0.55 * max(base, heur["length_score"]))
    completeness = max(heur["length_ratio_score"], 0.5 if heur["length_words"] >= 1 else 0.0)
    coherence = 0.55 if heur["length_words"] < 5 else 0.75
    correctness = base if semantic.cosine is not None else (0.5 * relevance + 0.5 * completeness)
    if semantic.cosine is not None and semantic.cosine >= 0.55:
        correctness = max(correctness, semantic.cosine)
        relevance = max(relevance, 0.6)
    groundedness = None
    if context:
        overlap = compute_heuristics(answer, context, None)["question_overlap"]
        groundedness = round(min(1.0, overlap + 0.2), 4)
    return Dimensions(
        correctness=round(correctness, 4),
        relevance=round(relevance, 4),
        completeness=round(completeness, 4),
        coherence=round(coherence, 4),
        groundedness=groundedness,
    )


def _fallback_pairwise(
    question: str,
    answer_a: str,
    answer_b: str,
    reference: str | None,
) -> Literal["A", "B", "tie"]:
    heur_a = compute_heuristics(answer_a, question, reference)
    heur_b = compute_heuristics(answer_b, question, reference)
    score_a = heur_a["question_overlap"] + 0.3 * heur_a["length_score"]
    score_b = heur_b["question_overlap"] + 0.3 * heur_b["length_score"]
    if abs(score_a - score_b) < 0.05:
        return "tie"
    return "A" if score_a > score_b else "B"
