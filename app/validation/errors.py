from __future__ import annotations

from app.evaluators.common import lexical_mean
from app.evaluators.lexical import tokenize
from app.schemas import ErrorCase, LabeledItem, QualityReport


def analyze_errors(
    items: list[LabeledItem],
    reports: list[QualityReport],
    *,
    top_n: int = 8,
    gap: float = 0.25,
) -> tuple[list[ErrorCase], list[ErrorCase]]:
    human_high: list[ErrorCase] = []
    auto_high: list[ErrorCase] = []
    for item, report in zip(items, reports):
        delta = report.overall - item.human_score
        case = ErrorCase(
            id=item.id,
            case_type=item.case_type,
            question=item.question,
            answer=item.answer[:180],
            human=item.human_score,
            auto=report.overall,
            delta=round(delta, 4),
            reasons=_reasons(item, report, delta),
        )
        if delta <= -gap:
            human_high.append(case)
        elif delta >= gap:
            auto_high.append(case)
    human_high.sort(key=lambda row: row.delta)
    auto_high.sort(key=lambda row: row.delta, reverse=True)
    return human_high[:top_n], auto_high[:top_n]


def _reasons(item: LabeledItem, report: QualityReport, delta: float) -> list[str]:
    reasons: list[str] = []
    ref_len = len(tokenize(item.reference or ""))
    ans_len = int(report.heuristics.get("length_words", 0))
    if report.flags.judge_fallback:
        reasons.append("fallback")
    if report.flags.high_disagreement:
        reasons.append("disagreement")
    if report.flags.position_bias_detected:
        reasons.append("position-bias")
    if ref_len and ref_len <= 3:
        reasons.append("short-reference")
    if ref_len and ans_len >= max(12, 4 * ref_len):
        reasons.append("verbosity")
    cosine = report.semantic.cosine
    lexical = lexical_mean(report.lexical)
    if cosine is not None and lexical is not None and cosine - lexical >= 0.25 and delta < 0:
        reasons.append("paraphrase-under-lexical")
    if cosine is not None and lexical is not None and lexical - cosine >= 0.25 and delta > 0:
        reasons.append("lexical-overlap")
    if item.case_type:
        reasons.append(item.case_type)
    if report.factcheck and report.factcheck.unsupported_numbers:
        reasons.append("unsupported-numbers")
    if not reasons:
        reasons.append("residual")
    return reasons
