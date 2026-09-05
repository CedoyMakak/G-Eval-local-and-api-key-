from typing import Literal, Optional

from pydantic import BaseModel, Field


class EvaluateRequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    reference: Optional[str] = None
    context: Optional[str] = None
    skip_judge: bool = False


class BatchEvaluateRequest(BaseModel):
    items: list[EvaluateRequest] = Field(..., min_length=1)
    skip_judge: bool = False


class PairwiseRequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer_a: str = Field(..., min_length=1)
    answer_b: str = Field(..., min_length=1)
    reference: Optional[str] = None
    context: Optional[str] = None


class Dimensions(BaseModel):
    correctness: float
    relevance: float
    completeness: float
    coherence: float
    groundedness: Optional[float] = None


class LexicalMetrics(BaseModel):
    rouge_l: Optional[float] = None
    bleu: Optional[float] = None


class SemanticMetrics(BaseModel):
    cosine: Optional[float] = None
    method: Optional[str] = None


class JudgeResult(BaseModel):
    provider: str
    model: str
    rationale: str = ""
    raw_scores: Optional[Dimensions] = None
    used_logprobs: bool = False
    fallback: bool = False


class QualityFlags(BaseModel):
    no_reference: bool
    no_context: bool
    high_disagreement: bool
    empty_or_too_short: bool
    judge_fallback: bool = False


class QualityReport(BaseModel):
    overall: float
    confidence: float
    dimensions: Dimensions
    lexical: LexicalMetrics
    semantic: SemanticMetrics
    judge: JudgeResult
    flags: QualityFlags
    heuristics: dict[str, float] = Field(default_factory=dict)


class BatchEvaluateResponse(BaseModel):
    results: list[QualityReport]


class PairwiseJudgment(BaseModel):
    winner: Literal["A", "B", "tie"]
    rationale: str = ""
    order: str


class PairwiseResponse(BaseModel):
    first_pass: PairwiseJudgment
    swapped_pass: PairwiseJudgment
    consistent: bool
    position_bias_detected: bool
    preferred: Literal["A", "B", "tie", "inconsistent"]


class LabeledItem(BaseModel):
    id: str
    question: str
    answer: str
    human_score: float = Field(..., ge=0.0, le=1.0)
    reference: Optional[str] = None
    context: Optional[str] = None


class ValidateRequest(BaseModel):
    items: list[LabeledItem] = Field(..., min_length=2)
    skip_judge: bool = False


class CorrelationReport(BaseModel):
    n: int
    pearson: Optional[float] = None
    spearman: Optional[float] = None
    kendall: Optional[float] = None


class BiasReport(BaseModel):
    verbosity_pearson: Optional[float] = None
    verbosity_spearman: Optional[float] = None
    mean_disagreement: Optional[float] = None
    high_disagreement_rate: Optional[float] = None


class ValidateResponse(BaseModel):
    overall_vs_human: CorrelationReport
    judge_vs_human: CorrelationReport
    semantic_vs_human: CorrelationReport
    lexical_vs_human: CorrelationReport
    biases: BiasReport
    predictions: list[QualityReport]


class HealthResponse(BaseModel):
    status: str
    judge_provider: str
    judge_model: str
    embedding_model: str
