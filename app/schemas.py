from typing import Literal, Optional

from pydantic import BaseModel, Field


class EvaluateRequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    reference: Optional[str] = None
    context: Optional[str] = None
    skip_judge: bool = False
    id: Optional[str] = None


class BatchEvaluateRequest(BaseModel):
    items: list[EvaluateRequest] = Field(..., min_length=1, max_length=100)
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


class HumanDimensions(BaseModel):
    correctness: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    relevance: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    completeness: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    coherence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    groundedness: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class LexicalMetrics(BaseModel):
    rouge_l: Optional[float] = None
    bleu: Optional[float] = None
    chrf: Optional[float] = None


class SemanticMetrics(BaseModel):
    cosine: Optional[float] = None
    method: Optional[str] = None
    bertscore: Optional[float] = None


class FactCheckMetrics(BaseModel):
    score: Optional[float] = None
    context_overlap: Optional[float] = None
    unsupported_numbers: int = 0
    method: Optional[str] = None


class JudgeMember(BaseModel):
    provider: str
    model: str
    fallback: bool = False
    rationale: str = ""
    score: Optional[float] = None


class JudgeResult(BaseModel):
    provider: str
    model: str
    rationale: str = ""
    raw_scores: Optional[Dimensions] = None
    used_logprobs: bool = False
    fallback: bool = False
    ensemble: Optional[str] = None
    members: list[JudgeMember] = Field(default_factory=list)
    position_bias_detected: bool = False
    position_delta: Optional[float] = None


class QualityFlags(BaseModel):
    no_reference: bool
    no_context: bool
    high_disagreement: bool
    empty_or_too_short: bool
    judge_fallback: bool = False
    position_bias_detected: bool = False
    ensemble_disagreement: bool = False
    correctness_capped: bool = False


class QualityReport(BaseModel):
    overall: float
    confidence: float
    dimensions: Dimensions
    lexical: LexicalMetrics
    semantic: SemanticMetrics
    judge: JudgeResult
    flags: QualityFlags
    heuristics: dict[str, float] = Field(default_factory=dict)
    factcheck: Optional[FactCheckMetrics] = None


class BatchEvaluateResponse(BaseModel):
    results: list[QualityReport]
    n: int = 0
    mean_overall: Optional[float] = None


class ImportDocumentsRequest(BaseModel):
    text: str = Field(..., min_length=1)
    filename: str = ""


class ImportDocumentsResponse(BaseModel):
    items: list[EvaluateRequest]
    n: int


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
    stability: Optional[float] = None


class LabeledItem(BaseModel):
    id: str = ""
    question: str
    answer: str
    human_score: float = Field(..., ge=0.0, le=1.0)
    reference: Optional[str] = None
    context: Optional[str] = None
    human_dimensions: Optional[HumanDimensions] = None
    case_type: Optional[str] = None
    notes: Optional[str] = None


class SaveLabelResponse(BaseModel):
    item: LabeledItem
    total: int


class LabelsInfoResponse(BaseModel):
    total: int
    path: str


class LabelsListResponse(BaseModel):
    total: int
    items: list[LabeledItem]


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


class ErrorCase(BaseModel):
    id: str
    case_type: Optional[str] = None
    question: str
    answer: str
    human: float
    auto: float
    delta: float
    reasons: list[str] = Field(default_factory=list)


class WeightSearchResult(BaseModel):
    weight_semantic: float
    weight_lexical: float
    weight_judge: float
    spearman: float
    pearson: Optional[float] = None
    n: int


class ValidateResponse(BaseModel):
    overall_vs_human: CorrelationReport
    judge_vs_human: CorrelationReport
    semantic_vs_human: CorrelationReport
    lexical_vs_human: CorrelationReport
    biases: BiasReport
    predictions: list[QualityReport]
    human_high_auto_low: list[ErrorCase] = Field(default_factory=list)
    human_low_auto_high: list[ErrorCase] = Field(default_factory=list)
    recommended_weights: Optional[WeightSearchResult] = None
    semantic_method: Optional[str] = None
    fallback_rate: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    judge_provider: str
    judge_model: str
    embedding_model: str
    semantic_method: Optional[str] = None
    ensemble: Optional[str] = None


class JudgeSettingsView(BaseModel):
    provider: str
    model: str
    openai_model: str
    openai_base_url: str
    openai_key_set: bool
    openrouter_model: str
    openrouter_base_url: str
    openrouter_key_set: bool
    ollama_base_url: str
    ollama_model: str


class JudgeSettingsUpdate(BaseModel):
    provider: Literal["openrouter", "openai", "ollama"]
    openai_model: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_api_key: Optional[str] = None
    openrouter_model: Optional[str] = None
    openrouter_base_url: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    ollama_base_url: Optional[str] = None
    ollama_model: Optional[str] = None


class JudgeTestResponse(BaseModel):
    ok: bool
    provider: str
    model: str
    message: str
    sample: str = ""
