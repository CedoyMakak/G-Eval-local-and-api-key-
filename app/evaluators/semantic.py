from __future__ import annotations

import logging
import re
from functools import lru_cache

import numpy as np

from app.schemas import SemanticMetrics

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)

_model = None
_model_name: str | None = None
_model_failed = False


def compute_semantic(
    answer: str,
    reference: str | None,
    model_name: str,
) -> SemanticMetrics:
    if not reference or not reference.strip():
        return SemanticMetrics(cosine=None, method=None)

    embedding = _try_embedding_cosine(answer, reference, model_name)
    if embedding is not None:
        return SemanticMetrics(cosine=embedding, method="sentence-transformers")

    return SemanticMetrics(
        cosine=_bow_cosine(answer, reference),
        method="bow-cosine",
    )


def _try_embedding_cosine(answer: str, reference: str, model_name: str) -> float | None:
    global _model, _model_name, _model_failed

    if _model_failed and _model_name == model_name:
        return None

    try:
        model = _load_embedding_model(model_name)
        vectors = model.encode(
            [answer, reference],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        cosine = float(np.clip(np.dot(vectors[0], vectors[1]), -1.0, 1.0))
        return max(0.0, cosine)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Не удалось посчитать эмбеддинги (%s), fallback на BoW: %s", model_name, exc)
        _model_failed = True
        _model_name = model_name
        return None


@lru_cache(maxsize=1)
def _load_embedding_model(model_name: str):
    global _model, _model_name
    from sentence_transformers import SentenceTransformer

    _model = SentenceTransformer(model_name)
    _model_name = model_name
    return _model


def _bow_cosine(answer: str, reference: str) -> float:
    tokens_a = _TOKEN_RE.findall(answer.lower())
    tokens_b = _TOKEN_RE.findall(reference.lower())
    if not tokens_a or not tokens_b:
        return 0.0

    vocab = sorted(set(tokens_a) | set(tokens_b))
    index = {token: i for i, token in enumerate(vocab)}
    vec_a = np.zeros(len(vocab), dtype=float)
    vec_b = np.zeros(len(vocab), dtype=float)
    for token in tokens_a:
        vec_a[index[token]] += 1.0
    for token in tokens_b:
        vec_b[index[token]] += 1.0

    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return float(np.clip(np.dot(vec_a, vec_b) / denom, 0.0, 1.0))
