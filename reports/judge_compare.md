# Сравнение судей: heuristics vs Gemma vs Ollama

- Размер выборки: **59**
- Семантика: `sentence-transformers`
- Один и тот же корпус, одни и те же лексика/cosine; меняется только судья.

## Spearman / Pearson к человеку

| Режим | fallback | overall ρ | overall r | judge ρ | cosine ρ | lexical ρ | chrF ρ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Heuristics / fallback | 100% | 0.459 | 0.494 | 0.461 | 0.487 | 0.540 | 0.430 |
| Gemma (`google/gemma-4-31b-it`) | 0% | 0.913 | 0.977 | 0.926 | 0.487 | 0.540 | 0.430 |
| Ollama (`llama3.2`) | 0% | 0.659 | 0.698 | 0.660 | 0.487 | 0.540 | 0.430 |
| Ensemble mean | 0% | 0.868 | 0.929 | 0.869 | 0.487 | 0.540 | 0.430 |
| Ensemble min | 0% | 0.873 | 0.831 | 0.875 | 0.487 | 0.540 | 0.430 |

## Слои внутри каждого режима

### Heuristics / fallback

| Слой | n | Pearson | Spearman | Kendall |
|---|---:|---:|---:|---:|
| overall | 59 | 0.494 | 0.459 | 0.328 |
| judge | 59 | 0.490 | 0.461 | 0.332 |
| cosine | 59 | 0.489 | 0.487 | 0.348 |
| lexical | 59 | 0.406 | 0.540 | 0.399 |
| ROUGE-L | 59 | 0.476 | 0.556 | 0.416 |
| BLEU | 59 | 0.218 | 0.300 | 0.217 |
| chrF | 59 | 0.405 | 0.430 | 0.321 |

### Gemma (`google/gemma-4-31b-it`)

| Слой | n | Pearson | Spearman | Kendall |
|---|---:|---:|---:|---:|
| overall | 59 | 0.977 | 0.913 | 0.796 |
| judge | 59 | 0.978 | 0.926 | 0.820 |
| cosine | 59 | 0.489 | 0.487 | 0.348 |
| lexical | 59 | 0.406 | 0.540 | 0.399 |
| ROUGE-L | 59 | 0.476 | 0.556 | 0.416 |
| BLEU | 59 | 0.218 | 0.300 | 0.217 |
| chrF | 59 | 0.405 | 0.430 | 0.321 |

### Ollama (`llama3.2`)

| Слой | n | Pearson | Spearman | Kendall |
|---|---:|---:|---:|---:|
| overall | 59 | 0.698 | 0.659 | 0.494 |
| judge | 59 | 0.698 | 0.660 | 0.496 |
| cosine | 59 | 0.489 | 0.487 | 0.348 |
| lexical | 59 | 0.406 | 0.540 | 0.399 |
| ROUGE-L | 59 | 0.476 | 0.556 | 0.416 |
| BLEU | 59 | 0.218 | 0.300 | 0.217 |
| chrF | 59 | 0.405 | 0.430 | 0.321 |

### Ensemble mean

| Слой | n | Pearson | Spearman | Kendall |
|---|---:|---:|---:|---:|
| overall | 59 | 0.929 | 0.868 | 0.687 |
| judge | 59 | 0.930 | 0.869 | 0.693 |
| cosine | 59 | 0.489 | 0.487 | 0.348 |
| lexical | 59 | 0.406 | 0.540 | 0.399 |
| ROUGE-L | 59 | 0.476 | 0.556 | 0.416 |
| BLEU | 59 | 0.218 | 0.300 | 0.217 |
| chrF | 59 | 0.405 | 0.430 | 0.321 |

### Ensemble min

| Слой | n | Pearson | Spearman | Kendall |
|---|---:|---:|---:|---:|
| overall | 59 | 0.831 | 0.873 | 0.720 |
| judge | 59 | 0.832 | 0.875 | 0.723 |
| cosine | 59 | 0.489 | 0.487 | 0.348 |
| lexical | 59 | 0.406 | 0.540 | 0.399 |
| ROUGE-L | 59 | 0.476 | 0.556 | 0.416 |
| BLEU | 59 | 0.218 | 0.300 | 0.217 |
| chrF | 59 | 0.405 | 0.430 | 0.321 |

## Подбор весов агрегатора

- По heuristics: WEIGHT_SEMANTIC=0.20 WEIGHT_LEXICAL=0.75 WEIGHT_JUDGE=0.05 (Spearman 0.561, n=59)
- По Gemma: WEIGHT_SEMANTIC=0.00 WEIGHT_LEXICAL=0.00 WEIGHT_JUDGE=1.00 (Spearman 0.926, n=59)
- Рекомендация в `.env`: `WEIGHT_SEMANTIC=0.00 WEIGHT_LEXICAL=0.00 WEIGHT_JUDGE=1.00 (Spearman 0.926, n=59)`

## Разбор ошибок на рабочем судье

### Человек высокий — автомат низкий

| id | тип | human | auto | Δ | причины |
|---|---|---:|---:|---:|---|
| `q16-approx` | partial | 0.45 | 0.19 | -0.26 | disagreement, paraphrase-under-lexical, partial |
|  | _Чему равно число π с точностью до двух знаков?_ |  |  |  | Примерно 3.1 |
| `q31-contradiction` | contradiction | 0.25 | 0.00 | -0.25 | disagreement, short-reference, paraphrase-under-lexical, contradiction |
|  | _Что такое столица Франции?_ |  |  |  | Столица Франции — Париж. Впрочем, столица всё-таки Лион. |

### Человек низкий — автомат высокий

| id | тип | human | auto | Δ | причины |
|---|---|---:|---:|---:|---|
| `q35-water-evaporate` | partial | 0.55 | 0.83 | +0.28 | disagreement, partial |
|  | _Что произойдёт с водой при 100 °C на уровне моря?_ |  |  |  | Вода начнёт испаряться. |

## Вывод

Gemma vs heuristics: Δρ = +0.454; Ollama vs heuristics: Δρ = +0.200; cosine Spearman 0.487 vs ROUGE-L 0.556; Лучший overall Spearman: **gemma** (0.913)

## Как читать

- Живой G-Eval — факт, не заявление: fallback у Gemma и Ollama 0%.
- Судья Gemma Spearman **0.926**, overall **0.913**; heuristics overall **0.459**.
- Ollama llama3.2 overall **0.659**: лучше эвристик, хуже Gemma.
- Ensemble mean overall **0.868** — слабее одной Gemma, потому что llama3.2 тянет среднее вниз. По умолчанию `JUDGE_ENSEMBLE=off`.
- Cosine MiniLM **0.487**, ROUGE-L **0.556**, BLEU **0.300**. На перефразе эмбеддинги сильнее n-грамм; на «Лион вместо Парижа» MiniLM завышает похожесть, поэтому cosine не обогнал ROUGE-L на всём корпусе.
- Grid search на HITL выбрал `WEIGHT_JUDGE=1.00`: смесь с cosine/lexical на этой выборке только шумела.
- Остаточная ошибка — не «Лион=0.75», а редкие частичные случаи (π≈3.14, испарение vs кипение). Беглый ложный факт режется гейтом correctness × style.
