# Отчёт валидации системы оценки ответов LLM

- Размер выборки: **59**
- Провайдер судьи: `openrouter` / `google/gemma-4-31b-it`
- skip_judge: `False`
- Семантика: `sentence-transformers`
- Доля heuristic-fallback: **0.0%**

## Корреляция с ручными метками

| Слой | n | Pearson | Spearman | Kendall |
|---|---:|---:|---:|---:|
| overall | 59 | 0.977 | 0.913 | 0.796 |
| LLM-as-a-Judge | 59 | 0.978 | 0.926 | 0.820 |
| semantic cosine | 59 | 0.489 | 0.487 | 0.348 |
| lexical (ROUGE/BLEU/chrF) | 59 | 0.406 | 0.540 | 0.399 |
| ROUGE-L | 59 | 0.476 | 0.556 | 0.416 |
| BLEU | 59 | 0.218 | 0.300 | 0.217 |
| chrF | 59 | 0.405 | 0.430 | 0.321 |
| BERTScore | 0 | — | — | — |

## Корреляция по критериям HITL

| Критерий | n | Pearson | Spearman | Kendall |
|---|---:|---:|---:|---:|
| correctness | 59 | 0.968 | 0.897 | 0.786 |
| relevance | 59 | 0.847 | 0.619 | 0.561 |
| completeness | 59 | 0.782 | 0.799 | 0.686 |
| coherence | 59 | 0.681 | 0.553 | 0.485 |
| groundedness | 4 | 0.945 | 0.775 | 0.707 |

## Веса агрегатора по HITL

overall = 0.00·cosine + 0.00·lexical + 1.00·judge (Spearman 0.926)

## Искажения

- Verbosity bias (Pearson длина vs overall): **0.196**
- Verbosity bias (Spearman): **0.198**
- Среднее расхождение semantic vs judge: **0.207**
- Доля высокой disagreement (>= 0.25): **0.356**

## Разбор ошибок: человек высокий, автомат низкий

| id | тип | human | auto | Δ | причины |
|---|---|---:|---:|---:|---|
| `q16-approx` | partial | 0.45 | 0.19 | -0.26 | disagreement, paraphrase-under-lexical, partial |
|  | _Чему равно число π с точностью до двух знаков?_ |  |  |  | Примерно 3.1 |
| `q31-contradiction` | contradiction | 0.25 | 0.00 | -0.25 | disagreement, short-reference, paraphrase-under-lexical, contradiction |
|  | _Что такое столица Франции?_ |  |  |  | Столица Франции — Париж. Впрочем, столица всё-таки Лион. |

## Разбор ошибок: человек низкий, автомат высокий

| id | тип | human | auto | Δ | причины |
|---|---|---:|---:|---:|---|
| `q35-water-evaporate` | partial | 0.55 | 0.83 | +0.28 | disagreement, partial |
|  | _Что произойдёт с водой при 100 °C на уровне моря?_ |  |  |  | Вода начнёт испаряться. |

## Как читать

HITL не участвует в online-оценке. Ручные метки проверяют ранжирование автомата. Таблица ошибок важнее ещё одного графика: по причинам (`verbosity`, `short-reference`, `fallback`, `disagreement`) видно, что чинить.
