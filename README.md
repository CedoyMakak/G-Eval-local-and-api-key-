# Система автоматизированной оценки качества ответов LLM

Учебный сервис для практики: оценивает ответ языковой модели по вопросу и возвращает структурированный отчёт о качестве.

Онлайн-контур считает дешёвые метрики и вызывает LLM-as-a-Judge. Ручные метки не участвуют в каждом запросе: они нужны только для офлайн-валидации (корреляция, веса агрегатора, разбор ошибок).

## Архитектура

```
Вход: question + answer + optional reference + optional context
        │
        ├── лексика: ROUGE-L, BLEU, chrF     (нужен эталон)
        ├── семантика: cosine MiniLM         (нужен эталон)
        ├── фактчек: overlap с context
        └── LLM-as-a-Judge: G-Eval           (один судья или ensemble)
                    │
                    ▼
              агрегатор → Quality Report
                    │
                    ▼
     офлайн HITL (~60 меток, критерии отдельно) → Spearman / ошибки / веса
```

Ограничения, которые система учитывает явно:

- без `reference` ROUGE/BLEU/chrF/cosine не считаются;
- ROUGE-L считается по Unicode-токенам: стандартный `rouge-score` выкидывает кириллицу;
- один `overall` недостаточен — есть критерии и отдельная HITL-разметка по ним;
- HITL — контур калибровки, а не третий скорер в проде;
- один LLM-судья нестабилен: поэтому есть ensemble (mean/min) и pairwise swap.

## Быстрый старт

```bash
python scripts/start.py
```

Если окружения ещё нет, скрипт сам создаст `.venv` и поставит зависимости, включая `sentence-transformers`. Порт выбирается автоматически (обычно `8001`). Остановка: `Ctrl+C`.

Документация API: http://127.0.0.1:8001/docs

Семантика берёт `paraphrase-multilingual-MiniLM-L12-v2`. Если веса лежат в `models/minilm/`, они читаются с диска и HuggingFace не нужен. Если модели нет — fallback на BoW-cosine, это видно в поле `semantic.method`.

## Провайдер судьи

Вкладка **Судья** переключает провайдер на лету: OpenRouter (Gemma и другие), OpenAI-совместимый API, Ollama.

Для устойчивости двух независимых G-Eval:

```
JUDGE_ENSEMBLE=mean
JUDGE_ENSEMBLE_PROVIDERS=openrouter,ollama
JUDGE_SWAP_CHECK=false
POSITION_BIAS_PENALTY=0.08
```

`mean` усредняет критерии, `min` берёт пессимистичную оценку. Если `JUDGE_SWAP_CHECK=true`, pointwise гоняется второй раз с переставленными блоками эталон/ответ: при большом Δ overall снижается.

Logprobs, если провайдер их отдаёт, калибруют шкалу 1–5 → 0–1 через ожидание по `top_logprobs`, а не через сжатие к 0.6.

## Агрегатор

При эталоне:

```
overall = w_sem * cosine + w_lex * lexical + w_judge * judge
```

Веса в `.env`. Их подбирает `scripts/run_validation.py` grid search по Spearman к HITL, а не «на глаз». На текущем корпусе это `0.00 / 0.00 / 1.00`: смесь с cosine и лексикой снижала ρ относительно чистого судьи.

## Офлайн-валидация

В `data/human_labels.json` — около 60 пар: полные эталоны-предложения, метки по критериям, спорные случаи (многословие, частично верно, галлюцинация, перефраз, лексическая ловушка).

```powershell
python scripts/run_validation.py --skip-judge
python scripts/run_validation.py --compare
```

`--compare` гоняет **тот же корпус** через heuristics, Gemma (OpenRouter) и Ollama, пишет `reports/judge_compare.md`. На 59 парах (fallback 0%): overall Spearman heuristics **0.459**, Ollama **0.659**, Gemma **0.913** (Pearson 0.977). Гейт `correctness × style` не даёт беглости завышать ложный факт. Grid search выставил `WEIGHT_JUDGE=1.00`. Cosine MiniLM (0.487) не обогнал ROUGE-L (0.556) на всём корпусе: near-miss вроде «Лион» завышается. Ответы судей кэшируются в `reports/judge_cache/`.

BERTScore считается только если установлен пакет `bert-score`; chrF считается всегда (sacrebleu).

## Эндпоинты

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/health` | провайдер, модель, метод семантики |
| GET | `/settings` | текущий судья, ключи замаскированы |
| PUT | `/settings` | сменить провайдера / модель / ключ |
| POST | `/settings/test` | короткий пробный запрос к судье |
| GET | `/samples` | демо-пары из `data/sample_eval.json` |
| POST | `/evaluate/import` | разобрать JSON/JSONL/CSV в пары |
| POST | `/evaluate` | оценка одного ответа |
| POST | `/evaluate/batch` | пакет до 100 пар |
| POST | `/evaluate/pairwise` | сравнение A/B со свапом позиций |
| POST | `/validate` | корреляция, веса, таблица ошибок |

Пример запроса:

```json
{
  "question": "Что такое столица Франции?",
  "answer": "Столица Франции — Париж.",
  "reference": "Столица Франции — Париж."
}
```

`skip_judge: true` отключает LLM и считает лексику, семантику и эвристики.

## Что писать в пояснительной

1. Почему BLEU/ROUGE — baseline, а cosine MiniLM должен быть ближе к человеку на перефразе.
2. Почему «качество судьи» проверяется сравнением heuristics vs Gemma vs Ollama на одном корпусе, а не декларацией.
3. Почему веса агрегатора взяты из grid search по Spearman, а не из `.env` «потому что так принято».
4. Почему корпус усилен эталонами-предложениями и спорными случаями: иначе метрика упирается в шум разметки.
5. Как ensemble и position-swap бьют по нестабильности LLM-as-a-Judge.
6. Почему разбор ошибок сильнее нового графика.
