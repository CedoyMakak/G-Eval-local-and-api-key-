# Система автоматизированной оценки качества ответов LLM

Учебный сервис для практики: оценивает ответ языковой модели по вопросу и возвращает структурированный отчёт о качестве.

Онлайн-контур считает дешёвые метрики и вызывает LLM-as-a-Judge. Ручные метки не участвуют в каждом запросе: они нужны только для офлайн-валидации (корреляция и искажения).

## Архитектура

```
Вход: question + answer + optional reference + optional context
        │
        ├── лексика: ROUGE-L, BLEU          (нужен эталон)
        ├── семантика: cosine эмбеддингов   (нужен эталон)
        └── LLM-as-a-Judge: G-Eval          (OpenRouter / OpenAI / Ollama)
                    │
                    ▼
              агрегатор → Quality Report
                    │
                    ▼
     офлайн: 40 ручных меток → корреляция + bias
```

Ограничения, которые система учитывает явно:

- без `reference` ROUGE/BLEU/cosine не считаются;
- ROUGE-L считается по Unicode-токенам: стандартный `rouge-score` выкидывает кириллицу;
- один `overall` недостаточен, поэтому есть критерии correctness / relevance / completeness / coherence (+ groundedness при `context`);
- HITL — контур калибровки, а не третий скорер в проде.

## Быстрый старт

```powershell
cd "c:\Users\User\Desktop\Практика_Тюхменев_БФБО-05-24"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

Документация API: http://127.0.0.1:8000/docs

Если OpenAI-ключ или Ollama недоступны, судья автоматически падает в эвристический fallback. Для отчёта практики этого достаточно; для «настоящего» G-Eval нужен один из провайдеров.

Семантика по умолчанию считает cosine на мешке слов. Для более сильного сигнала:

```powershell
pip install sentence-transformers
```

Модель `paraphrase-multilingual-MiniLM-L12-v2` подхватывается сама.

## Провайдер судьи

В `.env`:

```
JUDGE_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
```

или

```
JUDGE_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=google/gemma-4-31b-it
```

или OpenAI / Ollama:

```
JUDGE_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

Logprobs используются только если их отдаёт провайдер. OpenRouter/Gemma обычно возвращают дискретную оценку 1–5 в JSON.

## Эндпоинты

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/health` | провайдер и модель судьи |
| POST | `/evaluate` | оценка одного ответа |
| POST | `/evaluate/batch` | пакетная оценка |
| POST | `/evaluate/pairwise` | сравнение A/B со свапом позиций |
| POST | `/validate` | корреляция с ручными метками + bias |

Пример запроса:

```json
{
  "question": "Что такое столица Франции?",
  "answer": "Столица Франции — Париж.",
  "reference": "Париж"
}
```

Поле `skip_judge: true` отключает вызов LLM и считает только лексику, семантику и эвристики.

## Офлайн-валидация

В `data/human_labels.json` — 40 размеченных пар (хорошие, неполные, многословные и неверные ответы).

```powershell
python scripts/run_validation.py --skip-judge
```

Отчёт появится в `reports/validation.md`. Без `--skip-judge` скрипт пойдёт в выбранный LLM-провайдер.

## Структура

```
app/
  api/routes.py          эндпоинты
  evaluators/            лексика, семантика, G-Eval, агрегатор
  providers/             OpenAI / Ollama
  validation/            корреляции и bias
data/
  sample_eval.json
  human_labels.json
scripts/run_validation.py
```

## Что можно написать в пояснительной

1. Почему BLEU/ROUGE оставлены как baseline, а основной reference-based сигнал — семантический cosine.
2. Почему G-Eval идёт с рубрикой и JSON, а logprobs — опциональны.
3. Почему 30–50 ручных меток стоят отдельно от online-пайплайна.
4. Какие искажения измеряются: verbosity (длина vs оценка), position bias (свап в pairwise), disagreement между слоями.
