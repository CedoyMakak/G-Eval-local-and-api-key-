# Полное описание проекта для разбора

Этот файл можно целиком отдать в Gemini. Задача собеседника: объяснять непонятные места, задавать вопросы на понимание, помогать готовить защиту практики. Не выдумывать кода, которого здесь нет.

**Тема практики:** разработка компонентов системы автоматизированной оценки качества ответов больших языковых моделей.

**Репозиторий:** учебный FastAPI-сервис. Система не отвечает на вопросы пользователя. Она берёт уже готовый ответ какой-то LLM и ставит ему оценку качества.

**Автор/контекст:** практика студента, стек Python + FastAPI. Судья по умолчанию — Gemma 4 31B через OpenRouter (`google/gemma-4-31b-it`). Есть запасные пути: OpenAI, локальная Ollama и эвристический fallback.

**Секреты в этот файл не входят.** Ключи лежат только в локальном `.env` и в git не коммитятся.

---

## 1. Зачем проект существует

Большие языковые модели ошибаются, льют воду, путают факты, иногда выдумывают источники. Проверять каждый ответ руками дорого. Нужен автомат, который:

1. быстро отсекает явный мусор дешёвыми метриками;
2. глубже оценивает смысл через модель-судью (LLM-as-a-Judge, метод G-Eval);
3. иногда сверяется с человеком, чтобы понять, не врёт ли автомат.

Это не чат-бот и не генератор ответов. Это **оценщик**.

Исходная учебная схема студента была такой: вход Question + Answer, три параллельные ветки (метрики / судья / человек), на выходе один Quality Score и анализ bias. В реализации схему поправили:

- вход шире: ещё эталон и контекст;
- человек не считает балл на каждый запрос, а стоит в офлайн-контуре;
- один overall недостаточен — есть несколько критериев;
- появился явный агрегатор;
- bias измеряется конкретными числами, а не «на словах».

---

## 2. Архитектура: два контура

### 2.1. Онлайн-контур (каждый запрос)

```
question + answer + optional reference + optional context
        │
        ├── лексика: ROUGE-L, BLEU, chrF    (только если есть эталон)
        ├── семантика: cosine MiniLM        (только если есть эталон)
        ├── эвристики: длина, пересечение с вопросом
        └── LLM-as-a-Judge: G-Eval          (или heuristic fallback)
                    │
                    ▼
              агрегатор
                    │
                    ▼
         QualityReport (overall, критерии, флаги, обоснование)
```

Человек здесь не участвует.

### 2.2. Офлайн-контур (HITL)

В `data/human_labels.json` лежат ручные оценки `human_score` от 0 до 1. Их ставит студент (или второй разметчик) — с страницы в браузере или сразу в JSON.

Скрипт `scripts/run_validation.py` или эндпоинт `POST /validate` прогоняет те же пары через автомат и считает:

- корреляции Pearson / Spearman / Kendall между автоматом и человеком;
- verbosity bias (длина ответа vs оценка);
- среднее расхождение semantic vs judge.

**Важная формулировка для защиты:** HITL — не третий скорер в проде. Это калибровка и проверка системы.

---

## 3. Входные данные одного запроса

Поля `EvaluateRequest` (`app/schemas.py`):

| Поле | Обязательно | Смысл |
|---|---|---|
| `question` | да | вопрос, на который отвечала модель |
| `answer` | да | ответ-кандидат, который оцениваем |
| `reference` | нет | эталон («правильный» ответ). Без него ROUGE, BLEU и cosine = null |
| `context` | нет | фрагмент документа. Если есть, у судьи появляется критерий groundedness |
| `skip_judge` | нет, по умолчанию false | не звать LLM-судью, сразу эвристики |

Почему эталон обязателен для метрик: ROUGE/BLEU/cosine — это сравнение двух текстов. Не с чем сравнивать — считать нечего. Вопрос сам по себе эталоном не является.

---

## 4. Выход: QualityReport

Все оценки в отчёте в диапазоне **0..1**, кроме сырых 1–5 у судьи, которые сразу нормализуются.

```json
{
  "overall": 0.78,
  "confidence": 0.64,
  "dimensions": {
    "correctness": 0.8,
    "relevance": 0.9,
    "completeness": 0.7,
    "coherence": 0.8,
    "groundedness": null
  },
  "lexical": {"rouge_l": 0.41, "bleu": 0.22},
  "semantic": {"cosine": 0.73, "method": "bow-cosine"},
  "judge": {
    "provider": "openrouter",
    "model": "google/gemma-4-31b-it",
    "rationale": "...",
    "used_logprobs": false,
    "fallback": false
  },
  "flags": {
    "no_reference": false,
    "no_context": true,
    "high_disagreement": false,
    "empty_or_too_short": false,
    "judge_fallback": false
  },
  "heuristics": {
    "length_words": 12,
    "question_overlap": 0.4,
    "length_score": 0.69,
    "length_ratio_score": 0.8
  }
}
```

Критерии `dimensions`:

- **correctness** — фактическая корректность;
- **relevance** — отвечает ли на заданный вопрос;
- **completeness** — достаточно ли полно;
- **coherence** — связность и язык;
- **groundedness** — опирается ли на `context`, без выдумок. Есть только при ненулевом контексте.

Флаги:

- `no_reference` / `no_context` — чего не передали;
- `high_disagreement` — |cosine − среднее судьи| ≥ 0.25;
- `empty_or_too_short` — в ответе 0 слов;
- `judge_fallback` — LLM-судья не отработал, сработали правила.

---

## 5. Как считается один запрос (pipeline)

Файл: `app/evaluators/pipeline.py`, функция `evaluate_answer`.

Порядок строго такой:

1. `compute_lexical(answer, reference)`
2. `compute_semantic(answer, reference, embedding_model)`
3. `compute_heuristics(answer, question, reference)`
4. если `skip_judge` — провайдер не создаётся;
5. иначе `get_judge_provider(settings)` (при ошибке конфига — заглушка, которая падает при вызове);
6. `judge_pointwise(...)` — либо G-Eval, либо fallback;
7. `aggregate(...)` — собирает overall, confidence, флаги.

Дальше отчёт уходит в API или на HTML-страницу.

---

## 6. Слой A. Лексика

Файл: `app/evaluators/lexical.py`.

Если эталона нет — `rouge_l = null`, `bleu = null`.

### ROUGE-L

Собственная реализация. Стандартный пакет `rouge-score` выкидывает кириллицу (оставляет только латиницу и цифры), поэтому его не используем.

Алгоритм:

1. токенизация Unicode-словами и числами, нижний регистр;
2. длина LCS (наибольшая общая подпоследовательность токенов);
3. precision = LCS / длина ответа, recall = LCS / длина эталона;
4. F-мера с beta = 1.2 (recall чуть важнее precision, как в классическом ROUGE-L).

### BLEU

Библиотека `sacrebleu`, `sentence_bleu`, токенизатор `intl`. Сырой score 0..100 делится на 100.

BLEU и ROUGE слабы на перефразе. Пример из жизни проекта: эталон «Париж», ответ «Столица Франции — Париж.» — смысл верный, BLEU ≈ 0.11, ROUGE-L ≈ 0.55, итоговый overall без живого судьи ≈ 0.57. Это не баг кнопки, а свойство n-грамм.

---

## 7. Слой B. Семантика

Файл: `app/evaluators/semantic.py`.

Если эталона нет — `cosine = null`.

Сначала пытается загрузить `sentence-transformers` и модель из `.env` (`EMBEDDING_MODEL`, по умолчанию `paraphrase-multilingual-MiniLM-L12-v2`). Если веса лежат в `models/minilm/`, они читаются с диска. Векторы нормализуются, считается скалярное произведение, отрицательные значения обрезаются до 0.

Пакет стоит в `requirements.txt`. Если импорт или файл модели падает, используется **мешок слов (BoW cosine)**. В отчёте поле `method` равно `sentence-transformers` или `bow-cosine`.

BoW не понимает синонимы: «столица Франции» и «Париж» дадут умеренный cosine. MiniLM на перефразе «Главный город этой страны — Париж» vs «Столица Франции — Париж» дал cosine 0.93 при ROUGE-L 0.26. На всём HITL-корпусе cosine Spearman 0.487 всё же ниже ROUGE-L 0.556: модель завышает near-miss вроде «Столица Франции — Лион» (cosine ~0.90). Это в `reports/judge_compare.md`, не декларация.

---

## 8. Эвристики

Файл: `app/evaluators/heuristics.py`.

Считаются всегда, даже без эталона.

| Ключ | Смысл |
|---|---|
| `length_words` | число слов в ответе |
| `question_overlap` | доля слов вопроса, которые встретились в ответе |
| `length_score` | грубая «нормальность» длины. Пусто = 0; 1–2 слова при эталоне = 0.7, без эталона = 0.45; иначе `min(1, 0.45 + words/50)` |
| `length_ratio_score` | насколько длина ответа близка к длине эталона. Если ответа не больше чем в 6 раз длиннее эталона, нижняя граница 0.55 |

Короткие фактические ответы («323», «Париж») специально не обнуляются: иначе автомат наказывал бы правильные однословные факты.

---

## 9. Слой C. LLM-as-a-Judge (G-Eval)

Файл: `app/evaluators/llm_judge.py`.

### Идея G-Eval

Статья Liu et al., 2023. Судья не ставит число сразу. Сначала кратко рассуждает по рубрике, потом возвращает структурированную оценку. Температура 0.

В проекте два режима:

1. **Pointwise** — один ответ, баллы 1–5 по критериям + rationale, формат JSON.
2. **Pairwise** — два ответа A и B, победитель `A` / `B` / `tie`.

### Нормализация 1–5 → 0–1

```
unit = (score - 1) / 4
```

1 → 0.0, 3 → 0.5, 5 → 1.0. Если модель уже вернула число 0..1, оно принимается как есть.

### Logprobs

Если провайдер отдаёт logprob токенов «1»..«5», оценки чуть сжимаются к 0.6 при низкой уверенности. OpenRouter/Gemma обычно logprobs не даёт — тогда используется только JSON. Это честно описано как ограничение.

### Когда судья не вызывается

- `skip_judge=true` (на HTML-странице так, пока не включена галочка «Спросить судью»);
- провайдер не создался;
- HTTP/API ошибка, таймаут, битый JSON.

Тогда `fallback=true` и критерии собираются из эвристик + cosine (если он был). На странице это надо читать как «оценила не модель-судья, а простая программа».

### Pairwise и position bias

`evaluate_pairwise`:

1. спрашивает победителя в порядке A, B;
2. спрашивает ещё раз, но тексты меняет местами (бывший B становится первым);
3. инвертирует второго победителя обратно в координаты исходных A/B;
4. если победители совпали — `consistent=true`, `position_bias_detected=false`;
5. если нет — судья зависит от позиции, `preferred=inconsistent`.

Это классический тест на position bias у LLM-судей.

---

## 10. Агрегатор

Файл: `app/evaluators/aggregator.py`.

`judge_score` = не среднее четырёх критериев, а **гейт по факту**: `correctness × среднее(relevance, completeness, coherence[, groundedness])`. Беглый, но ложный ответ (correctness=0, остальные 1) даёт overall 0, а не 0.75. На HITL это подняло Spearman overall с 0.854 до ~0.93.

**Если есть эталон и cosine:**

```
overall = w_sem * cosine + w_lex * lexical + w_judge * judge_score
```

`lexical` — среднее ROUGE-L, BLEU и chrF. Веса читаются из `.env` и **подбираются grid search по Spearman** на HITL (`scripts/run_validation.py --compare`). На 59 живых оценках Gemma максимум дал `WEIGHT_JUDGE=1.00` (ρ=0.869): cosine и лексика в overall на этой выборке только шумели. Слои всё равно считаются и видны в отчёте.

Если два судьи в ensemble сильно разошлись — confidence режется. Если `position_bias_detected` — overall умножается на `(1 - POSITION_BIAS_PENALTY)`.

**Если эталона нет:**

```
overall = 0.85 * judge_score + 0.15 * length_score
confidence = 0.7  (или 0.55, если fallback)
```

При context отдельно считается фактчек (пересечение токенов + штраф за числа, которых нет в контексте). Он слегка подмешивается в overall и режет балл, если в ответе появились «лишние» числа.

Дополнительно:

- при fallback confidence не выше 0.5;
- пустой ответ: overall ≤ 0.15, confidence ≤ 0.3;
- `high_disagreement`, если disagreement ≥ 0.25.

Поэтому без эталона и без Gemma оценка «про курс доллара» легко получается около 0.68: нет ROUGE/BLEU/cosine, остаются длина и пересечение с вопросом.

---

## 11. Провайдеры судьи

Файлы: `app/providers/`.

Общий интерфейс: `complete(prompt, temperature) -> text + optional logprobs`.

| `JUDGE_PROVIDER` | Что происходит |
|---|---|
| `openrouter` | OpenAI-совместимый клиент, `base_url=https://openrouter.ai/api/v1`, модель `google/gemma-4-31b-it`, заголовки HTTP-Referer и X-Title |
| `openai` | официальный OpenAI, модель из `OPENAI_MODEL` |
| `ollama` | локальный `POST {OLLAMA_BASE_URL}/api/chat` |

Клиент OpenAI/OpenRouter создаётся с `httpx.AsyncClient(trust_env=False)`, чтобы системный SOCKS Windows не ломал запросы. Сначала пробует запрос с `logprobs=true`, при ошибке — без них.

Типичная проблема студента: из РФ OpenRouter часто отвечает `403 Access denied by security policy` (Cloudflare). Ключ при этом может быть верным. Через туннель (Happ и т.п.) запросы проходят. Если 403 — код ловит исключение и включает heuristic fallback. Это выглядит как «Gemma не работает», хотя падает сеть, не рубрика.

---

## 12. Конфигурация

`app/config.py` + файл `.env` (шаблон `.env.example`).

Сейчас в локальном `.env` провайдер `openrouter`, модель `google/gemma-4-31b-it`.

Другие важные переменные:

- `WEIGHT_SEMANTIC=0.00`
- `WEIGHT_LEXICAL=0.00`
- `WEIGHT_JUDGE=1.00` (grid search по Spearman к HITL, живая Gemma)
- `DISAGREEMENT_THRESHOLD=0.25`
- `JUDGE_TEMPERATURE=0.0`
- `EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2`
- `JUDGE_ENSEMBLE=off` (mean на этом корпусе хуже одной Gemma)

`get_settings()` кэшируется (`lru_cache`). Вкладка «Судья» пишет выбранный провайдер и ключ в `.env` через `PUT /settings` и вызывает `reload_settings()` — перезапуск сервера не нужен. `GET /settings` не возвращает ключ, только флаг `key_set`. Настройки судьи принимаются только с localhost. Пустое поле ключа не затирает уже сохранённый.

---

## 13. HTTP API

Приложение: `app/main.py` (FastAPI). Корневой `GET /` отдаёт HTML-форму `app/static/index.html`. Swagger: `/docs`.

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/` | страница оценки + HITL-ползунок |
| GET | `/health` | провайдер и имя модели судьи |
| GET | `/settings` | текущий судья, ключи замаскированы |
| PUT | `/settings` | сменить OpenRouter / OpenAI / Ollama, модель, ключ |
| POST | `/settings/test` | пробный запрос «ответь OK» |
| POST | `/evaluate` | один ответ → QualityReport |
| POST | `/evaluate/import` | разобрать JSON/JSONL/CSV |
| GET | `/samples` | демо из `data/sample_eval.json` |
| POST | `/evaluate/batch` | до 100 пар; без судьи — параллельно |
| POST | `/evaluate/pairwise` | A vs B + свап позиций |
| GET | `/labels` | сколько меток в JSON |
| POST | `/labels` | сохранить ручную оценку |
| POST | `/validate` | прогнать размеченные пары, корреляции и bias |

Реализация маршрутов: `app/api/routes.py`.

---

## 14. Веб-страница

`app/static/index.html`.

Поля: вопрос, ответ модели, эталон. Кнопка «Оценить автоматически» бьёт в `POST /evaluate`.

Вкладка **Пакет**: файл JSON/JSONL/CSV или демо `sample_eval.json`. Без G-Eval пакет уходит в `POST /evaluate/batch`. С судьёй пары идут по одной, чтобы был прогресс и можно было остановить. Клик по строке открывает пару в Pointwise.

Вкладка «Судья»: три карточки OpenRouter / OpenAI-совместимый API / Ollama. Можно вставить ключ, base URL и имя любой модели (пресеты — подсказка, поле свободное). Чип в шапке показывает текущего судью и открывает эту вкладку.

Галочка «Спросить судью ({модель})»:

- выключена → `skip_judge=true` → эвристики;
- включена → вызывается выбранный провайдер; при ошибке сети/ключа снова fallback.

Под оценкой пишется, кто считал: эвристики или `{provider} / {model}`.

Ниже блок HITL: ползунок 0..100, кнопка «Сохранить мою оценку» → `POST /labels` → дописывается `data/human_labels.json`. Счётчик меток берётся из `GET /labels`.

После автооценки ползунок подставляется в значение системы — это только стартовая подсказка, человек может сдвинуть.

---

## 15. Данные и валидация

### `data/sample_eval.json`

Около 25 пар вопрос/ответ/эталон для демо. Есть и хорошие, и специально плохие ответы (Лион вместо Парижа, 330 вместо 323, галлюцинация срока хранения документа).

### `data/human_labels.json`

Стартово около 60 размеченных пар. У каждой:

- `id`;
- `question`, `answer`, `reference` (полное предложение, не одно слово вроде «Париж»);
- иногда `context`;
- `human_score` 0..1;
- `human_dimensions` — correctness / relevance / completeness / coherence (+ groundedness при контексте);
- `case_type`: good, wrong, partial, verbose, paraphrase, hallucination, lexical-trap и т.д.

В корпусе специально есть спорные случаи: многословие при верном факте, частично верно, галлюцинация поверх верного срока, перефраз, лексическое пересечение при неверном факте, непроверяемое время.

### Скрипт `scripts/run_validation.py`

```bash
python scripts/run_validation.py --skip-judge
python scripts/run_validation.py --compare
```

`--compare` прогоняет **тот же корпус** через heuristics, Gemma (OpenRouter) и Ollama и пишет `reports/judge_compare.md`. Без этого «качество судьи» — заявление, а не факт. Кэш LLM — `reports/judge_cache/` (только оценки судьи; cosine пересчитывается заново).

Фактический прогон, n=59, fallback 0% у обоих LLM, после гейта `correctness × style`:

| Режим | overall Spearman | overall Pearson | judge Spearman |
|---|---:|---:|---:|
| Heuristics / fallback | 0.459 | 0.494 | 0.461 |
| Gemma 4 31B (OpenRouter) | 0.913 | 0.977 | 0.926 |
| Ollama llama3.2 | 0.659 | 0.698 | 0.660 |
| Ensemble mean | 0.868 | 0.929 | 0.869 |

Gemma vs heuristics: Δρ overall **+0.454**. До гейта overall Gemma был 0.854 / Pearson 0.851: беглый ложный факт (Лион, Достоевский) получал 0.75. После гейта таких случаев с |Δ|≥0.25 осталось три, все частичные. Cosine MiniLM 0.487, ROUGE-L 0.556, BLEU 0.300.

В отчёте также: подобранные веса, таблицы ошибок «человек высокий — автомат низкий» и наоборот.

---

## 16. Запуск

```bash
python scripts/start.py
```

Скрипт `scripts/start.py`:

1. переходит в корень проекта;
2. если нет `.venv` — создаёт и ставит `requirements.txt`;
3. если нет `.env` — копирует `.env.example`;
4. берёт первый свободный порт из 8001, 8000, 8002, 8010;
5. стартует `uvicorn app.main:app`;
6. когда `/` отвечает, открывает браузер.

Остановка: Ctrl+C. Пересобирать ничего не нужно: это обычный Python-скрипт.

Зависимости: fastapi, uvicorn, pydantic, pydantic-settings, openai, httpx, sacrebleu, numpy, scipy, python-dotenv. Опционально sentence-transformers.

`.gitignore` скрывает `.env`, `.venv`, кэши. Отчёт `reports/validation.md` можно хранить в репозитории.

---

## 17. Карта файлов

```
app/main.py                      FastAPI, GET /
app/static/index.html            форма оценки и HITL
app/api/routes.py                все эндпоинты
app/config.py                    настройки из .env
app/schemas.py                   Pydantic-модели

app/evaluators/pipeline.py       склейка слоёв
app/ingest.py                    разбор JSON/CSV пакета
app/evaluators/lexical.py        ROUGE-L + BLEU + chrF
app/evaluators/semantic.py       cosine MiniLM / BoW fallback
app/evaluators/heuristics.py     длина и overlap
app/evaluators/factcheck.py      overlap с context
app/evaluators/llm_judge.py      G-Eval, JSON, fallback, ensemble merge
app/evaluators/aggregator.py     overall и флаги

app/providers/base.py            CompletionResult
app/providers/factory.py         выбор провайдера
app/providers/openai_provider.py OpenAI + OpenRouter
app/providers/ollama_provider.py локальная Ollama

app/validation/metrics.py        корреляции
app/validation/biases.py         verbosity и disagreement
app/validation/weights.py        grid search весов
app/validation/errors.py         таблицы расхождений
app/validation/labels_store.py   чтение/запись меток

data/human_labels.json
data/sample_eval.json
scripts/start.py
scripts/run_validation.py
reports/validation.md
reports/judge_compare.md
requirements.txt
.env.example
```

Цепочка клика «Оценить»:

браузер → `POST /evaluate` → `evaluate_answer` → lexical + semantic + heuristics + judge → `aggregate` → JSON → карточка на странице.

Цепочка «Сохранить мою оценку»:

браузер → `POST /labels` → `save_label` → `data/human_labels.json`.

---

## 18. Ограничения и честные слабые места

1. Без эталона lexical/semantic молчат. Остаются судья или эвристики.
2. MiniLM на перефразе сильнее ROUGE/BLEU, но завышает near-miss с общим лексическим каркасом («Столица Франции — Лион»).
3. BLEU/ROUGE занижают верный длинный ответ при коротком эталоне; эталоны в корпусе выровнены до предложений, однословные «Париж» убраны.
4. G-Eval зависит от сети и ключа. 403 ≠ неверный промпт.
5. Logprobs у Gemma через OpenRouter часто нет; калибровка 1–5 → 0–1 тогда не срабатывает.
6. Ensemble Gemma+Ollama на этой выборке хуже одной Gemma: слабый судья тянет среднее вниз.
7. 59 меток, один разметчик. Для практики хватает, для публикации мало; спорные случаи уже добавлены, чтобы метрика не упиралась только в «хорошо/плохо».
8. Без гейта Gemma ставила 0.75 беглому ложному ответу (correctness=0, остальные 1). Гейт `correctness × style` это закрывает; остаются спорные частичные случаи.
9. Нет обучения reward-модели, BERTScore не ставился (пакет опционален). Pairwise swap в API есть, на 59 парах pointwise-swap не гонялся (удвоил бы число вызовов).

---

## 19. Что говорить на защите

1. Тема — компоненты оценки ответов LLM, не чат.
2. Три технических слоя + отдельный HITL.
3. G-Eval: рубрика, рассуждение, JSON, температура 0.
4. HITL: 59 меток, корреляция, не online-скорер. «Качество судьи» — таблица heuristics vs Gemma vs Ollama, не лозунг.
5. Почему BLEU/ROUGE оставлены как baseline: ROUGE-L ρ=0.556, судья Gemma ρ=0.926.
6. Почему один overall мало — четыре критерия; у Gemma correctness ρ=0.90, coherence слабее.
7. Как ловим bias: verbosity (после живого судьи Spearman длины 0.13), position-swap в pairwise, disagreement слоёв.
8. Ограничение сети OpenRouter и fallback — часть эксперимента. На боевом прогоне fallback был 0%.
9. Ensemble не «всегда лучше»: mean 0.868 < Gemma 0.913. Веса агрегатора — grid search, не вкус. Overall режется гейтом факта: беглость не спасает ложный ответ.

---

## 20. Словарь

- **LLM** — большая языковая модель (Gemma, GPT, Llama).
- **Эталон / reference** — текст, с которым сравнивают ответ.
- **LLM-as-a-Judge** — одна модель оценивает ответ другой.
- **G-Eval** — судья с рубрикой и цепочкой рассуждения, затем число.
- **Pointwise** — оценка одного ответа по шкале.
- **Pairwise** — выбор лучшего из двух.
- **HITL** — человек в контуре; здесь только валидация и разметка.
- **Fallback** — запасной путь, если судья недоступен.
- **ROUGE-L** — похожесть по самой длинной общей цепочке слов.
- **BLEU** — похожесть по n-граммам, из машинного перевода.
- **Cosine** — угол между двумя векторами текстов.
- **BoW** — мешок слов, частоты токенов без порядка и синонимов.
- **Logprobs** — логарифм вероятности токена; прокси уверенности модели.
- **Overall** — итоговый балл 0..1 после агрегации.
- **Confidence** — насколько слои согласны между собой, не «вероятность истины».
- **Корреляция** — насколько автомат ранжирует как человек. 1 идеал, 0 хаос.
- **Pearson** — линейная связь сырых чисел.
- **Spearman** — связь рангов (порядок «лучше/хуже»).
- **Kendall** — доля согласных пар порядка.
- **Bias** — систематический перекос, не случайная ошибка.
- **Verbosity bias** — длинное кажется лучше короткого.
- **Position bias** — первый в паре чаще побеждает.
- **Groundedness** — ответ опирается на выданный контекст.
- **OpenRouter** — единое API ко многим моделям, совместимо с OpenAI SDK.

---

## 21. Инструкция для Gemini

Ты помогаешь студенту разобрать учебный проект по этому конспекту.

Правила:

- опирайся только на этот текст, не придумывай несуществующие файлы и цифры;
- объясняй простыми словами, затем можно точнее;
- если студент путает «кто отвечает» и «кто оценивает» — поправляй: система оценивает чужой ответ;
- если спрашивает, почему балл низкий при верном Париже — объясни n-граммы и короткий эталон;
- если спрашивает, оценивает ли Gemma — смотри skip_judge, fallback и 403;
- после объяснения задай 1 короткий вопрос на понимание;
- помогай формулировать абзацы для пояснительной записки, если попросят.

Вопросы студенту на проверку:

1. Чем HITL отличается от третьего скорера в проде?
2. Почему без эталона нет ROUGE и cosine?
3. Почему верный ответ «Столица Франции — Париж» при эталоне «Париж» получил около 57 без судьи?
4. Что такое G-Eval тремя предложениями?
5. Зачем свапать A и B в pairwise?
6. Как из оценок 1–5 получается 0–1?
7. Что делает агрегатор, если cosine и судья сильно разошлись?
8. Кто ставит балл на странице, если галочка Gemma выключена?
9. Что означает `judge.fallback=true`?
10. Чем Spearman полезнее Pearson для этой практики?
