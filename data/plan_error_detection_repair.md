# План реализации системы детекции и исправления RTL по методологии VeriRAG (arXiv:2507.15664v2)

Документ опирается на статью VeriRAG (`data/2507.15664v2.pdf`), сравнение в `data/comparison_VeriRAG_vs_tokenier.md` и текущую структуру проекта tokenier. Включает **полный пайплайн** из статьи и **создание собственной LLM** на базе ваших модулей (TRANSFORMER/GPT, EMBEDDING_LAYER, train_gpt).

---

## Часть I. Пайплайн VeriRAG (из статьи) — что повторяем

### 1.1 Общая схема VeriRAG

```
[Verilog с ошибками] 
    → Представление (у них: Yosys→JSON→TF-IDF→512d → Autoencoder→128d)
    → Retrieval: поиск k ближайших эталонов по cosine similarity (reference: 35 пар broken/fixed)
    → LLM: промпт = система + few-shot (пары broken→fixed) + текущий RTL
    → [Исправленный Verilog]
    → (Опционально) Итерация: компилятор + DFT + LEC → при неуспехе повторный запрос к LLM
```

- **Датасет VeriDFT:** 437 файлов; метки — тип DFT-ошибки (ACNCPI, CLKNPI, CDFDAT, FFCKNP). Разбиение: 85 train (энкодер), 35 reference (для RAG), 317 test.
- **Успех:** синтезируемость + отсутствие DFT-нарушений + LEC. Метрика: доля успешно отремонтированных RTL (в статье 53.76% vs 6.96% zero-shot).

### 1.2 Адаптация под tokenier (без Yosys/EDA)

| Этап VeriRAG | В tokenier |
|--------------|------------|
| Представление RTL | **Текст RTL** → токенизатор (.pkl) → эмбеддинги (.pth) → фиксированный вектор 256d (`text_to_embeddings` + `embeddings_to_fixed_vector`). |
| Обучение представления | Эмбеддинги уже обучены (Layer_ML/EMBEDDING_LAYER). При необходимости дообучить с метками типа ошибки (контрастный loss / классификация) — отдельная задача. |
| Retrieval | По косинусной близости 256d-векторов искать k ближайших эталонов из reference-базы (пары broken/fixed). |
| LLM | **Собственная LLM** (GPT из TRANSFORMER + обучение через train_gpt) или внешний API. Промпт: система + few-shot пары (broken→fixed) + текущий RTL → модель генерирует исправленный код. |
| Валидация | По желанию: синтаксис (iverilog/yosys в Docker); при наличии EDA — синтез/DFT/LEC. |

---

## Часть II. Полный пайплайн реализации в проекте

### Фаза 0: Подготовка данных (как в VeriRAG)

| Шаг | Действие | Где в проекте / новые артефакты |
|-----|----------|----------------------------------|
| 0.1 | Определить типы ошибок | Например: `correct`, `error_syntax`, `error_width`, `error_latch`, при необходимости `error_dft` и др. |
| 0.2 | Собрать/сгенерировать RTL с метками | **correct:** подмножество из `Layer_ML/lerning_data/`. **С ошибками:** копии модулей с внесёнными типовыми ошибками (скрипт или вручную). |
| 0.3 | Разложить по папкам для детекции | Структура: `data_errors/lerning_data/correct/`, `data_errors/lerning_data/error_syntax/`, `error_width/`, … (один .txt на файл). |
| 0.4 | Сформировать reference для RAG (как 35 в VeriRAG) | Пары (broken, fixed): `data_errors/reference/<error_type>/` с файлами вида `id_broken.txt`, `id_fixed.txt` или один файл с двумя блоками. Минимум 20–35 пар на тип ошибки. |
| 0.5 | Подготовить данные для обучения LLM | См. раздел «Данные для собственной LLM» ниже. |

### Фаза 1: Детекция типа ошибки (классификация)

| Шаг | Действие | Компоненты |
|-----|----------|-------------|
| 1.1 | Векторизация RTL по классам ошибок | `Layer_ML/build_train_data.py` с `--input-dir data_errors/lerning_data`, `--output-dir train_data_errors`, те же `--model`, `--tokenizer`, `--output-dim 256`, `--method mean`. |
| 1.2 | Сбор датасета для классификатора | `Learn_ML/create_dataset.py --input-dir train_data_errors --output Learn_ML/dataset_errors.csv`. |
| 1.3 | Обучение классификатора | `Learn_ML/train_classifiers.py`: загрузка dataset_errors, разбиение train/test, обучение (SVM/XGBoost/RF и т.д.), сохранение в `Learn_ML/models/error_detector.joblib`. |
| 1.4 | CLI детекции | Обёртка над `Learn_ML/use_model.py`: вход — путь к .v/.txt, модель `error_detector.joblib`, вывод — метка (correct / error_syntax / …) и при необходимости вероятность. |

### Фаза 2: Retrieval (поиск похожих эталонов)

| Шаг | Действие | Компоненты |
|-----|----------|-------------|
| 2.1 | Построение индекса reference | Для каждого файла в `data_errors/reference/` (broken или оба broken+fixed — по выбранной схеме) вызвать `text_to_embeddings` + `embeddings_to_fixed_vector`; сохранить векторы и метаданные (путь к broken, путь к fixed, тип ошибки) в `Repair_ML/reference_index.npz` (или .npy + JSON). |
| 2.2 | Модуль поиска k-NN | Новый модуль `Repair_ML/retrieval.py`: загрузка индекса, функция `retrieve(vector, k=3, error_type=None)` — cosine similarity, возврат списка пар (broken_text, fixed_text) и при необходимости error_type. |
| 2.3 | Интеграция с пайплайном | Для входящего RTL: текст → вектор (Layer_ML) → при необходимости детектор ошибки (Фаза 1) → retrieval по вектору (и опционально по error_type) → пары для промпта LLM. |

### Фаза 3: Собственная LLM для исправления кода

Идея по статье: LLM получает контекст (системный промпт + few-shot примеры broken→fixed) и генерирует исправленный RTL. В проекте уже есть **GPT** (`TRANSFORMER/gpt_model.py`) и **обучение** (`train_gpt.py`), можно использовать ваши эмбеддинги из .pth.

#### Вариант A: GPT как генератор правки (decoder-only)

- **Вход модели:** последовательность токенов вида  
  `[SYSTEM]... [BROKEN] код1 [FIXED] исправленный_код1 ... [BROKEN] код_текущий [FIXED]`  
  цель — достроить токены после `[FIXED]` (исправленный код для текущего модуля).
- **Обучение:** датасет пар (broken, fixed). Каждый пример — одна последовательность:  
  `[BROKEN]\n{broken}\n[FIXED]\n{fixed}`  
  Обучать next-token prediction (как в `train_gpt.py`), чтобы модель научилась продолжать после `[FIXED]`.
- **Данные:** директория с парами `data_llm_repair/train/` (файлы `*.broken.txt`, `*.fixed.txt` или один формат с разделителями). Скрипт загрузки формирует строки для `TextDataset` в `train_gpt.py`.
- **Инференс:** подать в GPT: системный промпт + 1–3 пары (broken, fixed) из retrieval + `[BROKEN]\n{текущий_код}\n[FIXED]\n`; генерировать токены до стоп-символа (например `\n\n` или `[END]`), декодировать в текст — это и есть исправленный RTL.

#### Вариант B: Few-shot без дообучения (как в статье)

- Не обучать свою LLM, а формировать текстовый промпт и вызывать **внешний API** (OpenAI, local Ollama и т.д.). Промпт = система + 1–3 пары (broken, fixed) из retrieval + текущий RTL; ответ API — исправленный код.
- Собственная LLM тогда используется только если вы дообучите GPT (Вариант A) и замените вызов API на вызов своей модели.

#### Где какие модули использовать для своей LLM

| Задача | Модуль / файл |
|--------|----------------|
| Токенизация RTL | Тот же BPE/словный токенизатор (.pkl), что и для эмбеддингов. |
| Эмбеддинги для GPT | Загрузить из `Layer_ML/final_model.pth` в `EmbeddingLayer` и передать в `GPTModel` (как в `train_gpt.py` при выборе «использовать обученные эмбеддинги»). |
| Модель | `TRANSFORMER/gpt_model.py` — `GPTModel` (embedding + decoder blocks + lm_head). |
| Обучение | `train_gpt.py`: либо интерактивно, либо CLI с `--data` — файл, где каждая строка = один пример формата `[BROKEN]\n...\n[FIXED]\n...`. Подготовить такой файл из пар (broken, fixed). |
| Генерация (инференс) | Новый скрипт `Repair_ML/generate_repair.py`: загрузка чекпоинта GPT, формирование последовательности (система + few-shot + текущий RTL + `[FIXED]\n`), автодополнение по токенам (greedy или sampling) до стоп-токена, декодирование в текст. |
| Сохранение чекпоинта LLM | Отдельный путь, например `Repair_ML/checkpoints/gpt_repair.pth`, чтобы не путать с `final_model.pth` (эмбеддинги) и с классификаторами. |

### Фаза 4: Связывание детекции, retrieval и LLM

| Шаг | Действие |
|-----|----------|
| 4.1 | Единый вход: путь к RTL-файлу или строка кода. |
| 4.2 | Векторизация (Layer_ML) → опционально детектор ошибки (error_detector.joblib) → тип ошибки. |
| 4.3 | Retrieval по вектору (и по типу ошибки) → 1–3 пары (broken, fixed). |
| 4.4 | Формирование промпта для LLM (система + пары + текущий код). |
| 4.5 | Вызов собственной GPT (Repair_ML/generate_repair.py) или внешнего API → получение исправленного кода. |
| 4.6 | Возврат пользователю: метка ошибки (если делали детекцию) + исправленный RTL. |

### Фаза 5: Валидация (опционально, по статье — компилятор + DFT + LEC)

- Проверка синтаксиса: iverilog или yosys в Docker.
- При наличии EDA: запуск синтеза, DFT-проверки, LEC; при неуспехе — повторный запрос к LLM (итеративный цикл как в VeriRAG).

---

## Часть III. Данные для обучения и где их применять

### 3.1 Для детекции ошибок

| Данные | Формат | Куда класть | Где использовать |
|--------|--------|-------------|-------------------|
| Корректные RTL | Один модуль на .txt | `data_errors/lerning_data/correct/` | Векторизация → train_data_errors → create_dataset → train_classifiers (target=correct). |
| RTL с ошибкой типа X | Один модуль на .txt | `data_errors/lerning_data/error_syntax/`, `error_width/`, … | То же; класс = имя папки. |
| Минимум объёма | ≥ 50–100 примеров на класс | — | Балансировка при обучении (веса классов / oversampling). |

### 3.2 Для retrieval (reference)

| Данные | Формат | Куда класть | Где использовать |
|--------|--------|-------------|-------------------|
| Пары (broken, fixed) | Два файла на пару или один с разделителями | `data_errors/reference/<error_type>/` (например `ref_001_broken.txt`, `ref_001_fixed.txt`) | Построение индекса (Repair_ML): векторы от broken или от обоих; при retrieval возвращать пары для промпта. |

### 3.3 Для обучения собственной LLM (GPT repair)

| Данные | Формат | Куда класть | Где использовать |
|--------|--------|-------------|-------------------|
| Пары (broken, fixed) | Тексты RTL | Каталог пар, например `data_llm_repair/train/` | Скрипт преобразует в строки вида `[BROKEN]\n{broken}\n[FIXED]\n{fixed}` и записывает в один текстовый файл (по одной строке на пару или по одному блоку на пару — в зависимости от того, как `train_gpt.py` читает данные). |
| Рекомендация | Строка = один полный пример для LM | Один большой .txt, каждая «строка» — один пример (можно использовать переносы как `\n` внутри примера и разделитель `\n\n` между примерами) | `train_gpt.py --data data_llm_repair/repair_examples.txt` (после адаптации загрузки под формат). |

---

## Часть IV. Какие модули есть и что добавить

### 4.1 Используемые без изменений

- **Layer_ML:** `text_to_embeddings.py`, `embedding_to_fixed_vector.py`, `build_train_data.py`.
- **Learn_ML:** `create_dataset.py`, `train_classifiers.py`, `use_model.py`.
- **TRANSFORMER:** `gpt_model.py` (GPTModel с опцией загрузки эмбеддингов из .pth).
- **EMBEDDING_LAYER:** `embedding_layer.py`, `create_embedding_from_tokenizer`.
- **Корень:** `train_gpt.py` (обучение GPT с возможностью подгрузки эмбеддингов), BPE/токенизатор.

### 4.2 Новые или адаптированные модули

| Модуль | Назначение |
|--------|------------|
| **Repair_ML/build_reference_index.py** | Обход `data_errors/reference/`, вызов text_to_embeddings + embeddings_to_fixed_vector, сохранение векторов и метаданных (пути broken/fixed, error_type). |
| **Repair_ML/retrieval.py** | Загрузка индекса, поиск k ближайших по cosine similarity, возврат пар (broken_text, fixed_text). |
| **Repair_ML/generate_repair.py** | Загрузка GPT-чекпоинта (и токенизатора), формирование промпта (система + few-shot + текущий RTL + `[FIXED]\n`), генерация токенов до стоп-символа, декодирование в текст исправленного RTL. |
| **Repair_ML/prepare_llm_data.py** | Чтение пар из `data_errors/reference/` или `data_llm_repair/train/`, запись в формат для `train_gpt.py` (одна строка/блок на пример). |
| **Learn_ML/run_error_detection.py** (или аналог) | CLI: путь к файлу RTL → вектор → error_detector.joblib → вывод метки ошибки (и при необходимости вероятностей). |
| **Repair_ML/run_repair_pipeline.py** | Оркестрация: RTL → вектор → (детекция) → retrieval → промпт → LLM (своя GPT или API) → исправленный код; опционально проверка синтаксиса. |

### 4.3 Конфигурация и пути

- `data_errors/` — корень данных по ошибкам (lerning_data для детекции, reference для RAG).
- `train_data_errors/` — .npy векторы для классификатора ошибок.
- `Learn_ML/models/error_detector.joblib` — модель детекции.
- `Repair_ML/reference_index.npz` (или .npy + .json) — индекс для retrieval.
- `Repair_ML/checkpoints/gpt_repair.pth` — чекпоинт GPT для задачи repair (если обучаете свою LLM).

---

## Часть V. Порядок реализации (краткий чек-лист)

1. **Данные:** собрать correct + 2–3 типа ошибок в `data_errors/lerning_data/`; собрать 20–35 пар (broken, fixed) в `data_errors/reference/`.
2. **Детекция:** build_train_data → train_data_errors; create_dataset → dataset_errors.csv; train_classifiers → error_detector.joblib; скрипт run_error_detection.
3. **Retrieval:** build_reference_index; retrieval.py с k-NN по cosine.
4. **LLM:** подготовить данные для GPT (prepare_llm_data); при необходимости дообучить GPT через train_gpt на примерах `[BROKEN]...[FIXED]`; реализовать generate_repair.py (промпт + генерация).
5. **Пайплайн:** run_repair_pipeline: RTL → вектор → детекция → retrieval → промпт → LLM → исправленный код.
6. **Валидация (опционально):** вызов iverilog/yosys в Docker для проверки синтаксиса исправленного кода.

---

Итог: план повторяет **полный пайплайн VeriRAG** (представление → retrieval → LLM → при необходимости итерация/валидация), адаптированный под **текстовые эмбеддинги tokenier** и **собственную LLM** на базе ваших модулей (TRANSFORMER/GPT + train_gpt + при необходимости отдельный скрипт генерации исправлений).
