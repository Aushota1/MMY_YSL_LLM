# Tokenier - LLM Project

Проект для создания и обучения языковых моделей с поддержкой GPT и TRM архитектур.

## 📁 Структура проекта

```
tokenier/
├── BPE_STUCTUR.py              # BPE токенизатор
├── tokenizer_trainer.py         # Обучение токенизатора
├── train_gpt.py                 # Обучение GPT модели
├── train_trm.py                 # Обучение TRM модели
├── create_embedding_dataset.py # Создание датасета для эмбеддингов
│
├── EMBEDDING_LAYER/             # Модуль эмбеддингов
│   ├── embedding_layer.py
│   ├── embedding_trainer.py
│   ├── embedding_evaluator.py
│   └── *.md                     # Документация модуля
│
├── TRANSFORMER/                 # GPT архитектура
│   ├── gpt_model.py
│   ├── decoder_block.py
│   ├── attention.py
│   ├── feed_forward.py
│   └── *.md                     # Документация модуля
│
├── TRM/                         # Tiny Recursive Model
│   ├── trm_model.py
│   ├── tiny_recursive_network.py
│   ├── deep_supervision.py
│   └── *.md                     # Документация модуля
│
├── data/                        # Текстовые данные для обучения
│   ├── *.txt
│
├── tests/                       # Все тесты
│   ├── test_embeddings.py
│   ├── test_transformer.py
│   ├── test_trm.py
│   └── test_embedding.py
│
└── docs/                        # Общая документация
    ├── TRM_MIGRATION_PLAN.md
    └── *.pdf
```

## 🚀 Быстрый старт

### 1. Обучение токенизатора
```bash
python tokenizer_trainer.py
```

### 2. Обучение GPT модели
```bash
python train_gpt.py --interactive
```

### 3. Обучение TRM модели
```bash
python train_trm.py --interactive
```

## 📚 Документация

- **EMBEDDING_LAYER/**: Документация по эмбеддингам
- **TRANSFORMER/**: Документация по GPT архитектуре
- **TRM/**: Документация по TRM модели
- **docs/**: Общая документация и планы

## 🧪 Тестирование

Все тесты находятся в папке `tests/`:

```bash
# Тест эмбеддингов
python tests/test_embeddings.py

# Тест трансформера
python tests/test_transformer.py

# Тест TRM
python tests/test_trm.py
```

## 📝 Лицензия

Проект для обучения и исследований.






