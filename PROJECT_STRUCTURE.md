# Структура проекта Tokenier

## 📂 Организация файлов

### Корневая директория
- `BPE_STUCTUR.py` - BPE токенизатор
- `tokenizer_trainer.py` - Обучение токенизатора
- `train_gpt.py` - Обучение GPT модели
- `train_trm.py` - Обучение TRM модели
- `create_embedding_dataset.py` - Создание датасета
- `README.md` - Главный README
- `.gitignore` - Игнорируемые файлы

### 📁 data/
Текстовые данные для обучения:
- `*.txt` - Все текстовые файлы с данными
- `README.md` - Описание данных

### 📁 tests/
Все тесты проекта:
- `test_embeddings.py` - Тесты эмбеддингов
- `test_embedding.py` - Тесты embedding layer
- `test_transformer.py` - Тесты трансформера
- `test_trm.py` - Тесты TRM модели
- `gpt_tester.py` - Тестер GPT модели

### 📁 docs/
Общая документация:
- `2510.04871v1.pdf` - Статья TRM
- `README_TRAINER.md` - Документация по обучению

### 📁 EMBEDDING_LAYER/
Модуль эмбеддингов:
- `embedding_layer.py` - Основной модуль
- `embedding_trainer.py` - Обучение эмбеддингов
- `embedding_evaluator.py` - Оценка эмбеддингов
- `auto_evaluation_system.py` - Автоматическая оценка
- `*.md` - Вся документация модуля

### 📁 TRANSFORMER/
GPT архитектура:
- `gpt_model.py` - Полная GPT модель
- `decoder_block.py` - Decoder блок
- `attention.py` - Multi-head attention
- `feed_forward.py` - Feed-forward network
- `*.md` - Документация модуля

### 📁 TRM/
Tiny Recursive Model:
- `trm_model.py` - Полная TRM модель
- `tiny_recursive_network.py` - Маленькая сеть
- `deep_supervision.py` - Deep supervision
- `latent_recursion.py` - Рекурсивное обновление
- `*.md` - Документация модуля

## 🗂️ Игнорируемые файлы (.gitignore)

- `*.pth`, `*.pt`, `*.ckpt` - Модели и чекпоинты
- `*.pkl` - Сериализованные объекты
- `checkpoints/`, `my_checkpoints/` - Папки с чекпоинтами
- `__pycache__/` - Кэш Python
- `*.pdf` - Большие PDF файлы
- `*.log` - Логи

## 📝 Правила организации

1. **Документация** - в папках соответствующих модулей
2. **Тесты** - все в папке `tests/`
3. **Данные** - все `.txt` файлы в `data/`
4. **Модели** - не коммитятся в Git (в .gitignore)




