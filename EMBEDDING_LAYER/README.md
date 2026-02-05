# Модуль Embedding Layer

## Описание

Модуль для работы с эмбеддингами в проекте LLM. Содержит реализацию Token Embedding и Positional Encoding для преобразования токенов в векторные представления.

## Структура модуля

```
EMBEDDING_LAYER/
├── __init__.py              # Инициализация модуля
├── embedding_layer.py       # Основная реализация
├── test_embedding.py         # Тесты
├── example_usage.py          # Примеры использования
└── README.md                 # Этот файл
```

## Компоненты

### 1. TokenEmbedding
Базовый слой для преобразования токенов (ID) в векторные представления.
- Инициализация весов как в GPT-2/GPT-3
- Масштабирование эмбеддингов для стабилизации

### 2. SinusoidalPositionalEncoding
Синусоидальное позиционное кодирование (как в оригинальном Transformer).
- Точная реализация из "Attention Is All You Need"
- Формула: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
- Поддержка нечетных размерностей

### 3. LearnablePositionalEncoding
Обучаемое позиционное кодирование (альтернатива синусоидальному).
- Используется в некоторых современных моделях
- Позволяет модели обучать позиционные представления

### 4. PositionalEncoding
Универсальный класс, поддерживающий оба типа позиционного кодирования.

### 5. EmbeddingLayer
Полный слой, объединяющий Token Embedding и Positional Encoding.
- **Layer Normalization** (как в GPT-2/GPT-3) - опционально
- Поддержка синусоидального и обучаемого позиционного кодирования
- Dropout для регуляризации

### 6. create_embedding_from_tokenizer()
Функция-помощник для создания Embedding Layer из BPETokenizer.

## Использование

### Базовое использование

```python
from BPE_STUCTUR import BPETokenizer
from EMBEDDING_LAYER import create_embedding_from_tokenizer
import torch

# Загрузка токенизатора
tokenizer = BPETokenizer()
tokenizer.load("chekpoint.pkl")

# Создание Embedding Layer
embedding = create_embedding_from_tokenizer(
    tokenizer,
    embedding_dim=256,
    max_seq_len=512
)

# Кодирование текста
text = "Привет, мир!"
token_ids = tokenizer.encode(text)
token_tensor = torch.tensor([token_ids], dtype=torch.long)

# Получение эмбеддингов
embeddings = embedding(token_tensor)
print(f"Размер: {embeddings.shape}")  # [1, seq_len, 256]
```

### Прямой импорт классов

```python
from EMBEDDING_LAYER import EmbeddingLayer, TokenEmbedding, PositionalEncoding

# Создание напрямую
embedding = EmbeddingLayer(
    vocab_size=10000,
    embedding_dim=256,
    max_seq_len=512
)
```

## Тестирование

Запуск тестов:

```bash
python EMBEDDING_LAYER/test_embedding.py
```

Запуск примеров:

```bash
python EMBEDDING_LAYER/example_usage.py
```

## Параметры

### EmbeddingLayer

- `vocab_size`: Размер словаря (из `tokenizer.get_vocab_size()`)
- `embedding_dim`: Размерность эмбеддингов (256, 512, 768 и т.д.)
- `max_seq_len`: Максимальная длина последовательности (512, 1024 и т.д.)
- `dropout`: Вероятность dropout (обычно 0.1)
- `padding_idx`: ID токена паддинга (обычно 0 для `<PAD>`)
- `learnable_pos`: Если True, использует обучаемое позиционное кодирование (по умолчанию False - синусоидальное)
- `layer_norm`: Если True, добавляет Layer Normalization (как в GPT-2/GPT-3, по умолчанию True)
- `layer_norm_eps`: Эпсилон для Layer Normalization (по умолчанию 1e-5)

### Примеры использования параметров

```python
# Синусоидальное кодирование с Layer Norm (как в GPT-2)
embedding = create_embedding_from_tokenizer(
    tokenizer,
    embedding_dim=256,
    learnable_pos=False,  # Синусоидальное
    layer_norm=True        # С Layer Norm
)

# Обучаемое позиционное кодирование без Layer Norm
embedding = create_embedding_from_tokenizer(
    tokenizer,
    embedding_dim=256,
    learnable_pos=True,    # Обучаемое
    layer_norm=False       # Без Layer Norm
)
```

## Интеграция с проектом

Модуль полностью интегрирован с вашим BPE токенизатором:

- Использует `tokenizer.get_vocab_size()` для размера словаря
- Использует `tokenizer.special_tokens['<PAD>']` для паддинга
- Совместим с методами `encode()` и `encode_batch()`

## Следующие шаги

После использования Embedding Layer:

1. Создание Transformer блоков для обработки эмбеддингов
2. Создание Language Model Head для предсказания токенов
3. Обучение полной модели LLM

## Современные возможности

Модуль реализует современные практики из LLM:

- ✅ **Синусоидальное позиционное кодирование** - точная реализация из Transformer
- ✅ **Обучаемое позиционное кодирование** - как в GPT-2/GPT-3
- ✅ **Layer Normalization** - как в GPT-2/GPT-3 (включена по умолчанию)
- ✅ **Улучшенная инициализация** - как в GPT-2/GPT-3

Подробнее см. `MODERN_LLM_FEATURES.md`

## Версия

2.0.0 - Улучшенная реализация с современными возможностями

