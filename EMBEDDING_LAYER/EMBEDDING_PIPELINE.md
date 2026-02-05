# Пайплайн реализации Embedding Layer для вашего LLM проекта

## Описание

Этот документ содержит пошаговый пайплайн для реализации Embedding Layer в вашем проекте создания LLM. Следуйте инструкциям последовательно для правильной интеграции.

---

## Содержание

1. [Структура проекта](#структура-проекта)
2. [Шаг 1: Подготовка окружения](#шаг-1-подготовка-окружения)
3. [Шаг 2: Создание файлов](#шаг-2-создание-файлов)
4. [Шаг 3: Реализация компонентов](#шаг-3-реализация-компонентов)
5. [Шаг 4: Интеграция с токенизатором](#шаг-4-интеграция-с-токенизатором)
6. [Шаг 5: Тестирование](#шаг-5-тестирование)
7. [Шаг 6: Использование в модели](#шаг-6-использование-в-модели)

---

## Структура проекта

### Текущая структура:

```
tokenier/
├── BPE_STUCTUR.py          # Ваш BPE токенизатор
├── tokenizer_trainer.py     # Приложение для обучения токенизатора
├── chekpoint.pkl            # Обученный токенизатор
├── EMBEDDING_LAYER_GUIDE.md # Теоретическое руководство
└── ...
```

### Целевая структура (после реализации):

```
tokenier/
├── BPE_STUCTUR.py              # Ваш BPE токенизатор
├── tokenizer_trainer.py         # Приложение для обучения токенизатора
├── chekpoint.pkl                # Обученный токенизатор
├── embedding_layer.py           # ⭐ НОВЫЙ: Реализация Embedding Layer
├── test_embedding.py            # ⭐ НОВЫЙ: Тесты для Embedding Layer
├── example_usage.py             # ⭐ НОВЫЙ: Примеры использования
├── EMBEDDING_LAYER_GUIDE.md     # Теоретическое руководство
└── EMBEDDING_PIPELINE.md        # ⭐ НОВЫЙ: Этот файл (пайплайн)
```

---

## Шаг 1: Подготовка окружения

### 1.1 Установка зависимостей

Убедитесь, что у вас установлены необходимые библиотеки:

```bash
pip install torch numpy
```

Проверьте версию PyTorch:

```bash
python -c "import torch; print(torch.__version__)"
```

**Требования:**
- Python 3.7+
- PyTorch 1.8+
- NumPy (обычно устанавливается с PyTorch)

### 1.2 Проверка токенизатора

Убедитесь, что ваш токенизатор работает:

```bash
python -c "from BPE_STUCTUR import BPETokenizer; t = BPETokenizer(); t.load('chekpoint.pkl'); print('OK')"
```

Если ошибка - сначала обучите токенизатор через `tokenizer_trainer.py`.

---

## Шаг 2: Создание файлов

### 2.1 Создайте файл `embedding_layer.py`

Этот файл будет содержать всю реализацию Embedding Layer.

**Действие:** Создайте новый файл `embedding_layer.py` в корне проекта.

### 2.2 Создайте файл `test_embedding.py`

Этот файл будет содержать тесты для проверки работы Embedding Layer.

**Действие:** Создайте новый файл `test_embedding.py` в корне проекта.

### 2.3 Создайте файл `example_usage.py`

Этот файл будет содержать примеры использования Embedding Layer.

**Действие:** Создайте новый файл `example_usage.py` в корне проекта.

---

## Шаг 3: Реализация компонентов

### 3.1 Реализация Token Embedding

**Файл:** `embedding_layer.py`

**Действие:** Добавьте следующий код в начало файла:

```python
"""
Embedding Layer для LLM проекта
Реализация Token Embedding и Positional Encoding
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class TokenEmbedding(nn.Module):
    """
    Базовый слой эмбеддингов для токенов
    
    Args:
        vocab_size: Размер словаря (из tokenizer.get_vocab_size())
        embedding_dim: Размерность эмбеддингов (например, 256, 512, 768)
        padding_idx: ID токена паддинга (обычно 0 для <PAD>)
    """
    def __init__(self, vocab_size: int, embedding_dim: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx  # Игнорирует градиенты для паддинга
        )
        self.embedding_dim = embedding_dim
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов эмбеддингов"""
        # Инициализация по нормальному распределению
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        
        # Обнуление эмбеддинга паддинга
        if self.embedding.padding_idx is not None:
            nn.init.constant_(
                self.embedding.weight[self.embedding.padding_idx], 
                0.0
            )
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Преобразование токенов в эмбеддинги
        
        Args:
            token_ids: [batch_size, seq_len] - ID токенов
        
        Returns:
            [batch_size, seq_len, embedding_dim] - эмбеддинги
        """
        # Масштабирование для стабилизации (как в Transformer)
        embeddings = self.embedding(token_ids) * math.sqrt(self.embedding_dim)
        return embeddings
```

### 3.2 Реализация Positional Encoding

**Файл:** `embedding_layer.py`

**Действие:** Добавьте следующий код после класса `TokenEmbedding`:

```python
class PositionalEncoding(nn.Module):
    """
    Позиционное кодирование для добавления информации о позиции
    
    Args:
        embedding_dim: Размерность эмбеддингов
        max_seq_len: Максимальная длина последовательности (например, 512, 1024)
        dropout: Вероятность dropout (опционально)
    """
    def __init__(self, embedding_dim: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Создание позиционного кодирования
        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        # Вычисление div_term для синусоидального кодирования
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() *
            -(math.log(10000.0) / embedding_dim)
        )
        
        # Применение синуса к четным позициям
        pe[:, 0::2] = torch.sin(position * div_term)
        # Применение косинуса к нечетным позициям
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Добавление размерности batch
        pe = pe.unsqueeze(0)  # [1, max_seq_len, embedding_dim]
        
        # Регистрация как буфер (не обучаемый параметр)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Добавление позиционного кодирования
        
        Args:
            x: [batch_size, seq_len, embedding_dim] - эмбеддинги токенов
        
        Returns:
            [batch_size, seq_len, embedding_dim] - эмбеддинги с позиционной информацией
        """
        # Добавляем позиционное кодирование
        x = x + self.pe[:, :x.size(1)]
        # Применяем dropout
        return self.dropout(x)
```

### 3.3 Реализация полного Embedding Layer

**Файл:** `embedding_layer.py`

**Действие:** Добавьте следующий код после класса `PositionalEncoding`:

```python
class EmbeddingLayer(nn.Module):
    """
    Полный слой эмбеддингов с токенами и позициями
    
    Args:
        vocab_size: Размер словаря
        embedding_dim: Размерность эмбеддингов
        max_seq_len: Максимальная длина последовательности
        dropout: Вероятность dropout
        padding_idx: ID токена паддинга
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        padding_idx: int = 0
    ):
        super().__init__()
        
        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        
        self.positional_encoding = PositionalEncoding(
            embedding_dim=embedding_dim,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Преобразование токенов в эмбеддинги с позиционной информацией
        
        Args:
            token_ids: [batch_size, seq_len] - ID токенов
        
        Returns:
            [batch_size, seq_len, embedding_dim] - эмбеддинги
        """
        # Токенные эмбеддинги
        x = self.token_embedding(token_ids)
        
        # Добавление позиционного кодирования
        x = self.positional_encoding(x)
        
        return x
    
    def get_embedding_dim(self) -> int:
        """Получение размерности эмбеддингов"""
        return self.embedding_dim
```

### 3.4 Финальная структура файла `embedding_layer.py`

**Действие:** Убедитесь, что файл содержит все три класса в правильном порядке:

1. `TokenEmbedding`
2. `PositionalEncoding`
3. `EmbeddingLayer`

---

## Шаг 4: Интеграция с токенизатором

### 4.1 Создание функции-помощника

**Файл:** `embedding_layer.py`

**Действие:** Добавьте в конец файла функцию для создания Embedding Layer из токенизатора:

```python
def create_embedding_from_tokenizer(tokenizer, embedding_dim: int = 256, 
                                    max_seq_len: int = 512, dropout: float = 0.1):
    """
    Создание Embedding Layer из BPETokenizer
    
    Args:
        tokenizer: Обученный BPETokenizer
        embedding_dim: Размерность эмбеддингов
        max_seq_len: Максимальная длина последовательности
        dropout: Вероятность dropout
    
    Returns:
        EmbeddingLayer готовый к использованию
    """
    vocab_size = tokenizer.get_vocab_size()
    padding_idx = tokenizer.special_tokens.get('<PAD>', 0)
    
    embedding_layer = EmbeddingLayer(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        max_seq_len=max_seq_len,
        dropout=dropout,
        padding_idx=padding_idx
    )
    
    return embedding_layer
```

---

## Шаг 5: Тестирование

### 5.1 Базовый тест

**Файл:** `test_embedding.py`

**Действие:** Создайте файл со следующим содержимым:

```python
"""
Тесты для Embedding Layer
"""

import torch
from BPE_STUCTUR import BPETokenizer
from embedding_layer import EmbeddingLayer, create_embedding_from_tokenizer


def test_basic_embedding():
    """Базовый тест Embedding Layer"""
    print("=" * 70)
    print("ТЕСТ 1: Базовый Embedding Layer")
    print("=" * 70)
    
    # Загрузка токенизатора
    tokenizer = BPETokenizer()
    tokenizer.load("chekpoint.pkl")
    
    vocab_size = tokenizer.get_vocab_size()
    print(f"Размер словаря: {vocab_size}")
    
    # Создание Embedding Layer
    embedding = EmbeddingLayer(
        vocab_size=vocab_size,
        embedding_dim=256,
        max_seq_len=512,
        dropout=0.1,
        padding_idx=tokenizer.special_tokens['<PAD>']
    )
    
    # Тестовый текст
    text = "Привет, мир!"
    token_ids = tokenizer.encode(text)
    print(f"Текст: {text}")
    print(f"Токены: {token_ids}")
    
    # Преобразование в тензор
    token_tensor = torch.tensor([token_ids], dtype=torch.long)
    print(f"Размер входного тензора: {token_tensor.shape}")
    
    # Получение эмбеддингов
    embeddings = embedding(token_tensor)
    print(f"Размер эмбеддингов: {embeddings.shape}")
    print(f"Ожидаемый размер: [1, {len(token_ids)}, 256]")
    
    assert embeddings.shape == (1, len(token_ids), 256), "Неверный размер эмбеддингов!"
    print("✓ Тест пройден!")
    print()


def test_with_helper_function():
    """Тест с функцией-помощником"""
    print("=" * 70)
    print("ТЕСТ 2: Использование функции-помощника")
    print("=" * 70)
    
    # Загрузка токенизатора
    tokenizer = BPETokenizer()
    tokenizer.load("chekpoint.pkl")
    
    # Создание через функцию-помощник
    embedding = create_embedding_from_tokenizer(
        tokenizer,
        embedding_dim=256,
        max_seq_len=512
    )
    
    # Тестовый текст
    text = "Искусственный интеллект"
    token_ids = tokenizer.encode(text)
    
    token_tensor = torch.tensor([token_ids], dtype=torch.long)
    embeddings = embedding(token_tensor)
    
    print(f"Текст: {text}")
    print(f"Размер эмбеддингов: {embeddings.shape}")
    assert embeddings.shape[0] == 1, "Неверный batch size!"
    assert embeddings.shape[2] == 256, "Неверная размерность эмбеддингов!"
    print("✓ Тест пройден!")
    print()


def test_batch_processing():
    """Тест пакетной обработки"""
    print("=" * 70)
    print("ТЕСТ 3: Пакетная обработка")
    print("=" * 70)
    
    tokenizer = BPETokenizer()
    tokenizer.load("chekpoint.pkl")
    
    embedding = create_embedding_from_tokenizer(tokenizer, embedding_dim=256)
    
    # Несколько текстов
    texts = [
        "Мама мыла раму.",
        "Привет, как дела?",
        "Сегодня хорошая погода."
    ]
    
    # Кодирование
    token_ids_list = tokenizer.encode_batch(texts)
    
    # Выравнивание до максимальной длины
    max_len = max(len(ids) for ids in token_ids_list)
    pad_id = tokenizer.special_tokens['<PAD>']
    
    padded_tokens = []
    for ids in token_ids_list:
        padded = ids + [pad_id] * (max_len - len(ids))
        padded_tokens.append(padded[:max_len])
    
    # Преобразование в тензор
    token_tensor = torch.tensor(padded_tokens, dtype=torch.long)
    print(f"Размер входного тензора: {token_tensor.shape}")
    
    # Получение эмбеддингов
    embeddings = embedding(token_tensor)
    print(f"Размер эмбеддингов: {embeddings.shape}")
    print(f"Ожидаемый размер: [{len(texts)}, {max_len}, 256]")
    
    assert embeddings.shape == (len(texts), max_len, 256), "Неверный размер!"
    print("✓ Тест пройден!")
    print()


def test_gpu_support():
    """Тест поддержки GPU"""
    print("=" * 70)
    print("ТЕСТ 4: Поддержка GPU")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")
    
    tokenizer = BPETokenizer()
    tokenizer.load("chekpoint.pkl")
    
    embedding = create_embedding_from_tokenizer(tokenizer, embedding_dim=256)
    embedding = embedding.to(device)
    
    text = "Тест на GPU"
    token_ids = tokenizer.encode(text)
    token_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
    
    embeddings = embedding(token_tensor)
    print(f"Эмбеддинги на устройстве: {embeddings.device}")
    print(f"Размер: {embeddings.shape}")
    
    assert embeddings.device.type == device.type, "Неверное устройство!"
    print("✓ Тест пройден!")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ЗАПУСК ТЕСТОВ EMBEDDING LAYER")
    print("=" * 70 + "\n")
    
    try:
        test_basic_embedding()
        test_with_helper_function()
        test_batch_processing()
        test_gpu_support()
        
        print("=" * 70)
        print("✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("=" * 70)
    except Exception as e:
        print(f"\n✗ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
```

### 5.2 Запуск тестов

**Действие:** Запустите тесты:

```bash
python test_embedding.py
```

**Ожидаемый результат:**
- Все тесты должны пройти успешно
- Не должно быть ошибок
- Вывод должен показывать "✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!"

---

## Шаг 6: Использование в модели

### 6.1 Создание примера использования

**Файл:** `example_usage.py`

**Действие:** Создайте файл со следующим содержимым:

```python
"""
Примеры использования Embedding Layer
"""

import torch
from BPE_STUCTUR import BPETokenizer
from embedding_layer import create_embedding_from_tokenizer


def example_basic_usage():
    """Базовый пример использования"""
    print("=" * 70)
    print("ПРИМЕР 1: Базовое использование")
    print("=" * 70)
    
    # 1. Загрузка токенизатора
    tokenizer = BPETokenizer()
    tokenizer.load("chekpoint.pkl")
    print(f"✓ Токенизатор загружен. Размер словаря: {tokenizer.get_vocab_size()}")
    
    # 2. Создание Embedding Layer
    embedding = create_embedding_from_tokenizer(
        tokenizer,
        embedding_dim=256,
        max_seq_len=512
    )
    print(f"✓ Embedding Layer создан. Размерность: {embedding.get_embedding_dim()}")
    
    # 3. Кодирование текста
    text = "Привет, мир! Это пример использования Embedding Layer."
    token_ids = tokenizer.encode(text)
    print(f"✓ Текст закодирован. Токенов: {len(token_ids)}")
    
    # 4. Преобразование в эмбеддинги
    token_tensor = torch.tensor([token_ids], dtype=torch.long)
    embeddings = embedding(token_tensor)
    
    print(f"\nРезультат:")
    print(f"  Входной текст: '{text}'")
    print(f"  Количество токенов: {len(token_ids)}")
    print(f"  Размер эмбеддингов: {embeddings.shape}")
    print(f"  Размерность каждого токена: {embeddings.shape[2]}")
    print()


def example_batch_processing():
    """Пример пакетной обработки"""
    print("=" * 70)
    print("ПРИМЕР 2: Пакетная обработка")
    print("=" * 70)
    
    tokenizer = BPETokenizer()
    tokenizer.load("chekpoint.pkl")
    
    embedding = create_embedding_from_tokenizer(tokenizer, embedding_dim=256)
    
    # Несколько текстов
    texts = [
        "Мама мыла раму.",
        "Привет, как дела?",
        "Сегодня хорошая погода.",
        "Искусственный интеллект меняет мир."
    ]
    
    print(f"Обрабатываем {len(texts)} текстов...")
    
    # Кодирование всех текстов
    token_ids_list = tokenizer.encode_batch(texts)
    
    # Выравнивание до максимальной длины
    max_len = max(len(ids) for ids in token_ids_list)
    pad_id = tokenizer.special_tokens['<PAD>']
    
    padded_tokens = []
    for ids in token_ids_list:
        padded = ids + [pad_id] * (max_len - len(ids))
        padded_tokens.append(padded[:max_len])
    
    # Преобразование в тензор
    token_tensor = torch.tensor(padded_tokens, dtype=torch.long)
    
    # Получение эмбеддингов
    embeddings = embedding(token_tensor)
    
    print(f"\nРезультат:")
    print(f"  Количество текстов: {len(texts)}")
    print(f"  Максимальная длина: {max_len} токенов")
    print(f"  Размер эмбеддингов: {embeddings.shape}")
    print(f"  Общее количество векторов: {embeddings.shape[0] * embeddings.shape[1]}")
    print()


def example_gpu_usage():
    """Пример использования GPU"""
    print("=" * 70)
    print("ПРИМЕР 3: Использование GPU")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")
    
    tokenizer = BPETokenizer()
    tokenizer.load("chekpoint.pkl")
    
    embedding = create_embedding_from_tokenizer(tokenizer, embedding_dim=256)
    embedding = embedding.to(device)
    
    text = "Этот пример работает на GPU, если доступно"
    token_ids = tokenizer.encode(text)
    token_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
    
    embeddings = embedding(token_tensor)
    
    print(f"\nРезультат:")
    print(f"  Устройство эмбеддингов: {embeddings.device}")
    print(f"  Размер: {embeddings.shape}")
    print()


def example_integration_with_model():
    """Пример интеграции с моделью"""
    print("=" * 70)
    print("ПРИМЕР 4: Интеграция с моделью")
    print("=" * 70)
    
    import torch.nn as nn
    
    tokenizer = BPETokenizer()
    tokenizer.load("chekpoint.pkl")
    
    # Создание Embedding Layer
    embedding = create_embedding_from_tokenizer(tokenizer, embedding_dim=256)
    
    # Простая модель (пример)
    class SimpleModel(nn.Module):
        def __init__(self, embedding_layer):
            super().__init__()
            self.embedding = embedding_layer
            self.linear = nn.Linear(256, 128)
            self.output = nn.Linear(128, tokenizer.get_vocab_size())
        
        def forward(self, token_ids):
            x = self.embedding(token_ids)
            x = x.mean(dim=1)  # Усреднение по последовательности
            x = torch.relu(self.linear(x))
            x = self.output(x)
            return x
    
    model = SimpleModel(embedding)
    
    # Тестовый текст
    text = "Пример интеграции"
    token_ids = tokenizer.encode(text)
    token_tensor = torch.tensor([token_ids], dtype=torch.long)
    
    # Forward pass
    output = model(token_tensor)
    
    print(f"\nРезультат:")
    print(f"  Входной текст: '{text}'")
    print(f"  Размер выхода модели: {output.shape}")
    print(f"  Размер словаря: {tokenizer.get_vocab_size()}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ EMBEDDING LAYER")
    print("=" * 70 + "\n")
    
    try:
        example_basic_usage()
        example_batch_processing()
        example_gpu_usage()
        example_integration_with_model()
        
        print("=" * 70)
        print("✓ ВСЕ ПРИМЕРЫ ВЫПОЛНЕНЫ УСПЕШНО!")
        print("=" * 70)
    except Exception as e:
        print(f"\n✗ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
```

### 6.2 Запуск примеров

**Действие:** Запустите примеры:

```bash
python example_usage.py
```

---

## Чеклист реализации

Используйте этот чеклист для отслеживания прогресса:

### Подготовка
- [ ] Установлен PyTorch
- [ ] Проверена работа токенизатора
- [ ] Создан файл `embedding_layer.py`
- [ ] Создан файл `test_embedding.py`
- [ ] Создан файл `example_usage.py`

### Реализация
- [ ] Реализован класс `TokenEmbedding`
- [ ] Реализован класс `PositionalEncoding`
- [ ] Реализован класс `EmbeddingLayer`
- [ ] Добавлена функция `create_embedding_from_tokenizer`

### Тестирование
- [ ] Запущены тесты (`test_embedding.py`)
- [ ] Все тесты прошли успешно
- [ ] Запущены примеры (`example_usage.py`)
- [ ] Примеры работают корректно

### Интеграция
- [ ] Embedding Layer интегрирован в проект
- [ ] Документация обновлена
- [ ] Код готов к использованию в модели

---

## Следующие шаги

После успешной реализации Embedding Layer:

1. **Создание Transformer блоков** - следующий компонент для LLM
2. **Создание Language Model Head** - слой для предсказания токенов
3. **Обучение модели** - полный цикл обучения LLM
4. **Инференс** - генерация текста

---

## Решение проблем

### Проблема: "ModuleNotFoundError: No module named 'embedding_layer'"

**Решение:** Убедитесь, что файл `embedding_layer.py` находится в той же директории, что и скрипт, который его импортирует.

### Проблема: "RuntimeError: CUDA out of memory"

**Решение:** 
- Уменьшите `batch_size`
- Уменьшите `embedding_dim`
- Используйте CPU вместо GPU

### Проблема: "IndexError: index out of range"

**Решение:** 
- Проверьте, что `vocab_size` соответствует размеру словаря токенизатора
- Используйте `tokenizer.get_vocab_size()` для получения правильного размера

### Проблема: "AssertionError" в тестах

**Решение:** 
- Проверьте размеры тензоров
- Убедитесь, что токенизатор правильно закодировал текст
- Проверьте параметры Embedding Layer

---

## Заключение

После выполнения всех шагов у вас будет:

1. ✅ Полностью реализованный Embedding Layer
2. ✅ Рабочие тесты
3. ✅ Примеры использования
4. ✅ Готовый к интеграции код

**Важные файлы:**
- `embedding_layer.py` - основная реализация
- `test_embedding.py` - тесты
- `example_usage.py` - примеры

**Следующий этап:** Создание Transformer блоков для обработки эмбеддингов.

Удачи в реализации! 🚀

