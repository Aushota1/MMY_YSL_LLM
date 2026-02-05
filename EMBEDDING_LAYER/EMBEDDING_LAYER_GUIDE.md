# Руководство по реализации Embedding Layer для LLM

## Описание

Этот документ описывает реализацию Embedding Layer для вашего проекта создания LLM на основе BPE токенизатора. Embedding Layer преобразует токены (ID) в векторные представления, которые могут использоваться нейронной сетью.

---

## Содержание

1. [Обзор](#обзор)
2. [Архитектура Embedding Layer](#архитектура-embedding-layer)
3. [Интеграция с BPE токенизатором](#интеграция-с-bpe-токенизатором)
4. [Реализация](#реализация)
5. [Использование](#использование)
6. [Оптимизация](#оптимизация)
7. [Примеры](#примеры)

---

## Обзор

### Что такое Embedding Layer?

Embedding Layer - это слой нейронной сети, который преобразует дискретные токены (ID) в плотные векторные представления. В контексте вашего проекта:

- **Вход**: Список токенов (ID) от BPE токенизатора
- **Выход**: Матрица эмбеддингов (векторные представления токенов)

### Зачем это нужно?

1. **Семантическое представление**: Близкие по смыслу токены получают близкие векторы
2. **Размерность**: Сокращает размерность (vocab_size → embedding_dim)
3. **Обучаемость**: Векторы обучаются в процессе тренировки модели

---

## Архитектура Embedding Layer

### Базовый Embedding Layer

```
Input:  [batch_size, seq_len] - токены (ID)
         ↓
Embedding Layer
         ↓
Output: [batch_size, seq_len, embedding_dim] - векторы
```

### Компоненты

1. **Token Embedding**: Преобразование токенов в векторы
2. **Positional Encoding**: Добавление информации о позиции
3. **Layer Normalization**: Нормализация (опционально)
4. **Dropout**: Регуляризация (опционально)

---

## Интеграция с BPE токенизатором

### Структура вашего токенизатора

Ваш `BPETokenizer` предоставляет:

```python
tokenizer.vocab_size        # Максимальный размер словаря
tokenizer.get_vocab_size()  # Текущий размер словаря
tokenizer.vocab             # Словарь {id: token}
tokenizer.inverse_vocab     # Обратный словарь {token: id}
tokenizer.special_tokens    # Специальные токены
tokenizer.encode(text)      # Кодирование текста в ID
tokenizer.decode(token_ids) # Декодирование ID в текст
```

### Специальные токены

Ваш токенизатор использует следующие специальные токены:

- `<PAD>`: 0 - Паддинг (для выравнивания последовательностей)
- `<UNK>`: 1 - Неизвестный токен
- `<BOS>`: 2 - Начало последовательности
- `<EOS>`: 3 - Конец последовательности
- `<SEP>`: 4 - Разделитель

### Важные моменты

1. **Размер словаря**: Используйте `tokenizer.get_vocab_size()` для получения реального размера
2. **ID токенов**: ID начинаются с 0 и идут последовательно
3. **Специальные токены**: Уже включены в словарь с фиксированными ID

---

## Реализация

### 1. Базовый Token Embedding

```python
import torch
import torch.nn as nn
import math

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
        
        # Инициализация весов (опционально)
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

### 2. Positional Encoding

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

### 3. Полный Embedding Layer

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
```

---

## Использование

### Интеграция с вашим токенизатором

```python
from BPE_STUCTUR import BPETokenizer
import torch

# Загрузка обученного токенизатора
tokenizer = BPETokenizer()
tokenizer.load("chekpoint.pkl")  # или ваш путь к чекпоинту

# Получение размера словаря
vocab_size = tokenizer.get_vocab_size()
print(f"Размер словаря: {vocab_size}")

# Создание Embedding Layer
embedding_dim = 256  # Размерность эмбеддингов
max_seq_len = 512    # Максимальная длина последовательности

embedding_layer = EmbeddingLayer(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    max_seq_len=max_seq_len,
    dropout=0.1,
    padding_idx=tokenizer.special_tokens['<PAD>']  # 0
)

# Пример использования
text = "Привет, мир!"
token_ids = tokenizer.encode(text)
print(f"Токены: {token_ids}")

# Преобразование в тензор
token_tensor = torch.tensor([token_ids], dtype=torch.long)  # [1, seq_len]

# Получение эмбеддингов
embeddings = embedding_layer(token_tensor)  # [1, seq_len, embedding_dim]
print(f"Размер эмбеддингов: {embeddings.shape}")
```

### Пакетная обработка

```python
# Несколько текстов
texts = [
    "Мама мыла раму.",
    "Привет, как дела?",
    "Сегодня хорошая погода."
]

# Кодирование всех текстов
token_ids_list = tokenizer.encode_batch(texts)

# Выравнивание до максимальной длины (padding)
max_len = max(len(ids) for ids in token_ids_list)
padded_tokens = []
for ids in token_ids_list:
    padded = ids + [tokenizer.special_tokens['<PAD>']] * (max_len - len(ids))
    padded_tokens.append(padded[:max_len])

# Преобразование в тензор
token_tensor = torch.tensor(padded_tokens, dtype=torch.long)  # [batch_size, seq_len]

# Получение эмбеддингов
embeddings = embedding_layer(token_tensor)  # [batch_size, seq_len, embedding_dim]
```

---

## Оптимизация

### 1. Использование GPU

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_layer = embedding_layer.to(device)
token_tensor = token_tensor.to(device)
embeddings = embedding_layer(token_tensor)
```

### 2. Градиентный чекипойнтинг (для больших моделей)

```python
from torch.utils.checkpoint import checkpoint

# Для экономии памяти при обратном проходе
embeddings = checkpoint(embedding_layer, token_tensor)
```

### 3. Кэширование позиционного кодирования

Позиционное кодирование уже кэшируется как буфер, но можно оптимизировать дальше:

```python
# Использование более эффективного кодирования для длинных последовательностей
# (например, learnable positional embeddings вместо синусоидальных)
```

---

## Примеры

### Пример 1: Базовое использование

```python
from BPE_STUCTUR import BPETokenizer
import torch
from embedding_layer import EmbeddingLayer

# 1. Загрузка токенизатора
tokenizer = BPETokenizer()
tokenizer.load("chekpoint.pkl")

# 2. Создание Embedding Layer
vocab_size = tokenizer.get_vocab_size()
embedding = EmbeddingLayer(
    vocab_size=vocab_size,
    embedding_dim=256,
    max_seq_len=512
)

# 3. Кодирование текста
text = "Искусственный интеллект меняет мир"
token_ids = tokenizer.encode(text)

# 4. Преобразование в эмбеддинги
token_tensor = torch.tensor([token_ids], dtype=torch.long)
embeddings = embedding(token_tensor)

print(f"Текст: {text}")
print(f"Токены: {token_ids}")
print(f"Эмбеддинги: {embeddings.shape}")  # [1, seq_len, 256]
```

### Пример 2: Интеграция с DataLoader

```python
from torch.utils.data import Dataset, DataLoader
from BPE_STUCTUR import BPETokenizer
import torch

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        token_ids = self.tokenizer.encode(text)
        
        # Обрезка или паддинг
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            pad_id = self.tokenizer.special_tokens['<PAD>']
            token_ids = token_ids + [pad_id] * (self.max_length - len(token_ids))
        
        # Input: все токены кроме последнего
        # Target: все токены кроме первого (shifted)
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(token_ids[1:], dtype=torch.long)
        
        return input_ids, target_ids

# Использование
tokenizer = BPETokenizer()
tokenizer.load("chekpoint.pkl")

texts = ["Текст 1", "Текст 2", "Текст 3"]
dataset = TextDataset(texts, tokenizer, max_length=512)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Создание Embedding Layer
vocab_size = tokenizer.get_vocab_size()
embedding = EmbeddingLayer(vocab_size=vocab_size, embedding_dim=256)

# Обучение
for input_ids, target_ids in dataloader:
    # Получение эмбеддингов
    embeddings = embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
    
    # Дальнейшая обработка в модели...
```

### Пример 3: Полная модель с Embedding Layer

```python
import torch
import torch.nn as nn
from BPE_STUCTUR import BPETokenizer

class SimpleLLM(nn.Module):
    def __init__(self, tokenizer, embedding_dim=256, num_heads=8, num_layers=6):
        super().__init__()
        
        vocab_size = tokenizer.get_vocab_size()
        
        # Embedding Layer
        self.embedding = EmbeddingLayer(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_seq_len=512
        )
        
        # Transformer блоки (упрощенный пример)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Language Model Head
        self.lm_head = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, token_ids):
        # Эмбеддинги
        x = self.embedding(token_ids)  # [batch, seq, dim]
        
        # Transformer
        x = self.transformer(x)  # [batch, seq, dim]
        
        # Предсказание следующего токена
        logits = self.lm_head(x)  # [batch, seq, vocab_size]
        
        return logits

# Использование
tokenizer = BPETokenizer()
tokenizer.load("chekpoint.pkl")

model = SimpleLLM(tokenizer, embedding_dim=256)
text = "Привет, мир!"
token_ids = tokenizer.encode(text)
token_tensor = torch.tensor([token_ids], dtype=torch.long)

logits = model(token_tensor)
print(f"Logits shape: {logits.shape}")  # [1, seq_len, vocab_size]
```

---

## Рекомендации по параметрам

### Размерность эмбеддингов (embedding_dim)

- **Маленькие модели**: 128-256
- **Средние модели**: 256-512
- **Большие модели**: 512-1024
- **Очень большие**: 1024-2048

### Максимальная длина последовательности (max_seq_len)

- **Короткие тексты**: 128-256
- **Средние тексты**: 512
- **Длинные тексты**: 1024-2048

### Dropout

- **Обычно**: 0.1
- **Для регуляризации**: 0.2-0.3
- **Для больших моделей**: 0.1-0.15

---

## Важные замечания

### 1. Совместимость с токенизатором

- Всегда используйте `tokenizer.get_vocab_size()` для получения реального размера словаря
- Используйте `tokenizer.special_tokens['<PAD>']` для padding_idx
- Убедитесь, что max_seq_len соответствует вашим данным

### 2. Обработка паддинга

- Паддинг должен быть на позиции `tokenizer.special_tokens['<PAD>']` (обычно 0)
- Эмбеддинг паддинга не участвует в обучении (padding_idx)

### 3. Память

- Для больших словарей и длинных последовательностей используйте GPU
- Рассмотрите градиентный чекипойнтинг для экономии памяти

### 4. Инициализация

- Веса эмбеддингов инициализируются случайно
- Можно использовать предобученные эмбеддинги (если доступны)

---

## Следующие шаги

После реализации Embedding Layer:

1. **Transformer Blocks**: Добавьте self-attention и feed-forward слои
2. **Language Model Head**: Слой для предсказания следующего токена
3. **Training Loop**: Цикл обучения модели
4. **Inference**: Генерация текста

---

## Заключение

Embedding Layer - это первый и важный компонент вашей LLM. Он преобразует дискретные токены в непрерывные векторные представления, которые могут обрабатываться нейронной сетью.

Интеграция с вашим BPE токенизатором проста:
- Используйте `tokenizer.get_vocab_size()` для размера словаря
- Используйте `tokenizer.encode()` для получения токенов
- Используйте специальные токены для паддинга и других целей

Удачи в создании вашей LLM! 🚀

