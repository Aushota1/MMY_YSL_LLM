# Transformer Decoder Block для LLM

## Описание

Модуль `TRANSFORMER/` содержит реализацию Transformer Decoder Block, максимально приближенную к современным LLM (GPT-2, GPT-3, GPT-4). Модуль интегрируется с существующим `EMBEDDING_LAYER/` и позволяет построить полноценную GPT-подобную модель.

## Структура модуля

```
TRANSFORMER/
├── __init__.py              # Экспорт всех компонентов
├── attention.py             # Multi-Head Self-Attention
├── feed_forward.py          # Feed-Forward Network
├── decoder_block.py         # Transformer Decoder Block
├── gpt_model.py             # Полная GPT модель
├── test_transformer.py      # Unit тесты
├── example_usage.py         # Примеры использования
└── README.md                # Этот файл
```

## Компоненты

### 1. MultiHeadSelfAttention

Multi-Head Self-Attention с causal masking для autoregressive моделей.

**Особенности:**
- Scaled Dot-Product Attention
- Causal masking (предотвращает "заглядывание в будущее")
- Multi-head конкатенация
- Инициализация весов как в GPT-2/GPT-3

**Использование:**
```python
from TRANSFORMER import MultiHeadSelfAttention

attention = MultiHeadSelfAttention(
    embedding_dim=512,
    num_heads=8,
    dropout=0.1,
    causal=True
)

# x: [batch_size, seq_len, embedding_dim]
output = attention(x)
```

### 2. FeedForward

Feed-Forward Network с GELU активацией (как в GPT-2/GPT-3).

**Особенности:**
- Двухслойная сеть: `embedding_dim → ff_dim → embedding_dim`
- GELU активация
- Dropout для регуляризации
- Инициализация весов как в GPT

**Использование:**
```python
from TRANSFORMER import FeedForward

ffn = FeedForward(
    embedding_dim=512,
    ff_dim=2048,  # Обычно 4 × embedding_dim
    dropout=0.1,
    activation='gelu'
)

# x: [batch_size, seq_len, embedding_dim]
output = ffn(x)
```

### 3. TransformerDecoderBlock

Transformer Decoder Block с Pre-Norm архитектурой (как в GPT-2).

**Архитектура:**
```
Input → LayerNorm → MultiHeadAttention → Residual → 
       LayerNorm → FeedForward → Residual → Output
```

**Особенности:**
- Pre-Norm архитектура (нормализация перед подблоком)
- Residual connections
- Causal attention
- GELU активация в FFN

**Использование:**
```python
from TRANSFORMER import TransformerDecoderBlock

block = TransformerDecoderBlock(
    embedding_dim=512,
    num_heads=8,
    ff_dim=2048,
    dropout=0.1
)

# x: [batch_size, seq_len, embedding_dim]
output = block(x)
```

### 4. GPTModel

Полная GPT модель, объединяющая Embedding Layer и N Decoder Blocks.

**Архитектура:**
```
Token IDs → Embedding Layer → N × Decoder Blocks → 
           Final Layer Norm → Language Model Head → Logits
```

**Использование:**
```python
from TRANSFORMER import GPTModel
from EMBEDDING_LAYER import create_embedding_from_tokenizer

# С токенизатором
model = GPTModel(
    vocab_size=10000,
    embedding_dim=512,
    num_layers=12,
    num_heads=8,
    tokenizer=tokenizer
)

# Или с существующим EmbeddingLayer
embedding = create_embedding_from_tokenizer(tokenizer, embedding_dim=512)
model = GPTModel(
    vocab_size=10000,
    embedding_dim=512,
    num_layers=12,
    num_heads=8,
    embedding_layer=embedding
)

# Forward pass
token_ids = torch.tensor([[1, 2, 3, 4, 5]])  # [batch, seq_len]
logits = model(token_ids)  # [batch, seq_len, vocab_size]
```

## Сравнение с GPT-2/GPT-3

### Реализованные особенности GPT-2/GPT-3:

✅ **Pre-Norm архитектура** - Layer Normalization перед подблоками  
✅ **Causal Masking** - Авторегрессивное внимание  
✅ **GELU активация** - Вместо ReLU  
✅ **Инициализация весов** - `std=0.02` для всех Linear слоев  
✅ **Bias=False** - В проекциях внимания и FFN (как в GPT-2)  
✅ **Layer Norm eps=1e-5** - Как в GPT-2  
✅ **Residual connections** - После каждого подблока  

### Отличия от оригинального Transformer:

- **Pre-Norm** вместо Post-Norm (более стабильное обучение)
- **GELU** вместо ReLU (лучше для языкового моделирования)
- **Causal masking** для autoregressive генерации
- **Bias=False** в большинстве слоев (как в GPT-2)

## Рекомендуемые параметры

### Малая модель (для тестирования):
```python
model = GPTModel(
    vocab_size=10000,
    embedding_dim=256,
    num_layers=6,
    num_heads=8,
    ff_dim=1024,
    max_seq_len=512
)
# Параметров: ~5-10M
```

### Средняя модель:
```python
model = GPTModel(
    vocab_size=10000,
    embedding_dim=512,
    num_layers=12,
    num_heads=8,
    ff_dim=2048,
    max_seq_len=1024
)
# Параметров: ~50-100M
```

### Большая модель:
```python
model = GPTModel(
    vocab_size=10000,
    embedding_dim=768,
    num_layers=24,
    num_heads=12,
    ff_dim=3072,
    max_seq_len=2048
)
# Параметров: ~200-500M
```

## Интеграция с EmbeddingLayer

Модуль автоматически интегрируется с существующим `EMBEDDING_LAYER/`:

```python
from TRANSFORMER import GPTModel
from EMBEDDING_LAYER import create_embedding_from_tokenizer
from BPE_STUCTUR import BPETokenizer

# Загрузка токенизатора
tokenizer = BPETokenizer()
tokenizer.load("chekpoint.pkl")

# Создание модели (автоматически создаст EmbeddingLayer)
model = GPTModel(
    vocab_size=tokenizer.get_vocab_size(),
    embedding_dim=512,
    num_layers=12,
    num_heads=8,
    tokenizer=tokenizer
)
```

## Тестирование

Запуск тестов:
```bash
python TRANSFORMER/test_transformer.py
```

Тесты проверяют:
- Корректность размеров выходных тензоров
- Работу causal masking
- Residual connections
- Интеграцию с EmbeddingLayer

## Примеры использования

Запуск примеров:
```bash
python TRANSFORMER/example_usage.py
```

Примеры демонстрируют:
- Использование отдельных компонентов
- Создание полной GPT модели
- Интеграцию с токенизатором
- Стекирование блоков

## Использование в обучении

### Замена SimpleLanguageModel на GPTModel:

```python
# Было (в embedding_trainer.py):
from EMBEDDING_LAYER.embedding_trainer import SimpleLanguageModel

model = SimpleLanguageModel(embedding_layer, vocab_size, hidden_dim)

# Стало:
from TRANSFORMER import GPTModel

model = GPTModel(
    vocab_size=vocab_size,
    embedding_dim=512,
    num_layers=12,
    num_heads=8,
    embedding_layer=embedding_layer
)
```

## Архитектурные детали

### Causal Masking

Causal mask предотвращает доступ к будущим токенам:
```python
# Маска: [seq_len, seq_len]
# -inf для будущих позиций, 0 для прошлых и текущей
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
mask = mask.masked_fill(mask == 1, float('-inf'))
```

### Pre-Norm vs Post-Norm

**Pre-Norm (используется в GPT-2):**
```python
x = x + attention(norm(x))
x = x + ffn(norm(x))
```

**Post-Norm (оригинальный Transformer):**
```python
x = norm(x + attention(x))
x = norm(x + ffn(x))
```

Pre-Norm более стабилен при обучении глубоких сетей.

### Инициализация весов

Все Linear слои инициализируются как в GPT-2:
```python
nn.init.normal_(weight, mean=0.0, std=0.02)
nn.init.zeros_(bias)  # Если используется
```

## Производительность

### Память

Для модели с параметрами:
- `embedding_dim=512`, `num_layers=12`, `num_heads=8`
- `batch_size=32`, `seq_len=512`

Приблизительное использование памяти:
- **FP32**: ~2-4 GB
- **FP16**: ~1-2 GB
- **BF16**: ~1-2 GB

### Скорость

На GPU (NVIDIA RTX 3090):
- Forward pass: ~50-100 ms для batch_size=32, seq_len=512
- Backward pass: ~100-200 ms

## Оптимизации (будущие улучшения)

1. **Gradient Checkpointing** - Экономия памяти за счет пересчета
2. **Mixed Precision Training** - FP16/BF16 для ускорения
3. **Flash Attention** - Оптимизированное внимание
4. **Weight Tying** - Общие веса embedding и LM head
5. **Sparse Attention** - Для длинных последовательностей

## Следующие шаги

После реализации Transformer Decoder Block:

1. **Обучение модели** - Language Modeling задача
2. **Inference** - Генерация текста (autoregressive)
3. **Fine-tuning** - Дообучение на специфичных задачах
4. **Оптимизации** - Улучшение производительности

## Ссылки

- GPT-2 Paper: "Language Models are Unsupervised Multitask Learners"
- GPT-3 Paper: "Language Models are Few-Shot Learners"
- Transformer Paper: "Attention Is All You Need"

## Лицензия

Часть проекта tokenier для создания LLM.

---

*Модуль создан для максимальной совместимости с GPT-2/GPT-3 архитектурой*

