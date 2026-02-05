# Быстрый старт с Transformer Decoder Block

## Проверка установки

```bash
# Проверка импорта
python -c "from TRANSFORMER import GPTModel; print('✓ Модуль работает')"
```

## Базовое использование

### Создание GPT модели

```python
from TRANSFORMER import GPTModel
from BPE_STUCTUR import BPETokenizer

# Загрузка токенизатора
tokenizer = BPETokenizer()
tokenizer.load("chekpoint.pkl")

# Создание модели
model = GPTModel(
    vocab_size=tokenizer.get_vocab_size(),
    embedding_dim=512,
    num_layers=12,
    num_heads=8,
    tokenizer=tokenizer
)

# Использование
token_ids = torch.tensor([[1, 2, 3, 4, 5]])
logits = model(token_ids)  # [1, 5, vocab_size]
```

## Тестирование

```bash
python TRANSFORMER/test_transformer.py
```

## Примеры

```bash
python TRANSFORMER/example_usage.py
```

## Интеграция с обучением

Замените `SimpleLanguageModel` на `GPTModel` в вашем коде обучения.

---

Подробная документация: `TRANSFORMER/README.md`

