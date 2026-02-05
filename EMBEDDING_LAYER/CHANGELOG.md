# Changelog - Embedding Layer

## Версия 2.0.0 - Улучшенная реализация (современные LLM)

### Новые возможности:

1. **Улучшенное синусоидальное позиционное кодирование**
   - Точная реализация из "Attention Is All You Need"
   - Правильная обработка нечетных размерностей
   - Оптимизированное вычисление

2. **Обучаемое позиционное кодирование**
   - Новый класс `LearnablePositionalEncoding`
   - Альтернатива синусоидальному кодированию
   - Используется в некоторых современных моделях

3. **Layer Normalization**
   - Добавлена опция Layer Normalization (как в GPT-2/GPT-3)
   - Настраиваемый epsilon
   - Включена по умолчанию

4. **Улучшенная инициализация**
   - Инициализация весов как в GPT-2/GPT-3
   - Правильная обработка паддинга

### Изменения API:

- `EmbeddingLayer` теперь поддерживает параметры:
  - `learnable_pos`: выбор типа позиционного кодирования
  - `layer_norm`: включение/выключение Layer Normalization
  - `layer_norm_eps`: настройка epsilon для Layer Norm

- `create_embedding_from_tokenizer()` обновлена с новыми параметрами

### Обратная совместимость:

✅ Полная обратная совместимость с версией 1.0.0
✅ Все старые вызовы работают без изменений
✅ Новые параметры имеют разумные значения по умолчанию

### Примеры использования:

```python
# Синусоидальное кодирование с Layer Norm (по умолчанию, как GPT-2)
embedding = create_embedding_from_tokenizer(tokenizer, embedding_dim=256)

# Обучаемое позиционное кодирование
embedding = create_embedding_from_tokenizer(
    tokenizer, 
    embedding_dim=256,
    learnable_pos=True
)

# Без Layer Norm
embedding = create_embedding_from_tokenizer(
    tokenizer,
    embedding_dim=256,
    layer_norm=False
)
```

---

## Версия 1.0.0 - Первая реализация

- Базовый Token Embedding
- Синусоидальное позиционное кодирование
- Базовая интеграция с токенизатором

