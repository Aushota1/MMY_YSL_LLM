# 🧪 GPT Tester - Интерфейс для тестирования и оценки GPT модели

## Описание

`gpt_tester.py` - полнофункциональный интерфейс для тестирования и оценки обученной GPT модели. Включает генерацию текста, вычисление метрик качества и интерактивный режим.

## Возможности

### ✨ Основные функции:

1. **Генерация текста** с различными стратегиями:
   - Greedy decoding
   - Top-k sampling
   - Top-p (nucleus) sampling
   - Temperature control
   - Repetition penalty

2. **Оценка качества модели**:
   - **Perplexity** - основная метрика для language models
   - **Token Accuracy** - точность предсказания следующего токена
   - **Sequence Accuracy** - точность предсказания всей последовательности

3. **Интерактивный режим** для тестирования генерации

4. **Оценка на файле** с сохранением отчета

## Использование

### Базовый запуск

```bash
python TRANSFORMER/gpt_tester.py
```

Или из Python кода:

```python
from TRANSFORMER.gpt_tester import GPTTester

# Создание тестера
tester = GPTTester(
    model_path="gpt_model_checkpoint.pth",
    tokenizer_path="chekpoint.pkl"
)

# Генерация текста
generated = tester.generate("Машинное обучение - это")
print(generated)

# Оценка на файле
report = tester.evaluate_on_file("test_data.txt", save_report="evaluation_report.json")
print(f"Perplexity: {report['perplexity']:.2f}")
```

## Методы класса GPTTester

### `generate()` - Генерация текста

```python
generated = tester.generate(
    prompt="Машинное обучение",
    max_length=100,              # Максимальная длина
    temperature=1.0,             # Температура (0.1-2.0)
    top_k=50,                    # Top-k sampling
    top_p=0.9,                   # Nucleus sampling
    repetition_penalty=1.0,      # Штраф за повторения
    stop_tokens=None             # Токены для остановки
)
```

**Параметры:**
- `prompt`: Начальный текст (промпт)
- `max_length`: Максимальная длина генерируемого текста
- `temperature`: Контроль случайности (ниже = более детерминировано)
- `top_k`: Количество топ токенов для рассмотрения (0 = все)
- `top_p`: Cumulative probability для nucleus sampling (0.0-1.0)
- `repetition_penalty`: Штраф за повторения (>1.0 уменьшает повторения)
- `stop_tokens`: Список токенов для остановки генерации

**Примеры:**

```python
# Детерминированная генерация
text = tester.generate("Привет", temperature=0.1, top_k=1)

# Творческая генерация
text = tester.generate("Однажды", temperature=1.5, top_k=100)

# Без повторений
text = tester.generate("Расскажи", repetition_penalty=1.2)
```

### `calculate_perplexity()` - Вычисление Perplexity

```python
perplexity = tester.calculate_perplexity(
    texts=["Текст 1", "Текст 2", ...],
    batch_size=8,
    max_length=512
)
```

**Perplexity** - основная метрика для language models:
- Чем меньше, тем лучше
- Perplexity = exp(average_loss)
- Хорошие значения: 10-100 (зависит от задачи)

### `calculate_accuracy()` - Вычисление Accuracy

```python
accuracy = tester.calculate_accuracy(
    texts=["Текст 1", "Текст 2", ...],
    batch_size=8,
    max_length=512
)

print(f"Token Accuracy: {accuracy['token_accuracy']:.4f}")
print(f"Sequence Accuracy: {accuracy['sequence_accuracy']:.4f}")
```

**Метрики:**
- `token_accuracy`: Процент правильно предсказанных токенов
- `sequence_accuracy`: Процент полностью правильных последовательностей

### `evaluate_on_file()` - Полная оценка на файле

```python
report = tester.evaluate_on_file(
    file_path="test_data.txt",
    max_lines=1000,                    # Максимум строк
    save_report="evaluation.json"     # Сохранение отчета
)
```

**Возвращает словарь:**
```json
{
  "perplexity": 45.23,
  "token_accuracy": 0.3245,
  "sequence_accuracy": 0.0123,
  "num_texts": 1000,
  "vocab_size": 10000,
  "model_params": 9850000
}
```

### `interactive_mode()` - Интерактивный режим

```python
tester.interactive_mode()
```

Интерактивный режим для тестирования генерации:
- Ввод промпта и получение результата
- Изменение параметров генерации на лету
- Команды: `exit`, `quit`, `params`, `set <param> <value>`

## Примеры использования

### Пример 1: Простая генерация

```python
from TRANSFORMER.gpt_tester import GPTTester

tester = GPTTester("gpt_model_checkpoint.pth", "chekpoint.pkl")

# Генерация
text = tester.generate("Машинное обучение - это", max_length=50)
print(text)
```

### Пример 2: Оценка модели

```python
from TRANSFORMER.gpt_tester import GPTTester

tester = GPTTester("gpt_model_checkpoint.pth", "chekpoint.pkl")

# Оценка на тестовом файле
report = tester.evaluate_on_file(
    "test_data.txt",
    max_lines=500,
    save_report="evaluation_report.json"
)

print(f"Perplexity: {report['perplexity']:.2f}")
print(f"Token Accuracy: {report['token_accuracy']*100:.2f}%")
```

### Пример 3: Сравнение параметров генерации

```python
prompt = "Искусственный интеллект"

# Детерминированная генерация
text1 = tester.generate(prompt, temperature=0.1, top_k=1)
print(f"Детерминированная: {text1}")

# Творческая генерация
text2 = tester.generate(prompt, temperature=1.5, top_k=100)
print(f"Творческая: {text2}")

# С nucleus sampling
text3 = tester.generate(prompt, top_p=0.9, temperature=1.0)
print(f"Nucleus sampling: {text3}")
```

### Пример 4: Пакетная оценка

```python
# Загрузка текстов
texts = []
with open("test_data.txt", "r", encoding="utf-8") as f:
    for line in f:
        texts.append(line.strip())

# Вычисление метрик
perplexity = tester.calculate_perplexity(texts, batch_size=16)
accuracy = tester.calculate_accuracy(texts, batch_size=16)

print(f"Perplexity: {perplexity:.2f}")
print(f"Token Accuracy: {accuracy['token_accuracy']*100:.2f}%")
```

## Интерфейс командной строки

При запуске `python TRANSFORMER/gpt_tester.py` доступно меню:

```
1. Интерактивная генерация текста
2. Оценка на файле (Perplexity + Accuracy)
3. Быстрый тест генерации
4. Выход
```

### Интерактивный режим

В интерактивном режиме доступны команды:

- `exit` / `quit` - выход
- `params` - показать текущие параметры
- `set <param> <value>` - изменить параметр

Пример:
```
💬 Промпт: set temperature 1.5
✓ Параметр temperature установлен в 1.5

💬 Промпт: Машинное обучение
🤖 Генерация...
📝 Результат: Машинное обучение - это подраздел...
```

## Параметры генерации

### Temperature (0.1 - 2.0)

- **Низкая (0.1-0.5)**: Детерминированная генерация, повторяющиеся паттерны
- **Средняя (0.7-1.0)**: Баланс между детерминированностью и творчеством
- **Высокая (1.5-2.0)**: Творческая генерация, больше разнообразия

### Top-k (1 - vocab_size)

- **Низкая (1-10)**: Более детерминированная генерация
- **Средняя (20-50)**: Баланс
- **Высокая (100+)**: Более разнообразная генерация

### Top-p / Nucleus Sampling (0.0 - 1.0)

- **Низкая (0.1-0.5)**: Более детерминированная
- **Средняя (0.7-0.9)**: Баланс
- **Высокая (0.95-1.0)**: Более разнообразная

### Repetition Penalty (0.5 - 2.0)

- **< 1.0**: Увеличивает повторения
- **1.0**: Нейтрально
- **> 1.0**: Уменьшает повторения (рекомендуется 1.1-1.3)

## Метрики качества

### Perplexity

**Что это:**
- Мера "удивления" модели при предсказании следующего токена
- Perplexity = exp(average_loss)

**Интерпретация:**
- **< 10**: Отличная модель
- **10-50**: Хорошая модель
- **50-100**: Приемлемая модель
- **> 100**: Модель нуждается в улучшении

**Зависит от:**
- Размера словаря
- Сложности задачи
- Качества данных

### Token Accuracy

**Что это:**
- Процент правильно предсказанных токенов

**Интерпретация:**
- **> 50%**: Отличная модель
- **30-50%**: Хорошая модель
- **10-30%**: Приемлемая модель
- **< 10%**: Модель нуждается в улучшении

### Sequence Accuracy

**Что это:**
- Процент полностью правильных последовательностей

**Интерпретация:**
- Обычно очень низкая для language models (это нормально)
- Полезно для сравнения моделей

## Рекомендации

### Для тестирования генерации:

1. Начните с низкой temperature (0.7-1.0)
2. Используйте top-k=50 и top-p=0.9
3. Установите repetition_penalty=1.1-1.2
4. Экспериментируйте с параметрами

### Для оценки качества:

1. Используйте отдельный тестовый набор данных
2. Вычисляйте perplexity на достаточно большом датасете (1000+ текстов)
3. Сравнивайте метрики между разными моделями
4. Сохраняйте отчеты для отслеживания прогресса

### Для отладки:

1. Проверьте, что модель загружена правильно
2. Убедитесь, что токенизатор совместим с моделью
3. Проверьте размеры входных данных
4. Используйте интерактивный режим для быстрого тестирования

## Устранение проблем

### Ошибка: "CUDA out of memory"

→ Уменьшите `batch_size` в методах оценки

### Ошибка: "Model and tokenizer mismatch"

→ Убедитесь, что используете правильный токенизатор, на котором обучалась модель

### Генерация слишком повторяющаяся

→ Увеличьте `temperature` или `repetition_penalty`

### Генерация слишком случайная

→ Уменьшите `temperature` или используйте `top_k=1` для greedy decoding

## Следующие шаги

После тестирования модели:

1. Анализируйте метрики и сравнивайте с другими моделями
2. Экспериментируйте с параметрами генерации
3. Fine-tune модель на специфичных задачах
4. Оптимизируйте для production использования

