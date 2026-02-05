# 🚀 Быстрый старт: Обучение GPT модели

## Описание

`train_gpt.py` - простой скрипт для тестового обучения GPT модели на ваших данных.

## Требования

1. **Обученный токенизатор** - файл `chekpoint.pkl` (создается через `tokenizer_trainer.py`)
2. **Текстовые данные** - файл с текстом для обучения (например, `data_1.txt`)

## Использование

### Базовый запуск

```bash
python train_gpt.py
```

### Настройка параметров

Откройте `train_gpt.py` и измените параметры в функции `main()`:

```python
# Пути к файлам
tokenizer_path = "chekpoint.pkl"      # Путь к токенизатору
data_file = "data_1.txt"              # Путь к данным

# Параметры данных
max_lines = 1000                       # Максимум строк для загрузки
max_length = 128                       # Длина последовательности
batch_size = 4                         # Размер батча

# Параметры обучения
num_epochs = 3                         # Количество эпох
learning_rate = 3e-4                   # Скорость обучения

# Параметры модели (малая модель для теста)
embedding_dim = 256                     # Размерность embeddings
num_layers = 4                         # Количество слоев
num_heads = 8                          # Количество голов внимания
max_seq_len = 512                      # Максимальная длина последовательности
```

## Параметры модели

### Малая модель (для теста)
- `embedding_dim`: 256
- `num_layers`: 4
- `num_heads`: 8
- Параметров: ~5-10M

### Средняя модель
- `embedding_dim`: 512
- `num_layers`: 8
- `num_heads`: 8
- Параметров: ~30-50M

### Большая модель
- `embedding_dim`: 768
- `num_layers`: 12
- `num_heads`: 12
- Параметров: ~100M+

## Что делает скрипт

1. ✅ Загружает токенизатор из `chekpoint.pkl`
2. ✅ Загружает текстовые данные из файла
3. ✅ Создает Dataset и DataLoader
4. ✅ Создает GPT модель с указанными параметрами
5. ✅ Обучает модель на данных
6. ✅ Сохраняет чекпоинт после каждой эпохи в `gpt_model_checkpoint.pth`

## Выходные файлы

- `gpt_model_checkpoint.pth` - чекпоинт модели после обучения

Чекпоинт содержит:
- `model_state_dict` - веса модели
- `optimizer_state_dict` - состояние оптимизатора
- `epoch` - номер эпохи
- `loss` - значение loss
- Параметры модели (vocab_size, embedding_dim, и т.д.)

## Использование обученной модели

```python
import torch
from TRANSFORMER import GPTModel
from BPE_STUCTUR import BPETokenizer

# Загрузка токенизатора
tokenizer = BPETokenizer()
tokenizer.load("chekpoint.pkl")

# Загрузка чекпоинта
checkpoint = torch.load("gpt_model_checkpoint.pth")

# Создание модели
model = GPTModel(
    vocab_size=checkpoint['vocab_size'],
    embedding_dim=checkpoint['embedding_dim'],
    num_layers=checkpoint['num_layers'],
    num_heads=checkpoint['num_heads'],
    tokenizer=tokenizer
)

# Загрузка весов
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Использование
text = "Машинное обучение"
token_ids = tokenizer.encode(text)
input_ids = torch.tensor([token_ids])

with torch.no_grad():
    logits = model(input_ids)
    # logits: [1, seq_len, vocab_size]
```

## Прерывание обучения

Нажмите `Ctrl+C` для прерывания. Последний чекпоинт будет сохранен.

## Рекомендации

1. **Для первого теста**: Используйте малую модель и небольшой датасет
2. **Для реального обучения**: Увеличьте `num_epochs`, `batch_size`, и размер модели
3. **GPU**: Скрипт автоматически использует GPU, если доступен
4. **Память**: Если не хватает памяти, уменьшите `batch_size` или `max_length`

## Примеры использования

### Тест на маленьком датасете
```python
max_lines = 100
num_epochs = 2
embedding_dim = 128
num_layers = 2
```

### Обучение на большом датасете
```python
max_lines = None  # Загрузить все строки
num_epochs = 10
embedding_dim = 512
num_layers = 12
batch_size = 16
```

## Устранение проблем

### Ошибка: "файл chekpoint.pkl не найден"
→ Сначала обучите токенизатор через `tokenizer_trainer.py`

### Ошибка: "CUDA out of memory"
→ Уменьшите `batch_size` или `max_length`

### Ошибка: "нет данных для обучения"
→ Проверьте путь к файлу данных и его содержимое

## Следующие шаги

После обучения модели:
1. Протестируйте модель на новых данных
2. Реализуйте генерацию текста (inference)
3. Fine-tune на специфичных задачах
4. Экспериментируйте с гиперпараметрами

