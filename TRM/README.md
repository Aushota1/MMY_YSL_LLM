# Tiny Recursive Model (TRM)

Реализация Tiny Recursive Model на основе статьи "Less is More: Recursive Reasoning with Tiny Networks".

## Описание

TRM - это рекурсивная модель рассуждения, которая использует очень маленькую сеть (2 слоя, ~7M параметров) с рекурсивным применением для решения сложных задач рассуждения.

## Структура модулей

```
TRM/
├── __init__.py                  # Экспорт всех компонентов
├── utils.py                     # Утилиты: RMSNorm, SwiGLU, Rotary Embeddings
├── tiny_recursive_network.py   # Маленькая сеть (2 слоя)
├── latent_recursion.py          # Рекурсивное обновление латентного состояния
├── output_refinement.py         # Улучшение ответа
├── heads.py                     # OutputHead, QHead
├── losses.py                    # StableMaxLoss, BinaryCE
├── deep_supervision.py          # Deep supervision для обучения
├── trm_model.py                 # Полная TRM модель
├── test_trm.py                 # Тесты
└── README.md                    # Этот файл
```

## Использование

### Базовое использование

```python
from TRM import TRMModel
from BPE_STUCTUR import BPETokenizer

# Загрузка токенизатора
tokenizer = BPETokenizer()
tokenizer.load("chekpoint.pkl")

# Создание модели
model = TRMModel(
    vocab_size=tokenizer.get_vocab_size(),
    embedding_dim=512,
    hidden_dim=512,
    tokenizer=tokenizer
)

# Forward pass
x_input = torch.randint(0, vocab_size, (2, 10))
y_hat, q_hat = model(x_input, n=6, T=3, N_sup=16)

# Генерация ответа
tokens = model.generate_answer(x_input, max_steps=16)
```

### Обучение

```bash
# Интерактивный режим
python train_trm.py --interactive

# Командная строка
python train_trm.py \
    --tokenizer chekpoint.pkl \
    --data data.txt \
    --epochs 10 \
    --batch-size 8 \
    --lr 1e-4 \
    --embedding-lr 1e-2 \
    --n 6 \
    --T 3 \
    --N-sup 16
```

## Параметры

### Гиперпараметры модели

- `embedding_dim`: Размерность эмбеддингов (по умолчанию 512)
- `hidden_dim`: Размерность скрытого слоя (по умолчанию 512)
- `max_seq_len`: Максимальная длина последовательности (по умолчанию 512)

### Гиперпараметры обучения

- `n`: Количество рекурсивных шагов (по умолчанию 6)
- `T`: Количество глубоких рекурсий (по умолчанию 3)
- `N_sup`: Максимальное количество шагов супервизии (по умолчанию 16)
- `learning_rate`: Learning rate для основных параметров (по умолчанию 1e-4)
- `embedding_lr`: Learning rate для embeddings (по умолчанию 1e-2)

## Отслеживание бенчмарков

Модель автоматически отслеживает метрики во время обучения:

- Loss на каждом шаге
- Accuracy
- Early stopping rate
- Количество шагов
- Время выполнения

Метрики сохраняются в `trm_benchmarks/benchmark_YYYYMMDD_HHMMSS.json`

## Тестирование

```bash
python TRM/test_trm.py
```

## Особенности

- ✅ Рекурсивное рассуждение с маленькой сетью
- ✅ Deep supervision для улучшения обучения
- ✅ Early stopping для экономии вычислений
- ✅ Стабильная функция потерь (stable-max loss)
- ✅ Отслеживание бенчмарков
- ✅ Интеграция с существующим EmbeddingLayer и BPETokenizer

## Сравнение с GPT

| Характеристика | GPT | TRM |
|----------------|-----|-----|
| Параметры | 10-100M+ | ~7M |
| Слои | 4-12 | 2 |
| Рекурсия | Нет | Да |
| Deep Supervision | Нет | Да |
| Early Stopping | Нет | Да |
| Задачи | Language Modeling | Reasoning |

## Литература

- "Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871)

