# План миграции проекта в Tiny Recursive Model (TRM)

## Введение

Этот документ описывает различия между текущим GPT-проектом и архитектурой Tiny Recursive Model (TRM) из статьи "Less is More: Recursive Reasoning with Tiny Networks", а также содержит план реализации переделки проекта.

---

## 1. Архитектурные различия

### 1.1 Текущий проект (GPT-подобная архитектура)

**Архитектура:**
```
Input (token_ids) 
  → Embedding Layer 
  → N × Transformer Decoder Blocks (4-12 слоев)
  → Final Layer Norm 
  → Language Model Head 
  → Output (logits)
```

**Характеристики:**
- **Параметры**: Зависит от размера (обычно 10-100M+ параметров)
- **Архитектура**: Feedforward через последовательные слои
- **Обучение**: Один forward pass, один backward pass
- **Сеть**: Большая глубокая сеть (много слоев)
- **Рекурсия**: Отсутствует
- **Deep Supervision**: Отсутствует

**Ключевые компоненты:**
- `EmbeddingLayer`: Токенные и позиционные эмбеддинги
- `TransformerDecoderBlock`: Multi-head attention + FFN
- `GPTModel`: Объединяет все компоненты
- Стандартный CrossEntropyLoss

### 1.2 TRM (Tiny Recursive Model)

**Архитектура:**
```
Input (x) → Embedding
Initial: y (answer), z (latent)

For each supervision step (N_sup = 16):
  For T-1 times (no gradients):
    z = latent_recursion(x, y, z, n=6)  # Рекурсивное обновление z
    y = refine_output(y, z)             # Улучшение ответа
  
  For 1 time (with gradients):
    z = latent_recursion(x, y, z, n=6)
    y = refine_output(y, z)
    loss = stable_max_loss(y_hat, y_true) + binary_ce(q_hat, correct)
  
  z = z.detach()  # Отсоединение градиентов
  if q[0] > 0: break  # Early stopping
```

**Характеристики:**
- **Параметры**: Всего 7M параметров (2 слоя)
- **Архитектура**: Рекурсивная (одна маленькая сеть применяется много раз)
- **Обучение**: Deep supervision (множественные шаги улучшения)
- **Сеть**: Одна крошечная сеть (2 слоя)
- **Рекурсия**: n=6 шагов рекурсии, T=3 глубоких рекурсии
- **Deep Supervision**: N_sup=16 шагов супервизии

**Ключевые компоненты:**
- `InputEmbedding`: Встраивание вопроса
- `TinyRecursiveNetwork`: Маленькая сеть (2 слоя) для рекурсивного рассуждения
- `LatentRecursion`: Рекурсивное обновление латентного состояния z
- `OutputRefinement`: Улучшение ответа y на основе z
- `DeepSupervision`: Множественные шаги улучшения с градиентами
- `StableMaxLoss`: Стабильная функция потерь
- `QHead`: Голова для предсказания правильности ответа (early stopping)

---

## 2. Плюсы и минусы

### 2.1 Текущий проект (GPT)

**Плюсы:**
- ✅ Стандартная архитектура, хорошо изучена
- ✅ Простая реализация и отладка
- ✅ Хорошо работает для языкового моделирования
- ✅ Много готовых решений и примеров
- ✅ Прямой forward pass, легко понять поток данных
- ✅ Эффективное использование GPU (batch processing)

**Минусы:**
- ❌ Требует много параметров для хорошего качества
- ❌ Может переобучаться на малых данных
- ❌ Не использует рекурсивное рассуждение
- ❌ Один проход = один ответ (нет итеративного улучшения)
- ❌ Плохо работает на сложных задачах рассуждения (ARC-AGI, Sudoku)

### 2.2 TRM

**Плюсы:**
- ✅ Очень мало параметров (7M vs 100M+)
- ✅ Отличная обобщающая способность на малых данных (~1000 примеров)
- ✅ Рекурсивное рассуждение позволяет итеративно улучшать ответ
- ✅ Deep supervision помогает обучению
- ✅ Превосходит LLM на задачах рассуждения (ARC-AGI, Sudoku, Maze)
- ✅ Early stopping экономит вычисления
- ✅ Параметро-эффективный подход

**Минусы:**
- ❌ Сложнее реализация (рекурсия, градиенты)
- ❌ Медленнее обучение (множественные шаги)
- ❌ Больше памяти (нужно хранить промежуточные состояния)
- ❌ Менее изученная архитектура
- ❌ Может быть нестабильным при неправильной настройке
- ❌ Сложнее отладка (рекурсивные шаги)

---

## 3. План реализации

### Этап 1: Создание базовых модулей TRM

#### 3.1.1 Модуль: `TRM/tiny_recursive_network.py`

**Назначение**: Маленькая сеть (2 слоя) для рекурсивного рассуждения

**Компоненты:**
- `TinyRecursiveNetwork`: 2-слойная сеть с RMSNorm, SwiGLU, rotary embeddings
- Вход: `(x, y, z)` - вопрос, ответ, латентное состояние
- Выход: Обновленные `z` или `y`

**Технические детали:**
- 2 слоя (как в статье)
- Hidden size: 512
- RMSNorm (без bias)
- SwiGLU активация
- Rotary embeddings для позиций

#### 3.1.2 Модуль: `TRM/latent_recursion.py`

**Назначение**: Рекурсивное обновление латентного состояния

**Функции:**
- `latent_recursion(x, y, z, n=6)`: n шагов рекурсивного обновления z
- `deep_recursion(x, y, z, n=6, T=3)`: T глубоких рекурсий (T-1 без градиентов, 1 с градиентами)

**Логика:**
```python
def latent_recursion(x, y, z, n=6):
    for i in range(n):
        z = net(x, y, z)  # Обновление z
    return z

def deep_recursion(x, y, z, n=6, T=3):
    # T-1 раз без градиентов
    with torch.no_grad():
        for j in range(T-1):
            z = latent_recursion(x, y, z, n)
            y = refine_output(y, z)
    
    # 1 раз с градиентами
    z = latent_recursion(x, y, z, n)
    y = refine_output(y, z)
    return y.detach(), z.detach(), output_head(y), q_head(y)
```

#### 3.1.3 Модуль: `TRM/output_refinement.py`

**Назначение**: Улучшение ответа на основе латентного состояния

**Компоненты:**
- `OutputRefinement`: Сеть для обновления ответа y
- Использует ту же архитектуру, что и `TinyRecursiveNetwork`
- Вход: `(y, z)` - текущий ответ и латентное состояние
- Выход: Обновленный `y`

#### 3.1.4 Модуль: `TRM/heads.py`

**Назначение**: Выходные головы для предсказания

**Компоненты:**
- `OutputHead`: Предсказание ответа (softmax over vocab)
- `QHead`: Предсказание правильности ответа (binary classification для early stopping)

### Этап 2: Deep Supervision и обучение

#### 3.2.1 Модуль: `TRM/deep_supervision.py`

**Назначение**: Реализация deep supervision для обучения

**Компоненты:**
- `DeepSupervisionTrainer`: Класс для обучения с deep supervision
- Методы:
  - `forward_step(x, y_init, z_init)`: Один шаг супервизии
  - `compute_loss(y_hat, y_true, q_hat)`: Вычисление loss (stable-max + binary CE)
  - `early_stopping(q_hat)`: Проверка условия early stopping

**Логика обучения:**
```python
for step in range(N_sup=16):
    y, z, y_hat, q_hat = deep_recursion(x, y, z)
    loss = stable_max_loss(y_hat, y_true) + binary_ce(q_hat, (y_hat == y_true))
    loss.backward()
    optimizer.step()
    z = z.detach()  # Отсоединение для следующего шага
    if q_hat[0] > 0:  # Early stopping
        break
```

#### 3.2.2 Модуль: `TRM/losses.py`

**Назначение**: Специальные функции потерь

**Компоненты:**
- `StableMaxLoss`: Стабильная функция потерь (из статьи)
- `BinaryCrossEntropy`: Для Q-head (early stopping)

### Этап 3: Основная модель TRM

#### 3.3.1 Модуль: `TRM/trm_model.py`

**Назначение**: Полная модель TRM

**Компоненты:**
- `TRMModel`: Главный класс модели
- Интеграция всех компонентов:
  - Input embedding
  - Tiny recursive network
  - Latent recursion
  - Output refinement
  - Deep supervision
  - Output heads

**Архитектура:**
```python
class TRMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, ...):
        self.input_embedding = InputEmbedding(...)
        self.tiny_net = TinyRecursiveNetwork(...)
        self.output_refinement = OutputRefinement(...)
        self.output_head = OutputHead(...)
        self.q_head = QHead(...)
    
    def forward(self, x_input, y_init=None, z_init=None):
        x = self.input_embedding(x_input)
        # Deep supervision loop
        ...
```

### Этап 4: Интеграция с существующим проектом

#### 3.4.1 Адаптация Embedding Layer

**Изменения:**
- Использовать существующий `EmbeddingLayer` для input embedding
- Добавить поддержку инициализации `y` и `z` эмбеддингов
- Возможно, создать отдельный `InputEmbedding` на основе существующего

#### 3.4.2 Новый тренировочный скрипт

**Файл**: `train_trm.py`

**Функционал:**
- Загрузка данных (совместимо с существующим форматом)
- Создание TRM модели
- Обучение с deep supervision
- Сохранение чекпоинтов
- Интеграция с существующим токенизатором

#### 3.4.3 Универсальная модель

**Опция**: Создать `UnifiedModel`, который поддерживает оба режима:
- Режим GPT: Стандартный forward pass
- Режим TRM: Рекурсивное рассуждение с deep supervision

---

## 4. Новые модули и файлы

### 4.1 Структура директорий

```
tokenier/
├── TRANSFORMER/          # Существующий (GPT)
│   ├── gpt_model.py
│   └── ...
├── TRM/                  # НОВЫЙ: Tiny Recursive Model
│   ├── __init__.py
│   ├── tiny_recursive_network.py    # Маленькая сеть (2 слоя)
│   ├── latent_recursion.py          # Рекурсивное обновление z
│   ├── output_refinement.py         # Улучшение ответа y
│   ├── heads.py                     # OutputHead, QHead
│   ├── deep_supervision.py          # Deep supervision логика
│   ├── losses.py                    # StableMaxLoss, BinaryCE
│   ├── trm_model.py                 # Полная TRM модель
│   └── README.md
├── EMBEDDING_LAYER/      # Существующий
├── train_gpt.py          # Существующий
├── train_trm.py          # НОВЫЙ: Обучение TRM
└── ...
```

### 4.2 Детальное описание модулей

#### 4.2.1 `TRM/tiny_recursive_network.py`

**Классы:**
- `TinyRecursiveNetwork(nn.Module)`
  - `__init__(embedding_dim=512, hidden_dim=512, ...)`
  - `forward(x, y, z)` → обновленный z или y

**Технические детали:**
- 2 слоя
- RMSNorm (без bias)
- SwiGLU активация
- Rotary embeddings
- Dropout для регуляризации

#### 4.2.2 `TRM/latent_recursion.py`

**Функции:**
- `latent_recursion(x, y, z, net, n=6)`: n шагов рекурсии
- `deep_recursion(x, y, z, net, refine_net, n=6, T=3)`: T глубоких рекурсий

**Параметры:**
- `n`: Количество рекурсивных шагов (по умолчанию 6)
- `T`: Количество глубоких рекурсий (по умолчанию 3)

#### 4.2.3 `TRM/output_refinement.py`

**Классы:**
- `OutputRefinement(nn.Module)`: Сеть для улучшения ответа
  - Аналогична `TinyRecursiveNetwork`
  - Вход: `(y, z)`
  - Выход: обновленный `y`

#### 4.2.4 `TRM/heads.py`

**Классы:**
- `OutputHead(nn.Module)`: Предсказание ответа
  - Linear layer: `embedding_dim → vocab_size`
  - Softmax для вероятностей
  
- `QHead(nn.Module)`: Предсказание правильности
  - Linear layer: `embedding_dim → 1`
  - Sigmoid для бинарной классификации

#### 4.2.5 `TRM/deep_supervision.py`

**Классы:**
- `DeepSupervisionTrainer`: Обучение с deep supervision
  - `train_step(x, y_true, y_init, z_init)`: Один шаг обучения
  - `forward_step(x, y, z)`: Forward pass с deep recursion

**Параметры:**
- `N_sup`: Максимальное количество шагов супервизии (16)
- `n`: Количество рекурсивных шагов (6)
- `T`: Количество глубоких рекурсий (3)

#### 4.2.6 `TRM/losses.py`

**Функции:**
- `stable_max_loss(y_hat, y_true)`: Стабильная функция потерь
- `binary_cross_entropy(q_hat, is_correct)`: Для Q-head

**Детали:**
- Stable-max loss из статьи (улучшенная стабильность)
- Binary cross-entropy для early stopping

#### 4.2.7 `TRM/trm_model.py`

**Классы:**
- `TRMModel(nn.Module)`: Полная модель TRM
  - Интеграция всех компонентов
  - Forward pass с deep supervision
  - Поддержка early stopping

**Методы:**
- `forward(x_input, y_init=None, z_init=None)`: Forward pass
- `generate_answer(x_input)`: Генерация ответа
- `get_num_params()`: Подсчет параметров

---

## 5. План реализации по этапам

### Этап 1: Подготовка (1-2 дня)

1. ✅ Создать директорию `TRM/`
2. ✅ Изучить детали реализации из статьи
3. ✅ Подготовить интерфейсы модулей

### Этап 2: Базовые компоненты (3-5 дней)

1. ✅ Реализовать `TinyRecursiveNetwork` (2 слоя, RMSNorm, SwiGLU)
2. ✅ Реализовать `LatentRecursion` (рекурсивное обновление)
3. ✅ Реализовать `OutputRefinement` (улучшение ответа)
4. ✅ Реализовать `OutputHead` и `QHead`
5. ✅ Unit тесты для каждого компонента

### Этап 3: Deep Supervision (2-3 дня)

1. ✅ Реализовать `DeepSupervisionTrainer`
2. ✅ Реализовать `StableMaxLoss`
3. ✅ Интегрировать early stopping
4. ✅ Тесты для deep supervision

### Этап 4: Полная модель (2-3 дня)

1. ✅ Реализовать `TRMModel`
2. ✅ Интегрировать с существующим `EmbeddingLayer`
3. ✅ Поддержка инициализации y и z
4. ✅ Тесты для полной модели

### Этап 5: Обучение (2-3 дня)

1. ✅ Создать `train_trm.py`
2. ✅ Интеграция с существующим токенизатором
3. ✅ Поддержка загрузки данных
4. ✅ Сохранение/загрузка чекпоинтов
5. ✅ Логирование и мониторинг

### Этап 6: Тестирование и оптимизация (3-5 дней)

1. ✅ Тесты на простых задачах
2. ✅ Сравнение с GPT моделью
3. ✅ Оптимизация гиперпараметров
4. ✅ Документация и примеры

**Общее время: 13-21 день**

---

## 6. Технические детали реализации

### 6.1 Гиперпараметры TRM

Из статьи:
- `embedding_dim`: 512
- `hidden_dim`: 512
- `n`: 6 (количество рекурсивных шагов)
- `T`: 3 (количество глубоких рекурсий)
- `N_sup`: 16 (максимальное количество шагов супервизии)
- `learning_rate`: 1e-4 (для embeddings: 1e-2)
- `weight_decay`: 0.1 (для ARC-AGI), 1.0 (для Sudoku/Maze)
- `batch_size`: 768
- `beta1`: 0.9, `beta2`: 0.95
- `EMA`: 0.999 (Exponential Moving Average)

### 6.2 Архитектура TinyRecursiveNetwork

```
Input: (x, y, z) - [B, L, D]
  ↓
RMSNorm
  ↓
Linear (D → hidden_dim)
  ↓
SwiGLU activation
  ↓
Dropout
  ↓
Linear (hidden_dim → D)
  ↓
RMSNorm
  ↓
Output: z или y - [B, L, D]
```

**SwiGLU формула:**
```
SwiGLU(x) = Swish(xW + b) ⊙ (xV + c)
где Swish(x) = x * sigmoid(x)
```

### 6.3 Рекурсивная логика

**Latent Recursion:**
```python
def latent_recursion(x, y, z, net, n=6):
    for i in range(n):
        z = net(x, y, z)  # Обновление z
    return z
```

**Deep Recursion:**
```python
def deep_recursion(x, y, z, net, refine_net, n=6, T=3):
    # T-1 раз без градиентов (warm-up)
    with torch.no_grad():
        for j in range(T-1):
            z = latent_recursion(x, y, z, net, n)
            y = refine_net(y, z)
    
    # 1 раз с градиентами
    z = latent_recursion(x, y, z, net, n)
    y = refine_net(y, z)
    
    # Выходные головы
    y_hat = output_head(y)
    q_hat = q_head(y)
    
    return y.detach(), z.detach(), y_hat, q_hat
```

### 6.4 Deep Supervision

```python
for step in range(N_sup=16):
    # Deep recursion
    y, z, y_hat, q_hat = deep_recursion(x, y, z, ...)
    
    # Loss
    loss = stable_max_loss(y_hat, y_true)
    loss += binary_ce(q_hat, (y_hat == y_true))
    
    # Backward
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Detach для следующего шага
    z = z.detach()
    
    # Early stopping
    if q_hat[0] > 0:  # Правильный ответ найден
        break
```

---

## 7. Интеграция с существующим проектом

### 7.1 Использование существующих компонентов

**Можно переиспользовать:**
- ✅ `EmbeddingLayer` для input embedding
- ✅ `BPETokenizer` для токенизации
- ✅ Существующие утилиты загрузки данных
- ✅ Система чекпоинтов

**Нужно создать:**
- ❌ `TinyRecursiveNetwork` (новая архитектура)
- ❌ `LatentRecursion` (новая логика)
- ❌ `DeepSupervision` (новый подход к обучению)
- ❌ `StableMaxLoss` (новая функция потерь)

### 7.2 Универсальная архитектура (опционально)

Можно создать `UnifiedModel`, который поддерживает оба режима:

```python
class UnifiedModel(nn.Module):
    def __init__(self, mode='gpt', ...):
        if mode == 'gpt':
            self.model = GPTModel(...)
        elif mode == 'trm':
            self.model = TRMModel(...)
    
    def forward(self, x, mode=None):
        if mode == 'gpt' or isinstance(self.model, GPTModel):
            return self.model(x)  # Стандартный forward
        elif mode == 'trm' or isinstance(self.model, TRMModel):
            return self.model(x, y_init, z_init)  # TRM forward
```

---

## 8. Тестирование

### 8.1 Unit тесты

Для каждого модуля:
- `test_tiny_recursive_network.py`
- `test_latent_recursion.py`
- `test_output_refinement.py`
- `test_deep_supervision.py`
- `test_trm_model.py`

### 8.2 Интеграционные тесты

- Тест на простых задачах (например, арифметика)
- Сравнение с GPT моделью
- Тест на малых данных (~1000 примеров)

### 8.3 Бенчмарки

- Sudoku-Extreme
- Maze-Hard
- ARC-AGI-1
- ARC-AGI-2

---

## 9. Риски и митигация

### 9.1 Риски

1. **Сложность реализации**: Рекурсивная логика сложнее стандартного forward pass
   - *Митигация*: Тщательное тестирование каждого компонента

2. **Нестабильность обучения**: Deep supervision может быть нестабильным
   - *Митигация*: Использование stable-max loss, gradient clipping, EMA

3. **Память**: Множественные шаги требуют больше памяти
   - *Митигация*: Detach промежуточных состояний, уменьшение batch size

4. **Время обучения**: Множественные шаги замедляют обучение
   - *Митигация*: Early stopping, оптимизация кода

### 9.2 Зависимости

**Новые зависимости:**
- Возможно, нужны дополнительные библиотеки для RMSNorm, SwiGLU, Rotary embeddings
- Или реализовать самостоятельно (рекомендуется)

---

## 10. Заключение

### 10.1 Ключевые выводы

1. **TRM** - это принципиально другой подход к архитектуре моделей
2. **Преимущества**: Меньше параметров, лучшее обобщение на малых данных
3. **Недостатки**: Сложнее реализация, медленнее обучение
4. **Применение**: Лучше для задач рассуждения (ARC-AGI, Sudoku, Maze)

### 10.2 Рекомендации

1. **Начать с простых компонентов**: Сначала реализовать базовые модули
2. **Тестировать на каждом этапе**: Unit тесты для каждого компонента
3. **Сравнивать с GPT**: Постоянно сравнивать результаты с существующей моделью
4. **Документировать**: Подробная документация для каждого модуля

### 10.3 Следующие шаги

1. ✅ Создать структуру директорий `TRM/`
2. ✅ Реализовать базовые компоненты
3. ✅ Интегрировать с существующим проектом
4. ✅ Тестировать на простых задачах
5. ✅ Оптимизировать и улучшать

---

## Приложение A: Псевдокод из статьи

### A.1 TRM с single z

```python
def latent_recursion(x, z, n=6):
    for i in range(n-1):  # latent recursion
        z = net(x, z)
    return z

def deep_recursion(x, z, n=6, T=3):
    # recursing T-1 times to improve z (no gradients needed)
    with torch.no_grad():
        for j in range(T-1):
            z = latent_recursion(x, z, n)
    # recursing once to improve z
    z = latent_recursion(x, z, n)
    return z.detach(), output_head(y), Q_head(y)

# Deep Supervision
for x_input, y_true in train_dataloader:
    z = z_init
    for step in range(N_supervision):
        x = input_embedding(x_input)
        z, y_hat, q_hat = deep_recursion(x, z)
        loss = softmax_cross_entropy(y_hat, y_true)
        loss += binary_cross_entropy(q_hat, (y_hat == y_true))
        z = z.detach()
        loss.backward()
        opt.step()
        opt.zero_grad()
        if q[0] > 0:  # early-stopping
            break
```

### A.2 TRM с multi-scale z

```python
def latent_recursion(x, y, z, n=6):
    for i in range(n):  # latent recursion
        z[i] = net(x, y, z[0], ..., z[n-1])
    y = net(y, z[0], ..., z[n-1])  # refine output answer
    return y, z

def deep_recursion(x, y, z, n=6, T=3):
    # recursing T-1 times to improve y and z (no gradients needed)
    with torch.no_grad():
        for j in range(T-1):
            y, z = latent_recursion(x, y, z, n)
    # recursing once to improve y and z
    y, z = latent_recursion(x, y, z, n)
    return (y.detach(), z.detach(), output_head(y), Q_head(y))

# Deep Supervision
for x_input, y_true in train_dataloader:
    y, z = y_init, z_init
    for step in range(N_supervision):
        x = input_embedding(x_input)
        (y, z), y_hat, q_hat = deep_recursion(x, y, z)
        loss = softmax_cross_entropy(y_hat, y_true)
        loss += binary_cross_entropy(q_hat, (y_hat == y_true))
        loss.backward()
        opt.step()
        opt.zero_grad()
        if q[0] > 0:  # early-stopping
            break
```

---

## Приложение B: Сравнительная таблица

| Характеристика | GPT (текущий) | TRM |
|----------------|---------------|-----|
| Параметры | 10-100M+ | 7M |
| Слои | 4-12 | 2 |
| Рекурсия | Нет | Да (n=6, T=3) |
| Deep Supervision | Нет | Да (N_sup=16) |
| Обучение | 1 forward pass | Множественные шаги |
| Early Stopping | Нет | Да (Q-head) |
| Loss | CrossEntropy | StableMax + BinaryCE |
| Данные | Большие | Малые (~1000) |
| Задачи | Language Modeling | Reasoning (ARC, Sudoku) |
| Сложность реализации | Простая | Сложная |
| Скорость обучения | Быстрая | Медленная |
| Память | Средняя | Высокая |

---

**Дата создания**: 2025-01-XX  
**Версия**: 1.0  
**Автор**: Анализ проекта tokenier

