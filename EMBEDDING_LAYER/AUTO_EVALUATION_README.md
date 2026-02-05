# Автоматическая система обучения и оценки Embeddings

## 📋 Описание

Автоматическая система для обучения embeddings с встроенной оценкой качества и генерацией рекомендаций на основе метрик из `EMBEDDING_TRAINING_METHODS.md`.

## 🎯 Возможности

✅ **Автоматическое обучение** - Обучение модели с настраиваемыми параметрами  
✅ **Автоматическая оценка** - Оценка качества через заданные интервалы  
✅ **Множество метрик** - Комплексная оценка по различным критериям  
✅ **Автоматические рекомендации** - Генерация рекомендаций на основе метрик  
✅ **Детальные отчеты** - Подробные отчеты в консоли и JSON  
✅ **Сохранение чекпоинтов** - Автоматическое сохранение лучших моделей  

## 📊 Реализованные метрики

### Внутренние метрики (Intrinsic Metrics)

1. **Косинусное сходство**
   - Статистика (mean, std, min, max, median)
   - Анализ распределения сходства между токенами

2. **Нормы embeddings**
   - Статистика норм векторов
   - Выявление проблем со стабильностью

3. **Качество кластеризации**
   - Silhouette Score
   - Intra-cluster и Inter-cluster расстояния
   - Оценка структуры пространства embeddings

4. **Анализ размерности**
   - Эффективная размерность (90%, 95%, 99% дисперсии)
   - PCA анализ
   - Выявление избыточной размерности

5. **Alignment и Uniformity**
   - Метрики из Wang & Isola, 2020
   - Оценка качества распределения embeddings

## 🚀 Использование

### Базовое использование

```python
from EMBEDDING_LAYER.auto_evaluation_system import AutoEmbeddingSystem

# Создание системы
system = AutoEmbeddingSystem(tokenizer_path="chekpoint.pkl")

# Создание модели
system.create_model(
    embedding_dim=256,
    max_seq_len=512,
    hidden_dim=256,
    learnable_pos=False,
    layer_norm=True
)

# Загрузка данных
texts = ["текст 1", "текст 2", ...]

# Обучение с автоматической оценкой
results = system.train_with_auto_evaluation(
    texts=texts,
    num_epochs=10,
    batch_size=32,
    learning_rate=0.001,
    eval_interval=2,  # Оценка каждые 2 эпохи
    save_checkpoints=True,
    checkpoint_dir="embedding_checkpoints"
)
```

### Командная строка

```bash
# Базовое использование
python EMBEDDING_LAYER/auto_evaluation_system.py --data war_and_peace.ru.txt --epochs 10

# С настройками
python EMBEDDING_LAYER/auto_evaluation_system.py 
    --tokenizer chekpoint.pkl 
    --data war_and_peace.ru.txt 
    --epochs 20 
    --batch-size 64 
    --lr 0.0005 
    --embedding-dim 512 
    --eval-interval 3 
    --checkpoint-dir my_checkpoints
```

### Параметры командной строки

- `--tokenizer` - Путь к токенизатору (по умолчанию: `chekpoint.pkl`)
- `--data` - Путь к файлу с данными для обучения
- `--epochs` - Количество эпох (по умолчанию: 10)
- `--batch-size` - Размер батча (по умолчанию: 32)
- `--lr` - Learning rate (по умолчанию: 0.001)
- `--embedding-dim` - Размерность embeddings (по умолчанию: 256)
- `--eval-interval` - Интервал оценки в эпохах (по умолчанию: 2)
- `--checkpoint-dir` - Директория для чекпоинтов (по умолчанию: `embedding_checkpoints`)

## 📈 Интерпретация результатов

### Косинусное сходство

- **Среднее > 0.3**: Embeddings слишком похожи, возможно переобучение
- **Среднее < -0.1**: Embeddings слишком разрозненные, возможно недообучение
- **Оптимально**: -0.1 до 0.2

### Silhouette Score

- **< 0.2**: Плохая структура кластеров
- **0.2 - 0.5**: Умеренная структура
- **> 0.5**: Отличная структура кластеров

### Alignment и Uniformity

- **Alignment**: Меньше = лучше (близкие примеры должны быть близки)
- **Uniformity**: Больше = лучше (равномерное распределение в пространстве)

### Эффективная размерность

- Если эффективная размерность (95%) < 50% исходной → можно уменьшить размерность без потери качества

## 💡 Рекомендации

Система автоматически генерирует рекомендации на основе метрик:

### Приоритеты рекомендаций

- 🔴 **HIGH** - Критические проблемы, требующие немедленного внимания
- 🟡 **MEDIUM** - Важные улучшения, но не критичные
- 🟢 **LOW** - Оптимизации и улучшения

### Категории рекомендаций

- **quality** - Качество embeddings
- **stability** - Стабильность обучения
- **structure** - Структура пространства embeddings
- **efficiency** - Эффективность использования ресурсов

## 📁 Структура выходных файлов

После обучения создается следующая структура:

```
embedding_checkpoints/
├── best_model_epoch_N.pth      # Лучшая модель (по loss)
├── final_model.pth              # Финальная модель
└── final_evaluation_report.json # Полный отчет оценки
```

### Формат отчета JSON

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "vocab_size": 10000,
  "embedding_dim": 256,
  "cosine_similarity": {...},
  "embedding_norms": {...},
  "clustering": {...},
  "dimensionality": {...},
  "alignment_uniformity": {...},
  "recommendations": [...]
}
```

## 🔧 Использование EmbeddingEvaluator отдельно

Если нужно оценить уже обученную модель:

```python
from EMBEDDING_LAYER.embedding_evaluator import EmbeddingEvaluator
from EMBEDDING_LAYER.auto_evaluation_system import AutoEmbeddingSystem

# Загрузка модели
system = AutoEmbeddingSystem(tokenizer_path="chekpoint.pkl")
system.load_model("embedding_model.pth")

# Оценка
results = system.evaluator.evaluate_all(sample_size=2000, n_clusters=20)

# Печать отчета
system.evaluator.print_report(results)

# Сохранение отчета
system.evaluator.save_report("evaluation_report.json", results)
```

## 📊 Пример вывода

```
================================================================================
🚀 АВТОМАТИЧЕСКОЕ ОБУЧЕНИЕ С ОЦЕНКОЙ
================================================================================

📊 Параметры обучения:
   Эпох: 10
   Batch size: 32
   Learning rate: 0.001
   Оценка каждые: 2 эпох
   Всего батчей: 156

================================================================================
📈 ЭПОХА 1/10
================================================================================
  [██████████████████████████████████████████████████] 100.0% | Loss: 8.2341

🔍 Автоматическая оценка после эпохи 1...
🔍 Начало автоматической оценки embeddings...
================================================================================
📊 Вычисление статистики косинусного сходства...
📏 Вычисление статистики норм embeddings...
🔗 Анализ качества кластеризации...
📐 Анализ эффективной размерности...
⚖️  Вычисление Alignment и Uniformity...
💡 Генерация рекомендаций...

  📊 Краткие результаты (эпоха 1):
     Косинусное сходство (среднее): 0.1234
     Silhouette Score: 0.2345
     Alignment: 0.4567, Uniformity: -0.7890
     💡 Топ рекомендации:
        1. [HIGH] Низкая Uniformity
        2. [MEDIUM] Большой разброс норм embeddings
        3. [LOW] Отличное качество кластеризации
```

## 🎓 Ссылки

- Подробное описание метрик: `EMBEDDING_TRAINING_METHODS.md`
- Руководство по embedding layer: `EMBEDDING_LAYER_GUIDE.md`
- Примеры использования: `example_usage.py`

## ⚙️ Требования

- PyTorch
- NumPy
- scikit-learn (для метрик кластеризации и PCA)
- scipy (опционально, для дополнительных метрик)

## 📝 Примечания

- Оценка может занять некоторое время, особенно для больших словарей
- Рекомендуется использовать `sample_size=1000-2000` для баланса между скоростью и точностью
- Для production использования увеличьте `sample_size` и `n_clusters`

---

*Система создана на основе метрик из EMBEDDING_TRAINING_METHODS.md*

