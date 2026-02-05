# Современные методы обучения Embeddings: Полное руководство

## 📚 Содержание

1. [Введение](#введение)
2. [Историческая эволюция методов](#историческая-эволюция-методов)
3. [Современные подходы к обучению](#современные-подходы-к-обучению)
4. [Метрики качества embeddings](#метрики-качества-embeddings)
5. [Анализ качества и валидация](#анализ-качества-и-валидация)
6. [Практические рекомендации](#практические-рекомендации)
7. [Сравнение методов](#сравнение-методов)
8. [Применение в LLM](#применение-в-llm)

---

## Введение

Embeddings (векторные представления) - это числовые векторы, которые кодируют семантическую и синтаксическую информацию о словах, фразах или документах в многомерном пространстве. Современные методы обучения embeddings позволяют моделям понимать контекст, семантику и отношения между словами.

### Основные принципы

1. **Распределенное представление**: Слова с похожим значением имеют близкие векторы в пространстве
2. **Контекстное обучение**: Embeddings обучаются на основе контекста использования слов
3. **Масштабируемость**: Методы должны работать с большими корпусами текстов
4. **Универсальность**: Один набор embeddings может использоваться для различных задач

---

## Историческая эволюция методов

### 1. Word2Vec (2013, Google)

**Архитектуры:**

#### CBOW (Continuous Bag of Words)
- **Принцип**: Предсказывает целевое слово по контексту окружающих слов
- **Формула**: `P(w_t | w_{t-k}, ..., w_{t-1}, w_{t+1}, ..., w_{t+k})`
- **Преимущества**: 
  - Быстрее обучение
  - Лучше работает с частыми словами
  - Меньше вычислительных ресурсов
- **Недостатки**:
  - Менее эффективен для редких слов
  - Теряет порядок слов в контексте

#### Skip-Gram
- **Принцип**: Использует целевое слово для предсказания контекстных слов
- **Формула**: `P(w_{t-k}, ..., w_{t+k} | w_t)`
- **Преимущества**:
  - Лучше работает с редкими словами
  - Более точные embeddings для малочастотных контекстов
  - Лучше улавливает семантические отношения
- **Недостатки**:
  - Медленнее обучение
  - Требует больше вычислительных ресурсов

**Математическая основа:**

```
Objective: Maximize log P(w_{t-k}, ..., w_{t+k} | w_t)

Для Skip-Gram:
P(w_{t-k}, ..., w_{t+k} | w_t) = ∏ P(w_{t+j} | w_t)

P(w_O | w_I) = exp(v'_{w_O}^T v_{w_I}) / Σ exp(v'_{w}^T v_{w_I})
```

**Оптимизации:**

1. **Negative Sampling**: Вместо вычисления softmax по всему словарю, выбираются k отрицательных примеров
   ```
   log σ(v'_{w_O}^T v_{w_I}) + Σ log σ(-v'_{w_i}^T v_{w_I})
   ```

2. **Hierarchical Softmax**: Использует бинарное дерево для вычисления вероятностей (O(log V) вместо O(V))

### 2. GloVe (Global Vectors for Word Representation, 2014)

**Принцип**: Комбинирует локальный контекст (как Word2Vec) с глобальной статистикой корпуса

**Формула:**
```
J = Σ f(X_{ij}) (w_i^T w̃_j + b_i + b̃_j - log X_{ij})²
```

Где:
- `X_{ij}` - количество совместных вхождений слов i и j
- `f(X_{ij})` - взвешивающая функция
- `w_i, w̃_j` - векторы слов
- `b_i, b̃_j` - смещения

**Преимущества**:
- Учитывает глобальную статистику
- Лучше работает с синтаксическими отношениями
- Более стабильные результаты

### 3. FastText (Facebook AI Research, 2016)

**Принцип**: Расширение Word2Vec с поддержкой n-грамм символов

**Особенности**:
- Каждое слово представляется как сумма n-грамм
- Работает с OOV (out-of-vocabulary) словами
- Учитывает морфологию

**Формула:**
```
w = Σ_{g ∈ G_w} z_g
```

Где `G_w` - множество n-грамм слова w.

### 4. ELMo (Embeddings from Language Models, 2018)

**Принцип**: Контекстные embeddings на основе двунаправленных LSTM

**Особенности**:
- Контекстные представления (одно слово имеет разные embeddings в разных контекстах)
- Многоуровневые представления (из разных слоев LSTM)
- Взвешенная комбинация слоев

### 5. Transformer-based Embeddings (2017-настоящее время)

#### BERT (Bidirectional Encoder Representations from Transformers)
- **Архитектура**: Encoder-only Transformer
- **Обучение**: Masked Language Modeling (MLM) + Next Sentence Prediction (NSP)
- **Особенности**: Двунаправленное понимание контекста

#### GPT (Generative Pre-trained Transformer)
- **Архитектура**: Decoder-only Transformer
- **Обучение**: Autoregressive Language Modeling
- **Особенности**: Одностороннее, но мощное генеративное представление

#### T5 (Text-to-Text Transfer Transformer)
- **Архитектура**: Encoder-Decoder Transformer
- **Обучение**: Text-to-text задачи
- **Особенности**: Универсальная архитектура для различных задач

---

## Современные подходы к обучению

### 1. Self-Supervised Learning (SSL)

**Принцип**: Модель обучается на неразмеченных данных, создавая собственные задачи обучения

**Задачи SSL для embeddings:**

#### Masked Language Modeling (MLM)
```
Input:  "The cat sat on the [MASK]"
Target: "mat"
```
- Случайно маскируются токены
- Модель предсказывает замаскированные токены
- Используется в BERT

#### Next Sentence Prediction (NSP)
```
Sentence A: "The cat sat on the mat"
Sentence B: "It was very comfortable"
Label: IsNext (True/False)
```
- Модель учится понимать отношения между предложениями
- Используется в BERT

#### Autoregressive Language Modeling
```
Input:  "The cat sat"
Target: "on the mat"
```
- Модель предсказывает следующий токен
- Используется в GPT

#### Contrastive Learning
- Обучение на парах похожих/непохожих примеров
- Используется в Sentence-BERT, SimCSE

### 2. Fine-tuning на специфичных задачах

**Принцип**: Предобученная модель дообучается на целевой задаче

**Стратегии:**
- **Full fine-tuning**: Обновление всех параметров
- **Partial fine-tuning**: Обновление только верхних слоев
- **Adapter layers**: Добавление небольших адаптеров
- **LoRA (Low-Rank Adaptation)**: Обновление через низкоранговые матрицы

### 3. Multi-task Learning

**Принцип**: Одновременное обучение на нескольких задачах

**Преимущества**:
- Более универсальные embeddings
- Лучшая обобщающая способность
- Эффективное использование данных

### 4. Curriculum Learning

**Принцип**: Постепенное усложнение обучающих примеров

**Этапы**:
1. Простые примеры (короткие предложения, частые слова)
2. Средние примеры (средние предложения, средняя частота)
3. Сложные примеры (длинные предложения, редкие слова)

---

## Метрики качества embeddings

### 1. Внутренние метрики (Intrinsic Metrics)

#### Косинусное сходство (Cosine Similarity)
```
cos(θ) = (A · B) / (||A|| × ||B||)
```
- **Диапазон**: [-1, 1]
- **Интерпретация**: 
  - 1.0 = идентичные векторы
  - 0.0 = ортогональные векторы
  - -1.0 = противоположные векторы
- **Применение**: Измерение семантической близости

#### Евклидово расстояние (Euclidean Distance)
```
d = √Σ(a_i - b_i)²
```
- **Интерпретация**: Меньше расстояние = больше похожесть
- **Применение**: Кластеризация, поиск ближайших соседей

#### Манипуляции с векторами (Vector Arithmetic)
```
king - man + woman ≈ queen
```
- **Тест**: Аналогии (word analogies)
- **Метрика**: Точность на наборе аналогий

#### Семантические тесты
- **Word Similarity**: Корреляция с человеческими оценками
- **Word Relatedness**: Связанность слов
- **Synonym Detection**: Обнаружение синонимов

### 2. Внешние метрики (Extrinsic Metrics)

#### Задачи классификации
- **Sentiment Analysis**: Точность классификации тональности
- **Text Classification**: F1-score, Accuracy
- **Named Entity Recognition (NER)**: Precision, Recall, F1

#### Задачи регрессии
- **Semantic Textual Similarity (STS)**: Pearson/Spearman correlation
- **Paraphrase Detection**: Accuracy, F1-score

#### Задачи поиска
- **Information Retrieval**: 
  - Precision@K
  - Recall@K
  - Mean Reciprocal Rank (MRR)
  - Normalized Discounted Cumulative Gain (NDCG)

#### Задачи генерации
- **BLEU Score**: Для машинного перевода
- **ROUGE Score**: Для суммаризации
- **Perplexity**: Для языкового моделирования

### 3. Метрики для визуализации

#### t-SNE (t-Distributed Stochastic Neighbor Embedding)
- **Принцип**: Нелинейное снижение размерности
- **Применение**: Визуализация кластеров слов
- **Параметры**: perplexity (обычно 30-50)

#### PCA (Principal Component Analysis)
- **Принцип**: Линейное снижение размерности
- **Применение**: Быстрая визуализация основных компонент
- **Ограничения**: Теряет нелинейные отношения

#### UMAP (Uniform Manifold Approximation and Projection)
- **Принцип**: Сохраняет глобальную и локальную структуру
- **Преимущества**: Быстрее t-SNE, лучше сохраняет структуру

### 4. Специализированные метрики

#### Alignment и Uniformity (Wang & Isola, 2020)
- **Alignment**: Близость похожих примеров
- **Uniformity**: Равномерное распределение в пространстве
- **Формула Alignment**: `E_{p_pos} [||f(x) - f(x^+)||²]`
- **Формула Uniformity**: `log E_{x,y ~ p_data} [e^{-2||f(x) - f(y)||²}]`

#### Intra-cluster и Inter-cluster расстояния
- **Intra-cluster**: Среднее расстояние внутри кластера
- **Inter-cluster**: Среднее расстояние между кластерами
- **Цель**: Минимизировать intra, максимизировать inter

---

## Анализ качества и валидация

### 1. Процесс валидации

#### Этап 1: Подготовка данных
- **Train/Validation/Test split**: 80/10/10 или 70/15/15
- **Stratified sampling**: Сохранение распределения классов
- **Temporal split**: Для временных данных

#### Этап 2: Базовые проверки
- **Vocabulary coverage**: Процент слов из тестового набора в словаре
- **Embedding statistics**: Среднее, дисперсия, распределение норм
- **Dimensionality analysis**: Анализ эффективной размерности

#### Этап 3: Семантические тесты
- **Word analogy tasks**: 
  - Синтаксические: "quick → quickly" как "slow → slowly"
  - Семантические: "man → woman" как "king → queen"
- **Word similarity benchmarks**:
  - SimLex-999
  - WordSim-353
  - MEN (Multimodal Embeddings)

#### Этап 4: Задаче-специфичные тесты
- **Downstream tasks**: Тестирование на реальных задачах
- **Cross-lingual evaluation**: Для многоязычных моделей
- **Domain adaptation**: Тестирование на новых доменах

### 2. Методы анализа

#### Анализ кластеров
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Кластеризация
kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit_predict(embeddings)

# Оценка качества кластеризации
silhouette = silhouette_score(embeddings, clusters)
```

#### Анализ ближайших соседей
```python
def analyze_nearest_neighbors(embeddings, vocab, word, top_k=10):
    word_idx = vocab[word]
    word_embedding = embeddings[word_idx]
    
    # Косинусное сходство
    similarities = cosine_similarity([word_embedding], embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k-1:-1][::-1]
    
    return [(vocab[i], similarities[i]) for i in top_indices]
```

#### Анализ размерности
```python
from sklearn.decomposition import PCA

# PCA для анализа эффективной размерности
pca = PCA()
pca.fit(embeddings)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Находим количество компонент для 95% дисперсии
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
```

### 3. Бенчмарки для оценки

#### GLUE (General Language Understanding Evaluation)
- Задачи: CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE
- Метрика: Средний score по всем задачам

#### SuperGLUE
- Более сложные задачи
- Лучше различает современные модели

#### SentEval
- Набор задач для оценки sentence embeddings
- Включает: классификацию, поиск, семантическое сходство

#### MTEB (Massive Text Embedding Benchmark)
- 8 задач, 112 языков
- Современный стандарт для оценки embeddings

---

## Практические рекомендации

### 1. Выбор метода обучения

#### Для небольших корпусов (< 1M слов)
- **Word2Vec (Skip-Gram)**: Хороший баланс качества и скорости
- **FastText**: Если важна работа с OOV словами
- **GloVe**: Если нужны синтаксические отношения

#### Для средних корпусов (1M - 100M слов)
- **Word2Vec с Negative Sampling**: Эффективно и быстро
- **FastText**: Для морфологически богатых языков
- **BERT/GPT fine-tuning**: Если есть вычислительные ресурсы

#### Для больших корпусов (> 100M слов)
- **Transformer-based models**: BERT, GPT, T5
- **Distributed training**: Обучение на нескольких GPU/TPU
- **Efficient architectures**: DistilBERT, ALBERT

### 2. Гиперпараметры

#### Размерность embeddings
- **50-100**: Для простых задач, быстрых экспериментов
- **200-300**: Стандартный выбор, хороший баланс
- **512-768**: Для сложных задач, лучшего качества
- **1024+**: Для очень больших словарей, специализированных задач

#### Размер контекстного окна
- **Word2Vec**: 5-10 для Skip-Gram, 2-5 для CBOW
- **BERT**: 512 токенов (стандарт), до 4096 для некоторых моделей
- **GPT**: До 2048-8192 токенов в современных моделях

#### Learning rate
- **Word2Vec**: 0.025 (начальный), экспоненциальное затухание
- **BERT/GPT**: 1e-4 до 5e-4 (зависит от размера модели)
- **Fine-tuning**: 1e-5 до 5e-5 (меньше, чем предобучение)

#### Batch size
- **Word2Vec**: 100-1000
- **BERT/GPT**: 16-128 (зависит от GPU памяти)
- **Gradient accumulation**: Для больших эффективных batch size

### 3. Оптимизация обучения

#### Градиентный клиппинг
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
- Предотвращает взрыв градиентов
- Стандартное значение: 1.0

#### Learning rate scheduling
- **Warmup**: Постепенное увеличение LR в начале
- **Cosine annealing**: Плавное уменьшение LR
- **Reduce on plateau**: Уменьшение при отсутствии улучшений

#### Регуляризация
- **Dropout**: 0.1-0.3 для embeddings
- **Weight decay**: 0.01-0.1
- **Label smoothing**: Для classification задач

### 4. Обработка данных

#### Предобработка
- **Токенизация**: BPE, WordPiece, SentencePiece
- **Нормализация**: Lowercasing, Unicode normalization
- **Фильтрация**: Удаление слишком коротких/длинных текстов

#### Аугментация данных
- **Back-translation**: Для многоязычных моделей
- **Synonym replacement**: Замена синонимами
- **Random masking**: Для MLM задач

---

## Сравнение методов

### Таблица сравнения

| Метод | Год | Архитектура | Контекст | Размерность | Скорость обучения | Качество |
|-------|-----|-------------|----------|-------------|-------------------|----------|
| Word2Vec | 2013 | Neural Network | Локальный | 50-300 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| GloVe | 2014 | Matrix Factorization | Глобальный | 50-300 | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| FastText | 2016 | Neural Network + N-grams | Локальный | 50-300 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| ELMo | 2018 | BiLSTM | Контекстный | 512-1024 | ⭐⭐ | ⭐⭐⭐⭐ |
| BERT | 2018 | Transformer Encoder | Контекстный | 768-1024 | ⭐ | ⭐⭐⭐⭐⭐ |
| GPT-2/3 | 2019-2020 | Transformer Decoder | Контекстный | 768-12288 | ⭐ | ⭐⭐⭐⭐⭐ |
| T5 | 2020 | Transformer Encoder-Decoder | Контекстный | 768-1024 | ⭐ | ⭐⭐⭐⭐⭐ |

### Когда использовать какой метод

#### Word2Vec/GloVe/FastText
- ✅ Быстрое прототипирование
- ✅ Небольшие вычислительные ресурсы
- ✅ Статические embeddings достаточны
- ✅ Работа с большими корпусами без GPU

#### ELMo
- ✅ Нужны контекстные embeddings
- ✅ Задачи с пониманием контекста
- ✅ Средние вычислительные ресурсы

#### BERT/GPT/T5
- ✅ Максимальное качество
- ✅ Достаточные вычислительные ресурсы
- ✅ Сложные задачи NLP
- ✅ Нужны контекстные представления

---

## Применение в LLM

### 1. Архитектура Embedding Layer в LLM

#### Компоненты:
1. **Token Embedding**: Преобразование токенов в векторы
2. **Positional Encoding**: Добавление позиционной информации
3. **Layer Normalization**: Нормализация (опционально)
4. **Dropout**: Регуляризация (опционально)

#### Реализация (как в проекте):
```python
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_len, ...):
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_seq_len)
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, token_ids):
        x = self.token_embedding(token_ids)
        x = self.positional_encoding(x)
        x = self.layer_norm(x)
        return x
```

### 2. Обучение embeddings в LLM

#### Pre-training подход
1. **Self-supervised learning** на большом корпусе
2. **Language modeling** задача
3. **Multi-task learning** (опционально)

#### Fine-tuning подход
1. Загрузка предобученных embeddings
2. Дообучение на целевой задаче
3. Адаптация под специфичный домен

### 3. Метрики для LLM embeddings

#### Perplexity
```
PPL = exp(cross_entropy_loss)
```
- Мера неопределенности модели
- Ниже = лучше
- Хорошие значения: 20-50 для языкового моделирования

#### Embedding quality metrics
- **Intrinsic**: Косинусное сходство, аналогии
- **Extrinsic**: Performance на downstream задачах

#### Training metrics
- **Loss**: Cross-entropy, должен уменьшаться
- **Learning rate**: Должна адаптивно изменяться
- **Gradient norm**: Должна быть стабильной

### 4. Оптимизация для LLM

#### Memory efficiency
- **Gradient checkpointing**: Торговля памятью на вычисления
- **Mixed precision training**: FP16/BF16 для экономии памяти
- **Model parallelism**: Распределение модели по устройствам

#### Training speed
- **Data parallelism**: Обучение на нескольких GPU
- **Optimized optimizers**: AdamW, LAMB
- **Efficient attention**: Sparse attention, Linear attention

---

## Заключение

Современные методы обучения embeddings эволюционировали от простых статистических подходов (Word2Vec) к сложным контекстным моделям (BERT, GPT). Выбор метода зависит от:

1. **Размера данных**: Больше данных → более сложные модели
2. **Вычислительных ресурсов**: GPU/TPU → Transformer models
3. **Задачи**: Статические embeddings vs контекстные
4. **Требований к качеству**: Прототипирование vs production

Ключевые принципы успешного обучения:
- ✅ Правильная предобработка данных
- ✅ Адекватный выбор гиперпараметров
- ✅ Регулярный мониторинг метрик
- ✅ Валидация на реальных задачах
- ✅ Итеративное улучшение

---

## Ссылки и ресурсы

### Бенчмарки
- GLUE: https://gluebenchmark.com/
- SuperGLUE: https://super.gluebenchmark.com/
- SentEval: https://github.com/facebookresearch/SentEval
- MTEB: https://github.com/embeddings-benchmark/mteb

### Документация
- Word2Vec: https://code.google.com/archive/p/word2vec/
- BERT: https://github.com/google-research/bert
- GPT: https://github.com/openai/gpt-2
- Hugging Face: https://huggingface.co/

### Статьи
- Word2Vec: Mikolov et al., 2013
- GloVe: Pennington et al., 2014
- BERT: Devlin et al., 2018
- GPT: Radford et al., 2019
- T5: Raffel et al., 2020

---

*Документ создан на основе анализа современных методов обучения embeddings и практического опыта работы с LLM моделями.*

