# Layer_ML

Получение массива эмбеддингов для текста из файла с помощью обученной модели (.pth).

## Использование

```bash
python Layer_ML/text_to_embeddings.py <файл_с_текстом.txt> <модель.pth> --tokenizer <токенизатор.pkl> [--output embeddings.npy]
```

**Обязательные аргументы:**
- `text` — путь к файлу с текстом (.txt)
- `model` — путь к обученной модели эмбеддингов (.pth)
- `--tokenizer` / `-t` — путь к токенизатору (.pkl), совместимому с моделью (BPE или словный)

**Опции:**
- `--output` / `-o` — сохранить массив в .npy
- `--encoding` — кодировка файла (по умолчанию utf-8)
- `--device` — cpu или cuda
- `--max-seq-len` — максимальная длина последовательности (по умолчанию 512)

**Выход:** массив формы `(число_токенов, embedding_dim)` — по одному вектору на каждый токен в тексте.

## Пример

```bash
python Layer_ML/text_to_embeddings.py data/verylog_module.txt embedding_model.pth -t verylog.pkl -o out_embeddings.npy
```

В коде:

```python
from Layer_ML.text_to_embeddings import text_to_embeddings

embeddings = text_to_embeddings(
    text_path="data/text.txt",
    pth_path="embedding_model.pth",
    tokenizer_path="verylog.pkl",
    output_npy="emb.npy",
)
# embeddings: np.ndarray shape (num_tokens, embedding_dim)
```

---

## Преобразование в вектор фиксированной размерности

Модуль `embedding_to_fixed_vector` превращает массив эмбеддингов переменной длины `(n, d)` в один вектор **строго фиксированной** размерности `(output_dim,)` для классификаторов, регрессии и т.д.

**Вход:** массив формы `(n, d)` (например, из `embeddings.npy`), где `n` может быть любым (0, 1, тысячи).  
**Выход:** вектор формы `(output_dim,)` — всегда одна и та же размерность.

### Методы (по умолчанию — без потери уникальности)

| method     | Описание |
|-----------|----------|
| **`sequence`** (по умолчанию) | **Сохраняет уникальность**: последовательность обрезается или дополняется нулями до `max_len = output_dim // d` позиций, затем разворачивается в вектор. Разные тексты дают разные векторы. Рекомендуется задавать `output_dim` кратным `d`. |
| `mean`    | Среднее по токенам (схлопывает последовательность) |
| `max`     | Максимум по токенам |
| `first`   | Вектор первого токена |
| `last`    | Вектор последнего токена |
| `mean_max`| Конкатенация [mean; max] (размер 2*d), затем приведение к `output_dim` |

При `sequence` порядок токенов сохраняется; при агрегации (mean/max/…) один вектор на весь текст — быстрее, но разные тексты могут дать похожие векторы.

### Пример из файла .npy

```bash
# Сохранить уникальность (по умолчанию): output_dim лучше кратен d (например 256 при d=256 → 1 токен, 512 → 2 токена и т.д.)
python Layer_ML/embedding_to_fixed_vector.py embeddings.npy --output-dim 2048 -o vector.npy

# Агрегация (как раньше)
python Layer_ML/embedding_to_fixed_vector.py embeddings.npy --output-dim 256 --method mean -o vector.npy
```

### В коде

```python
import numpy as np
from Layer_ML.embedding_to_fixed_vector import embeddings_to_fixed_vector, embeddings_to_fixed_vector_from_file

embeddings = np.load("embeddings.npy")  # (n, d), например d=256

# По умолчанию method="sequence" — уникальность сохраняется (pad/truncate последовательности)
vector = embeddings_to_fixed_vector(embeddings, output_dim=2048)
# output_dim кратен d: max_len = 2048 // 256 = 8 — в векторе первые 8 токенов (остальные обрезаны или дополнены нулями)

# Агрегация (одно представление на весь текст)
vector = embeddings_to_fixed_vector(embeddings, output_dim=256, method="mean")
```

**Крайние случаи:** пустой массив `(0, d)` → вектор из нулей длины `output_dim`; при `sequence` короткие тексты дополняются нулями, длинные обрезаются до `max_len`.

---

## Визуализация нескольких .npy (приложение)

Приложение с окном для загрузки нескольких файлов `.npy` и отрисовки векторов на одном графике (каждый файл — своим цветом).

**Запуск:**
```bash
python Layer_ML/visualize_npy_app.py
```

**Как пользоваться:**
1. Нажать «Добавить файлы…» и выбрать один или несколько `.npy` (массивы формы `(n, d)`).
2. При необходимости удалить файлы из списка или очистить его.
3. Выбрать метод уменьшения размерности: **PCA** или **t-SNE**.
4. Выбрать отображение: **2D** или **3D**.
5. При большом числе точек можно ограничить «Макс. точек на файл» (сэмплирование).
6. Нажать «Построить график». Векторы из всех файлов отображаются разными цветами с легендой по именам файлов.
7. «Сохранить рисунок…» — сохранить текущий график в PNG/PDF.

**Зависимости:**
- `scikit-learn` — PCA и t-SNE (обязательно): `pip install scikit-learn`
- `umap-learn` — UMAP, лучше сохраняет косинусное сходство: `pip install umap-learn`
- `plotly` — интерактивный график в браузере: `pip install plotly`

**Учёт косинусного сходства:** включите опцию «Учёт косинусного сходства (L2-norm)» — тогда векторы с высоким косинусным сходством (0.8, 0.9) будут отображаться рядом на графике. Рекомендуется метод UMAP с этой опцией.
