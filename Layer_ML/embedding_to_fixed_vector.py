"""
Преобразование массива эмбеддингов переменной длины (n, d) в один вектор
строго фиксированной размерности для использования в машинном обучении.

Вход: массив формы (n, d), n — произвольное (0, 1, ...).
Выход: вектор формы (output_dim,) с заданной output_dim.

Два подхода:
- Агрегация (mean/max/first/last/mean_max): схлопывает последовательность в один вектор —
  быстрее, но разные тексты могут дать похожие векторы.
- sequence: сохраняет порядок токенов (pad/truncate + flatten) — уникальность текста сохраняется,
  разные последовательности дают разные векторы.
"""

from typing import Literal

import numpy as np

# Агрегация (схлопывает последовательность)
PoolingMethod = Literal["mean", "max", "first", "last", "mean_max"]
# Все методы включая сохраняющий уникальность
Method = Literal["mean", "max", "first", "last", "mean_max", "sequence"]


def embeddings_to_fixed_vector(
    embeddings: np.ndarray,
    output_dim: int,
    method: Method = "sequence",
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    Преобразует массив эмбеддингов переменной длины в один вектор фиксированной размерности.

    Параметры
    ---------
    embeddings : np.ndarray
        Массив формы (n, d), где n — число токенов (может быть 0, 1 или любым),
        d — размерность одного эмбеддинга.
    output_dim : int
        Строго фиксированная размерность выходного вектора. Должна быть > 0.
        Для method="sequence" лучше задавать кратным d (тогда max_len = output_dim // d).
    method : str
        - "sequence" (по умолчанию) — сохраняет уникальность: последовательность обрезается
          до max_len = output_dim // d или дополняется нулями, затем разворачивается в вектор.
          Разные тексты дают разные векторы.
        - "mean" — среднее по токенам (схлопывает последовательность);
        - "max" — максимум по оси 0;
        - "first" — вектор первого токена;
        - "last" — вектор последнего токена;
        - "mean_max" — конкатенация [mean; max], затем приведение к output_dim.
    pad_value : float
        Значение для дополнения (для агрегирующих методов и для хвоста при method="sequence").

    Возвращает
    ----------
    np.ndarray
        Вектор формы (output_dim,) dtype float32.
    """
    if output_dim <= 0:
        raise ValueError("output_dim должно быть положительным.")

    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"Ожидается массив формы (n, d), получена форма {embeddings.shape}.")

    n, d = embeddings.shape

    # Пустой массив
    if n == 0:
        return np.zeros(output_dim, dtype=np.float32)

    # Режим sequence: сохраняем порядок токенов (pad/truncate + flatten), не теряем уникальность
    if method == "sequence":
        return _sequence_to_fixed(embeddings, output_dim, d, pad_value)

    # Агрегирующие методы (схлопывают последовательность)
    if method == "mean":
        vec = np.mean(embeddings, axis=0)
    elif method == "max":
        vec = np.max(embeddings, axis=0)
    elif method == "first":
        vec = embeddings[0].copy()
    elif method == "last":
        vec = embeddings[-1].copy()
    elif method == "mean_max":
        mean_vec = np.mean(embeddings, axis=0)
        max_vec = np.max(embeddings, axis=0)
        vec = np.concatenate([mean_vec, max_vec], axis=0)
    else:
        raise ValueError(
            f"Неизвестный method='{method}'. Допустимы: mean, max, first, last, mean_max, sequence."
        )

    return _to_fixed_length(vec, output_dim, pad_value)


def _sequence_to_fixed(
    embeddings: np.ndarray, output_dim: int, d: int, pad_value: float
) -> np.ndarray:
    """
    Сохраняет уникальность: последовательность (n, d) обрезается или дополняется до
    max_len позиций (max_len = output_dim // d), затем разворачивается в вектор длины
    max_len*d; при необходимости дополняется до output_dim.
    """
    max_len = max(1, output_dim // d)
    slot_dim = max_len * d

    if embeddings.shape[0] >= max_len:
        # Обрезаем до первых max_len векторов
        block = embeddings[:max_len]
    else:
        # Дополняем нулями до max_len
        block = np.full((max_len, d), pad_value, dtype=np.float32)
        block[: embeddings.shape[0]] = embeddings

    vec = block.ravel()
    if len(vec) >= output_dim:
        return vec[:output_dim].copy()
    out = np.full(output_dim, pad_value, dtype=np.float32)
    out[: len(vec)] = vec
    return out


def _to_fixed_length(vec: np.ndarray, target_len: int, pad_value: float) -> np.ndarray:
    """Доводит вектор до длины target_len: обрезка или дополнение pad_value."""
    vec = np.asarray(vec, dtype=np.float32).ravel()
    if len(vec) == target_len:
        return vec
    if len(vec) > target_len:
        return vec[:target_len].copy()
    out = np.full(target_len, pad_value, dtype=np.float32)
    out[: len(vec)] = vec
    return out


def embeddings_to_fixed_vector_from_file(
    npy_path: str,
    output_dim: int,
    method: Method = "sequence",
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    Загружает массив эмбеддингов из .npy и возвращает вектор фиксированной размерности.

    Параметры
    ---------
    npy_path : str
        Путь к файлу .npy с массивом формы (n, d).
    output_dim, method, pad_value
        Как в embeddings_to_fixed_vector().

    Возвращает
    ----------
    np.ndarray
        Вектор формы (output_dim,).
    """
    arr = np.load(npy_path)
    return embeddings_to_fixed_vector(arr, output_dim=output_dim, method=method, pad_value=pad_value)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Преобразование массива эмбеддингов (n, d) в вектор фиксированной размерности"
    )
    parser.add_argument("input", help="Путь к .npy с массивом эмбеддингов (n, d)")
    parser.add_argument("--output-dim", "-d", type=int, required=True, help="Фиксированная размерность выхода")
    parser.add_argument(
        "--method",
        "-m",
        choices=["sequence", "mean", "max", "first", "last", "mean_max"],
        default="sequence",
        help="Метод: sequence (сохраняет уникальность) или агрегация (по умолчанию: sequence)",
    )
    parser.add_argument("--output", "-o", default=None, help="Путь для сохранения вектора .npy")
    parser.add_argument("--pad-value", type=float, default=0.0, help="Значение дополнения при pad")
    args = parser.parse_args()

    vec = embeddings_to_fixed_vector_from_file(
        args.input,
        output_dim=args.output_dim,
        method=args.method,
        pad_value=args.pad_value,
    )
    print(f"Выход: shape={vec.shape}, dtype={vec.dtype}")

    if args.output:
        np.save(args.output, vec)
        print(f"Сохранено: {args.output}")
