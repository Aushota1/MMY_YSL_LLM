"""
Временное разбиение на train/test (без shuffle).
"""

from typing import Tuple

import numpy as np


def time_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    train_ratio: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Разбивает данные по времени: первые train_ratio — train, остальные — test.

    Возвращает
    ----------
    X_train, X_test, y_train, y_test : np.ndarray
    test_dates : np.ndarray
        Даты тестовой выборки (для бэктеста).
    """
    n = len(X)
    if n == 0:
        feat_dim = X.shape[1] if X.ndim > 1 else 0
        return (
            np.zeros((0, feat_dim), dtype=X.dtype),
            np.zeros((0, feat_dim), dtype=X.dtype),
            np.zeros(0, dtype=y.dtype),
            np.zeros(0, dtype=y.dtype),
            np.array([], dtype=dates.dtype),
        )
    split_idx = int(n * train_ratio)
    if split_idx <= 0:
        split_idx = 1
    if split_idx >= n:
        split_idx = n - 1

    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    test_dates = dates[split_idx:]

    return X_train, X_test, y_train, y_test, test_dates
