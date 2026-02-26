"""
Функции обучения классификаторов для датасета из train_data.

Загрузка датасета (из папки .npy или из кэша CSV/parquet), разбиение train/test,
нормализация, обучение SVM, XGBoost, Random Forest, Logistic Regression, KNN.
Оценка модели, проверка переобучения, тепловая карта (confusion matrix), кросс-валидация.
Сохранение модели и LabelEncoder через joblib.

Зависимости: pip install scikit-learn xgboost pandas numpy joblib matplotlib seaborn
"""

import os
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# Импорт create_dataset из той же папки
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from create_dataset import create_dataset as _create_dataset


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dataset(
    cache_path: str | None = None,
    train_data_dir: str | None = None,
) -> pd.DataFrame:
    """
    Единая точка загрузки датасета.

    Параметры
    ---------
    cache_path : str, optional
        Путь к сохранённому датасету (CSV или parquet). Если задан и файл существует,
        загрузка идёт из кэша.
    train_data_dir : str, optional
        Папка с .npy (используется если cache_path не задан или файл не найден).
        По умолчанию — train_data в корне проекта.

    Возвращает
    ----------
    pd.DataFrame
        Таблица с признаками f0, f1, ... и столбцом target.
    """
    if cache_path and os.path.isfile(cache_path):
        if cache_path.lower().endswith(".parquet"):
            return pd.read_parquet(cache_path)
        return pd.read_csv(cache_path)

    data_dir = train_data_dir or os.path.join(_project_root(), "train_data")
    return _create_dataset(data_dir, output_path=None)


def prepare_splits(
    df: pd.DataFrame,
    target_col: str = "target",
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
):
    """
    Разбиение на X_train, X_test, y_train, y_test и опционально масштабирование.

    Возвращает
    ----------
    X_train, X_test, y_train, y_test : np.ndarray
    scaler : StandardScaler or None
    label_encoder : LabelEncoder (обученный на y)
    """
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].values
    y_raw = df[target_col].astype(str).values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, le


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec, rec, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    return {
        "accuracy": acc,
        "f1_weighted": f1,
        "precision_per_class": prec.tolist(),
        "recall_per_class": rec.tolist(),
        "f1_per_class": f1_per_class.tolist(),
    }


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    kernel: str = "rbf",
    probability: bool = True,
    **kwargs,
) -> tuple[Any, dict]:
    """Обучает SVC. Рекомендуется использовать после StandardScaler."""
    clf = SVC(kernel=kernel, probability=probability, **kwargs)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics = _compute_metrics(y_test, y_pred)
    return clf, metrics


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    **kwargs,
) -> tuple[Any, dict]:
    """Обучает XGBClassifier."""
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("Установите xgboost: pip install xgboost")

    clf = xgb.XGBClassifier(eval_metric="logloss", **kwargs)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics = _compute_metrics(y_test, y_pred)
    return clf, metrics


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = 100,
    **kwargs,
) -> tuple[Any, dict]:
    """Обучает RandomForestClassifier."""
    clf = RandomForestClassifier(n_estimators=n_estimators, **kwargs)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics = _compute_metrics(y_test, y_pred)
    return clf, metrics


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_iter: int = 1000,
    **kwargs,
) -> tuple[Any, dict]:
    """Обучает LogisticRegression. Рекомендуется после StandardScaler."""
    clf = LogisticRegression(max_iter=max_iter, **kwargs)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics = _compute_metrics(y_test, y_pred)
    return clf, metrics


def train_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_neighbors: int = 5,
    **kwargs,
) -> tuple[Any, dict]:
    """Обучает KNeighborsClassifier. Рекомендуется после StandardScaler."""
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics = _compute_metrics(y_test, y_pred)
    return clf, metrics


def train_gradient_boosting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = 100,
    **kwargs,
) -> tuple[Any, dict]:
    """Обучает GradientBoostingClassifier."""
    clf = GradientBoostingClassifier(n_estimators=n_estimators, **kwargs)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics = _compute_metrics(y_test, y_pred)
    return clf, metrics


CLASSIFIER_NAMES = ["svm", "xgboost", "rf", "random_forest", "logistic", "knn", "gb", "gradient_boosting"]


def train_classifier(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    **kwargs,
) -> tuple[Any, dict]:
    """
    Обёртка: по имени вызывает нужный классификатор.

    name: svm | xgboost | rf | random_forest | logistic | knn
    """
    name = name.strip().lower()
    if name == "svm":
        return train_svm(X_train, y_train, X_test, y_test, **kwargs)
    if name in ("xgboost", "xgb"):
        return train_xgboost(X_train, y_train, X_test, y_test, **kwargs)
    if name in ("rf", "random_forest"):
        return train_random_forest(X_train, y_train, X_test, y_test, **kwargs)
    if name in ("logistic", "lr", "logistic_regression"):
        return train_logistic_regression(X_train, y_train, X_test, y_test, **kwargs)
    if name == "knn":
        return train_knn(X_train, y_train, X_test, y_test, **kwargs)
    if name in ("gb", "gradient_boosting"):
        return train_gradient_boosting(X_train, y_train, X_test, y_test, **kwargs)
    raise ValueError(f"Неизвестный классификатор: {name}. Доступны: {CLASSIFIER_NAMES}")


def save_model(
    model: Any,
    path: str,
    scaler: StandardScaler | None = None,
    label_encoder: LabelEncoder | None = None,
) -> None:
    """
    Сохраняет модель и опционально scaler и label_encoder в один архив joblib
    (словарь с ключами model, scaler, label_encoder).
    """
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    bundle = {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
    }
    joblib.dump(bundle, path)
    print(f"Модель сохранена: {path}")


def evaluate_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder | None = None,
) -> dict[str, Any]:
    """
    Оценка модели: метрики на train и test, confusion matrix, отчёт по классам.
    """
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    train_metrics = _compute_metrics(y_train, y_pred_train)
    test_metrics = _compute_metrics(y_test, y_pred_test)
    cm = confusion_matrix(y_test, y_pred_test)
    labels = None if label_encoder is None else list(label_encoder.classes_)
    report = classification_report(
        y_test, y_pred_test, target_names=labels, zero_division=0
    )
    return {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
        "label_encoder": label_encoder,
    }


def check_overfitting(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """
    Проверка на переобучение: сравнение accuracy и F1 на train и test.
    """
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    train_f1 = f1_score(y_train, y_pred_train, average="weighted", zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)
    gap_acc = train_acc - test_acc
    gap_f1 = train_f1 - test_f1
    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_f1": train_f1,
        "test_f1": test_f1,
        "gap_accuracy": gap_acc,
        "gap_f1": gap_f1,
        "likely_overfitting": gap_acc > 0.1 or gap_f1 > 0.1,
    }


def plot_confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder | None = None,
    save_path: str | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """
    Строит тепловую карту (confusion matrix). Сохраняет в save_path или показывает окно.
    """
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except ImportError:
        sns = None
    cm = confusion_matrix(y_true, y_pred)
    labels = None
    if label_encoder is not None:
        labels = list(label_encoder.classes_)
    fig, ax = plt.subplots(figsize=figsize)
    if sns is not None:
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax,
        )
    else:
        ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        if labels:
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
    ax.set_xlabel("Предсказание")
    ax.set_ylabel("Истина")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Тепловая карта сохранена: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def cross_validate_classifier(
    name: str,
    df: pd.DataFrame,
    n_splits: int = 5,
    target_col: str = "target",
    scale: bool = True,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Кросс-валидация по имени классификатора. Возвращает mean ± std по accuracy и F1.
    """
    from sklearn.model_selection import StratifiedKFold
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].values
    y_raw = df[target_col].astype(str).values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    if scale:
        X = StandardScaler().fit_transform(X)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = ["accuracy", "f1_weighted"]
    name_lower = name.strip().lower()
    if name_lower == "svm":
        estimator = SVC(kernel="rbf", probability=False)
    elif name_lower in ("xgboost", "xgb"):
        try:
            import xgboost as xgb
            estimator = xgb.XGBClassifier(eval_metric="logloss")
        except ImportError:
            raise ImportError("Установите xgboost: pip install xgboost")
    elif name_lower in ("rf", "random_forest"):
        estimator = RandomForestClassifier(n_estimators=100)
    elif name_lower in ("logistic", "lr"):
        estimator = LogisticRegression(max_iter=1000)
    elif name_lower == "knn":
        estimator = KNeighborsClassifier(n_neighbors=5)
    elif name_lower in ("gb", "gradient_boosting"):
        estimator = GradientBoostingClassifier(n_estimators=100)
    else:
        raise ValueError(f"Неизвестный классификатор: {name}. Доступны: {CLASSIFIER_NAMES}")
    scores = cross_validate(estimator, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    return {
        "accuracy_mean": scores["test_accuracy"].mean(),
        "accuracy_std": scores["test_accuracy"].std(),
        "f1_weighted_mean": scores["test_f1_weighted"].mean(),
        "f1_weighted_std": scores["test_f1_weighted"].std(),
        "n_splits": n_splits,
    }


def load_model(path: str) -> dict[str, Any]:
    """Загружает бандл {model, scaler, label_encoder} из joblib."""
    return joblib.load(path)
