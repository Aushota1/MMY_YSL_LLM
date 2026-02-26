"""
Обучение классификатора (RF, SVM, XGBoost), StandardScaler, сохранение joblib.
"""

from typing import Any, Callable, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from config import (
    CLASSIFIER_CHOICES,
    LSTM_BATCH_SIZE,
    LSTM_EPOCHS,
    LSTM_UNITS,
    OVERFIT_ACC_DIFF_THRESHOLD,
    RANK_LABEL_PREFIX,
)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from model.ensemble import get_ensemble_model

try:
    from model.nn_models import HAS_KERAS, get_lstm_classifier
except ImportError:
    HAS_KERAS = False
    get_lstm_classifier = None


def _get_classifier(name: str, **kwargs) -> Any:
    name = name.strip()
    if "forest" in name.lower() or name.lower() == "rf":
        return RandomForestClassifier(n_estimators=100, **kwargs)
    if "svm" in name.lower():
        return SVC(kernel="rbf", probability=True, **kwargs)
    if "xgb" in name.lower() or "boost" in name.lower():
        if not HAS_XGB:
            raise ImportError("Установите xgboost: pip install xgboost")
        return xgb.XGBClassifier(eval_metric="logloss", **kwargs)
    if "logistic" in name.lower() or "lr" in name.lower():
        return LogisticRegression(max_iter=1000, **kwargs)
    return RandomForestClassifier(n_estimators=100, **kwargs)


def _run_cv(
    X_train: np.ndarray,
    y_train_le: np.ndarray,
    classifier_name: str,
    n_cv_splits: int,
    progress_callback: Optional[Callable[[str], None]],
) -> tuple:
    """Кросс-валидация по времени (только для не-LSTM классификаторов)."""
    if n_cv_splits < 2 or len(X_train) < n_cv_splits + 1:
        return [], [], 0.0, 0.0, 0.0, 0.0
    tscv = TimeSeriesSplit(n_splits=n_cv_splits)
    cv_train_scores = []
    cv_test_scores = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_train)):
        if progress_callback:
            progress_callback(f"CV fold {fold + 1}/{n_cv_splits}...\n")
        X_tr, X_te = X_train[train_idx], X_train[test_idx]
        y_tr, y_te = y_train_le[train_idx], y_train_le[test_idx]
        scaler_f = StandardScaler()
        X_tr_s = scaler_f.fit_transform(X_tr)
        X_te_s = scaler_f.transform(X_te)
        clf = get_ensemble_model(classifier_name) or _get_classifier(classifier_name)
        clf.fit(X_tr_s, y_tr)
        train_acc = float(np.mean(clf.predict(X_tr_s) == y_tr))
        test_acc = float(np.mean(clf.predict(X_te_s) == y_te))
        cv_train_scores.append(train_acc)
        cv_test_scores.append(test_acc)
    cv_train_mean = float(np.mean(cv_train_scores))
    cv_test_mean = float(np.mean(cv_test_scores))
    cv_train_std = float(np.std(cv_train_scores)) if len(cv_train_scores) > 1 else 0.0
    cv_test_std = float(np.std(cv_test_scores)) if len(cv_test_scores) > 1 else 0.0
    return cv_train_scores, cv_test_scores, cv_train_mean, cv_train_std, cv_test_mean, cv_test_std


def train_and_save(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classifier_name: str = "Random Forest",
    save_path: str = "model.joblib",
    feature_names: Optional[list] = None,
    window_len: Optional[int] = None,
    horizon: Optional[int] = None,
    n_quantiles: Optional[int] = None,
    predict_return: bool = False,
    y_train_return: Optional[np.ndarray] = None,
    y_test_return: Optional[np.ndarray] = None,
    use_volatility_regime: bool = False,
    y_train_vol: Optional[np.ndarray] = None,
    y_test_vol: Optional[np.ndarray] = None,
    epochs: Optional[int] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    n_cv_splits: int = 0,
) -> dict:
    """
    Обучает классификатор (или ансамбль), сохраняет бандл в joblib.
    Опционально: регрессор return, классификатор режима волатильности.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    le = LabelEncoder()
    y_train_le = le.fit_transform(y_train.astype(str))
    y_test_le = le.transform(y_test.astype(str))

    n_epochs = epochs if epochs is not None else LSTM_EPOCHS
    if n_epochs < 1:
        n_epochs = LSTM_EPOCHS

    def _lstm_progress(ep: int, total: int, loss: float, acc: float) -> None:
        if progress_callback:
            progress_callback(f"Epoch {ep}/{total} loss={loss:.4f} acc={acc:.4f}\n")

    ensemble_clf = get_ensemble_model(classifier_name)
    lstm_clf = None
    if (
        get_lstm_classifier is not None
        and "lstm" in classifier_name.lower()
    ):
        import os
        base = save_path.rsplit(".", 1)[0] if "." in save_path else save_path
        keras_path = base + ".keras"
        lstm_clf = get_lstm_classifier(
            X_train_s, y_train_le,
            window_len=window_len or 20,
            n_quantiles=n_quantiles or 5,
            save_path_keras=keras_path,
            epochs=n_epochs,
            batch_size=LSTM_BATCH_SIZE,
            units=LSTM_UNITS,
            progress_callback=_lstm_progress,
        )
    if lstm_clf is not None:
        clf = lstm_clf
        y_pred = clf.predict(X_test_s)
    elif ensemble_clf is not None:
        clf = ensemble_clf
        if progress_callback:
            progress_callback("Обучение ансамбля...\n")
        clf.fit(X_train_s, y_train_le)
        y_pred = clf.predict(X_test_s)
    else:
        clf = _get_classifier(classifier_name)
        if progress_callback:
            progress_callback("Обучение классификатора...\n")
        clf.fit(X_train_s, y_train_le)
        y_pred = clf.predict(X_test_s)

    acc = float(np.mean(y_pred == y_test_le))
    f1 = float(f1_score(y_test_le, y_pred, average="weighted", zero_division=0))
    precision_w = float(precision_score(y_test_le, y_pred, average="weighted", zero_division=0))
    recall_w = float(recall_score(y_test_le, y_pred, average="weighted", zero_division=0))

    y_train_pred = clf.predict(X_train_s)
    train_acc = float(np.mean(y_train_pred == y_train_le))
    train_f1 = float(f1_score(y_train_le, y_train_pred, average="weighted", zero_division=0))
    overfitting_warning = (train_acc - acc) > OVERFIT_ACC_DIFF_THRESHOLD

    result = {
        "accuracy": acc,
        "f1_weighted": f1,
        "train_accuracy": train_acc,
        "train_f1_weighted": train_f1,
        "precision_weighted": precision_w,
        "recall_weighted": recall_w,
        "overfitting_warning": overfitting_warning,
    }

    if n_cv_splits >= 2 and "lstm" not in classifier_name.lower():
        if progress_callback:
            progress_callback("Кросс-валидация (TimeSeriesSplit)...\n")
        cv_tr, cv_te, cv_tr_mean, cv_tr_std, cv_te_mean, cv_te_std = _run_cv(
            X_train, y_train_le, classifier_name, n_cv_splits, progress_callback
        )
        result["cv_train_scores"] = cv_tr
        result["cv_test_scores"] = cv_te
        result["cv_train_mean"] = cv_tr_mean
        result["cv_train_std"] = cv_tr_std
        result["cv_test_mean"] = cv_te_mean
        result["cv_test_std"] = cv_te_std

    bundle = {
        "model": clf,
        "scaler": scaler,
        "label_encoder": le,
        "feature_names": feature_names or [],
        "window_len": window_len,
        "horizon": horizon,
        "n_quantiles": n_quantiles,
    }

    if predict_return and y_train_return is not None and len(y_train_return) == len(y_train_le):
        from sklearn.linear_model import Ridge
        return_scaler = StandardScaler()
        yr = np.asarray(y_train_return, dtype=np.float64).reshape(-1, 1)
        yr_s = return_scaler.fit_transform(yr)
        return_reg = Ridge(alpha=1.0)
        return_reg.fit(X_train_s, yr_s.ravel())
        bundle["return_regressor"] = return_reg
        bundle["return_scaler"] = return_scaler
    else:
        bundle["return_regressor"] = None
        bundle["return_scaler"] = None

    if use_volatility_regime and y_train_vol is not None and len(y_train_vol) == len(y_train_le):
        vol_clf = RandomForestClassifier(n_estimators=50)
        vol_clf.fit(X_train_s, np.asarray(y_train_vol).ravel().astype(int))
        bundle["volatility_classifier"] = vol_clf
    else:
        bundle["volatility_classifier"] = None

    joblib.dump(bundle, save_path)
    return result
