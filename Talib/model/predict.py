"""
Загрузка бандла joblib и предсказание ранга, return, вероятностей, режима волатильности.
"""

from typing import Any, List, Optional, Tuple

import joblib
import numpy as np

from config import RANK_LABEL_PREFIX


def load_bundle(path: str) -> dict:
    """Загружает бандл (model, scaler, label_encoder, ...) из joblib."""
    return joblib.load(path)


def _prepare_vector(bundle: dict, vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    if vector.ndim == 1:
        vector = vector.reshape(1, -1)
    if vector.shape[0] != 1:
        vector = vector[-1:]
    scaler = bundle.get("scaler")
    if scaler is not None:
        vector = scaler.transform(vector)
    return vector


def predict(bundle: dict, vector: np.ndarray) -> str:
    """
    Предсказывает ранг по вектору признаков (одна строка или 2D массив из одной строки).
    """
    vec = _prepare_vector(bundle, vector)
    model = bundle["model"]
    le = bundle.get("label_encoder")
    pred = model.predict(vec)
    pred_id = int(pred[0])
    if le is not None:
        return str(le.inverse_transform([pred_id])[0])
    return f"{RANK_LABEL_PREFIX}{pred_id}"


def predict_return(bundle: dict, vector: np.ndarray) -> Optional[float]:
    """
    Возвращает точечную оценку форвард-доходности (сырую), если в бандле есть return_regressor.
    Иначе None.
    """
    reg = bundle.get("return_regressor")
    if reg is None:
        return None
    vec = _prepare_vector(bundle, vector)
    raw = float(reg.predict(vec)[0])
    scaler = bundle.get("return_scaler")
    if scaler is not None and hasattr(scaler, "inverse_transform"):
        raw = float(scaler.inverse_transform([[raw]])[0, 0])
    return raw


def predict_proba(bundle: dict, vector: np.ndarray) -> Optional[np.ndarray]:
    """Вероятности по классам (если модель поддерживает predict_proba). Иначе None."""
    vec = _prepare_vector(bundle, vector)
    model = bundle["model"]
    if not hasattr(model, "predict_proba"):
        return None
    return np.asarray(model.predict_proba(vec)[0])


def predict_volatility_regime(bundle: dict, vector: np.ndarray) -> Optional[str]:
    """Предсказание режима волатильности (низкая/средняя/высокая), если есть volatility_classifier."""
    vol_clf = bundle.get("volatility_classifier")
    if vol_clf is None:
        return None
    vec = _prepare_vector(bundle, vector)
    reg_id = int(vol_clf.predict(vec)[0])
    names = ["низкая", "средняя", "высокая"]
    return names[reg_id] if 0 <= reg_id < 3 else str(reg_id)
