"""
Признаки TA-Lib по OHLCV и построение окон с метками по квантилям форвард-доходности.
Опционально: время (hour, dayofweek), режим волатильности (0/1/2).
"""

from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from config import OHLCV_COLUMNS, USE_TIME_FEATURES


def compute_talib_features(df: pd.DataFrame, use_time_features: bool = False) -> pd.DataFrame:
    """
    По DataFrame OHLCV считает признаки: RSI, MACD, BBANDS, ATR, OBV, ret, range_hl.
    NaN обрабатываются: ffill, затем строки с оставшимися NaN отбрасываются.
    """
    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)
    v = df["volume"].values.astype(np.float64)

    out = pd.DataFrame(index=df.index)

    if HAS_TALIB:
        out["rsi"] = talib.RSI(c, timeperiod=14)
        macd, macdsig, macdhist = talib.MACD(c, 12, 26, 9)
        out["macd"] = macd
        out["macd_sig"] = macdsig
        out["macd_hist"] = macdhist
        bb_u, bb_m, bb_l = talib.BBANDS(c, 20, 2, 2)
        out["bb_upper"] = bb_u
        out["bb_mid"] = bb_m
        out["bb_lower"] = bb_l
        out["atr"] = talib.ATR(h, l, c, 14)
        out["obv"] = talib.OBV(c, v)
    else:
        out["rsi"] = 50.0
        out["macd"] = 0.0
        out["macd_sig"] = 0.0
        out["macd_hist"] = 0.0
        out["bb_upper"] = c
        out["bb_mid"] = c
        out["bb_lower"] = c
        out["atr"] = (h - l)
        out["obv"] = np.cumsum(v)

    out["ret"] = np.log(c / np.roll(c, 1))
    out.iloc[0, out.columns.get_loc("ret")] = np.nan
    out["range_hl"] = (h - l) / (c + 1e-8)

    atr_series = out["atr"]
    atr_roll = atr_series.rolling(20, min_periods=1).mean()
    atr_pct = atr_series / (atr_roll + 1e-8)
    try:
        vol_bins = pd.qcut(atr_pct.rank(method="first"), 3, labels=[0, 1, 2], duplicates="drop")
        out["volatility_regime"] = vol_bins.astype(float).fillna(1).astype(int)
    except Exception:
        out["volatility_regime"] = 1
    out["volatility_regime"] = out["volatility_regime"].ffill().bfill().astype(int)

    if use_time_features and hasattr(out.index, "hour"):
        out["hour"] = out.index.hour
        out["dayofweek"] = out.index.dayofweek
    elif use_time_features:
        out["hour"] = 12
        out["dayofweek"] = 2

    out = out.ffill().bfill()
    out = out.dropna(how="any")
    return out


def build_windows(
    features: pd.DataFrame,
    window_len: int,
    horizon: int,
    n_quantiles: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray]:
    """
    Строит окна и метки по квантилям форвард-доходности.
    Также возвращает сырую форвард-доходность и режим волатильности по каждому сэмплу.

    Возвращает
    ----------
    X, y, dates, feature_names : как раньше
    forward_returns : np.ndarray, длина = n_samples (сырая сумма log-return за horizon)
    vol_regime : np.ndarray, длина = n_samples (0/1/2)
    """
    features = features.dropna(how="all").ffill().dropna()
    if "ret" not in features.columns:
        raise ValueError("В features должен быть столбец 'ret'")

    fwd = sum(
        features["ret"].shift(-i) for i in range(1, horizon + 1)
    )
    fwd = fwd.dropna()
    try:
        labels = pd.qcut(
            fwd.rank(method="first"),
            n_quantiles,
            labels=False,
            duplicates="drop",
        )
    except Exception:
        labels = (fwd.rank(method="first") * (n_quantiles - 1) / max(1, fwd.rank(method="first").max())).astype(int).clip(0, n_quantiles - 1)

    has_vol = "volatility_regime" in features.columns
    feature_cols = [c for c in features.columns if c != "ret"]
    dates_index = []
    X_list = []
    y_list = []
    fwd_list = []
    vol_list = []

    for i in range(window_len, len(features) - horizon):
        idx_i = features.index[i]
        if idx_i not in labels.index:
            continue
        window = features.iloc[i - window_len : i][feature_cols]
        if window.isna().any().any():
            continue
        X_list.append(window.values.flatten())
        y_list.append(int(labels.loc[idx_i]))
        dates_index.append(idx_i)
        fwd_list.append(float(fwd.loc[idx_i]))
        vol_list.append(int(features.loc[idx_i, "volatility_regime"]) if has_vol else 1)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    dates = np.array(dates_index)
    forward_returns = np.array(fwd_list, dtype=np.float64)
    vol_regime = np.array(vol_list, dtype=np.int64)

    feature_names = [f"{col}_t{k}" for k in range(window_len) for col in feature_cols]
    return X, y, dates, feature_names, forward_returns, vol_regime
