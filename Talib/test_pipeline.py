"""
Проверка пайплайна без GUI: данные -> признаки -> окна -> обучение -> бэктест -> предсказание.
Запуск из папки Talib: python test_pipeline.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from data.features import compute_talib_features, build_windows
from model.dataset import time_train_test_split
from model.train import train_and_save
from model.predict import load_bundle, predict
from backtest.engine import run_backtest
from evaluation.metrics import classification_metrics, backtest_metrics

# Минимальный OHLCV (100 баров)
np.random.seed(42)
n = 100
t = pd.date_range("2024-01-01", periods=n, freq="1D")
close = 100 + np.cumsum(np.random.randn(n) * 0.5)
high = close + np.abs(np.random.randn(n))
low = close - np.abs(np.random.randn(n))
open_ = np.roll(close, 1)
open_[0] = close[0]
volume = np.random.randint(1000, 10000, n)
df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=t)
df.index.name = "date"

feats = compute_talib_features(df)
print("Features shape:", feats.shape)

X, y, dates, feature_names, forward_returns, vol_regime = build_windows(feats, window_len=10, horizon=3, n_quantiles=5)
print("X shape:", X.shape, "y shape:", y.shape)

X_train, X_test, y_train, y_test, test_dates = time_train_test_split(X, y, dates, train_ratio=0.7)
print("Train:", len(X_train), "Test:", len(X_test))

path = os.path.join(os.path.dirname(__file__), "models", "test_model.joblib")
os.makedirs(os.path.dirname(path), exist_ok=True)
train_and_save(
    X_train, y_train, X_test, y_test,
    classifier_name="Random Forest",
    save_path=path,
    feature_names=feature_names, window_len=10, horizon=3, n_quantiles=5,
)
print("Model saved")

bundle = load_bundle(path)
preds = bundle["model"].predict(bundle["scaler"].transform(X_test))
cm = classification_metrics(y_test, preds)
print("Accuracy:", cm["accuracy"])

close_series = df["close"]
test_close = close_series.reindex(test_dates).ffill().bfill().values
trades, equity = run_backtest(test_dates, test_close, preds, n_quantiles=5)
bt = backtest_metrics(trades, equity)
print("Backtest trades:", bt["n_trades"], "PnL:", bt["total_pnl"])

rank = predict(bundle, X_test[-1])
print("Last window prediction:", rank)
print("Pipeline OK")
