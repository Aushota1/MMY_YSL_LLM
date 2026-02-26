"""
Загрузка OHLCV: yfinance и чтение/запись CSV.
Колонки: open, high, low, close, volume; индекс datetime.
"""

import os
from typing import Optional

import pandas as pd

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

from config import OHLCV_COLUMNS


def fetch_ohlcv(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Загружает OHLCV по тикеру через yfinance.

    Параметры
    ---------
    ticker : str
        Тикер (например AAPL, BTC-USD).
    period : str
        Период (1mo, 3mo, 6mo, 1y, 2y, 5y).
    interval : str
        Интервал свечей (1d, 1h, 5m, 15m).
    save_path : str, optional
        Путь для сохранения CSV.

    Возвращает
    ----------
    pd.DataFrame
        Колонки open, high, low, close, volume; индекс DatetimeIndex.
    """
    if not HAS_YFINANCE:
        raise ImportError("Установите yfinance: pip install yfinance")

    obj = yf.Ticker(ticker)
    df = obj.history(period=period, interval=interval, auto_adjust=True)

    if df is None or df.empty:
        raise ValueError(f"Нет данных для тикера {ticker} (period={period}, interval={interval})")

    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    df = df[OHLCV_COLUMNS].copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if save_path:
        save_ohlcv_csv(df, save_path)

    return df


def save_ohlcv_csv(df: pd.DataFrame, path: str) -> None:
    """Сохраняет DataFrame OHLCV в CSV (индекс — дата)."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=True)


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    """
    Загружает OHLCV из CSV.
    Ожидаются колонки open, high, low, close, volume и индекс/колонка даты.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Файл не найден: {path}")

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.strip().lower() for c in df.columns]

    for col in OHLCV_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"В CSV отсутствует колонка: {col}. Найдены: {list(df.columns)}")

    df = df[OHLCV_COLUMNS].copy()
    df = df.sort_index()
    return df
