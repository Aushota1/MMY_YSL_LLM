"""
Вкладка «Данные»: загрузка рыночных данных (yfinance / CSV) и построение датасета.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd

from config import (
    YFINANCE_PERIODS,
    YFINANCE_INTERVALS,
    DEFAULT_WINDOW_LEN,
    DEFAULT_HORIZON,
    DEFAULT_N_QUANTILES,
    USE_TIME_FEATURES,
)
from data.fetcher import fetch_ohlcv, load_ohlcv_csv, save_ohlcv_csv
from data.features import compute_talib_features, build_windows
from model.dataset import time_train_test_split


class DataTab(ttk.Frame):
    def __init__(self, parent, state, set_status, **kwargs):
        super().__init__(parent, **kwargs)
        self.state = state
        self.set_status = set_status
        self._build_ui()

    def _build_ui(self):
        # Секция загрузки
        grp_fetch = ttk.LabelFrame(self, text="Загрузка рыночных данных", padding=5)
        grp_fetch.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(grp_fetch, text="Тикер:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.var_ticker = tk.StringVar(value="AAPL")
        ttk.Entry(grp_fetch, textvariable=self.var_ticker, width=15).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(grp_fetch, text="Период:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.var_period = tk.StringVar(value="6mo")
        cb_period = ttk.Combobox(grp_fetch, textvariable=self.var_period, values=YFINANCE_PERIODS, width=10)
        cb_period.grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(grp_fetch, text="Интервал:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.var_interval = tk.StringVar(value="1d")
        cb_interval = ttk.Combobox(grp_fetch, textvariable=self.var_interval, values=YFINANCE_INTERVALS, width=10)
        cb_interval.grid(row=2, column=1, padx=5, pady=2)

        ttk.Button(grp_fetch, text="Скачать", command=self._on_download).grid(row=3, column=0, columnspan=2, pady=5)
        ttk.Button(grp_fetch, text="Загрузить из CSV", command=self._on_load_csv).grid(row=4, column=0, columnspan=2, pady=2)

        # Построение датасета
        grp_build = ttk.LabelFrame(self, text="Построение датасета", padding=5)
        grp_build.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(grp_build, text="Окно (баров):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.var_window = tk.StringVar(value=str(DEFAULT_WINDOW_LEN))
        ttk.Entry(grp_build, textvariable=self.var_window, width=8).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(grp_build, text="Горизонт (баров):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.var_horizon = tk.StringVar(value=str(DEFAULT_HORIZON))
        ttk.Entry(grp_build, textvariable=self.var_horizon, width=8).grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(grp_build, text="Квантилей (классов):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.var_quantiles = tk.StringVar(value=str(DEFAULT_N_QUANTILES))
        ttk.Entry(grp_build, textvariable=self.var_quantiles, width=8).grid(row=2, column=1, padx=5, pady=2)

        ttk.Button(grp_build, text="Построить датасет", command=self._on_build_dataset).grid(row=3, column=0, columnspan=2, pady=5)

        self.txt_summary = tk.Text(self, height=12, width=70, wrap=tk.WORD)
        self.txt_summary.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _on_download(self):
        self.set_status("Загрузка данных...")
        try:
            ticker = self.var_ticker.get().strip()
            period = self.var_period.get().strip()
            interval = self.var_interval.get().strip()
            path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv")],
                initialdir=os.path.join(os.path.dirname(__file__), "..", ".."),
            )
            if not path:
                self.set_status("Отменено")
                return
            df = fetch_ohlcv(ticker, period=period, interval=interval, save_path=path)
            self.state.ohlcv_df = df
            self.set_status(f"Загружено {len(df)} баров. Сохранено: {path}")
            self.txt_summary.delete(1.0, tk.END)
            self.txt_summary.insert(tk.END, f"OHLCV: {len(df)} строк.\nКолонки: {list(df.columns)}\n")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            self.set_status("Ошибка загрузки")
        else:
            self.set_status("Данные загружены")

    def _on_load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path:
            return
        self.set_status("Чтение CSV...")
        try:
            df = load_ohlcv_csv(path)
            self.state.ohlcv_df = df
            self.set_status(f"Загружено {len(df)} баров из {path}")
            self.txt_summary.delete(1.0, tk.END)
            self.txt_summary.insert(tk.END, f"OHLCV: {len(df)} строк.\n")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            self.set_status("Ошибка")

    def _on_build_dataset(self):
        if self.state.ohlcv_df is None or self.state.ohlcv_df.empty:
            messagebox.showwarning("Нет данных", "Сначала загрузите OHLCV (Скачать или Загрузить из CSV).")
            return
        self.set_status("Построение датасета...")
        try:
            w = int(self.var_window.get())
            h = int(self.var_horizon.get())
            q = int(self.var_quantiles.get())
            feats = compute_talib_features(self.state.ohlcv_df, use_time_features=USE_TIME_FEATURES)
            X, y, dates, feature_names, forward_returns, vol_regime = build_windows(feats, w, h, n_quantiles=q)
            self.state.X = X
            self.state.y = y
            self.state.dates = dates
            self.state.feature_names = feature_names
            self.state.window_len = w
            self.state.horizon = h
            self.state.n_quantiles = q
            self.state.features_df = feats
            self.state.forward_returns = forward_returns
            self.state.vol_regime = vol_regime

            X_train, X_test, y_train, y_test, test_dates = time_train_test_split(X, y, dates, train_ratio=0.7)
            self.state.X_train = X_train
            self.state.X_test = X_test
            self.state.y_train = y_train
            self.state.y_test = y_test
            self.state.test_dates = test_dates
            close_series = self.state.ohlcv_df["close"]
            self.state.test_close = close_series.reindex(
                [pd.Timestamp(d) for d in test_dates]
            ).ffill().bfill().values
            split_idx = len(X_train)
            self.state.test_forward_returns = forward_returns[split_idx:] if forward_returns is not None and len(forward_returns) > split_idx else None

            _, counts = np.unique(y, return_counts=True)
            summary = f"Примеров: {len(X)}, признаков: {X.shape[1]}, классов: {q}\n"
            summary += f"Train: {len(X_train)}, Test: {len(X_test)}\n"
            summary += f"Распределение классов (всего): {dict(zip(range(q), counts))}\n"
            self.txt_summary.delete(1.0, tk.END)
            self.txt_summary.insert(tk.END, summary)
            self.set_status("Датасет построен")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            self.set_status("Ошибка построения датасета")
            import traceback
            traceback.print_exc()
