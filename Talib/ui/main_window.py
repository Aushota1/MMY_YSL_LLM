"""
Главное окно с вкладками и общим state для данных и модели.
"""

import tkinter as tk
from tkinter import ttk

from ui.tabs.data_tab import DataTab
from ui.tabs.train_tab import TrainTab
from ui.tabs.backtest_tab import BacktestTab
from ui.tabs.predict_tab import PredictTab
from ui.tabs.evaluation_tab import EvaluationTab


class AppState:
    """Общее состояние: OHLCV, датасет, модель, результаты бэктеста и метрик."""

    def __init__(self):
        self.ohlcv_df = None
        self.features_df = None
        self.X = None
        self.y = None
        self.dates = None
        self.feature_names = None
        self.window_len = None
        self.horizon = None
        self.n_quantiles = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_dates = None
        self.test_close = None
        self.forward_returns = None
        self.vol_regime = None
        self.test_forward_returns = None
        self.test_vol_regime = None
        self.model_path = None
        self.bundle = None
        self.backtest_trades = None
        self.backtest_equity = None
        self.classification_metrics_result = None
        self.backtest_metrics_result = None
        self.y_test_pred = None


class MainWindow(ttk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.state = AppState()
        self._build_ui()

    def _build_ui(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.data_tab = DataTab(self.notebook, self.state, self.set_status)
        self.notebook.add(self.data_tab, text="Данные")

        self.train_tab = TrainTab(self.notebook, self.state, self.set_status)
        self.notebook.add(self.train_tab, text="Обучение")

        self.backtest_tab = BacktestTab(self.notebook, self.state, self.set_status)
        self.notebook.add(self.backtest_tab, text="Бэктест")

        self.predict_tab = PredictTab(self.notebook, self.state, self.set_status)
        self.notebook.add(self.predict_tab, text="Предсказание")

        self.eval_tab = EvaluationTab(self.notebook, self.state, self.set_status)
        self.notebook.add(self.eval_tab, text="Оценка")

        self.status = ttk.Label(self, text="Готово", relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def set_status(self, text: str):
        self.status.config(text=text)
        self.update_idletasks()
