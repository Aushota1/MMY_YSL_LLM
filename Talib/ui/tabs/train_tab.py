"""
Вкладка «Обучение»: выбор классификатора, доля train, путь сохранения, кнопка Обучить.
Отслеживание обучения, метрики, кросс-валидация, проверка переобучения, эпохи LSTM.
"""

import os
import re
import threading
import tkinter as tk
from queue import Empty, Queue
from tkinter import ttk, filedialog, messagebox

import pandas as pd
from config import CLASSIFIER_CHOICES, DEFAULT_TRAIN_RATIO, LSTM_EPOCHS

def _classifier_choices():
    choices = list(CLASSIFIER_CHOICES)
    try:
        from model.nn_models import HAS_KERAS
        if HAS_KERAS:
            choices.append("LSTM")
    except Exception:
        pass
    return choices
from model.train import train_and_save
from model.predict import load_bundle
from evaluation.metrics import classification_metrics


class TrainTab(ttk.Frame):
    def __init__(self, parent, state, set_status, **kwargs):
        super().__init__(parent, **kwargs)
        self.state = state
        self.set_status = set_status
        self._train_queue = Queue()
        self._train_thread = None
        self._poll_after_id = None
        self._build_ui()

    def _build_ui(self):
        grp = ttk.LabelFrame(self, text="Параметры обучения", padding=5)
        grp.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(grp, text="Классификатор:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.var_clf = tk.StringVar(value=CLASSIFIER_CHOICES[0])
        cb = ttk.Combobox(grp, textvariable=self.var_clf, values=_classifier_choices(), width=24)
        cb.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(grp, text="Доля train (0–1):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.var_train_ratio = tk.StringVar(value=str(DEFAULT_TRAIN_RATIO))
        ttk.Entry(grp, textvariable=self.var_train_ratio, width=8).grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(grp, text="Эпохи (LSTM):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.var_epochs = tk.StringVar(value=str(LSTM_EPOCHS))
        ttk.Entry(grp, textvariable=self.var_epochs, width=8).grid(row=2, column=1, padx=5, pady=2)

        ttk.Label(grp, text="Кросс-валидация (разбиений, 0=выкл):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.var_cv_splits = tk.StringVar(value="0")
        ttk.Entry(grp, textvariable=self.var_cv_splits, width=8).grid(row=3, column=1, padx=5, pady=2)

        ttk.Label(grp, text="Путь модели:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.var_path = tk.StringVar(value=os.path.join(os.path.dirname(__file__), "..", "..", "models", "model.joblib"))
        ttk.Entry(grp, textvariable=self.var_path, width=50).grid(row=4, column=1, padx=5, pady=2)
        ttk.Button(grp, text="Обзор", command=self._browse).grid(row=5, column=1, sticky=tk.W, padx=5, pady=2)

        self.var_predict_return = tk.BooleanVar(value=False)
        ttk.Checkbutton(grp, text="Регрессия return (оценка размера движения)", variable=self.var_predict_return).grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=2)
        self.var_use_volatility = tk.BooleanVar(value=False)
        ttk.Checkbutton(grp, text="Режим волатильности (низкая/средняя/высокая)", variable=self.var_use_volatility).grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=2)

        ttk.Button(grp, text="Обучить", command=self._on_train).grid(row=8, column=0, columnspan=2, pady=10)

        self.progress_bar = ttk.Progressbar(self, mode="indeterminate")
        self.progress_bar.pack(fill=tk.X, padx=5, pady=2)

        self.txt_log = tk.Text(self, height=15, width=70, wrap=tk.WORD)
        self.txt_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _browse(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".joblib",
            filetypes=[("Joblib", "*.joblib")],
            initialdir=os.path.join(os.path.dirname(__file__), "..", "..", "models"),
        )
        if path:
            self.var_path.set(path)

    def _parse_epoch(self, msg):
        m = re.match(r"Epoch (\d+)/(\d+)", msg)
        if m:
            return int(m.group(1)), int(m.group(2))
        return None

    def _poll_progress(self):
        try:
            while True:
                item = self._train_queue.get_nowait()
                if isinstance(item, str):
                    self.txt_log.insert(tk.END, item)
                    self.txt_log.see(tk.END)
                    parsed = self._parse_epoch(item)
                    if parsed:
                        cur, total = parsed
                        self.progress_bar.stop()
                        self.progress_bar.config(mode="determinate", maximum=total, value=cur)
                elif isinstance(item, tuple):
                    if item[0] == "DONE":
                        self._train_thread = None
                        self.progress_bar.stop()
                        self.progress_bar.config(mode="indeterminate")
                        self._append_metrics_log(item[1])
                        self.state.model_path = item[2]
                        self.state.bundle = load_bundle(item[2])
                        import numpy as np
                        X_test = self.state.X_test
                        y_test = self.state.y_test
                        y_pred = self.state.bundle["model"].predict(
                            self.state.bundle["scaler"].transform(X_test)
                        )
                        self.state.y_test_pred = y_pred
                        self.state.classification_metrics_result = classification_metrics(
                            y_test, y_pred,
                            label_names=[f"rank_{i}" for i in range(self.state.n_quantiles)],
                        )
                        self.set_status("Обучение завершено")
                        messagebox.showinfo("Готово", f"Модель сохранена: {item[2]}")
                        return
                    elif item[0] == "ERROR":
                        self._train_thread = None
                        self.progress_bar.stop()
                        self.progress_bar.config(mode="indeterminate")
                        exc = item[1]
                        messagebox.showerror("Ошибка", str(exc))
                        self.txt_log.insert(tk.END, str(exc) + "\n")
                        self.set_status("Ошибка обучения")
                        import traceback
                        traceback.print_exc()
                        return
        except Empty:
            pass
        if self._train_thread is not None and self._train_thread.is_alive():
            self._poll_after_id = self.after(200, self._poll_progress)
        else:
            self._poll_after_id = None

    def _append_metrics_log(self, metrics):
        if not metrics:
            return
        self.txt_log.insert(tk.END, "\n--- Итоговые метрики ---\n")
        self.txt_log.insert(tk.END, f"Train accuracy: {metrics.get('train_accuracy', 0):.4f}\n")
        self.txt_log.insert(tk.END, f"Train F1 (weighted): {metrics.get('train_f1_weighted', 0):.4f}\n")
        self.txt_log.insert(tk.END, f"Test accuracy: {metrics.get('accuracy', 0):.4f}\n")
        self.txt_log.insert(tk.END, f"Test F1 (weighted): {metrics.get('f1_weighted', 0):.4f}\n")
        self.txt_log.insert(tk.END, f"Test precision (weighted): {metrics.get('precision_weighted', 0):.4f}\n")
        self.txt_log.insert(tk.END, f"Test recall (weighted): {metrics.get('recall_weighted', 0):.4f}\n")
        if metrics.get("overfitting_warning"):
            self.txt_log.insert(tk.END, "Внимание: возможное переобучение (train accuracy значительно выше test).\n")
        cv_test_mean = metrics.get("cv_test_mean")
        if cv_test_mean is not None:
            cv_test_std = metrics.get("cv_test_std") or 0.0
            cv_train_mean = metrics.get("cv_train_mean") or 0.0
            cv_train_std = metrics.get("cv_train_std") or 0.0
            self.txt_log.insert(tk.END, f"CV test accuracy: {cv_test_mean:.4f} ± {cv_test_std:.4f}\n")
            self.txt_log.insert(tk.END, f"CV train accuracy: {cv_train_mean:.4f} ± {cv_train_std:.4f}\n")

    def _run_train(self, path, train_ratio, epochs_val, n_cv_splits):
        try:
            from model.dataset import time_train_test_split
            X_train, X_test, y_train, y_test, test_dates = time_train_test_split(
                self.state.X, self.state.y, self.state.dates, train_ratio=train_ratio
            )
            split_idx = len(X_train)
            y_train_return = None
            y_test_return = None
            y_train_vol = None
            y_test_vol = None
            if self.state.forward_returns is not None and len(self.state.forward_returns) == len(self.state.y):
                y_train_return = self.state.forward_returns[:split_idx]
                y_test_return = self.state.forward_returns[split_idx:]
            if self.state.vol_regime is not None and len(self.state.vol_regime) == len(self.state.y):
                y_train_vol = self.state.vol_regime[:split_idx]
                y_test_vol = self.state.vol_regime[split_idx:]

            def progress_cb(msg):
                self._train_queue.put(msg)

            metrics = train_and_save(
                X_train, y_train, X_test, y_test,
                classifier_name=self.var_clf.get(),
                save_path=path,
                feature_names=self.state.feature_names,
                window_len=self.state.window_len,
                horizon=self.state.horizon,
                n_quantiles=self.state.n_quantiles,
                predict_return=self.var_predict_return.get(),
                y_train_return=y_train_return,
                y_test_return=y_test_return,
                use_volatility_regime=self.var_use_volatility.get(),
                y_train_vol=y_train_vol,
                y_test_vol=y_test_vol,
                epochs=epochs_val,
                progress_callback=progress_cb,
                n_cv_splits=n_cv_splits,
            )

            self.state.X_train = X_train
            self.state.X_test = X_test
            self.state.y_train = y_train
            self.state.y_test = y_test
            self.state.test_dates = test_dates
            close_series = self.state.ohlcv_df["close"]
            self.state.test_close = close_series.reindex(
                [pd.Timestamp(d) for d in test_dates]
            ).ffill().bfill().values
            if self.state.forward_returns is not None:
                self.state.test_forward_returns = self.state.forward_returns[split_idx:]
            if self.state.vol_regime is not None:
                self.state.test_vol_regime = self.state.vol_regime[split_idx:]

            self._train_queue.put(("DONE", metrics, path))
        except Exception as e:
            self._train_queue.put(("ERROR", e))

    def _on_train(self):
        if self.state.X is None or self.state.y is None:
            messagebox.showwarning("Нет данных", "Сначала постройте датасет на вкладке «Данные».")
            return
        path = self.var_path.get().strip()
        if not path:
            messagebox.showwarning("Путь", "Укажите путь сохранения модели.")
            return
        try:
            epochs_val = int(self.var_epochs.get())
            if epochs_val < 1:
                epochs_val = LSTM_EPOCHS
        except ValueError:
            epochs_val = LSTM_EPOCHS
        try:
            n_cv_splits = int(self.var_cv_splits.get())
            if n_cv_splits < 0:
                n_cv_splits = 0
        except ValueError:
            n_cv_splits = 0

        train_ratio = 0.7
        try:
            train_ratio = float(self.var_train_ratio.get())
            if train_ratio <= 0 or train_ratio >= 1:
                train_ratio = 0.7
        except ValueError:
            pass

        self.set_status("Обучение...")
        self.txt_log.delete(1.0, tk.END)
        if "lstm" in self.var_clf.get().lower():
            self.txt_log.insert(tk.END, f"LSTM (Keras), эпох: {epochs_val}. Обучение...\n")
        self.progress_bar.config(mode="indeterminate")
        self.progress_bar.start(8)
        self._train_thread = threading.Thread(
            target=self._run_train,
            args=(path, train_ratio, epochs_val, n_cv_splits),
            daemon=True,
        )
        self._train_thread.start()
        self._poll_after_id = self.after(200, self._poll_progress)
