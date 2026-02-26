"""
Вкладка «Предсказание»: выбор модели, источник данных (текущий датасет / CSV), кнопка Предсказать.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from config import USE_TIME_FEATURES
from data.features import compute_talib_features, build_windows
from data.fetcher import load_ohlcv_csv
from model.predict import load_bundle, predict, predict_return, predict_proba, predict_volatility_regime


class PredictTab(ttk.Frame):
    def __init__(self, parent, state, set_status, **kwargs):
        super().__init__(parent, **kwargs)
        self.state = state
        self.set_status = set_status
        self._build_ui()

    def _build_ui(self):
        grp = ttk.LabelFrame(self, text="Предсказание ранга", padding=5)
        grp.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(grp, text="Модель (joblib):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.var_model = tk.StringVar()
        ttk.Entry(grp, textvariable=self.var_model, width=50).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(grp, text="Обзор", command=self._browse).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(grp, text="Источник:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.var_source = tk.StringVar(value="current")
        f = ttk.Frame(grp)
        f.grid(row=2, column=1, sticky=tk.W, padx=5)
        ttk.Radiobutton(f, text="Текущий датасет (последнее окно)", variable=self.var_source, value="current").pack(anchor=tk.W)
        ttk.Radiobutton(f, text="Загрузить CSV и предсказать по последнему окну", variable=self.var_source, value="csv").pack(anchor=tk.W)

        ttk.Button(grp, text="Предсказать", command=self._on_predict).grid(row=3, column=0, columnspan=2, pady=10)

        grp_assess = ttk.LabelFrame(self, text="Полная оценка рынка", padding=5)
        grp_assess.pack(fill=tk.X, padx=5, pady=5)
        self.txt_assess = tk.Text(grp_assess, height=6, width=70, wrap=tk.WORD)
        self.txt_assess.pack(fill=tk.X)

        self.lbl_result = ttk.Label(self, text="Результат: —", font=("", 12))
        self.lbl_result.pack(pady=10, padx=5)
        self.txt_log = tk.Text(self, height=8, width=70, wrap=tk.WORD)
        self.txt_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _browse(self):
        path = filedialog.askopenfilename(filetypes=[("Joblib", "*.joblib")])
        if path:
            self.var_model.set(path)

    def _on_predict(self):
        path = self.var_model.get().strip() or self.state.model_path
        if not path or not os.path.isfile(path):
            messagebox.showwarning("Модель", "Выберите файл модели.")
            return
        self.txt_log.delete(1.0, tk.END)
        self.set_status("Предсказание...")
        try:
            bundle = load_bundle(path)
            if self.var_source.get() == "csv":
                fp = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
                if not fp:
                    self.set_status("Отменено")
                    return
                df = load_ohlcv_csv(fp)
                feats = compute_talib_features(df, use_time_features=USE_TIME_FEATURES)
                w = bundle.get("window_len", 20)
                h = bundle.get("horizon", 5)
                nq = bundle.get("n_quantiles", 5)
                X, _, _, _, _, _ = build_windows(feats, w, h, n_quantiles=nq)
                if len(X) == 0:
                    messagebox.showwarning("Нет окон", "Недостаточно данных для окна.")
                    return
                vector = X[-1]
            else:
                if self.state.X_test is None or len(self.state.X_test) == 0:
                    if self.state.X is not None and len(self.state.X) > 0:
                        vector = self.state.X[-1]
                    else:
                        messagebox.showwarning("Нет данных", "Постройте датасет на вкладке «Данные».")
                        return
                else:
                    vector = self.state.X_test[-1]
            rank = predict(bundle, vector)
            self.lbl_result.config(text=f"Ранг: {rank}")

            self.txt_assess.delete(1.0, tk.END)
            lines = [f"Ранг (направление): {rank}"]
            ret = predict_return(bundle, vector)
            if ret is not None:
                lines.append(f"Ожидаемая доходность (forward return): {ret:.6f}")
            vol = predict_volatility_regime(bundle, vector)
            if vol is not None:
                lines.append(f"Режим волатильности: {vol}")
            probs = predict_proba(bundle, vector)
            if probs is not None:
                lines.append("Вероятности по классам: " + ", ".join(f"rank_{i}: {p:.3f}" for i, p in enumerate(probs)))
            self.txt_assess.insert(tk.END, "\n".join(lines))

            self.txt_log.insert(tk.END, f"Вектор длины {len(vector)}\nПредсказанный класс: {rank}\n")
            self.set_status("Готово")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            self.txt_log.insert(tk.END, str(e))
            self.set_status("Ошибка")
