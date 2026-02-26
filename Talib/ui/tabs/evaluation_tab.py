"""
Вкладка «Оценка»: отчёт классификации (accuracy, F1, confusion matrix) и отчёт бэктеста.
"""

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np


class EvaluationTab(ttk.Frame):
    def __init__(self, parent, state, set_status, **kwargs):
        super().__init__(parent, **kwargs)
        self.state = state
        self.set_status = set_status
        self._build_ui()

    def _build_ui(self):
        grp = ttk.LabelFrame(self, text="Отчёты", padding=5)
        grp.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(grp, text="Показать отчёт классификации", command=self._show_classification).pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Button(grp, text="Показать отчёт бэктеста", command=self._show_backtest).pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Button(grp, text="График confusion matrix", command=self._plot_cm).pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Button(grp, text="Предсказание vs время", command=self._plot_pred_vs_time).pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Button(grp, text="Доходность по классу", command=self._plot_return_by_class).pack(side=tk.LEFT, padx=5, pady=2)

        self.txt_report = tk.Text(self, height=22, width=75, wrap=tk.WORD)
        self.txt_report.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _show_classification(self):
        self.txt_report.delete(1.0, tk.END)
        m = self.state.classification_metrics_result
        if m is None:
            self.txt_report.insert(tk.END, "Сначала обучите модель (вкладка «Обучение»).\n")
            return
        self.txt_report.insert(tk.END, f"Accuracy: {m['accuracy']:.4f}\n")
        self.txt_report.insert(tk.END, f"F1 weighted: {m['f1_weighted']:.4f}\n\n")
        self.txt_report.insert(tk.END, "Classification report:\n")
        self.txt_report.insert(tk.END, m["report_str"])
        self.set_status("Отчёт классификации")

    def _show_backtest(self):
        self.txt_report.delete(1.0, tk.END)
        m = self.state.backtest_metrics_result
        if m is None:
            self.txt_report.insert(tk.END, "Сначала запустите бэктест (вкладка «Бэктест»).\n")
            return
        self.txt_report.insert(tk.END, f"Сделок: {m['n_trades']}\n")
        self.txt_report.insert(tk.END, f"Суммарный PnL: {m['total_pnl']:.6f}\n")
        self.txt_report.insert(tk.END, f"Средний PnL на сделку: {m['avg_pnl_per_trade']:.6f}\n")
        self.txt_report.insert(tk.END, f"Sharpe ratio: {m['sharpe_ratio']:.4f}\n")
        self.txt_report.insert(tk.END, f"Max drawdown: {m['max_drawdown']:.6f}\n")
        self.txt_report.insert(tk.END, f"Max drawdown %: {m['max_drawdown_pct']:.2f}%\n")
        self.set_status("Отчёт бэктеста")

    def _plot_cm(self):
        m = self.state.classification_metrics_result
        if m is None:
            messagebox.showinfo("Нет данных", "Сначала обучите модель.")
            return
        cm = m["confusion_matrix"]
        if cm.size == 0:
            messagebox.showinfo("Нет данных", "Confusion matrix пуста.")
            return
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm, cmap="Blues")
            nq = cm.shape[0]
            ax.set_xticks(range(nq))
            ax.set_yticks(range(nq))
            ax.set_xticklabels([f"rank_{i}" for i in range(nq)])
            ax.set_yticklabels([f"rank_{i}" for i in range(nq)])
            ax.set_xlabel("Предсказание")
            ax.set_ylabel("Истина")
            for i in range(nq):
                for j in range(nq):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center")
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def _plot_pred_vs_time(self):
        if self.state.y_test_pred is None or self.state.test_dates is None:
            messagebox.showinfo("Нет данных", "Сначала обучите модель.")
            return
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            import pandas as pd
            dates = pd.to_datetime(self.state.test_dates)
            y_true = np.asarray(self.state.y_test).ravel()
            y_pred = np.asarray(self.state.y_test_pred).ravel()
            if len(dates) != len(y_true) or len(dates) != len(y_pred):
                messagebox.showwarning("Данные", "Длины дат и предсказаний не совпадают.")
                return
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(dates, y_true, label="Истинный ранг", alpha=0.8)
            ax.plot(dates, y_pred, label="Предсказанный ранг", alpha=0.8)
            ax.set_xlabel("Дата")
            ax.set_ylabel("Ранг (класс)")
            ax.legend()
            ax.set_title("Предсказание vs время")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            self.set_status("График: предсказание vs время")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def _plot_return_by_class(self):
        if self.state.y_test_pred is None or self.state.test_forward_returns is None:
            messagebox.showinfo("Нет данных", "Обучите модель и постройте датасет с forward returns.")
            return
        preds = np.asarray(self.state.y_test_pred).ravel()
        returns = np.asarray(self.state.test_forward_returns).ravel()
        if len(preds) != len(returns):
            messagebox.showwarning("Данные", "Длины предсказаний и доходностей не совпадают.")
            return
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            nq = int(np.max(preds)) + 1
            data_by_class = [returns[preds == k] for k in range(nq)]
            labels = [f"rank_{k}" for k in range(nq)]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.boxplot(data_by_class, labels=labels)
            ax.set_ylabel("Реальная форвард-доходность")
            ax.set_xlabel("Предсказанный класс")
            ax.set_title("Распределение доходности по предсказанному классу")
            plt.tight_layout()
            plt.show()
            self.set_status("График: доходность по классу")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
