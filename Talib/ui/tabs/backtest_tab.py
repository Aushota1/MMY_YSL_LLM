"""
Вкладка «Бэктест»: выбор модели, комиссия, проскальзывание, запуск бэктеста, график капитала.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

from config import (
    DEFAULT_COMMISSION_PCT,
    DEFAULT_SLIPPAGE_PCT,
    DEFAULT_MIN_RETURN_THRESHOLD,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_HOLD_BARS,
)
from model.predict import load_bundle, predict, predict_return, predict_proba
from backtest.engine import run_backtest
from rl.train_rl import build_states, train_agent, save_agent, load_agent
from rl.policy import Policy
from evaluation.metrics import backtest_metrics


class BacktestTab(ttk.Frame):
    def __init__(self, parent, state, set_status, **kwargs):
        super().__init__(parent, **kwargs)
        self.state = state
        self.set_status = set_status
        self._build_ui()

    def _build_ui(self):
        grp = ttk.LabelFrame(self, text="Параметры бэктеста", padding=5)
        grp.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(grp, text="Файл модели:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.var_model = tk.StringVar()
        ttk.Entry(grp, textvariable=self.var_model, width=45).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(grp, text="Обзор", command=self._browse_model).grid(row=1, column=1, sticky=tk.W, padx=5)

        ttk.Label(grp, text="Комиссия (%):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.var_comm = tk.StringVar(value=str(DEFAULT_COMMISSION_PCT * 100))
        ttk.Entry(grp, textvariable=self.var_comm, width=10).grid(row=2, column=1, padx=5, pady=2)

        ttk.Label(grp, text="Проскальзывание (%):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.var_slip = tk.StringVar(value=str(DEFAULT_SLIPPAGE_PCT * 100))
        ttk.Entry(grp, textvariable=self.var_slip, width=10).grid(row=3, column=1, padx=5, pady=2)

        ttk.Label(grp, text="Мин. return для входа (0=выкл):").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.var_min_ret = tk.StringVar(value=str(DEFAULT_MIN_RETURN_THRESHOLD))
        ttk.Entry(grp, textvariable=self.var_min_ret, width=10).grid(row=4, column=1, padx=5, pady=2)

        ttk.Label(grp, text="Мин. уверенность (0=выкл):").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.var_min_conf = tk.StringVar(value=str(DEFAULT_MIN_CONFIDENCE))
        ttk.Entry(grp, textvariable=self.var_min_conf, width=10).grid(row=5, column=1, padx=5, pady=2)

        ttk.Label(grp, text="Держать позицию (баров, 0=по сигналу):").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.var_hold = tk.StringVar(value=str(DEFAULT_HOLD_BARS or 0))
        ttk.Entry(grp, textvariable=self.var_hold, width=10).grid(row=6, column=1, padx=5, pady=2)

        ttk.Label(grp, text="RL агент (joblib, пусто=ранги):").grid(row=7, column=0, sticky=tk.W, pady=2)
        self.var_rl_path = tk.StringVar()
        ttk.Entry(grp, textvariable=self.var_rl_path, width=40).grid(row=7, column=1, padx=5, pady=2)
        ttk.Button(grp, text="Обзор RL", command=self._browse_rl).grid(row=8, column=1, sticky=tk.W, padx=5)
        ttk.Button(grp, text="Обучить RL и сохранить", command=self._on_train_rl).grid(row=9, column=0, columnspan=2, pady=2)

        ttk.Button(grp, text="Запустить бэктест", command=self._on_run).grid(row=10, column=0, columnspan=2, pady=10)
        ttk.Button(grp, text="График капитала", command=self._on_plot).grid(row=11, column=0, columnspan=2, pady=2)

        self.txt_result = tk.Text(self, height=14, width=70, wrap=tk.WORD)
        self.txt_result.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _browse_model(self):
        path = filedialog.askopenfilename(
            filetypes=[("Joblib", "*.joblib")],
            initialdir=os.path.join(os.path.dirname(__file__), "..", "..", "models"),
        )
        if path:
            self.var_model.set(path)

    def _browse_rl(self):
        path = filedialog.askopenfilename(
            filetypes=[("Joblib", "*.joblib")],
            initialdir=os.path.join(os.path.dirname(__file__), "..", "..", "models"),
        )
        if path:
            self.var_rl_path.set(path)

    def _on_run(self):
        if self.state.X_test is None or self.state.test_dates is None or self.state.test_close is None:
            messagebox.showwarning("Нет данных", "Постройте датасет и обучите модель.")
            return
        path = self.var_model.get().strip() or self.state.model_path
        if not path or not os.path.isfile(path):
            messagebox.showwarning("Модель", "Выберите файл модели.")
            return
        self.set_status("Бэктест...")
        self.txt_result.delete(1.0, tk.END)
        try:
            bundle = load_bundle(path)
            comm = float(self.var_comm.get()) / 100.0
            slip = float(self.var_slip.get()) / 100.0
            nq = bundle.get("n_quantiles", 5)

            preds = []
            pred_returns_list = []
            probs_list = []
            for i in range(len(self.state.X_test)):
                vec = self.state.X_test[i : i + 1]
                r = predict(bundle, vec)
                try:
                    preds.append(int(r.replace("rank_", "")))
                except ValueError:
                    preds.append(0)
                ret = predict_return(bundle, vec)
                pred_returns_list.append(ret if ret is not None else 0.0)
                prob = predict_proba(bundle, vec)
                probs_list.append(prob if prob is not None else None)
            test_ranks = np.array(preds)
            test_pred_returns = np.array(pred_returns_list, dtype=np.float64)
            has_probs = all(p is not None for p in probs_list)
            test_probs = np.array(probs_list) if has_probs and probs_list else None

            min_ret_val = None
            try:
                mr = float(self.var_min_ret.get())
                if mr > 0:
                    min_ret_val = mr
            except ValueError:
                pass
            min_conf_val = None
            try:
                mc = float(self.var_min_conf.get())
                if mc > 0 and test_probs is not None:
                    min_conf_val = mc
            except ValueError:
                pass
            hold_val = None
            try:
                h = int(self.var_hold.get())
                if h > 0:
                    hold_val = h
            except ValueError:
                pass

            rl_actions = None
            rl_path = self.var_rl_path.get().strip()
            if rl_path and os.path.isfile(rl_path):
                agent = load_agent(rl_path)
                states = build_states(
                    test_ranks,
                    test_pred_returns,
                    self.state.test_vol_regime,
                    nq,
                )
                policy = Policy(agent)
                rl_actions = np.array([policy.action(states[j]) for j in range(len(states))])

            trades, equity = run_backtest(
                self.state.test_dates,
                self.state.test_close,
                test_ranks,
                n_quantiles=nq,
                commission_pct=comm,
                slippage_pct=slip,
                hold_bars=hold_val,
                min_return_threshold=min_ret_val,
                min_confidence=min_conf_val,
                test_pred_returns=test_pred_returns if min_ret_val is not None else None,
                test_probs=test_probs if min_conf_val is not None else None,
                rl_actions=rl_actions,
            )
            self.state.backtest_trades = trades
            self.state.backtest_equity = equity
            self.state.backtest_metrics_result = backtest_metrics(trades, equity)

            m = self.state.backtest_metrics_result
            self.txt_result.insert(tk.END, f"Сделок: {m['n_trades']}\n")
            self.txt_result.insert(tk.END, f"Суммарный PnL: {m['total_pnl']:.6f}\n")
            self.txt_result.insert(tk.END, f"Средний PnL на сделку: {m['avg_pnl_per_trade']:.6f}\n")
            self.txt_result.insert(tk.END, f"Sharpe: {m['sharpe_ratio']:.4f}\n")
            self.txt_result.insert(tk.END, f"Max drawdown: {m['max_drawdown']:.6f} ({m['max_drawdown_pct']:.2f}%)\n")
            self.set_status("Бэктест завершён")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            self.txt_result.insert(tk.END, str(e))
            self.set_status("Ошибка бэктеста")
            import traceback
            traceback.print_exc()

    def _on_train_rl(self):
        if self.state.X_test is None or self.state.test_dates is None or self.state.test_close is None:
            messagebox.showwarning("Нет данных", "Постройте датасет и обучите модель.")
            return
        path = self.var_model.get().strip() or self.state.model_path
        if not path or not os.path.isfile(path):
            messagebox.showwarning("Модель", "Выберите файл модели для построения состояний.")
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension=".joblib",
            filetypes=[("Joblib", "*.joblib")],
            initialdir=os.path.join(os.path.dirname(__file__), "..", "..", "models"),
        )
        if not save_path:
            return
        self.set_status("Обучение RL...")
        try:
            bundle = load_bundle(path)
            nq = bundle.get("n_quantiles", 5)
            comm = float(self.var_comm.get()) / 100.0
            slip = float(self.var_slip.get()) / 100.0
            preds = []
            pred_returns_list = []
            for i in range(len(self.state.X_test)):
                vec = self.state.X_test[i : i + 1]
                r = predict(bundle, vec)
                try:
                    preds.append(int(r.replace("rank_", "")))
                except ValueError:
                    preds.append(0)
                ret = predict_return(bundle, vec)
                pred_returns_list.append(ret if ret is not None else 0.0)
            test_ranks = np.array(preds)
            test_pred_returns = np.array(pred_returns_list, dtype=np.float64)
            states = build_states(
                test_ranks,
                test_pred_returns,
                self.state.test_vol_regime,
                nq,
            )
            agent = train_agent(
                states,
                self.state.test_close,
                commission_pct=comm,
                slippage_pct=slip,
                n_episodes=3,
            )
            save_agent(agent, save_path)
            self.var_rl_path.set(save_path)
            self.set_status("RL агент обучен и сохранён")
            messagebox.showinfo("Готово", f"RL агент сохранён: {save_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            self.set_status("Ошибка обучения RL")
            import traceback
            traceback.print_exc()

    def _on_plot(self):
        if self.state.backtest_equity is None:
            messagebox.showinfo("Нет данных", "Сначала запустите бэктест.")
            return
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(self.state.backtest_equity)
            ax.set_title("Кривая капитала")
            ax.set_xlabel("Шаг")
            ax.set_ylabel("Капитал")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
