"""
Метрики классификации и бэктеста (PnL, Sharpe, max drawdown).
"""

from typing import List, Optional

import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[List[str]] = None,
) -> dict:
    """
    Метрики классификации: accuracy, f1_weighted, confusion_matrix, report_str.
    """
    if not HAS_SKLEARN:
        return {
            "accuracy": 0.0,
            "f1_weighted": 0.0,
            "confusion_matrix": np.zeros((1, 1)),
            "report_str": "sklearn not installed",
        }
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    report_str = classification_report(
        y_true, y_pred, target_names=label_names, zero_division=0
    )
    return {
        "accuracy": acc,
        "f1_weighted": f1,
        "confusion_matrix": cm,
        "report_str": report_str,
    }


def backtest_metrics(
    trades: List[dict],
    equity_curve: np.ndarray,
) -> dict:
    """
    Метрики бэктеста: total_pnl, n_trades, avg_pnl_per_trade, sharpe_ratio, max_drawdown, max_drawdown_pct.
    """
    n_trades = len(trades)
    if n_trades == 0:
        return {
            "total_pnl": 0.0,
            "n_trades": 0,
            "avg_pnl_per_trade": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
        }

    pnls = [t["pnl_netto"] for t in trades]
    total_pnl = sum(pnls)
    avg_pnl = total_pnl / n_trades

    if n_trades > 1 and np.std(pnls) > 1e-12:
        sharpe_ratio = float(np.mean(pnls) / np.std(pnls) * np.sqrt(n_trades))
    else:
        sharpe_ratio = 0.0

    curve = np.asarray(equity_curve, dtype=np.float64)
    peak = np.maximum.accumulate(curve)
    drawdown = peak - curve
    max_dd = float(np.max(drawdown))
    max_dd_pct = float(np.max(drawdown / (peak + 1e-12)) * 100) if np.max(peak) > 0 else 0.0

    return {
        "total_pnl": total_pnl,
        "n_trades": n_trades,
        "avg_pnl_per_trade": avg_pnl,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
    }
