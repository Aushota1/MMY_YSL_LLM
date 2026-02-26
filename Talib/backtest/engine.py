"""
Симуляция сделок с комиссией и проскальзыванием.
Вход: даты теста, close, предсказанные ранги; правило: long при rank_0, short при rank_K-1.
Выход: список сделок и кривая капитала.
"""

from typing import List, Tuple, Optional

import numpy as np

from config import DEFAULT_COMMISSION_PCT, DEFAULT_SLIPPAGE_PCT, LONG_RANK_INDEX, SHORT_RANK_INDEX


def run_backtest(
    test_dates: np.ndarray,
    test_close: np.ndarray,
    test_ranks: np.ndarray,
    n_quantiles: int = 5,
    commission_pct: float = DEFAULT_COMMISSION_PCT,
    slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
    hold_bars: Optional[int] = None,
    min_return_threshold: Optional[float] = None,
    min_confidence: Optional[float] = None,
    test_pred_returns: Optional[np.ndarray] = None,
    test_probs: Optional[np.ndarray] = None,
    rl_actions: Optional[np.ndarray] = None,
) -> Tuple[List[dict], np.ndarray]:
    """
    Запускает бэктест. Если задан rl_actions (массив 0/1/-1 по барам), используется RL-политика;
    иначе правило по рангам с опциональными фильтрами и hold_bars.
    """
    n = len(test_close)
    if n == 0:
        return [], np.array([1.0])

    equity = 1.0
    equity_curve = [1.0]
    position = 0
    entry_price = 0.0
    entry_time = None
    entry_idx = 0
    trades = []

    long_rank = LONG_RANK_INDEX if LONG_RANK_INDEX >= 0 else n_quantiles + LONG_RANK_INDEX
    short_rank = n_quantiles - 1 if SHORT_RANK_INDEX == -1 else SHORT_RANK_INDEX
    use_rl = rl_actions is not None and len(rl_actions) >= n

    for i in range(n):
        price = float(test_close[i])
        dt = test_dates[i]
        rank = int(test_ranks[i]) if i < len(test_ranks) else 0
        exit_now = False

        if use_rl:
            target = int(rl_actions[i]) if i < len(rl_actions) else 0
            if target not in (0, 1, -1):
                target = 0
            exit_now = position != 0 and (target == 0 or (position == 1 and target == -1) or (position == -1 and target == 1))
            if not exit_now and position == 0 and target == 1:
                position = 1
                entry_price = price * (1 + slippage_pct)
                entry_time = dt
                entry_idx = i
            elif not exit_now and position == 0 and target == -1:
                position = -1
                entry_price = price * (1 - slippage_pct)
                entry_time = dt
                entry_idx = i
        else:
            pred_ret = float(test_pred_returns[i]) if test_pred_returns is not None and i < len(test_pred_returns) else None
            probs_i = test_probs[i] if test_probs is not None and i < len(test_probs) else None
            confidence = float(np.max(probs_i)) if probs_i is not None and len(probs_i) else None

            if position == 0:
                ok_long = rank == long_rank
                ok_short = rank == short_rank
                if min_return_threshold is not None and pred_ret is not None:
                    ok_long = ok_long and pred_ret >= min_return_threshold
                    ok_short = ok_short and pred_ret <= -min_return_threshold
                if min_confidence is not None and confidence is not None:
                    ok_long = ok_long and confidence >= min_confidence
                    ok_short = ok_short and confidence >= min_confidence

                if ok_long:
                    position = 1
                    entry_price = price * (1 + slippage_pct)
                    entry_time = dt
                    entry_idx = i
                elif ok_short:
                    position = -1
                    entry_price = price * (1 - slippage_pct)
                    entry_time = dt
                    entry_idx = i
            else:
                if hold_bars is not None and (i - entry_idx) >= hold_bars:
                    exit_now = True
                elif position == 1 and rank == short_rank:
                    exit_now = True
                elif position == -1 and rank == long_rank:
                    exit_now = True

        if position != 0 and exit_now:
                exit_price = price * (1 - slippage_pct) if position == 1 else price * (1 + slippage_pct)
                if position == 1:
                    pnl_brutto = (exit_price - entry_price) / entry_price
                else:
                    pnl_brutto = (entry_price - exit_price) / entry_price
                notional = 1.0
                commission = 2 * commission_pct * notional
                slippage_cost = slippage_pct * 2 * notional
                pnl_netto = pnl_brutto - commission - slippage_cost
                equity *= 1 + pnl_netto
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": dt,
                    "side": position,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_brutto": pnl_brutto,
                    "commission": commission,
                    "slippage_cost": slippage_cost,
                    "pnl_netto": pnl_netto,
                })
                position = 0

        equity_curve.append(equity)

    return trades, np.array(equity_curve)
