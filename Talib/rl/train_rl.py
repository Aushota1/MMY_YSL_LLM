"""
Обучение RL-агента на исторических данных: симуляция по барам, награда = изменение капитала.
Состояние на каждом баре: [rank_norm, pred_return, vol_regime].
"""

from typing import List, Optional, Tuple

import joblib
import numpy as np

from rl.agent import QAgent


def build_states(
    test_ranks: np.ndarray,
    test_pred_returns: np.ndarray,
    test_vol_regime: Optional[np.ndarray],
    n_quantiles: int,
) -> np.ndarray:
    """Строит матрицу состояний (n_bars, 3): rank_norm, return, vol."""
    n = len(test_ranks)
    rank_norm = np.asarray(test_ranks, dtype=np.float64) / max(1, n_quantiles - 1)
    ret = np.asarray(test_pred_returns, dtype=np.float64)
    vol = np.ones(n, dtype=np.float64)
    if test_vol_regime is not None and len(test_vol_regime) == n:
        vol = np.asarray(test_vol_regime, dtype=np.float64)
    return np.column_stack([rank_norm, ret, vol])


def train_agent(
    states: np.ndarray,
    test_close: np.ndarray,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0005,
    n_episodes: int = 3,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon: float = 0.3,
) -> QAgent:
    """
    Обучает QAgent на истории: прогон по барам, награда = PnL за шаг (с комиссией).
    """
    n = len(test_close)
    agent = QAgent(alpha=alpha, gamma=gamma, epsilon=epsilon)

    for _ in range(n_episodes):
        position = 0
        entry_price = 0.0
        entry_idx = 0
        equity_prev = 1.0
        prev_s = None
        prev_a = None

        for i in range(n - 1):
            price = float(test_close[i])
            state = states[i]
            next_state = states[i + 1]
            next_price = float(test_close[i + 1])

            reward = 0.0
            if position != 0:
                exit_price = price * (1 - slippage_pct) if position == 1 else price * (1 + slippage_pct)
                if position == 1:
                    pnl_brutto = (exit_price - entry_price) / entry_price
                else:
                    pnl_brutto = (entry_price - exit_price) / entry_price
                commission = 2 * commission_pct
                slippage_cost = slippage_pct * 2
                reward = pnl_brutto - commission - slippage_cost
                position = 0

            if prev_s is not None and prev_a is not None:
                agent.update(prev_s, prev_a, reward, state)

            action = agent.action(state, deterministic=False)
            if action == 1 and position == 0:
                position = 1
                entry_price = price * (1 + slippage_pct)
                entry_idx = i
            elif action == -1 and position == 0:
                position = -1
                entry_price = price * (1 - slippage_pct)
                entry_idx = i

            equity_curr = equity_prev * (1 + reward) if position == 0 else equity_prev
            equity_prev = equity_curr
            prev_s = state
            prev_a = action

        if prev_s is not None and prev_a is not None:
            agent.update(prev_s, prev_a, 0.0, None)

    agent.epsilon = 0.1
    return agent


def save_agent(agent: QAgent, path: str):
    joblib.dump(agent.to_dict(), path)


def load_agent(path: str) -> QAgent:
    d = joblib.load(path)
    return QAgent.from_dict(d)
