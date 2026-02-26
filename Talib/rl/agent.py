"""
Простой Q-learning агент: состояние (rank_norm, return, vol) -> действие (flat/long/short).
Состояние дискретизуется по корзинам; Q хранится в словаре.
"""

from typing import Optional, Tuple

import numpy as np


class QAgent:
    """
    Табличный Q-learning: state = (rank_bin, return_bin, vol_bin).
    action: 0 = flat, 1 = long, 2 = short (внутри); наружу отдаём 0, 1, -1.
    """

    def __init__(
        self,
        n_rank_bins: int = 5,
        n_return_bins: int = 5,
        return_low: float = -0.02,
        return_high: float = 0.02,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.2,
    ):
        self.n_rank_bins = n_rank_bins
        self.n_return_bins = n_return_bins
        self.return_low = return_low
        self.return_high = return_high
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self._Q: dict = {}
        self._n_actions = 3

    def _discretize(self, state: np.ndarray) -> Tuple[int, int, int]:
        rank_norm = float(np.clip(state[0], 0, 1))
        ret = float(state[1]) if len(state) > 1 else 0.0
        vol = int(state[2]) if len(state) > 2 else 1
        vol = max(0, min(2, vol))
        r_bin = min(self.n_rank_bins - 1, int(rank_norm * self.n_rank_bins))
        ret_clip = np.clip(ret, self.return_low, self.return_high)
        ret_norm = (ret_clip - self.return_low) / (self.return_high - self.return_low + 1e-8)
        ret_bin = min(self.n_return_bins - 1, int(ret_norm * self.n_return_bins))
        return (r_bin, ret_bin, vol)

    def _get_Q(self, s: Tuple[int, int, int], a: int) -> float:
        return self._Q.get((s, a), 0.0)

    def _set_Q(self, s: Tuple[int, int, int], a: int, val: float):
        self._Q[(s, a)] = val

    def action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Возвращает действие: 0 = flat, 1 = long, -1 = short.
        """
        s = self._discretize(np.asarray(state).ravel())
        if not deterministic and np.random.rand() < self.epsilon:
            a_inner = int(np.random.randint(0, self._n_actions))
        else:
            q_vals = [self._get_Q(s, a) for a in range(self._n_actions)]
            a_inner = int(np.argmax(q_vals))
        return (0, 1, -1)[a_inner]

    def update(self, state: np.ndarray, action: int, reward: float, next_state: Optional[np.ndarray] = None):
        """Q(s,a) += alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))."""
        a_inner = {0: 0, 1: 1, -1: 2}.get(action, 0)
        s = self._discretize(np.asarray(state).ravel())
        q_old = self._get_Q(s, a_inner)
        if next_state is not None:
            s_next = self._discretize(np.asarray(next_state).ravel())
            q_max_next = max(self._get_Q(s_next, a) for a in range(self._n_actions))
            target = reward + self.gamma * q_max_next
        else:
            target = reward
        self._set_Q(s, a_inner, q_old + self.alpha * (target - q_old))

    def to_dict(self) -> dict:
        return {
            "Q": dict(self._Q),
            "n_rank_bins": self.n_rank_bins,
            "n_return_bins": self.n_return_bins,
            "return_low": self.return_low,
            "return_high": self.return_high,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QAgent":
        agent = cls(
            n_rank_bins=d.get("n_rank_bins", 5),
            n_return_bins=d.get("n_return_bins", 5),
            return_low=d.get("return_low", -0.02),
            return_high=d.get("return_high", 0.02),
            alpha=d.get("alpha", 0.1),
            gamma=d.get("gamma", 0.95),
            epsilon=d.get("epsilon", 0.1),
        )
        agent._Q = d.get("Q", {})
        return agent
