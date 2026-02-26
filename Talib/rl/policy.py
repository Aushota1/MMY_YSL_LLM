"""
Обёртка политики: state -> action (0, 1, -1).
Использует QAgent с deterministic=True при инференсе.
"""

from typing import Any

import numpy as np

from rl.agent import QAgent


class Policy:
    """Политика на основе обученного агента: action(state) -> 0 | 1 | -1."""

    def __init__(self, agent: QAgent):
        self.agent = agent

    def action(self, state: np.ndarray) -> int:
        return self.agent.action(state, deterministic=True)
