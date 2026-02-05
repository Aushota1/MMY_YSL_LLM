"""
Output Refinement - Улучшение ответа на основе латентного состояния
"""

import torch
import torch.nn as nn
from typing import Optional

from .tiny_recursive_network import TinyRecursiveNetwork


class OutputRefinement(nn.Module):
    """
    Сеть для улучшения ответа y на основе латентного состояния z
    Использует ту же архитектуру, что и TinyRecursiveNetwork
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        eps: float = 1e-5,
        bias: bool = False
    ):
        super().__init__()
        
        # Используем TinyRecursiveNetwork для улучшения ответа
        self.refinement_net = TinyRecursiveNetwork(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            eps=eps,
            bias=bias
        )
    
    def forward(
        self,
        y: torch.Tensor,
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Улучшение ответа y на основе латентного состояния z
        
        Args:
            y: [batch, seq_len, embedding_dim] - текущий ответ
            z: [batch, seq_len, embedding_dim] - латентное состояние
        
        Returns:
            [batch, seq_len, embedding_dim] - улучшенный ответ
        """
        # Используем z как "вопрос", y как "ответ"
        # В контексте refinement: обновляем y на основе z
        updated_y = self.refinement_net(x=z, y=y)
        return updated_y

