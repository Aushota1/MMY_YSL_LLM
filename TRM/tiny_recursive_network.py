"""
Tiny Recursive Network - Маленькая сеть (2 слоя) для рекурсивного рассуждения
"""

import torch
import torch.nn as nn
from typing import Optional

from .utils import RMSNorm, SwiGLU


class TinyRecursiveNetwork(nn.Module):
    """
    Маленькая рекурсивная сеть (2 слоя) для TRM
    Используется для обновления латентного состояния z или ответа y
    
    Архитектура:
        Input → RMSNorm → Linear → SwiGLU → Dropout → Linear → RMSNorm → Output
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
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Первый слой: RMSNorm → Linear → SwiGLU
        self.norm1 = RMSNorm(embedding_dim, eps=eps)
        self.swiglu = SwiGLU(embedding_dim, hidden_dim, bias=bias)
        self.dropout1 = nn.Dropout(dropout)
        
        # Второй слой: Linear → RMSNorm
        self.linear = nn.Linear(hidden_dim, embedding_dim, bias=bias)
        self.norm2 = RMSNorm(embedding_dim, eps=eps)
        self.dropout2 = nn.Dropout(dropout)
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов как в статье"""
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: [batch, seq_len, embedding_dim] - вопрос (input embedding)
            y: [batch, seq_len, embedding_dim] - ответ (опционально)
            z: [batch, seq_len, embedding_dim] - латентное состояние (опционально)
        
        Returns:
            [batch, seq_len, embedding_dim] - обновленное состояние
        """
        # Объединяем входы: x + y + z (если есть)
        if y is not None and z is not None:
            combined = x + y + z
        elif y is not None:
            combined = x + y
        elif z is not None:
            combined = x + z
        else:
            combined = x
        
        # Первый слой
        residual = combined
        combined = self.norm1(combined)
        combined = self.swiglu(combined)
        combined = self.dropout1(combined)
        
        # Второй слой
        combined = self.linear(combined)
        combined = self.dropout2(combined)
        combined = self.norm2(combined)
        
        # Residual connection
        output = residual + combined
        
        return output

