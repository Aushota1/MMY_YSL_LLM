"""
Feed-Forward Network для Transformer Decoder
Реализация максимально аналогична GPT-2/GPT-3
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Feed-Forward Network с GELU активацией
    Реализация аналогична GPT-2/GPT-3
    
    Args:
        embedding_dim: Размерность embeddings
        ff_dim: Размерность скрытого слоя (обычно 4 × embedding_dim)
        dropout: Вероятность dropout
        activation: Функция активации ('gelu' для GPT)
        bias: Использовать ли bias (False для GPT-2)
    """
    
    def __init__(
        self,
        embedding_dim: int,
        ff_dim: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
        bias: bool = False
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.ff_dim = ff_dim
        
        # Два линейных слоя
        self.fc1 = nn.Linear(embedding_dim, ff_dim, bias=bias)
        self.fc2 = nn.Linear(ff_dim, embedding_dim, bias=bias)
        
        # Активация
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Неизвестная активация: {activation}")
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Инициализация весов (как в GPT)
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов как в GPT-2/GPT-3"""
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: [batch_size, seq_len, embedding_dim] - входные embeddings
        
        Returns:
            [batch_size, seq_len, embedding_dim] - выходные embeddings
        """
        # Первый слой + активация
        x = self.fc1(x)  # [batch, seq_len, ff_dim]
        x = self.activation(x)
        x = self.dropout(x)
        
        # Второй слой
        x = self.fc2(x)  # [batch, seq_len, embedding_dim]
        x = self.dropout(x)
        
        return x

