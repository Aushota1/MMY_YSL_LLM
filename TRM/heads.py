"""
Output Heads - Выходные головы для предсказания ответа и правильности
"""

import torch
import torch.nn as nn


class OutputHead(nn.Module):
    """
    Голова для предсказания ответа (softmax over vocab)
    """
    
    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        bias: bool = False
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        
        # Линейный слой для предсказания токенов
        self.linear = nn.Linear(embedding_dim, vocab_size, bias=bias)
        
        # Инициализация
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов"""
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Предсказание ответа
        
        Args:
            y: [batch, seq_len, embedding_dim] - улучшенный ответ
        
        Returns:
            [batch, seq_len, vocab_size] - логиты для каждого токена
        """
        logits = self.linear(y)
        return logits


class QHead(nn.Module):
    """
    Голова для предсказания правильности ответа (binary classification для early stopping)
    """
    
    def __init__(
        self,
        embedding_dim: int,
        bias: bool = False
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Линейный слой для бинарной классификации
        self.linear = nn.Linear(embedding_dim, 1, bias=bias)
        
        # Инициализация
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов"""
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Предсказание правильности ответа
        
        Args:
            y: [batch, seq_len, embedding_dim] - улучшенный ответ
        
        Returns:
            [batch, seq_len, 1] - логиты для бинарной классификации (правильный/неправильный)
        """
        # Берем среднее по последовательности для получения одного значения на батч
        # Или можно взять последний токен
        if y.dim() == 3:
            # [batch, seq_len, embedding_dim] -> [batch, embedding_dim]
            y_pooled = y.mean(dim=1)  # Или y[:, -1, :] для последнего токена
        else:
            y_pooled = y
        
        logits = self.linear(y_pooled)  # [batch, 1]
        return logits

