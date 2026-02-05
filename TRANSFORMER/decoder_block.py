"""
Transformer Decoder Block
Реализация максимально аналогична GPT-2/GPT-3 с Pre-Norm архитектурой
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    from .attention import MultiHeadSelfAttention
    from .feed_forward import FeedForward
except ImportError:
    # Для случаев, когда импорт из того же модуля
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from attention import MultiHeadSelfAttention
    from feed_forward import FeedForward


class TransformerDecoderBlock(nn.Module):
    """
    Transformer Decoder Block с Pre-Norm архитектурой (как в GPT-2)
    
    Архитектура:
        Input → LayerNorm → MultiHeadAttention → Residual → 
               LayerNorm → FeedForward → Residual → Output
    
    Args:
        embedding_dim: Размерность embeddings
        num_heads: Количество голов внимания
        ff_dim: Размерность FFN (обычно 4 × embedding_dim)
        dropout: Вероятность dropout
        layer_norm_eps: Эпсилон для Layer Norm (1e-5 для GPT-2)
        bias: Использовать ли bias в проекциях (False для GPT-2)
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        bias: bool = False
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Если ff_dim не указан, используем 4 × embedding_dim (как в GPT)
        if ff_dim is None:
            ff_dim = 4 * embedding_dim
        
        # Pre-Norm Layer Normalization
        self.norm1 = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        
        # Multi-Head Self-Attention
        self.attention = MultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            causal=True  # Всегда causal для decoder
        )
        
        # Feed-Forward Network
        self.ffn = FeedForward(
            embedding_dim=embedding_dim,
            ff_dim=ff_dim,
            dropout=dropout,
            activation='gelu',
            bias=bias
        )
        
        # Dropout (опционально, для дополнительной регуляризации)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass с Pre-Norm архитектурой
        
        Args:
            x: [batch_size, seq_len, embedding_dim] - входные embeddings
            mask: Опциональная маска для attention
        
        Returns:
            [batch_size, seq_len, embedding_dim] - выходные embeddings
        """
        # Pre-Norm + Attention + Residual
        # x = x + attention(norm(x))
        residual = x
        x = self.norm1(x)
        x = self.attention(x, mask=mask)
        x = self.dropout(x)
        x = x + residual
        
        # Pre-Norm + FFN + Residual
        # x = x + ffn(norm(x))
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual
        
        return x

