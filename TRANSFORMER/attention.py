"""
Multi-Head Self-Attention для Transformer Decoder
Реализация максимально аналогична GPT-2/GPT-3
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention с causal masking
    Реализация аналогична GPT-2/GPT-3
    
    Args:
        embedding_dim: Размерность embeddings (должна быть кратна num_heads)
        num_heads: Количество голов внимания
        dropout: Вероятность dropout
        bias: Использовать ли bias в проекциях (False для GPT-2)
        causal: Использовать ли causal masking (True для GPT)
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = False,
        causal: bool = True
    ):
        super().__init__()
        
        assert embedding_dim % num_heads == 0, \
            f"embedding_dim ({embedding_dim}) должен быть кратен num_heads ({num_heads})"
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.causal = causal
        
        # Query, Key, Value проекции
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        
        # Output проекция
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Инициализация весов (как в GPT)
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов как в GPT-2/GPT-3"""
        # Инициализация всех Linear слоев
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Генерация causal mask для autoregressive модели
        
        Args:
            seq_len: Длина последовательности
            device: Устройство для создания тензора
        
        Returns:
            Mask тензор [seq_len, seq_len] с -inf для будущих позиций
        """
        # Создаем верхнюю треугольную матрицу
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        # Заменяем True на -inf
        mask = mask.float().masked_fill(mask, float('-inf'))
        return mask
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Разделение на головы
        
        Args:
            x: [batch_size, seq_len, embedding_dim]
        
        Returns:
            [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
    
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Объединение голов
        
        Args:
            x: [batch_size, num_heads, seq_len, head_dim]
        
        Returns:
            [batch_size, seq_len, embedding_dim]
        """
        batch_size, num_heads, seq_len, head_dim = x.size()
        x = x.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]
        x = x.contiguous().view(batch_size, seq_len, num_heads * head_dim)
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: [batch_size, seq_len, embedding_dim] - входные embeddings
            mask: Опциональная маска [seq_len, seq_len] или [batch, seq_len, seq_len]
        
        Returns:
            [batch_size, seq_len, embedding_dim] - выходные embeddings
        """
        batch_size, seq_len, _ = x.size()
        
        # Проекции Query, Key, Value
        q = self.q_proj(x)  # [batch, seq_len, embedding_dim]
        k = self.k_proj(x)  # [batch, seq_len, embedding_dim]
        v = self.v_proj(x)  # [batch, seq_len, embedding_dim]
        
        # Разделение на головы
        q = self._split_heads(q)  # [batch, num_heads, seq_len, head_dim]
        k = self._split_heads(k)  # [batch, num_heads, seq_len, head_dim]
        v = self._split_heads(v)  # [batch, num_heads, seq_len, head_dim]
        
        # Scaled Dot-Product Attention
        # Q @ K^T / sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch, num_heads, seq_len, seq_len]
        
        # Применение causal mask
        if self.causal:
            causal_mask = self._generate_causal_mask(seq_len, x.device)  # [seq_len, seq_len]
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)  # [batch, num_heads, seq_len, seq_len]
        
        # Применение дополнительной маски (если предоставлена)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)  # [batch, num_heads, seq_len, seq_len]
        attn_weights = self.dropout(attn_weights)
        
        # Применение attention к values
        attn_output = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len, head_dim]
        
        # Объединение голов
        attn_output = self._merge_heads(attn_output)  # [batch, seq_len, embedding_dim]
        
        # Output проекция
        output = self.out_proj(attn_output)  # [batch, seq_len, embedding_dim]
        
        return output

