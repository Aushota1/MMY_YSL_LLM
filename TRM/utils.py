"""
Утилиты для TRM: RMSNorm, SwiGLU, Rotary Embeddings
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    Без bias, как в статье TRM
    
    Формула: RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
    """
    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., dim] - входной тензор
        Returns:
            [..., dim] - нормализованный тензор
        """
        # Вычисляем RMS: sqrt(mean(x^2))
        # Для многомерных тензоров нормализуем по последнему измерению
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Нормализуем и масштабируем
        return (x / rms) * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU активация: SwiGLU(x) = Swish(xW + b) ⊙ (xV + c)
    где Swish(x) = x * sigmoid(x)
    """
    
    def __init__(self, dim: int, hidden_dim: int = None, bias: bool = False):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim
        
        # Два линейных слоя для SwiGLU
        self.w_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(dim, hidden_dim, bias=bias)
        
        # Инициализация
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов"""
        nn.init.normal_(self.w_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=0.02)
        if self.w_proj.bias is not None:
            nn.init.zeros_(self.w_proj.bias)
        if self.v_proj.bias is not None:
            nn.init.zeros_(self.v_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., dim] - входной тензор
        Returns:
            [..., hidden_dim] - выходной тензор
        """
        # Swish(xW + b) ⊙ (xV + c)
        swish = x * torch.sigmoid(self.w_proj(x))  # Swish(xW + b)
        gate = self.v_proj(x)  # xV + c
        return swish * gate  # Элементное умножение


def apply_rotary_pos_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Применение rotary positional embeddings (RoPE)
    Упрощенная версия для TRM
    
    Args:
        x: [batch, seq_len, num_heads, head_dim] - тензор для применения RoPE
        freqs_cis: [seq_len, head_dim] - частоты для rotary embeddings
    Returns:
        [batch, seq_len, num_heads, head_dim] - тензор с примененными RoPE
    """
    # Упрощенная версия: для TRM пока не используем полную реализацию RoPE
    # Можно добавить полную реализацию позже при необходимости
    return x


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Предвычисление частот для rotary embeddings
    
    Args:
        dim: размерность head
        max_seq_len: максимальная длина последовательности
        theta: базовая частота
    Returns:
        [max_seq_len, dim] - частоты
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # Комплексные числа
    return freqs_cis


def get_activation_fn(activation: str):
    """Получение функции активации по имени"""
    if activation == 'gelu':
        return nn.GELU()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'swiglu':
        return SwiGLU
    else:
        raise ValueError(f"Неизвестная активация: {activation}")

