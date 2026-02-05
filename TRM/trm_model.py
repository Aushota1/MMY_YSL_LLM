"""
Tiny Recursive Model (TRM) - Полная модель
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

import sys
import os

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from EMBEDDING_LAYER.embedding_layer import EmbeddingLayer, create_embedding_from_tokenizer
    HAS_EMBEDDING_LAYER = True
except ImportError:
    HAS_EMBEDDING_LAYER = False
    print("⚠️  EMBEDDING_LAYER не найден. TRMModel будет работать без предустановленного EmbeddingLayer.")

from .tiny_recursive_network import TinyRecursiveNetwork
from .output_refinement import OutputRefinement
from .heads import OutputHead, QHead


class TRMModel(nn.Module):
    """
    Tiny Recursive Model (TRM)
    Полная модель с рекурсивным рассуждением и deep supervision
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 512,
        hidden_dim: int = 512,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        eps: float = 1e-5,
        bias: bool = False,
        embedding_layer: Optional[nn.Module] = None,
        tokenizer = None
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # Input Embedding
        if embedding_layer is not None:
            self.input_embedding = embedding_layer
            if hasattr(embedding_layer, 'embedding_dim'):
                assert embedding_layer.embedding_dim == embedding_dim, \
                    f"embedding_dim не совпадает: {embedding_layer.embedding_dim} != {embedding_dim}"
        elif tokenizer is not None and HAS_EMBEDDING_LAYER:
            self.input_embedding = create_embedding_from_tokenizer(
                tokenizer,
                embedding_dim=embedding_dim,
                max_seq_len=max_seq_len,
                dropout=dropout,
                learnable_pos=False,
                layer_norm=True
            )
        else:
            if HAS_EMBEDDING_LAYER:
                from EMBEDDING_LAYER.embedding_layer import EmbeddingLayer
                self.input_embedding = EmbeddingLayer(
                    vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    max_seq_len=max_seq_len,
                    dropout=dropout,
                    padding_idx=0,
                    learnable_pos=False,
                    layer_norm=True,
                    layer_norm_eps=eps
                )
            else:
                raise ValueError(
                    "Необходимо предоставить embedding_layer, tokenizer, "
                    "или установить EMBEDDING_LAYER модуль"
                )
        
        # Tiny Recursive Network для обновления z
        self.tiny_net = TinyRecursiveNetwork(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            eps=eps,
            bias=bias
        )
        
        # Output Refinement для улучшения y
        self.output_refinement = OutputRefinement(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            eps=eps,
            bias=bias
        )
        
        # Output Heads
        self.output_head = OutputHead(
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            bias=bias
        )
        
        self.q_head = QHead(
            embedding_dim=embedding_dim,
            bias=bias
        )
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов"""
        # Веса уже инициализированы в компонентах
        pass
    
    def forward(
        self,
        x_input: torch.Tensor,
        y_init: Optional[torch.Tensor] = None,
        z_init: Optional[torch.Tensor] = None,
        n: int = 6,
        T: int = 3,
        N_sup: int = 16
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass с deep supervision
        
        Args:
            x_input: [batch, seq_len] - входные токены
            y_init: [batch, seq_len, embedding_dim] - начальный ответ (опционально)
            z_init: [batch, seq_len, embedding_dim] - начальное латентное состояние (опционально)
            n: количество рекурсивных шагов
            T: количество глубоких рекурсий
            N_sup: максимальное количество шагов супервизии
        
        Returns:
            Tuple:
                - y_hat: [batch, seq_len, vocab_size] - предсказанный ответ
                - q_hat: [batch, 1] - предсказание правильности
        """
        # Встраивание входа
        x = self.input_embedding(x_input)  # [batch, seq_len, embedding_dim]
        
        # Инициализация y и z
        if y_init is None:
            batch_size, seq_len, embedding_dim = x.shape
            y = torch.zeros_like(x)
        else:
            y = y_init
        
        if z_init is None:
            z = torch.zeros_like(x)
        else:
            z = z_init
        
        # Deep supervision loop
        for step in range(N_sup):
            # Импортируем здесь, чтобы избежать циклических импортов
            from .latent_recursion import deep_recursion
            
            # Deep recursion
            y, z, y_hat, q_hat = deep_recursion(
                x, y, z,
                self.tiny_net, self.output_refinement,
                self.output_head, self.q_head,
                n=n, T=T
            )
            
            # Early stopping
            q_prob = torch.sigmoid(q_hat).mean().item()
            if q_prob > 0.5:
                break
        
        return y_hat, q_hat
    
    def generate_answer(
        self,
        x_input: torch.Tensor,
        max_steps: int = 16
    ) -> torch.Tensor:
        """
        Генерация ответа
        
        Args:
            x_input: [batch, seq_len] - входные токены
            max_steps: максимальное количество шагов
        
        Returns:
            [batch, seq_len] - сгенерированные токены
        """
        self.eval()
        with torch.no_grad():
            y_hat, _ = self.forward(x_input, N_sup=max_steps)
            # Берем argmax для получения токенов
            tokens = y_hat.argmax(dim=-1)  # [batch, seq_len]
        return tokens
    
    def get_num_params(self) -> int:
        """Получение количества параметров модели"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_num_params_millions(self) -> float:
        """Получение количества параметров в миллионах"""
        return self.get_num_params() / 1_000_000

