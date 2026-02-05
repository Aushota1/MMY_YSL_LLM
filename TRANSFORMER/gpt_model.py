"""
Полная GPT модель
Интегрирует Embedding Layer и Transformer Decoder Blocks
"""

import torch
import torch.nn as nn
from typing import Optional

import sys
import os

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from EMBEDDING_LAYER.embedding_layer import EmbeddingLayer, create_embedding_from_tokenizer
    HAS_EMBEDDING_LAYER = True
except ImportError:
    HAS_EMBEDDING_LAYER = False
    print("⚠️  EMBEDDING_LAYER не найден. GPTModel будет работать без предустановленного EmbeddingLayer.")

from .decoder_block import TransformerDecoderBlock


class GPTModel(nn.Module):
    """
    Полная GPT модель (GPT-2/GPT-3 стиль)
    
    Архитектура:
        Token IDs → Embedding Layer → N × Decoder Blocks → 
                   Final Layer Norm → Language Model Head → Logits
    
    Args:
        vocab_size: Размер словаря
        embedding_dim: Размерность embeddings
        num_layers: Количество Decoder Blocks
        num_heads: Количество голов внимания
        ff_dim: Размерность FFN (None = 4 × embedding_dim)
        max_seq_len: Максимальная длина последовательности
        dropout: Вероятность dropout
        layer_norm_eps: Эпсилон для Layer Norm
        bias: Использовать ли bias в проекциях (False для GPT-2)
        embedding_layer: Существующий EmbeddingLayer (опционально)
        tokenizer: BPETokenizer для автоматического создания EmbeddingLayer (опционально)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: Optional[int] = None,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        bias: bool = False,
        embedding_layer: Optional[nn.Module] = None,
        tokenizer = None
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # Создание или использование существующего Embedding Layer
        if embedding_layer is not None:
            self.embedding = embedding_layer
            # Проверяем совместимость размерности
            if hasattr(embedding_layer, 'embedding_dim'):
                assert embedding_layer.embedding_dim == embedding_dim, \
                    f"embedding_dim не совпадает: {embedding_layer.embedding_dim} != {embedding_dim}"
        elif tokenizer is not None and HAS_EMBEDDING_LAYER:
            # Создаем EmbeddingLayer из токенизатора
            self.embedding = create_embedding_from_tokenizer(
                tokenizer,
                embedding_dim=embedding_dim,
                max_seq_len=max_seq_len,
                dropout=dropout,
                learnable_pos=False,
                layer_norm=True
            )
        else:
            # Создаем новый EmbeddingLayer
            if HAS_EMBEDDING_LAYER:
                from EMBEDDING_LAYER.embedding_layer import EmbeddingLayer
                self.embedding = EmbeddingLayer(
                    vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    max_seq_len=max_seq_len,
                    dropout=dropout,
                    padding_idx=0,
                    learnable_pos=False,
                    layer_norm=True,
                    layer_norm_eps=layer_norm_eps
                )
            else:
                raise ValueError(
                    "Необходимо предоставить embedding_layer, tokenizer, "
                    "или установить EMBEDDING_LAYER модуль"
                )
        
        # Transformer Decoder Blocks
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
                bias=bias
            )
            for _ in range(num_layers)
        ])
        
        # Final Layer Normalization
        self.final_norm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        
        # Language Model Head
        # В GPT-2/GPT-3 используется weight tying (общие веса с embedding)
        # Но для простоты используем отдельный слой
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов как в GPT-2/GPT-3"""
        # Инициализация Language Model Head
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        
        # Инициализация final layer norm
        nn.init.ones_(self.final_norm.weight)
        nn.init.zeros_(self.final_norm.bias)
    
    def forward(
        self,
        token_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            token_ids: [batch_size, seq_len] - ID токенов
            mask: Опциональная маска для attention
        
        Returns:
            [batch_size, seq_len, vocab_size] - логиты для каждого токена
        """
        # Embedding Layer
        x = self.embedding(token_ids)  # [batch, seq_len, embedding_dim]
        
        # Transformer Decoder Blocks
        for block in self.blocks:
            x = block(x, mask=mask)  # [batch, seq_len, embedding_dim]
        
        # Final Layer Normalization
        x = self.final_norm(x)  # [batch, seq_len, embedding_dim]
        
        # Language Model Head
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]
        
        return logits
    
    def get_num_params(self) -> int:
        """Получение количества параметров модели"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_num_params_millions(self) -> float:
        """Получение количества параметров в миллионах"""
        return self.get_num_params() / 1_000_000

