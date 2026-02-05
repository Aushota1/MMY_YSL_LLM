"""
Embedding Layer для LLM проекта
Реализация Token Embedding и Positional Encoding
Максимально аналогично современным LLM (GPT, BERT, Transformer)
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class TokenEmbedding(nn.Module):
    """
    Базовый слой эмбеддингов для токенов
    Реализация аналогична GPT-2/GPT-3
    
    Args:
        vocab_size: Размер словаря (из tokenizer.get_vocab_size())
        embedding_dim: Размерность эмбеддингов (например, 256, 512, 768)
        padding_idx: ID токена паддинга (обычно 0 для <PAD>)
    """
    def __init__(self, vocab_size: int, embedding_dim: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx  # Игнорирует градиенты для паддинга
        )
        self.embedding_dim = embedding_dim
        
        # Инициализация весов (как в GPT)
        self._init_weights()
    
    def _init_weights(self):
        """
        Инициализация весов эмбеддингов
        Использует инициализацию как в GPT-2/GPT-3
        """
        # Инициализация по нормальному распределению (как в GPT)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        
        # Обнуление эмбеддинга паддинга
        if self.embedding.padding_idx is not None:
            nn.init.constant_(
                self.embedding.weight[self.embedding.padding_idx], 
                0.0
            )
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Преобразование токенов в эмбеддинги
        
        Args:
            token_ids: [batch_size, seq_len] - ID токенов
        
        Returns:
            [batch_size, seq_len, embedding_dim] - эмбеддинги
        """
        # Масштабирование для стабилизации (как в оригинальном Transformer)
        embeddings = self.embedding(token_ids) * math.sqrt(self.embedding_dim)
        return embeddings


class SinusoidalPositionalEncoding(nn.Module):
    """
    Синусоидальное позиционное кодирование (как в оригинальном Transformer)
    Точная реализация из "Attention Is All You Need"
    
    Args:
        embedding_dim: Размерность эмбеддингов (должна быть четной)
        max_seq_len: Максимальная длина последовательности (например, 512, 1024, 2048)
        dropout: Вероятность dropout (опционально)
    """
    def __init__(self, embedding_dim: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(p=dropout)
        
        # Создание позиционного кодирования
        # Формула: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        #          PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Вычисление div_term для синусоидального кодирования
        # 10000^(2i/d_model) = exp(2i * log(10000) / d_model)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() *
            -(math.log(10000.0) / embedding_dim)
        )
        
        # Применение синуса к четным позициям (индексы 0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Применение косинуса к нечетным позициям (индексы 1, 3, 5, ...)
        # Обработка случая, когда embedding_dim нечетное
        if embedding_dim % 2 == 1:
            # Если размерность нечетная, используем div_term без последнего элемента
            if len(div_term) > 0:
                pe[:, 1::2] = torch.cos(position * div_term[:-1])
            # Если div_term пустой (embedding_dim = 1), ничего не делаем
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # Добавление размерности batch для эффективности
        pe = pe.unsqueeze(0)  # [1, max_seq_len, embedding_dim]
        
        # Регистрация как буфер (не обучаемый параметр)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Добавление позиционного кодирования
        
        Args:
            x: [batch_size, seq_len, embedding_dim] - эмбеддинги токенов
        
        Returns:
            [batch_size, seq_len, embedding_dim] - эмбеддинги с позиционной информацией
        """
        # Добавляем позиционное кодирование
        # Используем только нужную длину последовательности
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        
        # Применяем dropout (как в оригинальном Transformer)
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Обучаемое позиционное кодирование (как в некоторых современных моделях)
    Альтернатива синусоидальному кодированию
    
    Args:
        embedding_dim: Размерность эмбеддингов
        max_seq_len: Максимальная длина последовательности
        dropout: Вероятность dropout
    """
    def __init__(self, embedding_dim: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(p=dropout)
        
        # Обучаемые позиционные эмбеддинги
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        
        # Инициализация
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Добавление обучаемого позиционного кодирования
        
        Args:
            x: [batch_size, seq_len, embedding_dim] - эмбеддинги токенов
        
        Returns:
            [batch_size, seq_len, embedding_dim] - эмбеддинги с позиционной информацией
        """
        seq_len = x.size(1)
        
        # Создаем позиционные индексы
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
        
        # Получаем позиционные эмбеддинги
        pos_embeddings = self.position_embedding(positions)  # [1, seq_len, embedding_dim]
        
        # Добавляем позиционное кодирование
        x = x + pos_embeddings
        
        # Применяем dropout
        return self.dropout(x)


class PositionalEncoding(nn.Module):
    """
    Универсальный класс позиционного кодирования
    Поддерживает как синусоидальное, так и обучаемое кодирование
    
    Args:
        embedding_dim: Размерность эмбеддингов
        max_seq_len: Максимальная длина последовательности
        dropout: Вероятность dropout
        learnable: Если True, использует обучаемое кодирование, иначе синусоидальное
    """
    def __init__(
        self, 
        embedding_dim: int, 
        max_seq_len: int = 512, 
        dropout: float = 0.1,
        learnable: bool = False
    ):
        super().__init__()
        
        if learnable:
            self.pos_encoding = LearnablePositionalEncoding(
                embedding_dim=embedding_dim,
                max_seq_len=max_seq_len,
                dropout=dropout
            )
        else:
            self.pos_encoding = SinusoidalPositionalEncoding(
                embedding_dim=embedding_dim,
                max_seq_len=max_seq_len,
                dropout=dropout
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Добавление позиционного кодирования"""
        return self.pos_encoding(x)


class EmbeddingLayer(nn.Module):
    """
    Полный слой эмбеддингов с токенами и позициями
    Реализация максимально аналогична современным LLM (GPT-2, GPT-3, BERT)
    
    Args:
        vocab_size: Размер словаря
        embedding_dim: Размерность эмбеддингов
        max_seq_len: Максимальная длина последовательности
        dropout: Вероятность dropout
        padding_idx: ID токена паддинга
        learnable_pos: Если True, использует обучаемое позиционное кодирование
        layer_norm: Если True, добавляет Layer Normalization (как в GPT-2)
        layer_norm_eps: Эпсилон для Layer Normalization
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        padding_idx: int = 0,
        learnable_pos: bool = False,
        layer_norm: bool = True,
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        
        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        
        self.positional_encoding = PositionalEncoding(
            embedding_dim=embedding_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
            learnable=learnable_pos
        )
        
        # Layer Normalization (как в GPT-2/GPT-3)
        self.layer_norm = None
        if layer_norm:
            self.layer_norm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.learnable_pos = learnable_pos
        self.use_layer_norm = layer_norm
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Преобразование токенов в эмбеддинги с позиционной информацией
        
        Args:
            token_ids: [batch_size, seq_len] - ID токенов
        
        Returns:
            [batch_size, seq_len, embedding_dim] - эмбеддинги
        """
        # Токенные эмбеддинги
        x = self.token_embedding(token_ids)
        
        # Добавление позиционного кодирования
        x = self.positional_encoding(x)
        
        # Layer Normalization (как в GPT-2/GPT-3)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        
        return x
    
    def get_embedding_dim(self) -> int:
        """Получение размерности эмбеддингов"""
        return self.embedding_dim


def create_embedding_from_tokenizer(
    tokenizer, 
    embedding_dim: int = 256, 
    max_seq_len: int = 512, 
    dropout: float = 0.1,
    learnable_pos: bool = False,
    layer_norm: bool = True
):
    """
    Создание Embedding Layer из BPETokenizer
    Настройки по умолчанию аналогичны GPT-2/GPT-3
    
    Args:
        tokenizer: Обученный BPETokenizer
        embedding_dim: Размерность эмбеддингов
        max_seq_len: Максимальная длина последовательности
        dropout: Вероятность dropout
        learnable_pos: Если True, использует обучаемое позиционное кодирование
        layer_norm: Если True, добавляет Layer Normalization (рекомендуется)
    
    Returns:
        EmbeddingLayer готовый к использованию
    """
    vocab_size = tokenizer.get_vocab_size()
    padding_idx = tokenizer.special_tokens.get('<PAD>', 0)
    
    embedding_layer = EmbeddingLayer(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        max_seq_len=max_seq_len,
        dropout=dropout,
        padding_idx=padding_idx,
        learnable_pos=learnable_pos,
        layer_norm=layer_norm
    )
    
    return embedding_layer

