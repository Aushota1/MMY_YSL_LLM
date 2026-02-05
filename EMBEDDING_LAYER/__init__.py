"""
Модуль Embedding Layer для LLM проекта

Этот модуль содержит реализацию Embedding Layer для преобразования
токенов в векторные представления с позиционной информацией.
Реализация максимально аналогична современным LLM (GPT-2, GPT-3, BERT, Transformer).
"""

from .embedding_layer import (
    TokenEmbedding,
    SinusoidalPositionalEncoding,
    LearnablePositionalEncoding,
    PositionalEncoding,
    EmbeddingLayer,
    create_embedding_from_tokenizer
)

__all__ = [
    'TokenEmbedding',
    'SinusoidalPositionalEncoding',
    'LearnablePositionalEncoding',
    'PositionalEncoding',
    'EmbeddingLayer',
    'create_embedding_from_tokenizer'
]

__version__ = '2.0.0'

