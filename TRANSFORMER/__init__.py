"""
Модуль Transformer для LLM проекта

Этот модуль содержит реализацию Transformer Decoder Block для создания
GPT-подобных моделей. Реализация максимально аналогична современным LLM
(GPT-2, GPT-3, GPT-4).
"""

from .attention import MultiHeadSelfAttention
from .feed_forward import FeedForward
from .decoder_block import TransformerDecoderBlock
from .gpt_model import GPTModel

__all__ = [
    'MultiHeadSelfAttention',
    'FeedForward',
    'TransformerDecoderBlock',
    'GPTModel'
]

__version__ = '1.0.0'

