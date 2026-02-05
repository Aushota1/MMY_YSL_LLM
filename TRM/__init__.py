"""
Tiny Recursive Model (TRM) - Рекурсивная модель рассуждения
Реализация на основе статьи "Less is More: Recursive Reasoning with Tiny Networks"
"""

from .tiny_recursive_network import TinyRecursiveNetwork
from .latent_recursion import latent_recursion, deep_recursion
from .output_refinement import OutputRefinement
from .heads import OutputHead, QHead
from .losses import StableMaxLoss, binary_cross_entropy_with_logits
from .deep_supervision import DeepSupervisionTrainer
from .trm_model import TRMModel

__all__ = [
    'TinyRecursiveNetwork',
    'latent_recursion',
    'deep_recursion',
    'OutputRefinement',
    'OutputHead',
    'QHead',
    'StableMaxLoss',
    'binary_cross_entropy_with_logits',
    'DeepSupervisionTrainer',
    'TRMModel',
]

