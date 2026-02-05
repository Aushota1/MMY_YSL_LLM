"""
Latent Recursion - Рекурсивное обновление латентного состояния
"""

import torch
from typing import Callable, Tuple, Optional

from .tiny_recursive_network import TinyRecursiveNetwork
from .output_refinement import OutputRefinement


def latent_recursion(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    net: TinyRecursiveNetwork,
    n: int = 6
) -> torch.Tensor:
    """
    n шагов рекурсивного обновления латентного состояния z
    
    Args:
        x: [batch, seq_len, embedding_dim] - вопрос (input embedding)
        y: [batch, seq_len, embedding_dim] - ответ
        z: [batch, seq_len, embedding_dim] - латентное состояние
        net: TinyRecursiveNetwork для обновления
        n: количество рекурсивных шагов (по умолчанию 6)
    
    Returns:
        [batch, seq_len, embedding_dim] - обновленное латентное состояние z
    """
    for i in range(n):
        z = net(x, y, z)  # Обновление z
    return z


def deep_recursion(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    net: TinyRecursiveNetwork,
    refine_net: OutputRefinement,
    output_head: Callable,
    q_head: Callable,
    n: int = 6,
    T: int = 3
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    T глубоких рекурсий (T-1 без градиентов, 1 с градиентами)
    
    Args:
        x: [batch, seq_len, embedding_dim] - вопрос
        y: [batch, seq_len, embedding_dim] - ответ
        z: [batch, seq_len, embedding_dim] - латентное состояние
        net: TinyRecursiveNetwork для обновления z
        refine_net: OutputRefinement для улучшения y
        output_head: OutputHead для предсказания ответа
        q_head: QHead для предсказания правильности
        n: количество рекурсивных шагов (по умолчанию 6)
        T: количество глубоких рекурсий (по умолчанию 3)
    
    Returns:
        Tuple:
            - y_detached: [batch, seq_len, embedding_dim] - ответ (detached)
            - z_detached: [batch, seq_len, embedding_dim] - латентное состояние (detached)
            - y_hat: [batch, seq_len, vocab_size] - предсказанный ответ (logits)
            - q_hat: [batch, 1] - предсказание правильности (logits)
    """
    # T-1 раз без градиентов (warm-up)
    with torch.no_grad():
        for j in range(T - 1):
            z = latent_recursion(x, y, z, net, n)
            y = refine_net(y, z)
    
    # 1 раз с градиентами
    z = latent_recursion(x, y, z, net, n)
    y = refine_net(y, z)
    
    # Выходные головы
    y_hat = output_head(y)
    q_hat = q_head(y)
    
    # Detach для следующего шага deep supervision
    return y.detach(), z.detach(), y_hat, q_hat

