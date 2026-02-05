"""
Loss Functions - Функции потерь для TRM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StableMaxLoss(nn.Module):
    """
    Стабильная функция потерь (stable-max loss) из статьи TRM
    Улучшенная версия cross-entropy для стабильности обучения
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100
    ) -> torch.Tensor:
        """
        Вычисление stable-max loss
        
        Args:
            logits: [batch, seq_len, vocab_size] - логиты модели
            targets: [batch, seq_len] - целевые токены
            ignore_index: индекс для игнорирования (например, padding)
        
        Returns:
            Скалярный тензор - loss
        """
        # Reshape для удобства
        logits_flat = logits.view(-1, logits.size(-1))  # [batch*seq, vocab_size]
        targets_flat = targets.view(-1)  # [batch*seq]
        
        # Маска для игнорирования определенных индексов
        mask = (targets_flat != ignore_index)
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Применяем temperature scaling для стабильности
        logits_scaled = logits_flat / self.temperature
        
        # Стабильный softmax через log-sum-exp trick
        log_probs = F.log_softmax(logits_scaled, dim=-1)
        
        # Выбираем вероятности для целевых токенов
        selected_log_probs = log_probs[mask, targets_flat[mask]]
        
        # Negative log likelihood
        loss = -selected_log_probs.mean()
        
        return loss


def binary_cross_entropy_with_logits(
    q_hat: torch.Tensor,
    is_correct: torch.Tensor
) -> torch.Tensor:
    """
    Binary cross-entropy для Q-head (early stopping)
    
    Args:
        q_hat: [batch, 1] - логиты предсказания правильности
        is_correct: [batch] - бинарные метки (1 если правильный, 0 если нет)
    
    Returns:
        Скалярный тензор - loss
    """
    # Reshape для совместимости
    if q_hat.dim() > 1:
        q_hat = q_hat.squeeze(-1)  # [batch]
    
    if is_correct.dim() > 1:
        is_correct = is_correct.squeeze(-1)  # [batch]
    
    # Binary cross-entropy with logits
    loss = F.binary_cross_entropy_with_logits(
        q_hat,
        is_correct.float(),
        reduction='mean'
    )
    
    return loss

