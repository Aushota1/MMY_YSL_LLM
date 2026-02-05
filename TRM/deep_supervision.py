"""
Deep Supervision - Обучение с множественными шагами улучшения
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .latent_recursion import deep_recursion
from .losses import StableMaxLoss, binary_cross_entropy_with_logits


class DeepSupervisionTrainer:
    """
    Класс для обучения с deep supervision
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        n: int = 6,
        T: int = 3,
        N_sup: int = 16,
        device: str = 'cpu'
    ):
        self.model = model
        self.optimizer = optimizer
        self.n = n
        self.T = T
        self.N_sup = N_sup
        self.device = device
        
        # Функции потерь
        self.stable_max_loss = StableMaxLoss()
    
    def train_step(
        self,
        x_input: torch.Tensor,
        y_true: torch.Tensor,
        y_init: Optional[torch.Tensor] = None,
        z_init: Optional[torch.Tensor] = None
    ) -> Tuple[float, bool, int]:
        """
        Один шаг обучения с deep supervision
        
        Args:
            x_input: [batch, seq_len] - входные токены
            y_true: [batch, seq_len] - целевые токены
            y_init: [batch, seq_len, embedding_dim] - начальный ответ (опционально)
            z_init: [batch, seq_len, embedding_dim] - начальное латентное состояние (опционально)
        
        Returns:
            Tuple:
                - total_loss: общий loss за все шаги
                - early_stopped: был ли early stopping
                - num_steps: количество выполненных шагов
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Получаем компоненты модели
        input_embedding = self.model.input_embedding
        tiny_net = self.model.tiny_net
        refine_net = self.model.output_refinement
        output_head = self.model.output_head
        q_head = self.model.q_head
        
        # Встраивание входа
        x = input_embedding(x_input)  # [batch, seq_len, embedding_dim]
        
        # Инициализация y и z
        if y_init is None:
            batch_size, seq_len, embedding_dim = x.shape
            y = torch.zeros_like(x)  # Начальный ответ - нули
        else:
            y = y_init
        
        if z_init is None:
            z = torch.zeros_like(x)  # Начальное латентное состояние - нули
        else:
            z = z_init
        
        total_loss = 0.0
        early_stopped = False
        num_steps = 0
        
        # Deep supervision loop
        for step in range(self.N_sup):
            # Deep recursion
            y, z, y_hat, q_hat = deep_recursion(
                x, y, z,
                tiny_net, refine_net,
                output_head, q_head,
                n=self.n, T=self.T
            )
            
            # Вычисление loss
            loss = self.stable_max_loss(y_hat, y_true)
            
            # Проверка правильности для Q-head
            y_pred = y_hat.argmax(dim=-1)  # [batch, seq_len]
            is_correct = (y_pred == y_true).all(dim=-1).float()  # [batch] - все токены правильные
            
            # Binary CE для Q-head
            q_loss = binary_cross_entropy_with_logits(q_hat, is_correct)
            loss = loss + q_loss
            
            # Backward
            loss.backward()
            
            # Gradient clipping для стабильности
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            total_loss += loss.item()
            num_steps += 1
            
            # Early stopping: если q_hat предсказывает правильность
            # Проверяем, предсказал ли модель, что ответ правильный
            # Используем среднее значение по батчу для принятия решения
            q_prob = torch.sigmoid(q_hat).mean().item()  # Средняя вероятность правильности
            if q_prob > 0.5:  # Если средняя вероятность > 0.5, считаем что ответ найден
                early_stopped = True
                break
        
        avg_loss = total_loss / num_steps if num_steps > 0 else 0.0
        
        return avg_loss, early_stopped, num_steps
    
    def forward_step(
        self,
        x_input: torch.Tensor,
        y_init: Optional[torch.Tensor] = None,
        z_init: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass без обучения (для инференса)
        
        Args:
            x_input: [batch, seq_len] - входные токены
            y_init: [batch, seq_len, embedding_dim] - начальный ответ (опционально)
            z_init: [batch, seq_len, embedding_dim] - начальное латентное состояние (опционально)
        
        Returns:
            Tuple:
                - y_hat: [batch, seq_len, vocab_size] - предсказанный ответ
                - q_hat: [batch, 1] - предсказание правильности
        """
        self.model.eval()
        
        with torch.no_grad():
            # Получаем компоненты модели
            input_embedding = self.model.input_embedding
            tiny_net = self.model.tiny_net
            refine_net = self.model.output_refinement
            output_head = self.model.output_head
            q_head = self.model.q_head
            
            # Встраивание входа
            x = input_embedding(x_input)
            
            # Инициализация
            if y_init is None:
                y = torch.zeros_like(x)
            else:
                y = y_init
            
            if z_init is None:
                z = torch.zeros_like(x)
            else:
                z = z_init
            
            # Deep supervision loop (без градиентов)
            for step in range(self.N_sup):
                y, z, y_hat, q_hat = deep_recursion(
                    x, y, z,
                    tiny_net, refine_net,
                    output_head, q_head,
                    n=self.n, T=self.T
                )
                
                # Early stopping
                q_prob = torch.sigmoid(q_hat).mean().item()
                if q_prob > 0.5:
                    break
        
        return y_hat, q_hat

