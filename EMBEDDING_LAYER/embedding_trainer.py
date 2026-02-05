"""
Интерфейс для обучения Embedding Layer с визуализацией в реальном времени
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import sys
import os
import time
from typing import List, Optional, Dict
import threading
from collections import defaultdict

# Попытка импорта scipy (опционально)
try:
    from scipy.spatial.distance import pdist, squareform
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BPE_STUCTUR import BPETokenizer
from EMBEDDING_LAYER.embedding_layer import EmbeddingLayer, create_embedding_from_tokenizer


class TextDataset(Dataset):
    """Dataset для обучения эмбеддингов"""
    
    def __init__(self, texts: List[str], tokenizer: BPETokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        token_ids = self.tokenizer.encode(text)
        
        # Обрезка или паддинг
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            pad_id = self.tokenizer.special_tokens['<PAD>']
            token_ids = token_ids + [pad_id] * (self.max_length - len(token_ids))
        
        token_ids = token_ids[:self.max_length]
        
        # Input: все токены кроме последнего
        # Target: все токены кроме первого (shifted для language modeling)
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(token_ids[1:], dtype=torch.long)
        
        return input_ids, target_ids


class SimpleLanguageModel(nn.Module):
    """Простая модель для обучения эмбеддингов"""
    
    def __init__(self, embedding_layer: EmbeddingLayer, vocab_size: int, hidden_dim: int = 256):
        super().__init__()
        self.embedding = embedding_layer
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_layer.embedding_dim
        
        # Простой feed-forward слой
        self.ff = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, vocab_size)
        )
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Получаем эмбеддинги
        embeddings = self.embedding(token_ids)  # [batch, seq, dim]
        
        # Преобразуем в логиты
        logits = self.ff(embeddings)  # [batch, seq, vocab_size]
        
        return logits
    
    def get_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Получение эмбеддингов для визуализации"""
        with torch.no_grad():
            return self.embedding(token_ids)


class EmbeddingTrainer:
    """Класс для обучения эмбеддингов с визуализацией"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_history = {
            'loss': [],
            'epoch': [],
            'embeddings': []
        }
        self.is_training = False
        self.current_epoch = 0
        self.viz_fig = None
        self.viz_ax = None
        self.loss_fig = None
        self.loss_ax = None
        self.previous_embeddings = None  # Для сравнения между эпохами
        
    def clear_screen(self):
        """Очистка экрана"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self, title: str):
        """Печать заголовка"""
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80 + "\n")
    
    def load_tokenizer(self, path: str = "chekpoint.pkl"):
        """Загрузка токенизатора"""
        try:
            self.tokenizer = BPETokenizer()
            self.tokenizer.load(path)
            print(f"✓ Токенизатор загружен. Размер словаря: {self.tokenizer.get_vocab_size()}")
            return True
        except Exception as e:
            print(f"✗ Ошибка при загрузке токенизатора: {e}")
            return False
    
    def create_model(self, embedding_dim: int = 256, max_seq_len: int = 512, 
                     hidden_dim: int = 256, learnable_pos: bool = False, 
                     layer_norm: bool = True):
        """Создание модели"""
        if not self.tokenizer:
            print("✗ Сначала загрузите токенизатор")
            return False
        
        # Создание Embedding Layer
        embedding_layer = create_embedding_from_tokenizer(
            self.tokenizer,
            embedding_dim=embedding_dim,
            max_seq_len=max_seq_len,
            learnable_pos=learnable_pos,
            layer_norm=layer_norm
        )
        
        # Создание модели
        self.model = SimpleLanguageModel(
            embedding_layer=embedding_layer,
            vocab_size=self.tokenizer.get_vocab_size(),
            hidden_dim=hidden_dim
        ).to(self.device)
        
        print(f"✓ Модель создана")
        print(f"  Размерность эмбеддингов: {embedding_dim}")
        print(f"  Максимальная длина: {max_seq_len}")
        print(f"  Скрытая размерность: {hidden_dim}")
        print(f"  Обучаемое позиционное кодирование: {learnable_pos}")
        print(f"  Layer Normalization: {layer_norm}")
        print(f"  Устройство: {self.device}")
        
        return True
    
    def visualize_embeddings(self, embeddings: np.ndarray, tokens: List[str], 
                            epoch: int, method: str = 'tsne', fig=None, ax=None):
        """
        Красивая визуализация эмбеддингов с обновлением в реальном времени
        
        Args:
            embeddings: Массив эмбеддингов [n_tokens, embedding_dim]
            tokens: Список токенов для подписей
            epoch: Номер эпохи
            method: Метод визуализации ('tsne' или 'pca')
            fig: Фигура matplotlib (для обновления)
            ax: Оси matplotlib (для обновления)
        """
        if embeddings.shape[0] < 2:
            return
        
        # Снижение размерности
        if method == 'tsne':
            try:
                reducer = TSNE(n_components=2, random_state=42, 
                             perplexity=min(30, embeddings.shape[0]-1),
                             n_iter=300, verbose=0)
                embeddings_2d = reducer.fit_transform(embeddings)
            except:
                # Если t-SNE не работает, используем PCA
                reducer = PCA(n_components=2)
                embeddings_2d = reducer.fit_transform(embeddings)
        else:
            reducer = PCA(n_components=2)
            embeddings_2d = reducer.fit_transform(embeddings)
        
        # Создание или обновление графика
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(16, 12))
            fig.suptitle('Визуализация эмбеддингов в реальном времени', 
                        fontsize=16, fontweight='bold', y=0.98)
        else:
            ax.clear()
        
        # Вычисление расстояний для цветовой карты (близкие токены - похожие цвета)
        if HAS_SCIPY:
            try:
                distances = squareform(pdist(embeddings_2d))
                # Используем среднее расстояние до ближайших соседей
                k = min(5, len(embeddings_2d) - 1)
                if k > 0:
                    neighbor_distances = np.mean(np.partition(distances, kth=k, axis=1)[:, :k+1], axis=1)
                    colors = neighbor_distances
                else:
                    colors = range(len(embeddings_2d))
            except:
                colors = range(len(embeddings_2d))
        else:
            # Простая цветовая схема без scipy
            colors = range(len(embeddings_2d))
        
        # Красивая визуализация с градиентом
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=colors, cmap='plasma', 
                            alpha=0.7, s=150, edgecolors='white', 
                            linewidths=1.5, zorder=3)
        
        # Подписи для токенов (умный выбор - самые интересные)
        num_labels = min(30, len(tokens))
        # Выбираем токены с наибольшим разбросом
        if len(embeddings_2d) > num_labels:
            # Выбираем токены, которые далеко от центра
            center = np.mean(embeddings_2d, axis=0)
            distances_from_center = np.linalg.norm(embeddings_2d - center, axis=1)
            label_indices = np.argsort(distances_from_center)[-num_labels:]
        else:
            label_indices = range(len(tokens))
        
        for i in label_indices:
            token_display = tokens[i][:12] if len(tokens[i]) > 12 else tokens[i]
            ax.annotate(token_display, 
                       (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                       fontsize=9, alpha=0.8, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                               alpha=0.3, edgecolor='gray', linewidth=0.5),
                       zorder=4)
        
        # Красивое оформление
        ax.set_title(f'Эпоха {epoch} | Метод: {method.upper()} | Токенов: {len(embeddings_2d)}', 
                    fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('Первая компонента', fontsize=11, fontweight='bold')
        ax.set_ylabel('Вторая компонента', fontsize=11, fontweight='bold')
        
        # Цветовая шкала
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label('Расстояние до соседей', fontsize=10, fontweight='bold')
        
        # Сетка и стиль
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        ax.set_facecolor('#f8f9fa')
        
        # Добавляем информацию о статистике
        stats_text = f'Размерность: {embeddings.shape[1]}\n'
        stats_text += f'Среднее расстояние: {np.mean(colors):.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.05)
        
        return fig, ax
    
    def get_sample_embeddings(self, num_samples: int = 50) -> tuple:
        """Получение примеров эмбеддингов для визуализации"""
        if not self.model or not self.tokenizer:
            return None, None
        
        # Выбираем случайные токены из словаря
        vocab_size = self.tokenizer.get_vocab_size()
        sample_indices = np.random.choice(min(num_samples, vocab_size), 
                                         size=min(num_samples, vocab_size), 
                                         replace=False)
        
        # Создаем тензор с токенами
        token_tensor = torch.tensor([sample_indices], dtype=torch.long).to(self.device)
        
        # Получаем эмбеддинги
        with torch.no_grad():
            embeddings = self.model.get_embeddings(token_tensor)
            embeddings = embeddings[0].cpu().numpy()  # [seq_len, embedding_dim]
        
        # Получаем текстовые представления токенов
        tokens = []
        for idx in sample_indices:
            if idx in self.tokenizer.vocab:
                token = self.tokenizer.vocab[idx]
                tokens.append(token[:15])  # Ограничиваем длину
            else:
                tokens.append(f"ID_{idx}")
        
        return embeddings, tokens
    
    def train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module, epoch: int, num_epochs: int,
                   visualize: bool = True, viz_interval: int = 1):
        """Обучение одной эпохи"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # Forward pass
            logits = self.model(input_ids)
            
            # Reshape для loss
            logits = logits.view(-1, logits.size(-1))  # [batch*seq, vocab_size]
            target_ids = target_ids.view(-1)  # [batch*seq]
            
            # Loss
            loss = criterion(logits, target_ids)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Прогресс с красивым форматированием
            if batch_idx % 10 == 0:
                avg_loss = total_loss / num_batches
                progress = (batch_idx + 1) / len(dataloader) * 100
                progress_bar = "█" * int(progress / 2) + "░" * (50 - int(progress / 2))
                print(f"\rЭпоха {epoch+1}/{num_epochs} | "
                      f"[{progress_bar}] {progress:.1f}% | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Батч {batch_idx+1}/{len(dataloader)}", end='', flush=True)
        
        avg_loss = total_loss / num_batches
        
        # Визуализация
        if visualize and (epoch + 1) % viz_interval == 0:
            print(f"\n📊 Обновление визуализации...")
            embeddings, tokens = self.get_sample_embeddings(num_samples=50)
            if embeddings is not None:
                # Сохраняем для сравнения
                self.previous_embeddings = embeddings.copy() if self.previous_embeddings is None else embeddings.copy()
                
                self.viz_fig, self.viz_ax = self.visualize_embeddings(
                    embeddings, tokens, epoch + 1, method='tsne',
                    fig=self.viz_fig, ax=self.viz_ax
                )
        
        return avg_loss
    
    def train(self, texts: List[str], num_epochs: int = 10, batch_size: int = 32,
              learning_rate: float = 0.001, max_length: int = 512,
              visualize: bool = True, viz_interval: int = 1):
        """Обучение модели"""
        if not self.model:
            print("✗ Сначала создайте модель")
            return
        
        self.print_header("ОБУЧЕНИЕ ЭМБЕДДИНГОВ")
        
        # Создание датасета
        dataset = TextDataset(texts, self.tokenizer, max_length=max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Оптимизатор и loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.special_tokens['<PAD>']
        )
        
        # История
        self.training_history = {'loss': [], 'epoch': []}
        
        # Настройка matplotlib для интерактивного режима
        if visualize:
            plt.ion()
            # Создаем два окна: одно для эмбеддингов, одно для loss
            self.viz_fig, self.viz_ax = None, None
            self.loss_fig, self.loss_ax = plt.subplots(figsize=(10, 6))
            self.loss_ax.set_title('История Loss', fontsize=14, fontweight='bold')
            self.loss_ax.set_xlabel('Эпоха', fontsize=12)
            self.loss_ax.set_ylabel('Loss', fontsize=12)
            self.loss_ax.grid(True, alpha=0.3)
        
        print(f"Начинаем обучение:")
        print(f"  Эпох: {num_epochs}")
        print(f"  Размер батча: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Визуализация: {'Включена' if visualize else 'Выключена'}")
        print("-" * 80)
        
        try:
            for epoch in range(num_epochs):
                self.current_epoch = epoch
                loss = self.train_epoch(dataloader, optimizer, criterion, 
                                       epoch, num_epochs, visualize, viz_interval)
                
                self.training_history['loss'].append(loss)
                self.training_history['epoch'].append(epoch + 1)
                
                print(f"\n✓ Эпоха {epoch+1} завершена. Loss: {loss:.4f}")
                
                # Обновление графика loss в реальном времени
                if visualize:
                    self.update_loss_plot()
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Обучение прервано пользователем")
        
        finally:
            if visualize:
                # Сохранение финальных графиков
                if self.viz_fig is not None:
                    self.viz_fig.savefig('embeddings_visualization.png', dpi=150, bbox_inches='tight')
                    print("💾 Визуализация эмбеддингов сохранена: embeddings_visualization.png")
                
                if self.loss_fig is not None:
                    self.loss_fig.savefig('loss_history.png', dpi=150, bbox_inches='tight')
                    print("💾 График loss сохранен: loss_history.png")
                
                plt.ioff()
                print("\n💾 Графики сохранены. Нажмите Enter для закрытия окон...")
                input()
                plt.close('all')
        
        print("\n✓ Обучение завершено!")
    
    def update_loss_plot(self):
        """Обновление графика loss в реальном времени"""
        if len(self.training_history['loss']) == 0 or self.loss_ax is None:
            return
        
        self.loss_ax.clear()
        
        # Красивый график с градиентом
        epochs = self.training_history['epoch']
        losses = self.training_history['loss']
        
        # Градиентная линия
        for i in range(len(epochs) - 1):
            self.loss_ax.plot(epochs[i:i+2], losses[i:i+2], 
                            color=plt.cm.viridis(i / max(1, len(epochs)-1)),
                            linewidth=2.5, alpha=0.8)
        
        # Точки на графике
        self.loss_ax.scatter(epochs, losses, c=range(len(epochs)), 
                           cmap='viridis', s=80, zorder=3,
                           edgecolors='white', linewidths=1.5)
        
        # Оформление
        self.loss_ax.set_title(f'История Loss | Текущий Loss: {losses[-1]:.4f}', 
                              fontsize=13, fontweight='bold')
        self.loss_ax.set_xlabel('Эпоха', fontsize=11, fontweight='bold')
        self.loss_ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
        self.loss_ax.grid(True, alpha=0.2, linestyle='--')
        self.loss_ax.set_facecolor('#f8f9fa')
        
        # Добавляем тренд
        if len(epochs) > 1:
            z = np.polyfit(epochs, losses, 1)
            p = np.poly1d(z)
            self.loss_ax.plot(epochs, p(epochs), "r--", alpha=0.5, 
                            linewidth=1, label=f'Тренд (наклон: {z[0]:.4f})')
            self.loss_ax.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
    
    def interactive_menu(self):
        """Интерактивное меню"""
        while True:
            self.clear_screen()
            self.print_header("ОБУЧЕНИЕ ЭМБЕДДИНГОВ - ГЛАВНОЕ МЕНЮ")
            
            print("1. Загрузить токенизатор")
            print("2. Создать модель")
            print("3. Настроить параметры обучения")
            print("4. Начать обучение")
            print("5. Визуализировать текущие эмбеддинги")
            print("6. Показать историю обучения")
            print("7. Сохранить модель")
            print("8. Загрузить модель")
            print("0. Выход")
            print("\n" + "=" * 80)
            
            choice = input("\nВыберите действие: ").strip()
            
            if choice == '0':
                print("\nДо свидания!")
                break
            elif choice == '1':
                self.menu_load_tokenizer()
            elif choice == '2':
                self.menu_create_model()
            elif choice == '3':
                self.menu_configure_training()
            elif choice == '4':
                self.menu_start_training()
            elif choice == '5':
                self.menu_visualize_embeddings()
            elif choice == '6':
                self.menu_show_history()
            elif choice == '7':
                self.menu_save_model()
            elif choice == '8':
                self.menu_load_model()
            else:
                print("\n✗ Неверный выбор")
                input("Нажмите Enter для продолжения...")
    
    def menu_load_tokenizer(self):
        """Меню загрузки токенизатора"""
        self.print_header("ЗАГРУЗКА ТОКЕНИЗАТОРА")
        
        path = input("Путь к файлу токенизатора (Enter = chekpoint.pkl): ").strip()
        if not path:
            path = "chekpoint.pkl"
        
        if self.load_tokenizer(path):
            input("\nНажмите Enter для продолжения...")
    
    def menu_create_model(self):
        """Меню создания модели"""
        if not self.tokenizer:
            print("✗ Сначала загрузите токенизатор")
            input("Нажмите Enter для продолжения...")
            return
        
        self.print_header("СОЗДАНИЕ МОДЕЛИ")
        
        try:
            embedding_dim = int(input("Размерность эмбеддингов (по умолчанию 256): ").strip() or "256")
            max_seq_len = int(input("Максимальная длина последовательности (по умолчанию 512): ").strip() or "512")
            hidden_dim = int(input("Скрытая размерность (по умолчанию 256): ").strip() or "256")
            
            learnable_pos_input = input("Обучаемое позиционное кодирование? (y/n, по умолчанию n): ").strip().lower()
            learnable_pos = learnable_pos_input == 'y'
            
            layer_norm_input = input("Layer Normalization? (y/n, по умолчанию y): ").strip().lower()
            layer_norm = layer_norm_input != 'n'
            
            if self.create_model(embedding_dim, max_seq_len, hidden_dim, learnable_pos, layer_norm):
                input("\nНажмите Enter для продолжения...")
        except ValueError:
            print("✗ Ошибка: введите корректные числа")
            input("Нажмите Enter для продолжения...")
    
    def menu_configure_training(self):
        """Меню настройки параметров обучения"""
        self.print_header("НАСТРОЙКА ПАРАМЕТРОВ ОБУЧЕНИЯ")
        
        if not hasattr(self, 'training_config'):
            self.training_config = {
                'num_epochs': 10,
                'batch_size': 32,
                'learning_rate': 0.001,
                'max_length': 512,
                'visualize': True,
                'viz_interval': 1
            }
        
        print("Текущие параметры:")
        for key, value in self.training_config.items():
            print(f"  {key}: {value}")
        
        print("\nИзменить параметры? (y/n): ", end='')
        if input().strip().lower() == 'y':
            try:
                self.training_config['num_epochs'] = int(
                    input(f"Количество эпох (текущее: {self.training_config['num_epochs']}): ").strip() 
                    or str(self.training_config['num_epochs'])
                )
                self.training_config['batch_size'] = int(
                    input(f"Размер батча (текущее: {self.training_config['batch_size']}): ").strip() 
                    or str(self.training_config['batch_size'])
                )
                self.training_config['learning_rate'] = float(
                    input(f"Learning rate (текущее: {self.training_config['learning_rate']}): ").strip() 
                    or str(self.training_config['learning_rate'])
                )
                self.training_config['max_length'] = int(
                    input(f"Максимальная длина (текущее: {self.training_config['max_length']}): ").strip() 
                    or str(self.training_config['max_length'])
                )
                viz_input = input(f"Визуализация? (y/n, текущее: {'y' if self.training_config['visualize'] else 'n'}): ").strip().lower()
                self.training_config['visualize'] = viz_input != 'n'
                if self.training_config['visualize']:
                    self.training_config['viz_interval'] = int(
                        input(f"Интервал визуализации (каждые N эпох, текущее: {self.training_config['viz_interval']}): ").strip() 
                        or str(self.training_config['viz_interval'])
                    )
                print("✓ Параметры обновлены")
            except ValueError:
                print("✗ Ошибка: введите корректные значения")
        
        input("\nНажмите Enter для продолжения...")
    
    def menu_start_training(self):
        """Меню начала обучения"""
        if not self.model:
            print("✗ Сначала создайте модель")
            input("Нажмите Enter для продолжения...")
            return
        
        self.print_header("НАЧАЛО ОБУЧЕНИЯ")
        
        # Выбор источника данных
        print("Откуда загрузить тексты?")
        print("1. Ввод вручную")
        print("2. Из файла")
        choice = input("Выберите (1/2): ").strip()
        
        texts = []
        
        if choice == '2':
            # Загрузка из файла
            file_path = input("Путь к файлу с текстами: ").strip()
            if file_path and os.path.exists(file_path):
                try:
                    encoding = input("Кодировка (по умолчанию utf-8): ").strip() or 'utf-8'
                    with open(file_path, 'r', encoding=encoding) as f:
                        texts = [line.strip() for line in f if line.strip()]
                    print(f"✓ Загружено {len(texts)} текстов из файла")
                except Exception as e:
                    print(f"✗ Ошибка при чтении файла: {e}")
                    input("Нажмите Enter для продолжения...")
                    return
            else:
                print("✗ Файл не найден")
                input("Нажмите Enter для продолжения...")
                return
        else:
            # Ввод вручную
            print("\nВведите тексты для обучения (каждый текст с новой строки).")
            print("Для завершения ввода введите пустую строку или 'END':")
            print("-" * 80)
            
            line_num = 1
            while True:
                text = input(f"Текст {line_num}: ").strip()
                if not text or text.upper() == 'END':
                    break
                if text:
                    texts.append(text)
                    line_num += 1
        
        if not texts:
            print("✗ Не введено ни одного текста")
            input("Нажмите Enter для продолжения...")
            return
        
        # Получение параметров
        if not hasattr(self, 'training_config'):
            self.menu_configure_training()
        
        config = self.training_config
        
        # Обучение
        self.train(
            texts=texts,
            num_epochs=config['num_epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            max_length=config['max_length'],
            visualize=config['visualize'],
            viz_interval=config['viz_interval']
        )
        
        input("\nНажмите Enter для продолжения...")
    
    def menu_visualize_embeddings(self):
        """Меню визуализации эмбеддингов"""
        if not self.model:
            print("✗ Сначала создайте модель")
            input("Нажмите Enter для продолжения...")
            return
        
        self.print_header("ВИЗУАЛИЗАЦИЯ ЭМБЕДДИНГОВ")
        
        method = input("Метод визуализации (tsne/pca, по умолчанию tsne): ").strip().lower() or "tsne"
        num_samples = int(input("Количество токенов для визуализации (по умолчанию 50): ").strip() or "50")
        
        print("\n📊 Создание визуализации...")
        embeddings, tokens = self.get_sample_embeddings(num_samples=num_samples)
        
        if embeddings is not None:
            plt.ion()
            self.visualize_embeddings(embeddings, tokens, self.current_epoch, method=method)
            print("✓ Визуализация создана")
            input("\nНажмите Enter для закрытия графика...")
            plt.close('all')
        else:
            print("✗ Ошибка при создании визуализации")
            input("Нажмите Enter для продолжения...")
    
    def menu_show_history(self):
        """Меню показа истории"""
        if len(self.training_history['loss']) == 0:
            print("✗ История обучения пуста")
            input("Нажмите Enter для продолжения...")
            return
        
        self.print_header("ИСТОРИЯ ОБУЧЕНИЯ")
        
        print("Loss по эпохам:")
        for epoch, loss in zip(self.training_history['epoch'], self.training_history['loss']):
            print(f"  Эпоха {epoch}: {loss:.4f}")
        
        # Построение графика
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_history['epoch'], self.training_history['loss'], 
                'b-', linewidth=2, marker='o', markersize=6)
        plt.title('История Loss во время обучения', fontsize=14, fontweight='bold')
        plt.xlabel('Эпоха', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        input("\nНажмите Enter для продолжения...")
    
    def menu_save_model(self):
        """Меню сохранения модели"""
        if not self.model:
            print("✗ Нет модели для сохранения")
            input("Нажмите Enter для продолжения...")
            return
        
        self.print_header("СОХРАНЕНИЕ МОДЕЛИ")
        
        path = input("Путь для сохранения (по умолчанию embedding_model.pth): ").strip() or "embedding_model.pth"
        
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'embedding_dim': self.model.embedding_dim,
                'vocab_size': self.tokenizer.get_vocab_size() if self.tokenizer else None,
                'training_history': self.training_history
            }, path)
            print(f"✓ Модель сохранена: {path}")
        except Exception as e:
            print(f"✗ Ошибка при сохранении: {e}")
        
        input("\nНажмите Enter для продолжения...")
    
    def menu_load_model(self):
        """Меню загрузки модели"""
        self.print_header("ЗАГРУЗКА МОДЕЛИ")
        
        path = input("Путь к файлу модели: ").strip()
        
        if not path or not os.path.exists(path):
            print("✗ Файл не найден")
            input("Нажмите Enter для продолжения...")
            return
        
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            if not self.model:
                print("✗ Сначала создайте модель с теми же параметрами")
                input("Нажмите Enter для продолжения...")
                return
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'training_history' in checkpoint:
                self.training_history = checkpoint['training_history']
            
            print(f"✓ Модель загружена: {path}")
        except Exception as e:
            print(f"✗ Ошибка при загрузке: {e}")
        
        input("\nНажмите Enter для продолжения...")


def main():
    """Главная функция"""
    try:
        trainer = EmbeddingTrainer()
        trainer.interactive_menu()
    except KeyboardInterrupt:
        print("\n\nПрервано пользователем. Выход...")
    except Exception as e:
        print(f"\n✗ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

