"""
Автоматическая система обучения и тестирования Embeddings
Интегрирует обучение с автоматической оценкой и рекомендациями
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import os
from typing import List, Optional, Dict
import json
from datetime import datetime

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BPE_STUCTUR import BPETokenizer
from EMBEDDING_LAYER.embedding_layer import EmbeddingLayer, create_embedding_from_tokenizer
from EMBEDDING_LAYER.embedding_trainer import SimpleLanguageModel, TextDataset
from EMBEDDING_LAYER.embedding_evaluator import EmbeddingEvaluator


class AutoEmbeddingSystem:
    """
    Автоматическая система обучения и оценки embeddings
    Объединяет обучение и автоматическое тестирование
    """
    
    def __init__(self, tokenizer_path: str = "chekpoint.pkl", device: str = None):
        """
        Инициализация системы
        
        Args:
            tokenizer_path: Путь к токенизатору
            device: Устройство для вычислений
        """
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Загрузка токенизатора
        print("📖 Загрузка токенизатора...")
        self.tokenizer = BPETokenizer()
        self.tokenizer.load(tokenizer_path)
        print(f"✓ Токенизатор загружен. Размер словаря: {self.tokenizer.get_vocab_size()}")
        
        self.model = None
        self.evaluator = None
        self.training_history = []
        
    def create_model(self, embedding_dim: int = 256, max_seq_len: int = 512,
                     hidden_dim: int = 256, learnable_pos: bool = False,
                     layer_norm: bool = True):
        """
        Создание модели
        
        Args:
            embedding_dim: Размерность embeddings
            max_seq_len: Максимальная длина последовательности
            hidden_dim: Размерность скрытого слоя
            learnable_pos: Обучаемое позиционное кодирование
            layer_norm: Использовать Layer Normalization
        """
        print(f"\n🔧 Создание модели...")
        print(f"   Embedding dim: {embedding_dim}")
        print(f"   Max seq len: {max_seq_len}")
        print(f"   Hidden dim: {hidden_dim}")
        print(f"   Learnable pos: {learnable_pos}")
        print(f"   Layer norm: {layer_norm}")
        
        # Создание embedding layer
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
        
        print("✓ Модель создана")
        
        # Создание evaluator
        self.evaluator = EmbeddingEvaluator(self.model, self.tokenizer, self.device)
        
    def train_with_auto_evaluation(
        self,
        texts: List[str],
        num_epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        max_length: int = 512,
        eval_interval: int = 2,
        save_checkpoints: bool = True,
        checkpoint_dir: str = "embedding_checkpoints"
    ) -> Dict:
        """
        Обучение с автоматической оценкой через определенные интервалы
        
        Args:
            texts: Список текстов для обучения
            num_epochs: Количество эпох
            batch_size: Размер батча
            learning_rate: Learning rate
            max_length: Максимальная длина последовательности
            eval_interval: Интервал оценки (каждые N эпох)
            save_checkpoints: Сохранять ли чекпоинты
            checkpoint_dir: Директория для чекпоинтов
        
        Returns:
            Словарь с историей обучения и финальными результатами
        """
        if self.model is None:
            raise ValueError("Сначала создайте модель через create_model()")
        
        print("\n" + "=" * 80)
        print("🚀 АВТОМАТИЧЕСКОЕ ОБУЧЕНИЕ С ОЦЕНКОЙ")
        print("=" * 80)
        
        # Создание датасета
        dataset = TextDataset(texts, self.tokenizer, max_length=max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Оптимизатор и loss
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.special_tokens['<PAD>'])
        
        # История
        training_results = {
            'training_history': [],
            'evaluation_history': [],
            'final_evaluation': None,
            'best_epoch': 0,
            'best_loss': float('inf')
        }
        
        # Создание директории для чекпоинтов
        if save_checkpoints:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"\n📊 Параметры обучения:")
        print(f"   Эпох: {num_epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Оценка каждые: {eval_interval} эпох")
        print(f"   Всего батчей: {len(dataloader)}")
        
        # Обучение
        for epoch in range(num_epochs):
            print(f"\n{'='*80}")
            print(f"📈 ЭПОХА {epoch + 1}/{num_epochs}")
            print(f"{'='*80}")
            
            # Обучение эпохи
            epoch_loss = self._train_epoch(dataloader, optimizer, criterion, epoch, num_epochs)
            
            training_results['training_history'].append({
                'epoch': epoch + 1,
                'loss': epoch_loss
            })
            
            # Обновление лучшего результата
            if epoch_loss < training_results['best_loss']:
                training_results['best_loss'] = epoch_loss
                training_results['best_epoch'] = epoch + 1
                
                # Сохранение лучшего чекпоинта
                if save_checkpoints:
                    checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch+1}.pth")
                    self._save_checkpoint(checkpoint_path, epoch + 1, epoch_loss)
            
            # Автоматическая оценка
            if (epoch + 1) % eval_interval == 0 or epoch == num_epochs - 1:
                print(f"\n🔍 Автоматическая оценка после эпохи {epoch + 1}...")
                eval_results = self.evaluator.evaluate_all(sample_size=1000, n_clusters=10)
                
                training_results['evaluation_history'].append({
                    'epoch': epoch + 1,
                    'evaluation': eval_results
                })
                
                # Краткий отчет
                self._print_evaluation_summary(eval_results, epoch + 1)
        
        # Финальная оценка
        print(f"\n{'='*80}")
        print("🏁 ФИНАЛЬНАЯ ОЦЕНКА")
        print(f"{'='*80}")
        
        final_eval = self.evaluator.evaluate_all(sample_size=2000, n_clusters=20)
        training_results['final_evaluation'] = final_eval
        
        # Полный отчет
        self.evaluator.print_report(final_eval)
        
        # Сохранение финального отчета
        report_path = os.path.join(checkpoint_dir, "final_evaluation_report.json")
        self.evaluator.save_report(report_path, final_eval)
        
        # Сохранение финальной модели
        if save_checkpoints:
            final_model_path = os.path.join(checkpoint_dir, "final_model.pth")
            self._save_checkpoint(final_model_path, num_epochs, training_results['best_loss'])
        
        return training_results
    
    def _train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer,
                     criterion: nn.Module, epoch: int, num_epochs: int) -> float:
        """Обучение одной эпохи"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # Forward pass
            logits = self.model(input_ids)
            logits = logits.view(-1, logits.size(-1))
            target_ids = target_ids.view(-1)
            
            # Loss
            loss = criterion(logits, target_ids)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Прогресс
            if batch_idx % 10 == 0:
                avg_loss = total_loss / num_batches
                progress = (batch_idx + 1) / len(dataloader) * 100
                progress_bar = "█" * int(progress / 2) + "░" * (50 - int(progress / 2))
                print(f"\r  [{progress_bar}] {progress:.1f}% | Loss: {avg_loss:.4f}", end='', flush=True)
        
        avg_loss = total_loss / num_batches
        print(f"\r  {'█' * 50} 100.0% | Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _print_evaluation_summary(self, eval_results: Dict, epoch: int):
        """Печать краткого резюме оценки"""
        print(f"\n  📊 Краткие результаты (эпоха {epoch}):")
        
        if 'cosine_similarity' in eval_results:
            cs = eval_results['cosine_similarity']
            print(f"     Косинусное сходство (среднее): {cs.get('mean', 0):.4f}")
        
        if 'clustering' in eval_results and 'silhouette_score' in eval_results['clustering']:
            sil = eval_results['clustering']['silhouette_score']
            print(f"     Silhouette Score: {sil:.4f}")
        
        if 'alignment_uniformity' in eval_results:
            au = eval_results['alignment_uniformity']
            print(f"     Alignment: {au.get('alignment', 0):.4f}, Uniformity: {au.get('uniformity', 0):.4f}")
        
        # Показываем топ-3 рекомендации
        if 'recommendations' in eval_results:
            recs = eval_results['recommendations'][:3]
            if recs:
                print(f"     💡 Топ рекомендации:")
                for i, rec in enumerate(recs, 1):
                    priority = rec.get('priority', 'unknown')
                    issue = rec.get('issue', 'N/A')
                    print(f"        {i}. [{priority.upper()}] {issue}")
    
    def _save_checkpoint(self, filepath: str, epoch: int, loss: float):
        """Сохранение чекпоинта"""
        # Получаем embedding layer из модели
        embedding_layer = self.model.embedding if hasattr(self.model, 'embedding') else None
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'embedding_dim': embedding_layer.embedding_dim if embedding_layer else 256,
            'vocab_size': self.tokenizer.get_vocab_size(),
            'epoch': epoch,
            'loss': loss,
            'training_history': self.training_history
        }, filepath)
    
    def load_model(self, model_path: str):
        """Загрузка модели из чекпоинта"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Определяем параметры из checkpoint
        embedding_dim = checkpoint.get('embedding_dim', 256)
        vocab_size = checkpoint.get('vocab_size', self.tokenizer.get_vocab_size())
        hidden_dim = checkpoint.get('hidden_dim', 256)
        
        # Определяем learnable_pos из структуры
        state_dict = checkpoint['model_state_dict']
        learnable_pos_key = "embedding.positional_encoding.pos_encoding.position_embedding.weight"
        learnable_pos = learnable_pos_key in state_dict
        
        # Определяем max_seq_len
        if learnable_pos:
            max_seq_len = state_dict[learnable_pos_key].shape[0]
        else:
            max_seq_len = checkpoint.get('max_seq_len', 512)
        
        # Создаем модель с правильными параметрами
        self.create_model(
            embedding_dim=embedding_dim,
            max_seq_len=max_seq_len,
            hidden_dim=hidden_dim,
            learnable_pos=learnable_pos,
            layer_norm=True
        )
        
        # Загружаем веса
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Обновляем evaluator
        self.evaluator = EmbeddingEvaluator(self.model, self.tokenizer, self.device)
        
        print(f"✓ Модель загружена из {model_path}")


def print_header(title: str):
    """Печать заголовка"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def clear_screen():
    """Очистка экрана"""
    os.system('cls' if os.name == 'nt' else 'clear')


def select_file() -> Optional[str]:
    """Интерактивный выбор файла для обучения"""
    print_header("ВЫБОР ФАЙЛА ДЛЯ ОБУЧЕНИЯ")
    
    # Поиск текстовых файлов в текущей директории
    txt_files = [f for f in os.listdir('.') if f.endswith('.txt') and os.path.isfile(f)]
    
    if txt_files:
        print("📁 Найдены текстовые файлы в текущей директории:")
        for i, file in enumerate(txt_files, 1):
            size = os.path.getsize(file) / 1024  # KB
            print(f"   {i}. {file} ({size:.1f} KB)")
        print(f"   {len(txt_files) + 1}. Ввести путь вручную")
        print(f"   0. Отмена")
        
        choice = input("\nВыберите файл (номер): ").strip()
        
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(txt_files):
                return txt_files[choice_num - 1]
            elif choice_num == len(txt_files) + 1:
                file_path = input("Введите путь к файлу: ").strip()
                if os.path.exists(file_path):
                    return file_path
                else:
                    print(f"❌ Файл {file_path} не найден!")
                    return None
            elif choice_num == 0:
                return None
            else:
                print("❌ Неверный выбор!")
                return None
        except ValueError:
            print("❌ Введите число!")
            return None
    else:
        file_path = input("Введите путь к файлу с данными: ").strip()
        if file_path and os.path.exists(file_path):
            return file_path
        elif file_path:
            print(f"❌ Файл {file_path} не найден!")
            return None
        else:
            return None


def show_parameter_tips(param_name: str):
    """Показ советов по выбору параметра"""
    tips = {
        'embedding_dim': {
            'title': 'Размерность Embeddings',
            'tips': [
                '128-256: Малая модель, быстрая, для экспериментов',
                '256-512: Средняя модель, баланс качества и скорости',
                '512-768: Большая модель, лучшее качество, требует больше памяти',
                '768+: Очень большая модель, для production систем',
                '💡 Рекомендация: Начните с 256, увеличьте если есть ресурсы'
            ],
            'default': 256
        },
        'num_epochs': {
            'title': 'Количество Эпох',
            'tips': [
                '5-10: Быстрое обучение, для тестирования',
                '10-20: Стандартное обучение, хороший баланс',
                '20-50: Длительное обучение, лучшее качество',
                '50+: Очень длительное, может быть переобучение',
                '💡 Рекомендация: Начните с 10, увеличьте если loss еще падает'
            ],
            'default': 10
        },
        'batch_size': {
            'title': 'Размер Батча',
            'tips': [
                '8-16: Для малой памяти (CPU или слабая GPU)',
                '16-32: Стандартный размер, хороший баланс',
                '32-64: Для GPU с достаточной памятью',
                '64+: Для мощных GPU, быстрее обучение',
                '💡 Рекомендация: Максимальный размер, который помещается в память'
            ],
            'default': 32
        },
        'learning_rate': {
            'title': 'Learning Rate',
            'tips': [
                '0.0001-0.0005: Консервативный, медленное обучение',
                '0.0005-0.001: Стандартный, хороший баланс',
                '0.001-0.003: Агрессивный, быстрое обучение, риск нестабильности',
                '0.003+: Очень агрессивный, может быть нестабильным',
                '💡 Рекомендация: Начните с 0.001, уменьшите если loss скачет'
            ],
            'default': 0.001
        },
        'eval_interval': {
            'title': 'Интервал Оценки',
            'tips': [
                '1: Оценка после каждой эпохи, медленнее обучение',
                '2-3: Стандартный интервал, хороший баланс',
                '5-10: Редкая оценка, быстрее обучение',
                '💡 Рекомендация: 2-3 эпохи для отслеживания прогресса'
            ],
            'default': 2
        },
        'max_seq_len': {
            'title': 'Максимальная Длина Последовательности',
            'tips': [
                '128: Короткие тексты, быстрее обучение',
                '256-512: Стандартная длина, хороший баланс',
                '512-1024: Длинные тексты, требует больше памяти',
                '1024+: Очень длинные тексты, для специальных задач',
                '💡 Рекомендация: 512 для большинства задач'
            ],
            'default': 512
        },
        'hidden_dim': {
            'title': 'Размерность Скрытого Слоя',
            'tips': [
                '128-256: Малая модель, быстрая',
                '256-512: Средняя модель, баланс',
                '512-1024: Большая модель, лучшее качество',
                '💡 Рекомендация: Обычно равен embedding_dim'
            ],
            'default': 256
        }
    }
    
    if param_name in tips:
        info = tips[param_name]
        print(f"\n💡 Советы по выбору: {info['title']}")
        print("-" * 60)
        for tip in info['tips']:
            print(f"   {tip}")
        print(f"\n   По умолчанию: {info['default']}")
        print("-" * 60)


def configure_parameters() -> Dict:
    """Интерактивная настройка параметров обучения"""
    print_header("НАСТРОЙКА ПАРАМЕТРОВ ОБУЧЕНИЯ")
    
    params = {}
    
    # Embedding dimension
    print("\n1️⃣  Размерность Embeddings")
    show_parameter_tips('embedding_dim')
    embedding_dim = input("Введите размерность (Enter для 256): ").strip()
    params['embedding_dim'] = int(embedding_dim) if embedding_dim else 256
    
    # Number of epochs
    print("\n2️⃣  Количество Эпох")
    show_parameter_tips('num_epochs')
    num_epochs = input("Введите количество эпох (Enter для 10): ").strip()
    params['num_epochs'] = int(num_epochs) if num_epochs else 10
    
    # Batch size
    print("\n3️⃣  Размер Батча")
    show_parameter_tips('batch_size')
    batch_size = input("Введите размер батча (Enter для 32): ").strip()
    params['batch_size'] = int(batch_size) if batch_size else 32
    
    # Learning rate
    print("\n4️⃣  Learning Rate")
    show_parameter_tips('learning_rate')
    learning_rate = input("Введите learning rate (Enter для 0.001): ").strip()
    params['learning_rate'] = float(learning_rate) if learning_rate else 0.001
    
    # Evaluation interval
    print("\n5️⃣  Интервал Оценки")
    show_parameter_tips('eval_interval')
    eval_interval = input("Введите интервал оценки (Enter для 2): ").strip()
    params['eval_interval'] = int(eval_interval) if eval_interval else 2
    
    # Max sequence length
    print("\n6️⃣  Максимальная Длина Последовательности")
    show_parameter_tips('max_seq_len')
    max_seq_len = input("Введите максимальную длину (Enter для 512): ").strip()
    params['max_seq_len'] = int(max_seq_len) if max_seq_len else 512
    
    # Hidden dimension
    print("\n7️⃣  Размерность Скрытого Слоя")
    show_parameter_tips('hidden_dim')
    hidden_dim = input("Введите размерность скрытого слоя (Enter для 256): ").strip()
    params['hidden_dim'] = int(hidden_dim) if hidden_dim else 256
    
    # Checkpoint directory
    print("\n8️⃣  Директория для Чекпоинтов")
    checkpoint_dir = input("Введите путь к директории (Enter для 'embedding_checkpoints'): ").strip()
    params['checkpoint_dir'] = checkpoint_dir if checkpoint_dir else 'embedding_checkpoints'
    
    # Learnable positional encoding
    print("\n9️⃣  Обучаемое Позиционное Кодирование")
    print("   - False: Синусоидальное (стандартное, быстрее)")
    print("   - True: Обучаемое (может быть лучше, но медленнее)")
    learnable_pos = input("Использовать обучаемое? (y/n, Enter для n): ").strip().lower()
    params['learnable_pos'] = learnable_pos == 'y'
    
    return params


def show_summary(system: AutoEmbeddingSystem, file_path: str, params: Dict, num_texts: int):
    """Показ сводки перед началом обучения"""
    print_header("СВОДКА ПАРАМЕТРОВ ОБУЧЕНИЯ")
    
    print("📂 Файл с данными:")
    print(f"   {file_path}")
    print(f"   Текстов: {num_texts}")
    
    print("\n📊 Параметры модели:")
    print(f"   Размерность embeddings: {params['embedding_dim']}")
    print(f"   Размерность скрытого слоя: {params['hidden_dim']}")
    print(f"   Максимальная длина: {params['max_seq_len']}")
    print(f"   Обучаемое позиционное кодирование: {params['learnable_pos']}")
    
    print("\n⚙️  Параметры обучения:")
    print(f"   Эпох: {params['num_epochs']}")
    print(f"   Размер батча: {params['batch_size']}")
    print(f"   Learning rate: {params['learning_rate']}")
    print(f"   Интервал оценки: {params['eval_interval']}")
    
    print("\n💾 Сохранение:")
    print(f"   Директория чекпоинтов: {params['checkpoint_dir']}")
    
    print("\n📖 Токенизатор:")
    print(f"   Размер словаря: {system.tokenizer.get_vocab_size()}")
    
    print("\n" + "=" * 80)
    confirm = input("\nНачать обучение? (y/n): ").strip().lower()
    return confirm == 'y'


def interactive_main():
    """Интерактивный режим с меню"""
    clear_screen()
    print("=" * 80)
    print("  🚀 АВТОМАТИЧЕСКАЯ СИСТЕМА ОБУЧЕНИЯ И ОЦЕНКИ EMBEDDINGS")
    print("=" * 80)
    
    # Выбор токенизатора
    print("\n📖 Выбор токенизатора:")
    tokenizer_path = input("Введите путь к токенизатору (Enter для 'chekpoint.pkl'): ").strip()
    tokenizer_path = tokenizer_path if tokenizer_path else "chekpoint.pkl"
    
    if not os.path.exists(tokenizer_path):
        print(f"❌ Ошибка: файл {tokenizer_path} не найден!")
        input("Нажмите Enter для выхода...")
        return
    
    # Создание системы
    try:
        system = AutoEmbeddingSystem(tokenizer_path=tokenizer_path)
    except Exception as e:
        print(f"❌ Ошибка при загрузке токенизатора: {e}")
        input("Нажмите Enter для выхода...")
        return
    
    # Выбор файла
    file_path = select_file()
    if not file_path:
        print("❌ Файл не выбран. Выход.")
        return
    
    # Загрузка данных
    print(f"\n📂 Загрузка данных из {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"✓ Загружено {len(texts)} текстов")
        
        if len(texts) == 0:
            print("❌ Файл пуст или не содержит текстов!")
            return
    except Exception as e:
        print(f"❌ Ошибка при загрузке файла: {e}")
        return
    
    # Настройка параметров
    params = configure_parameters()
    
    # Создание модели
    print("\n🔧 Создание модели...")
    system.create_model(
        embedding_dim=params['embedding_dim'],
        max_seq_len=params['max_seq_len'],
        hidden_dim=params['hidden_dim'],
        learnable_pos=params['learnable_pos'],
        layer_norm=True
    )
    
    # Показ сводки
    if not show_summary(system, file_path, params, len(texts)):
        print("❌ Обучение отменено.")
        return
    
    # Обучение
    try:
        results = system.train_with_auto_evaluation(
            texts=texts,
            num_epochs=params['num_epochs'],
            batch_size=params['batch_size'],
            learning_rate=params['learning_rate'],
            max_length=params['max_seq_len'],
            eval_interval=params['eval_interval'],
            checkpoint_dir=params['checkpoint_dir']
        )
        
        print("\n" + "=" * 80)
        print("✅ ОБУЧЕНИЕ И ОЦЕНКА ЗАВЕРШЕНЫ!")
        print("=" * 80)
        print(f"\n📁 Результаты сохранены в {params['checkpoint_dir']}/")
        print(f"   - Лучшая модель: best_model_epoch_{results['best_epoch']}.pth")
        print(f"   - Финальная модель: final_model.pth")
        print(f"   - Отчет: final_evaluation_report.json")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Обучение прервано пользователем")
        print("   Последний чекпоинт сохранен (если был)")
    except Exception as e:
        print(f"\n❌ Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nНажмите Enter для выхода...")


def main():
    """Главная функция с поддержкой командной строки и интерактивного режима"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Автоматическая система обучения и оценки embeddings")
    parser.add_argument("--interactive", "-i", action="store_true", help="Интерактивный режим")
    parser.add_argument("--tokenizer", type=str, default="chekpoint.pkl", help="Путь к токенизатору")
    parser.add_argument("--data", type=str, help="Путь к файлу с данными для обучения")
    parser.add_argument("--epochs", type=int, default=10, help="Количество эпох")
    parser.add_argument("--batch-size", type=int, default=32, help="Размер батча")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--embedding-dim", type=int, default=256, help="Размерность embeddings")
    parser.add_argument("--eval-interval", type=int, default=2, help="Интервал оценки (эпохи)")
    parser.add_argument("--checkpoint-dir", type=str, default="embedding_checkpoints", help="Директория для чекпоинтов")
    
    args = parser.parse_args()
    
    # Интерактивный режим
    if args.interactive or (not args.data and len(sys.argv) == 1):
        interactive_main()
        return
    
    # Режим командной строки (старый функционал)
    # Создание системы
    system = AutoEmbeddingSystem(tokenizer_path=args.tokenizer)
    
    # Создание модели
    system.create_model(
        embedding_dim=args.embedding_dim,
        max_seq_len=512,
        hidden_dim=256,
        learnable_pos=False,
        layer_norm=True
    )
    
    # Загрузка данных
    if args.data and os.path.exists(args.data):
        print(f"\n📂 Загрузка данных из {args.data}...")
        with open(args.data, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"✓ Загружено {len(texts)} текстов")
    else:
        print("\n⚠️  Файл с данными не указан или не найден. Используются тестовые данные.")
        texts = [
            "Привет, как дела?",
            "Сегодня хорошая погода.",
            "Машинное обучение - это интересно.",
            "Python - отличный язык программирования.",
            "Нейронные сети используются везде."
        ] * 100  # Умножаем для большего объема
    
    # Обучение с автоматической оценкой
    results = system.train_with_auto_evaluation(
        texts=texts,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_interval=args.eval_interval,
        checkpoint_dir=args.checkpoint_dir
    )
    
    print("\n✅ Обучение и оценка завершены!")
    print(f"📁 Результаты сохранены в {args.checkpoint_dir}/")


if __name__ == "__main__":
    main()

