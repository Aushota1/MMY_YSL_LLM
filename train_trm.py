"""
Интерактивный скрипт для обучения TRM модели
С поддержкой deep supervision и отслеживания бенчмарков
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import sys
import json
from typing import List, Optional, Dict
from datetime import datetime

# Импорты
from BPE_STUCTUR import BPETokenizer
from TRM import TRMModel, DeepSupervisionTrainer

# Для отслеживания бенчмарков
import time


class TextDataset(Dataset):
    """Dataset для обучения TRM модели"""
    
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
            pad_id = self.tokenizer.special_tokens.get('<PAD>', 0)
            token_ids = token_ids + [pad_id] * (self.max_length - len(token_ids))
        
        token_ids = token_ids[:self.max_length]
        
        # Для TRM: input и target одинаковые (supervised learning)
        # В отличие от GPT, где используется shifted target
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        target_ids = torch.tensor(token_ids, dtype=torch.long)
        
        return input_ids, target_ids


def load_text_file(file_path: str, max_lines: int = None) -> List[str]:
    """Загрузка текста из файла"""
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            line = line.strip()
            if line and len(line) > 10:  # Пропускаем слишком короткие строки
                texts.append(line)
    return texts


class BenchmarkTracker:
    """Отслеживание бенчмарков и метрик"""
    
    def __init__(self, save_dir: str = "trm_benchmarks"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.metrics = {
            'epochs': [],
            'losses': [],
            'early_stops': [],
            'num_steps': [],
            'times': [],
            'accuracies': []
        }
    
    def log_step(
        self,
        epoch: int,
        loss: float,
        early_stopped: bool,
        num_steps: int,
        time_taken: float,
        accuracy: float = None
    ):
        """Логирование шага"""
        self.metrics['epochs'].append(epoch)
        self.metrics['losses'].append(loss)
        self.metrics['early_stops'].append(early_stopped)
        self.metrics['num_steps'].append(num_steps)
        self.metrics['times'].append(time_taken)
        if accuracy is not None:
            self.metrics['accuracies'].append(accuracy)
    
    def save(self, filename: str = None):
        """Сохранение метрик"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{timestamp}.json"
        
        filepath = os.path.join(self.save_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Метрики сохранены: {filepath}")
        return filepath
    
    def get_summary(self) -> Dict:
        """Получение сводки метрик"""
        if not self.metrics['losses']:
            return {}
        
        return {
            'total_epochs': len(self.metrics['epochs']),
            'avg_loss': sum(self.metrics['losses']) / len(self.metrics['losses']),
            'min_loss': min(self.metrics['losses']),
            'max_loss': max(self.metrics['losses']),
            'early_stop_rate': sum(self.metrics['early_stops']) / len(self.metrics['early_stops']) if self.metrics['early_stops'] else 0,
            'avg_steps': sum(self.metrics['num_steps']) / len(self.metrics['num_steps']) if self.metrics['num_steps'] else 0,
            'total_time': sum(self.metrics['times']),
            'avg_accuracy': sum(self.metrics['accuracies']) / len(self.metrics['accuracies']) if self.metrics['accuracies'] else None
        }


def compute_accuracy(y_hat: torch.Tensor, y_true: torch.Tensor, pad_id: int = 0) -> float:
    """Вычисление точности"""
    y_pred = y_hat.argmax(dim=-1)  # [batch, seq_len]
    
    # Маска для игнорирования padding
    mask = (y_true != pad_id)
    
    if mask.sum() == 0:
        return 0.0
    
    correct = (y_pred == y_true) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    
    return accuracy.item()


def train_model(
    model: TRMModel,
    dataloader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    embedding_lr: float = 1e-2,
    device: str = 'cpu',
    save_path: str = 'trm_model_checkpoint.pth',
    pad_id: int = 0,
    n: int = 6,
    T: int = 3,
    N_sup: int = 16,
    benchmark_tracker: Optional[BenchmarkTracker] = None
):
    """Обучение TRM модели с deep supervision"""
    
    model = model.to(device)
    model.train()
    
    # Разные learning rates для embeddings и остальных параметров
    embedding_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'input_embedding' in name:
            embedding_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = optim.AdamW(
        [
            {'params': embedding_params, 'lr': embedding_lr},
            {'params': other_params, 'lr': learning_rate}
        ],
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    # Deep Supervision Trainer
    trainer = DeepSupervisionTrainer(
        model=model,
        optimizer=optimizer,
        n=n,
        T=T,
        N_sup=N_sup,
        device=device
    )
    
    print(f"\n🚀 Начало обучения TRM на {device}")
    print(f"   Эпох: {num_epochs}")
    print(f"   Learning rate: {learning_rate} (embeddings: {embedding_lr})")
    print(f"   Параметров: {model.get_num_params():,} ({model.get_num_params_millions():.2f}M)")
    print(f"   n={n}, T={T}, N_sup={N_sup}")
    print("-" * 60)
    
    try:
        for epoch in range(num_epochs):
            total_loss = 0.0
            total_accuracy = 0.0
            num_batches = 0
            total_early_stops = 0
            total_steps = 0
            
            for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                
                # Обучение с deep supervision
                start_time = time.time()
                loss, early_stopped, num_steps = trainer.train_step(
                    x_input=input_ids,
                    y_true=target_ids
                )
                time_taken = time.time() - start_time
                
                # Вычисление точности
                with torch.no_grad():
                    y_hat, _ = model(input_ids, n=n, T=T, N_sup=N_sup)
                    accuracy = compute_accuracy(y_hat, target_ids, pad_id=pad_id)
                
                total_loss += loss
                total_accuracy += accuracy
                num_batches += 1
                total_early_stops += 1 if early_stopped else 0
                total_steps += num_steps
                
                # Логирование в benchmark tracker
                if benchmark_tracker:
                    benchmark_tracker.log_step(
                        epoch=epoch,
                        loss=loss,
                        early_stopped=early_stopped,
                        num_steps=num_steps,
                        time_taken=time_taken,
                        accuracy=accuracy
                    )
                
                # Прогресс каждые 10 батчей
                if (batch_idx + 1) % 10 == 0:
                    avg_loss = total_loss / num_batches
                    avg_accuracy = total_accuracy / num_batches
                    print(f"  Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(dataloader)} | "
                          f"Loss: {avg_loss:.4f} | Acc: {avg_accuracy:.4f} | "
                          f"Early stops: {total_early_stops}/{num_batches}")
            
            # Средние метрики за эпоху
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0.0
            early_stop_rate = total_early_stops / num_batches if num_batches > 0 else 0.0
            avg_steps = total_steps / num_batches if num_batches > 0 else 0.0
            
            print(f"\n✓ Epoch {epoch+1}/{num_epochs} завершена")
            print(f"   Средний Loss: {avg_loss:.4f}")
            print(f"   Средняя Accuracy: {avg_accuracy:.4f}")
            print(f"   Early stop rate: {early_stop_rate:.2%}")
            print(f"   Среднее количество шагов: {avg_steps:.2f}")
            
            # Сохранение чекпоинта
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'accuracy': avg_accuracy,
                'vocab_size': model.vocab_size,
                'embedding_dim': model.embedding_dim,
                'hidden_dim': model.hidden_dim,
                'n': n,
                'T': T,
                'N_sup': N_sup
            }
            torch.save(checkpoint, save_path)
            print(f"  💾 Чекпоинт сохранен: {save_path}")
            print("-" * 60)
        
        print(f"\n✅ Обучение завершено! Модель сохранена: {save_path}")
        
        # Сохранение бенчмарков
        if benchmark_tracker:
            benchmark_tracker.save()
            summary = benchmark_tracker.get_summary()
            print(f"\n📊 Сводка метрик:")
            for key, value in summary.items():
                print(f"   {key}: {value}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Обучение прервано пользователем (Ctrl+C)")
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'vocab_size': model.vocab_size,
                'embedding_dim': model.embedding_dim,
                'hidden_dim': model.hidden_dim,
            }
            torch.save(checkpoint, save_path)
            print(f"✅ Прогресс сохранен в: {save_path}")
            if benchmark_tracker:
                benchmark_tracker.save()
        raise


def interactive_main():
    """Интерактивный режим"""
    print("=" * 80)
    print("  🚀 ИНТЕРАКТИВНОЕ ОБУЧЕНИЕ TRM МОДЕЛИ")
    print("=" * 80)
    
    # Выбор токенизатора
    print("\n📖 Выбор токенизатора:")
    tokenizer_path = input("Введите путь к токенизатору (Enter для 'chekpoint.pkl'): ").strip()
    tokenizer_path = tokenizer_path if tokenizer_path else "chekpoint.pkl"
    
    if not os.path.exists(tokenizer_path):
        print(f"❌ Ошибка: файл {tokenizer_path} не найден!")
        return
    
    # Загрузка токенизатора
    try:
        tokenizer = BPETokenizer()
        tokenizer.load(tokenizer_path)
        vocab_size = tokenizer.get_vocab_size()
        print(f"✓ Токенизатор загружен. Размер словаря: {vocab_size}")
    except Exception as e:
        print(f"❌ Ошибка при загрузке токенизатора: {e}")
        return
    
    # Выбор файла
    print("\n📂 Выбор файла с данными:")
    file_path = input("Введите путь к файлу: ").strip()
    
    if not os.path.exists(file_path):
        print(f"❌ Файл {file_path} не найден!")
        return
    
    # Загрузка данных
    print(f"\n📚 Загрузка данных из {file_path}...")
    try:
        texts = load_text_file(file_path, max_lines=None)
        print(f"✓ Загружено {len(texts)} текстов")
        
        if len(texts) == 0:
            print("❌ Файл пуст или не содержит текстов!")
            return
    except Exception as e:
        print(f"❌ Ошибка при загрузке файла: {e}")
        return
    
    # Параметры модели
    print("\n⚙️  Параметры модели:")
    embedding_dim = int(input("Embedding dimension (Enter для 512): ").strip() or "512")
    hidden_dim = int(input("Hidden dimension (Enter для 512): ").strip() or "512")
    max_seq_len = int(input("Max sequence length (Enter для 512): ").strip() or "512")
    
    # Параметры обучения
    print("\n📊 Параметры обучения:")
    num_epochs = int(input("Количество эпох (Enter для 10): ").strip() or "10")
    batch_size = int(input("Batch size (Enter для 8): ").strip() or "8")
    learning_rate = float(input("Learning rate (Enter для 1e-4): ").strip() or "1e-4")
    embedding_lr = float(input("Embedding learning rate (Enter для 1e-2): ").strip() or "1e-2")
    max_length = int(input("Max length для dataset (Enter для 128): ").strip() or "128")
    
    # Параметры TRM
    print("\n🔄 Параметры TRM:")
    n = int(input("n (количество рекурсивных шагов, Enter для 6): ").strip() or "6")
    T = int(input("T (количество глубоких рекурсий, Enter для 3): ").strip() or "3")
    N_sup = int(input("N_sup (максимальное количество шагов супервизии, Enter для 16): ").strip() or "16")
    
    # Создание Dataset
    print(f"\n📦 Создание Dataset...")
    dataset = TextDataset(texts, tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"✓ Dataset создан. Батчей: {len(dataloader)}")
    
    # Создание модели
    print(f"\n🤖 Создание TRM модели...")
    try:
        model = TRMModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            tokenizer=tokenizer
        )
        print(f"✓ Модель создана")
        print(f"   Параметров: {model.get_num_params():,} ({model.get_num_params_millions():.2f}M)")
    except Exception as e:
        print(f"❌ Ошибка при создании модели: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Benchmark tracker
    benchmark_tracker = BenchmarkTracker()
    
    # Устройство
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n📱 Устройство: {device}")
    
    # Путь для сохранения
    save_path = input("\n💾 Путь для сохранения (Enter для 'trm_model_checkpoint.pth'): ").strip()
    save_path = save_path if save_path else 'trm_model_checkpoint.pth'
    
    # Подтверждение
    print("\n" + "=" * 80)
    print("Сводка параметров:")
    print(f"  Файл: {file_path}")
    print(f"  Текстов: {len(texts)}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Эпох: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate} (embeddings: {embedding_lr})")
    print(f"  n={n}, T={T}, N_sup={N_sup}")
    print("=" * 80)
    confirm = input("\nНачать обучение? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("❌ Обучение отменено.")
        return
    
    # Обучение
    try:
        pad_id = tokenizer.special_tokens.get('<PAD>', 0)
        train_model(
            model=model,
            dataloader=dataloader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            embedding_lr=embedding_lr,
            device=device,
            save_path=save_path,
            pad_id=pad_id,
            n=n,
            T=T,
            N_sup=N_sup,
            benchmark_tracker=benchmark_tracker
        )
        
        print("\n" + "=" * 80)
        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Обучение прервано пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Главная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Обучение TRM модели")
    parser.add_argument("--interactive", "-i", action="store_true", help="Интерактивный режим")
    parser.add_argument("--tokenizer", type=str, default="chekpoint.pkl", help="Путь к токенизатору")
    parser.add_argument("--data", type=str, help="Путь к файлу с данными")
    parser.add_argument("--epochs", type=int, default=10, help="Количество эпох")
    parser.add_argument("--batch-size", type=int, default=8, help="Размер батча")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--embedding-lr", type=float, default=1e-2, help="Embedding learning rate")
    parser.add_argument("--embedding-dim", type=int, default=512, help="Размерность embeddings")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Размерность hidden слоя")
    parser.add_argument("--max-length", type=int, default=128, help="Максимальная длина последовательности")
    parser.add_argument("--n", type=int, default=6, help="Количество рекурсивных шагов")
    parser.add_argument("--T", type=int, default=3, help="Количество глубоких рекурсий")
    parser.add_argument("--N-sup", type=int, default=16, help="Максимальное количество шагов супервизии")
    parser.add_argument("--save-path", type=str, default="trm_model_checkpoint.pth", help="Путь для сохранения")
    
    args = parser.parse_args()
    
    # Интерактивный режим
    if args.interactive or (not args.data and len(sys.argv) == 1):
        interactive_main()
        return
    
    # Режим командной строки
    print("=" * 60)
    print("🎯 ОБУЧЕНИЕ TRM МОДЕЛИ")
    print("=" * 60)
    
    # Загрузка токенизатора
    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()
    
    # Загрузка данных
    texts = load_text_file(args.data)
    
    # Создание Dataset
    dataset = TextDataset(texts, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Создание модели
    model = TRMModel(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        tokenizer=tokenizer
    )
    
    # Benchmark tracker
    benchmark_tracker = BenchmarkTracker()
    
    # Устройство
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Обучение
    pad_id = tokenizer.special_tokens.get('<PAD>', 0)
    train_model(
        model=model,
        dataloader=dataloader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        embedding_lr=args.embedding_lr,
        device=device,
        save_path=args.save_path,
        pad_id=pad_id,
        n=args.n,
        T=args.T,
        N_sup=args.N_sup,
        benchmark_tracker=benchmark_tracker
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Обучение прервано пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

