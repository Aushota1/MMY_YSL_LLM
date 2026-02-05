"""
Интерактивный скрипт для обучения GPT модели
С поддержкой выбора параметров и файлов с подсказками
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import sys
from typing import List, Optional, Dict

# Импорты
from BPE_STUCTUR import BPETokenizer
from TRANSFORMER import GPTModel


class TextDataset(Dataset):
    """Dataset для обучения GPT модели"""
    
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


def train_model(
    model: GPTModel,
    dataloader: DataLoader,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    device: str = 'cpu',
    save_path: str = 'gpt_model_checkpoint.pth',
    pad_id: int = 0
):
    """Обучение GPT модели"""
    
    model = model.to(device)
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)  # Игнорируем PAD токены
    
    print(f"\n🚀 Начало обучения на {device}")
    print(f"   Эпох: {num_epochs}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Параметров: {model.get_num_params():,} ({model.get_num_params_millions():.2f}M)")
    print("-" * 60)
    
    try:
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                
                # Forward pass
                logits = model(input_ids)  # [batch, seq_len, vocab_size]
                
                # Reshape для loss
                logits_flat = logits.view(-1, logits.size(-1))  # [batch*seq, vocab_size]
                target_flat = target_ids.view(-1)  # [batch*seq]
                
                # Loss
                loss = criterion(logits_flat, target_flat)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Прогресс каждые 10 батчей
                if (batch_idx + 1) % 10 == 0:
                    avg_loss = total_loss / num_batches
                    print(f"  Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(dataloader)} | Loss: {avg_loss:.4f}")
            
            # Средний loss за эпоху
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            print(f"\n✓ Epoch {epoch+1}/{num_epochs} завершена | Средний Loss: {avg_loss:.4f}")
            
            # Сохранение чекпоинта
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'vocab_size': model.embedding.vocab_size,
                'embedding_dim': model.embedding.embedding_dim,
                'num_layers': len(model.blocks),
                'num_heads': model.blocks[0].attention.num_heads,
            }
            torch.save(checkpoint, save_path)
            print(f"  💾 Чекпоинт сохранен: {save_path}")
            print("-" * 60)
        
        print(f"\n✅ Обучение завершено! Модель сохранена: {save_path}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Обучение прервано пользователем (Ctrl+C)")
        # Сохраняем текущий прогресс
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'vocab_size': model.embedding.vocab_size,
                'embedding_dim': model.embedding.embedding_dim,
                'num_layers': len(model.blocks),
                'num_heads': model.blocks[0].attention.num_heads,
            }
            torch.save(checkpoint, save_path)
            print(f"✅ Прогресс сохранен в: {save_path}")
            print(f"   Завершено эпох: {epoch + 1}/{num_epochs}")
        raise  # Пробрасываем исключение дальше


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


def select_embedding_file() -> Optional[str]:
    """Интерактивный выбор файла с обученными эмбеддингами"""
    print_header("ВЫБОР ЭМБЕДДИНГОВ")
    
    print("Выберите опцию:")
    print("  1. Использовать обученные эмбеддинги из .pth файла")
    print("  2. Создать новые эмбеддинги (обучение с нуля)")
    print("  0. Отмена")
    
    choice = input("\nВыберите опцию (номер): ").strip()
    
    if choice == "1":
        # Поиск .pth файлов
        pth_files = []
        
        # Текущая директория
        for f in os.listdir('.'):
            if f.endswith('.pth') and os.path.isfile(f):
                pth_files.append(f)
        
        # Директория my_checkpoints
        checkpoints_dir = "my_checkpoints"
        if os.path.exists(checkpoints_dir):
            for f in os.listdir(checkpoints_dir):
                if f.endswith('.pth'):
                    pth_files.append(os.path.join(checkpoints_dir, f))
        
        # EMBEDDING_LAYER директория (если есть)
        embedding_dir = "EMBEDDING_LAYER"
        if os.path.exists(embedding_dir):
            for f in os.listdir(embedding_dir):
                if f.endswith('.pth'):
                    pth_files.append(os.path.join(embedding_dir, f))
        
        if pth_files:
            print("\n📁 Найдены .pth файлы:")
            for i, file in enumerate(pth_files, 1):
                size = os.path.getsize(file) / 1024  # KB
                print(f"   {i}. {file} ({size:.1f} KB)")
            print(f"   {len(pth_files) + 1}. Ввести путь вручную")
            print(f"   0. Отмена")
            
            file_choice = input("\nВыберите файл (номер): ").strip()
            try:
                choice_num = int(file_choice)
                if 1 <= choice_num <= len(pth_files):
                    return pth_files[choice_num - 1]
                elif choice_num == len(pth_files) + 1:
                    file_path = input("Введите путь к файлу: ").strip()
                    if os.path.exists(file_path):
                        return file_path
                    else:
                        print(f"❌ Файл {file_path} не найден!")
                        return None
                elif choice_num == 0:
                    return None
            except ValueError:
                print("❌ Введите число!")
                return None
        else:
            file_path = input("Введите путь к .pth файлу: ").strip()
            if file_path and os.path.exists(file_path):
                return file_path
            elif file_path:
                print(f"❌ Файл {file_path} не найден!")
                return None
            return None
    elif choice == "2":
        return None  # Создать новые
    elif choice == "0":
        return None
    else:
        print("❌ Неверный выбор!")
        return None


def load_embedding_layer_from_checkpoint(
    checkpoint_path: str, 
    tokenizer: BPETokenizer,
    device: str = 'cpu'
):
    """Загрузка EmbeddingLayer из .pth чекпоинта"""
    try:
        print(f"\n📦 Загрузка эмбеддингов из {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Получение параметров из checkpoint
        embedding_dim = checkpoint.get('embedding_dim', 256)
        vocab_size = checkpoint.get('vocab_size', tokenizer.get_vocab_size())
        
        # Определяем параметры из state_dict
        state_dict = checkpoint['model_state_dict']
        
        # Определяем max_seq_len и learnable_pos
        max_seq_len = checkpoint.get('max_seq_len', 512)
        learnable_pos_key = "embedding.positional_encoding.pos_encoding.position_embedding.weight"
        sinusoidal_key = "embedding.positional_encoding.pos_encoding.pe"
        
        if learnable_pos_key in state_dict:
            learnable_pos = True
            max_seq_len = state_dict[learnable_pos_key].shape[0]
        elif sinusoidal_key in state_dict:
            learnable_pos = False
            pe_buffer = state_dict[sinusoidal_key]
            if len(pe_buffer.shape) >= 2:
                max_seq_len = pe_buffer.shape[1]
        else:
            learnable_pos = False
        
        # Определяем layer_norm
        layer_norm_key = "embedding.layer_norm.weight"
        layer_norm = layer_norm_key in state_dict
        
        # Создание EmbeddingLayer
        from EMBEDDING_LAYER.embedding_layer import create_embedding_from_tokenizer
        
        embedding_layer = create_embedding_from_tokenizer(
            tokenizer,
            embedding_dim=embedding_dim,
            max_seq_len=max_seq_len,
            learnable_pos=learnable_pos,
            layer_norm=layer_norm
        )
        
        # Загружаем веса эмбеддингов из checkpoint
        # Нужно извлечь только веса embedding layer
        embedding_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('embedding.'):
                # Убираем префикс 'embedding.'
                new_key = key[len('embedding.'):]
                embedding_state_dict[new_key] = value
        
        # Загружаем веса
        missing_keys, unexpected_keys = embedding_layer.load_state_dict(
            embedding_state_dict, strict=False
        )
        
        if missing_keys:
            print(f"⚠️  Отсутствующие ключи (игнорируются): {len(missing_keys)}")
        if unexpected_keys:
            print(f"⚠️  Неожиданные ключи (игнорируются): {len(unexpected_keys)}")
        
        print(f"✓ Эмбеддинги загружены успешно!")
        print(f"   Embedding dim: {embedding_dim}")
        print(f"   Max seq len: {max_seq_len}")
        print(f"   Learnable pos: {learnable_pos}")
        print(f"   Layer norm: {layer_norm}")
        
        return embedding_layer, embedding_dim, max_seq_len
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке эмбеддингов: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def show_parameter_tips(param_name: str):
    """Показ советов по выбору параметра"""
    tips = {
        'embedding_dim': {
            'title': 'Размерность Embeddings',
            'tips': [
                '128-256: Малая модель, быстрая, для экспериментов (~5-10M параметров)',
                '256-512: Средняя модель, баланс качества и скорости (~10-30M параметров)',
                '512-768: Большая модель, лучшее качество (~30-100M параметров)',
                '768+: Очень большая модель, для production систем (100M+ параметров)',
                '💡 Рекомендация: Начните с 256, увеличьте если есть ресурсы'
            ],
            'default': 256
        },
        'num_layers': {
            'title': 'Количество Слоев (Decoder Blocks)',
            'tips': [
                '2-4: Очень малая модель, для быстрых тестов',
                '4-8: Малая модель, для экспериментов',
                '8-12: Средняя модель, хороший баланс (как GPT-2 Small)',
                '12-24: Большая модель, лучшее качество (как GPT-2 Medium)',
                '24+: Очень большая модель, требует много ресурсов',
                '💡 Рекомендация: Начните с 4-6 слоев'
            ],
            'default': 4
        },
        'num_heads': {
            'title': 'Количество Голов Внимания',
            'tips': [
                '4-8: Для малых моделей (embedding_dim 128-256)',
                '8-12: Стандартное количество (embedding_dim 256-512)',
                '12-16: Для больших моделей (embedding_dim 512-768)',
                '16+: Для очень больших моделей',
                '💡 Важно: embedding_dim должен быть кратен num_heads!',
                '💡 Рекомендация: 8 голов для embedding_dim 256, 12 для 512'
            ],
            'default': 8
        },
        'num_epochs': {
            'title': 'Количество Эпох',
            'tips': [
                '3-5: Быстрое обучение, для тестирования',
                '5-10: Стандартное обучение, хороший баланс',
                '10-20: Длительное обучение, лучшее качество',
                '20-50: Очень длительное, для production',
                '50+: Может быть переобучение, нужен early stopping',
                '💡 Рекомендация: Начните с 10, увеличьте если loss еще падает'
            ],
            'default': 10
        },
        'batch_size': {
            'title': 'Размер Батча',
            'tips': [
                '1-4: Для малой памяти (CPU или слабая GPU)',
                '4-8: Для GPU с ограниченной памятью',
                '8-16: Стандартный размер, хороший баланс',
                '16-32: Для GPU с достаточной памятью',
                '32-64: Для мощных GPU, быстрее обучение',
                '64+: Для очень мощных GPU',
                '💡 Рекомендация: Максимальный размер, который помещается в память'
            ],
            'default': 8
        },
        'learning_rate': {
            'title': 'Learning Rate',
            'tips': [
                '0.0001-0.0003: Консервативный, медленное обучение',
                '0.0003-0.0005: Стандартный, хороший баланс',
                '0.0005-0.001: Агрессивный, быстрое обучение',
                '0.001-0.003: Очень агрессивный, риск нестабильности',
                '0.003+: Может быть нестабильным',
                '💡 Рекомендация: Начните с 0.0003-0.0005, уменьшите если loss скачет'
            ],
            'default': 0.0003
        },
        'max_length': {
            'title': 'Максимальная Длина Последовательности',
            'tips': [
                '64-128: Короткие тексты, быстрее обучение, меньше памяти',
                '128-256: Стандартная длина, хороший баланс',
                '256-512: Длинные тексты, требует больше памяти',
                '512-1024: Очень длинные тексты, для специальных задач',
                '1024+: Требует много памяти, для больших моделей',
                '💡 Рекомендация: 128-256 для большинства задач'
            ],
            'default': 128
        },
        'max_seq_len': {
            'title': 'Максимальная Длина для Модели',
            'tips': [
                '256: Для малых моделей',
                '512: Стандартная длина (как GPT-2)',
                '1024: Для средних моделей',
                '2048: Для больших моделей (как GPT-3)',
                '4096+: Для очень больших моделей',
                '💡 Рекомендация: 512-1024 для большинства случаев'
            ],
            'default': 512
        },
        'max_lines': {
            'title': 'Максимум Строк для Загрузки',
            'tips': [
                '100-500: Быстрая загрузка, для тестирования',
                '500-2000: Стандартная загрузка',
                '2000-10000: Большой датасет, лучшее качество',
                '10000+: Очень большой датасет, требует времени',
                'None: Загрузить все строки из файла',
                '💡 Рекомендация: Начните с 1000-2000, увеличьте если нужно'
            ],
            'default': 1000
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


def configure_parameters(loaded_embedding_dim: Optional[int] = None, 
                        loaded_max_seq_len: Optional[int] = None) -> Dict:
    """Интерактивная настройка параметров обучения"""
    print_header("НАСТРОЙКА ПАРАМЕТРОВ ОБУЧЕНИЯ")
    
    params = {}
    
    # Embedding dimension
    print("\n1️⃣  Размерность Embeddings")
    if loaded_embedding_dim:
        print(f"   ⚠️  ВНИМАНИЕ: Загружены эмбеддинги с embedding_dim={loaded_embedding_dim}")
        print(f"   Этот параметр будет автоматически установлен в {loaded_embedding_dim}")
    show_parameter_tips('embedding_dim')
    if loaded_embedding_dim:
        params['embedding_dim'] = loaded_embedding_dim
        print(f"   ✅ Установлено: {loaded_embedding_dim} (из загруженных эмбеддингов)")
    else:
        embedding_dim = input("Введите размерность (Enter для 256): ").strip()
        params['embedding_dim'] = int(embedding_dim) if embedding_dim else 256
    
    # Number of layers
    print("\n2️⃣  Количество Слоев (Decoder Blocks)")
    show_parameter_tips('num_layers')
    num_layers = input("Введите количество слоев (Enter для 4): ").strip()
    params['num_layers'] = int(num_layers) if num_layers else 4
    
    # Number of heads
    print("\n3️⃣  Количество Голов Внимания")
    show_parameter_tips('num_heads')
    num_heads = input("Введите количество голов (Enter для 8): ").strip()
    params['num_heads'] = int(num_heads) if num_heads else 8
    
    # Проверка кратности
    if params['embedding_dim'] % params['num_heads'] != 0:
        print(f"\n⚠️  Внимание: embedding_dim ({params['embedding_dim']}) не кратен num_heads ({params['num_heads']})!")
        print("   Автоматически корректирую num_heads...")
        # Находим ближайший делитель
        for h in [4, 8, 12, 16]:
            if params['embedding_dim'] % h == 0:
                params['num_heads'] = h
                print(f"   Установлено num_heads = {h}")
                break
    
    # Max sequence length
    print("\n4️⃣  Максимальная Длина Последовательности для Модели")
    if loaded_max_seq_len:
        print(f"   ⚠️  ВНИМАНИЕ: Загружены эмбеддинги с max_seq_len={loaded_max_seq_len}")
        print(f"   Этот параметр будет автоматически установлен в {loaded_max_seq_len}")
    show_parameter_tips('max_seq_len')
    if loaded_max_seq_len:
        params['max_seq_len'] = loaded_max_seq_len
        print(f"   ✅ Установлено: {loaded_max_seq_len} (из загруженных эмбеддингов)")
    else:
        max_seq_len = input("Введите максимальную длину (Enter для 512): ").strip()
        params['max_seq_len'] = int(max_seq_len) if max_seq_len else 512
    
    # Max length for training
    print("\n5️⃣  Максимальная Длина Последовательности для Обучения")
    show_parameter_tips('max_length')
    max_length = input("Введите максимальную длину (Enter для 128): ").strip()
    params['max_length'] = int(max_length) if max_length else 128
    
    # Number of epochs
    print("\n6️⃣  Количество Эпох")
    show_parameter_tips('num_epochs')
    num_epochs = input("Введите количество эпох (Enter для 10): ").strip()
    params['num_epochs'] = int(num_epochs) if num_epochs else 10
    
    # Batch size
    print("\n7️⃣  Размер Батча")
    show_parameter_tips('batch_size')
    batch_size = input("Введите размер батча (Enter для 8): ").strip()
    params['batch_size'] = int(batch_size) if batch_size else 8
    
    # Learning rate
    print("\n8️⃣  Learning Rate")
    show_parameter_tips('learning_rate')
    learning_rate = input("Введите learning rate (Enter для 0.0003): ").strip()
    params['learning_rate'] = float(learning_rate) if learning_rate else 0.0003
    
    # Max lines
    print("\n9️⃣  Максимум Строк для Загрузки")
    show_parameter_tips('max_lines')
    max_lines = input("Введите максимум строк (Enter для 1000, 'all' для всех): ").strip()
    if max_lines.lower() == 'all' or max_lines == '':
        params['max_lines'] = None
    else:
        params['max_lines'] = int(max_lines) if max_lines else 1000
    
    # Checkpoint path
    print("\n🔟  Путь для Сохранения Чекпоинта")
    save_path = input("Введите путь (Enter для 'gpt_model_checkpoint.pth'): ").strip()
    params['save_path'] = save_path if save_path else 'gpt_model_checkpoint.pth'
    
    # Tokenizer path
    print("\n1️⃣1️⃣  Путь к Токенизатору")
    tokenizer_path = input("Введите путь (Enter для 'chekpoint.pkl'): ").strip()
    params['tokenizer_path'] = tokenizer_path if tokenizer_path else 'chekpoint.pkl'
    
    return params


def show_summary(tokenizer: BPETokenizer, file_path: str, params: Dict, num_texts: int, 
                 embedding_loaded: bool = False, embedding_path: str = None):
    """Показ сводки перед началом обучения"""
    print_header("СВОДКА ПАРАМЕТРОВ ОБУЧЕНИЯ")
    
    print("📂 Файл с данными:")
    print(f"   {file_path}")
    print(f"   Текстов: {num_texts}")
    if params['max_lines']:
        print(f"   Загружено: {min(num_texts, params['max_lines'])} строк")
    
    print("\n📊 Параметры модели:")
    if embedding_loaded:
        print(f"   ✅ Используются ЗАГРУЖЕННЫЕ эмбеддинги")
        if embedding_path:
            print(f"   📁 Путь к эмбеддингам: {embedding_path}")
    else:
        print(f"   🆕 Создаются НОВЫЕ эмбеддинги")
    print(f"   Размерность embeddings: {params['embedding_dim']}")
    print(f"   Количество слоев: {params['num_layers']}")
    print(f"   Количество голов внимания: {params['num_heads']}")
    print(f"   Максимальная длина (модель): {params['max_seq_len']}")
    print(f"   Максимальная длина (обучение): {params['max_length']}")
    
    # Оценка размера модели
    estimated_params = estimate_model_size(
        vocab_size=tokenizer.get_vocab_size(),
        embedding_dim=params['embedding_dim'],
        num_layers=params['num_layers'],
        num_heads=params['num_heads']
    )
    print(f"   Примерный размер модели: ~{estimated_params/1e6:.2f}M параметров")
    
    print("\n⚙️  Параметры обучения:")
    print(f"   Эпох: {params['num_epochs']}")
    print(f"   Размер батча: {params['batch_size']}")
    print(f"   Learning rate: {params['learning_rate']}")
    
    print("\n💾 Сохранение:")
    print(f"   Путь к чекпоинту: {params['save_path']}")
    
    print("\n📖 Токенизатор:")
    print(f"   Путь: {params['tokenizer_path']}")
    print(f"   Размер словаря: {tokenizer.get_vocab_size()}")
    
    print("\n" + "=" * 80)
    confirm = input("\nНачать обучение? (y/n): ").strip().lower()
    return confirm == 'y'


def estimate_model_size(vocab_size: int, embedding_dim: int, num_layers: int, num_heads: int) -> int:
    """Оценка размера модели в параметрах"""
    # Embedding layer
    embedding_params = vocab_size * embedding_dim
    
    # Transformer blocks (приблизительно)
    # Attention: 4 * embedding_dim^2 (Q, K, V, out)
    # FFN: 2 * embedding_dim * (4 * embedding_dim) = 8 * embedding_dim^2
    # LayerNorm: 2 * embedding_dim
    params_per_block = 12 * embedding_dim * embedding_dim + 2 * embedding_dim
    
    # All blocks
    blocks_params = num_layers * params_per_block
    
    # Final layer norm
    final_norm_params = 2 * embedding_dim
    
    # Language model head
    lm_head_params = embedding_dim * vocab_size
    
    total = embedding_params + blocks_params + final_norm_params + lm_head_params
    return total


def interactive_main():
    """Интерактивный режим с меню"""
    clear_screen()
    print("=" * 80)
    print("  🚀 ИНТЕРАКТИВНОЕ ОБУЧЕНИЕ GPT МОДЕЛИ")
    print("=" * 80)
    
    # Выбор токенизатора
    print("\n📖 Выбор токенизатора:")
    tokenizer_path = input("Введите путь к токенизатору (Enter для 'chekpoint.pkl'): ").strip()
    tokenizer_path = tokenizer_path if tokenizer_path else "chekpoint.pkl"
    
    if not os.path.exists(tokenizer_path):
        print(f"❌ Ошибка: файл {tokenizer_path} не найден!")
        print("   Сначала обучите токенизатор используя tokenizer_trainer.py")
        input("Нажмите Enter для выхода...")
        return
    
    # Загрузка токенизатора
    try:
        tokenizer = BPETokenizer()
        tokenizer.load(tokenizer_path)
        vocab_size = tokenizer.get_vocab_size()
        print(f"✓ Токенизатор загружен. Размер словаря: {vocab_size}")
    except Exception as e:
        print(f"❌ Ошибка при загрузке токенизатора: {e}")
        input("Нажмите Enter для выхода...")
        return
    
    # Выбор эмбеддингов (ПЕРЕД настройкой параметров)
    embedding_path = select_embedding_file()
    embedding_layer = None
    loaded_embedding_dim = None
    loaded_max_seq_len = None
    
    if embedding_path:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        result = load_embedding_layer_from_checkpoint(
            embedding_path, tokenizer, device
        )
        if result[0] is not None:
            embedding_layer, loaded_embedding_dim, loaded_max_seq_len = result
            print(f"\n✅ Эмбеддинги загружены из: {embedding_path}")
        else:
            print("❌ Не удалось загрузить эмбеддинги. Создадим новые.")
            embedding_layer = None
    
    # Выбор файла
    file_path = select_file()
    if not file_path:
        print("❌ Файл не выбран. Выход.")
        return
    
    # Загрузка данных
    print(f"\n📂 Загрузка данных из {file_path}...")
    try:
        texts = load_text_file(file_path, max_lines=None)  # Загрузим все, потом ограничим
        print(f"✓ Загружено {len(texts)} текстов")
        
        if len(texts) == 0:
            print("❌ Файл пуст или не содержит текстов!")
            return
    except Exception as e:
        print(f"❌ Ошибка при загрузке файла: {e}")
        return
    
    # Настройка параметров (передаем загруженные параметры, если есть)
    params = configure_parameters(
        loaded_embedding_dim=loaded_embedding_dim,
        loaded_max_seq_len=loaded_max_seq_len
    )
    
    # Ограничение количества строк
    if params['max_lines']:
        texts = texts[:params['max_lines']]
        print(f"\n📊 Используется {len(texts)} текстов (ограничение: {params['max_lines']})")
    
    # Создание Dataset и DataLoader
    print(f"\n📦 Создание Dataset...")
    dataset = TextDataset(texts, tokenizer, max_length=params['max_length'])
    dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)
    print(f"✓ Dataset создан. Батчей: {len(dataloader)}")
    
    # Создание модели
    print(f"\n🤖 Создание GPT модели...")
    try:
        if embedding_layer is not None:
            # Используем загруженные эмбеддинги
            model = GPTModel(
                vocab_size=vocab_size,
                embedding_dim=params['embedding_dim'],
                num_layers=params['num_layers'],
                num_heads=params['num_heads'],
                max_seq_len=params['max_seq_len'],
                embedding_layer=embedding_layer  # Передаем загруженный embedding_layer
            )
            print(f"✓ Модель создана с загруженными эмбеддингами")
        else:
            # Создаем новые эмбеддинги
            model = GPTModel(
                vocab_size=vocab_size,
                embedding_dim=params['embedding_dim'],
                num_layers=params['num_layers'],
                num_heads=params['num_heads'],
                max_seq_len=params['max_seq_len'],
                tokenizer=tokenizer
            )
            print(f"✓ Модель создана с новыми эмбеддингами")
        
        print(f"   Параметров: {model.get_num_params():,} ({model.get_num_params_millions():.2f}M)")
    except Exception as e:
        print(f"❌ Ошибка при создании модели: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Показ сводки
    if not show_summary(tokenizer, file_path, params, len(texts), 
                       embedding_loaded=(embedding_layer is not None),
                       embedding_path=embedding_path if embedding_path else None):
        print("❌ Обучение отменено.")
        return
    
    # Устройство
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n📱 Устройство: {device}")
    
    # Обучение
    try:
        pad_id = tokenizer.special_tokens.get('<PAD>', 0)
        train_model(
            model=model,
            dataloader=dataloader,
            num_epochs=params['num_epochs'],
            learning_rate=params['learning_rate'],
            device=device,
            save_path=params['save_path'],
            pad_id=pad_id
        )
        
        print("\n" + "=" * 80)
        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print("=" * 80)
        print(f"\n📁 Модель сохранена: {params['save_path']}")
        print(f"\nДля использования модели:")
        print(f"  checkpoint = torch.load('{params['save_path']}')")
        print(f"  model.load_state_dict(checkpoint['model_state_dict'])")
        
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
    
    parser = argparse.ArgumentParser(description="Обучение GPT модели")
    parser.add_argument("--interactive", "-i", action="store_true", help="Интерактивный режим")
    parser.add_argument("--tokenizer", type=str, default="chekpoint.pkl", help="Путь к токенизатору")
    parser.add_argument("--data", type=str, help="Путь к файлу с данными")
    parser.add_argument("--epochs", type=int, default=3, help="Количество эпох")
    parser.add_argument("--batch-size", type=int, default=4, help="Размер батча")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--embedding-dim", type=int, default=256, help="Размерность embeddings")
    parser.add_argument("--num-layers", type=int, default=4, help="Количество слоев")
    parser.add_argument("--num-heads", type=int, default=8, help="Количество голов внимания")
    parser.add_argument("--max-length", type=int, default=128, help="Максимальная длина последовательности")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Максимальная длина для модели")
    parser.add_argument("--max-lines", type=int, default=1000, help="Максимум строк для загрузки")
    parser.add_argument("--save-path", type=str, default="gpt_model_checkpoint.pth", help="Путь для сохранения")
    
    args = parser.parse_args()
    
    # Интерактивный режим
    if args.interactive or (not args.data and len(sys.argv) == 1):
        interactive_main()
        return
    
    # Режим командной строки (старый функционал для обратной совместимости)
    print("=" * 60)
    print("🎯 ОБУЧЕНИЕ GPT МОДЕЛИ")
    print("=" * 60)
    
    # Параметры из аргументов
    tokenizer_path = args.tokenizer
    data_file = args.data
    max_lines = args.max_lines
    max_length = args.max_length
    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate = args.lr
    embedding_dim = args.embedding_dim
    num_layers = args.num_layers
    num_heads = args.num_heads
    max_seq_len = args.max_seq_len
    save_path = args.save_path
    
    # Устройство
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n📱 Устройство: {device}")
    
    # 1. Загрузка токенизатора
    print(f"\n📖 Загрузка токенизатора из {tokenizer_path}...")
    if not os.path.exists(tokenizer_path):
        print(f"❌ Ошибка: файл {tokenizer_path} не найден!")
        print("   Сначала обучите токенизатор используя tokenizer_trainer.py")
        return
    
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"✓ Токенизатор загружен. Размер словаря: {vocab_size}")
    
    # 2. Загрузка данных
    print(f"\n📚 Загрузка данных из {data_file}...")
    if not os.path.exists(data_file):
        print(f"❌ Ошибка: файл {data_file} не найден!")
        return
    
    texts = load_text_file(data_file, max_lines=max_lines)
    print(f"✓ Загружено {len(texts)} текстов")
    
    if len(texts) == 0:
        print("❌ Ошибка: нет данных для обучения!")
        return
    
    # 3. Создание Dataset и DataLoader
    print(f"\n📦 Создание Dataset...")
    dataset = TextDataset(texts, tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"✓ Dataset создан. Батчей: {len(dataloader)}")
    
    # 4. Создание модели
    print(f"\n🤖 Создание GPT модели...")
    model = GPTModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        tokenizer=tokenizer
    )
    print(f"✓ Модель создана")
    print(f"   Параметров: {model.get_num_params():,} ({model.get_num_params_millions():.2f}M)")
    
    # 5. Обучение
    pad_id = tokenizer.special_tokens.get('<PAD>', 0)
    train_model(
        model=model,
        dataloader=dataloader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        save_path=save_path,
        pad_id=pad_id
    )
    
    print("\n" + "=" * 60)
    print("✨ ГОТОВО!")
    print("=" * 60)
    print(f"\nДля использования модели загрузите чекпоинт:")
    print(f"  checkpoint = torch.load('{save_path}')")
    print(f"  model.load_state_dict(checkpoint['model_state_dict'])")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Обучение прервано пользователем")
        print("   Последний чекпоинт сохранен (если был)")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

