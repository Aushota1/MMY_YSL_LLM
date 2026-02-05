"""
Примеры использования Embedding Layer
"""

import torch
import sys
import os

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BPE_STUCTUR import BPETokenizer
from EMBEDDING_LAYER.embedding_layer import create_embedding_from_tokenizer


def example_basic_usage():
    """Базовый пример использования"""
    print("=" * 70)
    print("ПРИМЕР 1: Базовое использование")
    print("=" * 70)
    
    # 1. Загрузка токенизатора
    tokenizer = BPETokenizer()
    tokenizer.load("chekpoint.pkl")
    print(f"✓ Токенизатор загружен. Размер словаря: {tokenizer.get_vocab_size()}")
    
    # 2. Создание Embedding Layer
    embedding = create_embedding_from_tokenizer(
        tokenizer,
        embedding_dim=256,
        max_seq_len=512
    )
    print(f"✓ Embedding Layer создан. Размерность: {embedding.get_embedding_dim()}")
    
    # 3. Кодирование текста
    text = "Привет, мир! Это пример использования Embedding Layer."
    token_ids = tokenizer.encode(text)
    print(f"✓ Текст закодирован. Токенов: {len(token_ids)}")
    
    # 4. Преобразование в эмбеддинги
    token_tensor = torch.tensor([token_ids], dtype=torch.long)
    embeddings = embedding(token_tensor)
    
    print(f"\nРезультат:")
    print(f"  Входной текст: '{text}'")
    print(f"  Количество токенов: {len(token_ids)}")
    print(f"  Размер эмбеддингов: {embeddings.shape}")
    print(f"  Размерность каждого токена: {embeddings.shape[2]}")
    print()


def example_batch_processing():
    """Пример пакетной обработки"""
    print("=" * 70)
    print("ПРИМЕР 2: Пакетная обработка")
    print("=" * 70)
    
    tokenizer = BPETokenizer()
    tokenizer.load("chekpoint.pkl")
    
    embedding = create_embedding_from_tokenizer(tokenizer, embedding_dim=256)
    
    # Несколько текстов
    texts = [
        "Мама мыла раму.",
        "Привет, как дела?",
        "Сегодня хорошая погода.",
        "Искусственный интеллект меняет мир."
    ]
    
    print(f"Обрабатываем {len(texts)} текстов...")
    
    # Кодирование всех текстов
    token_ids_list = tokenizer.encode_batch(texts)
    
    # Выравнивание до максимальной длины
    max_len = max(len(ids) for ids in token_ids_list)
    pad_id = tokenizer.special_tokens['<PAD>']
    
    padded_tokens = []
    for ids in token_ids_list:
        padded = ids + [pad_id] * (max_len - len(ids))
        padded_tokens.append(padded[:max_len])
    
    # Преобразование в тензор
    token_tensor = torch.tensor(padded_tokens, dtype=torch.long)
    
    # Получение эмбеддингов
    embeddings = embedding(token_tensor)
    
    print(f"\nРезультат:")
    print(f"  Количество текстов: {len(texts)}")
    print(f"  Максимальная длина: {max_len} токенов")
    print(f"  Размер эмбеддингов: {embeddings.shape}")
    print(f"  Общее количество векторов: {embeddings.shape[0] * embeddings.shape[1]}")
    print()


def example_gpu_usage():
    """Пример использования GPU"""
    print("=" * 70)
    print("ПРИМЕР 3: Использование GPU")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")
    
    tokenizer = BPETokenizer()
    tokenizer.load("chekpoint.pkl")
    
    embedding = create_embedding_from_tokenizer(tokenizer, embedding_dim=256)
    embedding = embedding.to(device)
    
    text = "Этот пример работает на GPU, если доступно"
    token_ids = tokenizer.encode(text)
    token_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
    
    embeddings = embedding(token_tensor)
    
    print(f"\nРезультат:")
    print(f"  Устройство эмбеддингов: {embeddings.device}")
    print(f"  Размер: {embeddings.shape}")
    print()


def example_integration_with_model():
    """Пример интеграции с моделью"""
    print("=" * 70)
    print("ПРИМЕР 4: Интеграция с моделью")
    print("=" * 70)
    
    import torch.nn as nn
    
    tokenizer = BPETokenizer()
    tokenizer.load("chekpoint.pkl")
    
    # Создание Embedding Layer
    embedding = create_embedding_from_tokenizer(tokenizer, embedding_dim=256)
    
    # Простая модель (пример)
    class SimpleModel(nn.Module):
        def __init__(self, embedding_layer):
            super().__init__()
            self.embedding = embedding_layer
            self.linear = nn.Linear(256, 128)
            self.output = nn.Linear(128, tokenizer.get_vocab_size())
        
        def forward(self, token_ids):
            x = self.embedding(token_ids)
            x = x.mean(dim=1)  # Усреднение по последовательности
            x = torch.relu(self.linear(x))
            x = self.output(x)
            return x
    
    model = SimpleModel(embedding)
    
    # Тестовый текст
    text = "Пример интеграции"
    token_ids = tokenizer.encode(text)
    token_tensor = torch.tensor([token_ids], dtype=torch.long)
    
    # Forward pass
    output = model(token_tensor)
    
    print(f"\nРезультат:")
    print(f"  Входной текст: '{text}'")
    print(f"  Размер выхода модели: {output.shape}")
    print(f"  Размер словаря: {tokenizer.get_vocab_size()}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ EMBEDDING LAYER")
    print("=" * 70 + "\n")
    
    try:
        example_basic_usage()
        example_batch_processing()
        example_gpu_usage()
        example_integration_with_model()
        
        print("=" * 70)
        print("✓ ВСЕ ПРИМЕРЫ ВЫПОЛНЕНЫ УСПЕШНО!")
        print("=" * 70)
    except Exception as e:
        print(f"\n✗ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()

