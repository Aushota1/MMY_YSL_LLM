"""
Тесты для Embedding Layer
"""

import torch
import sys
import os

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BPE_STUCTUR import BPETokenizer
from EMBEDDING_LAYER.embedding_layer import (
    EmbeddingLayer, 
    create_embedding_from_tokenizer,
    SinusoidalPositionalEncoding,
    LearnablePositionalEncoding
)


def test_basic_embedding():
    """Базовый тест Embedding Layer"""
    print("=" * 70)
    print("ТЕСТ 1: Базовый Embedding Layer")
    print("=" * 70)
    
    # Загрузка токенизатора
    tokenizer = BPETokenizer()
    tokenizer.load("chekpoint.pkl")
    
    vocab_size = tokenizer.get_vocab_size()
    print(f"Размер словаря: {vocab_size}")
    
    # Создание Embedding Layer
    embedding = EmbeddingLayer(
        vocab_size=vocab_size,
        embedding_dim=256,
        max_seq_len=512,
        dropout=0.1,
        padding_idx=tokenizer.special_tokens['<PAD>']
    )
    
    # Тестовый текст
    text = "Привет, мир!"
    token_ids = tokenizer.encode(text)
    print(f"Текст: {text}")
    print(f"Токены: {token_ids}")
    
    # Преобразование в тензор
    token_tensor = torch.tensor([token_ids], dtype=torch.long)
    print(f"Размер входного тензора: {token_tensor.shape}")
    
    # Получение эмбеддингов
    embeddings = embedding(token_tensor)
    print(f"Размер эмбеддингов: {embeddings.shape}")
    print(f"Ожидаемый размер: [1, {len(token_ids)}, 256]")
    
    assert embeddings.shape == (1, len(token_ids), 256), "Неверный размер эмбеддингов!"
    print("✓ Тест пройден!")
    print()


def test_with_helper_function():
    """Тест с функцией-помощником"""
    print("=" * 70)
    print("ТЕСТ 2: Использование функции-помощника")
    print("=" * 70)
    
    # Загрузка токенизатора
    tokenizer = BPETokenizer()
    tokenizer.load("chekpoint.pkl")
    
    # Создание через функцию-помощник
    embedding = create_embedding_from_tokenizer(
        tokenizer,
        embedding_dim=256,
        max_seq_len=512
    )
    
    # Тестовый текст
    text = "Искусственный интеллект"
    token_ids = tokenizer.encode(text)
    
    token_tensor = torch.tensor([token_ids], dtype=torch.long)
    embeddings = embedding(token_tensor)
    
    print(f"Текст: {text}")
    print(f"Размер эмбеддингов: {embeddings.shape}")
    assert embeddings.shape[0] == 1, "Неверный batch size!"
    assert embeddings.shape[2] == 256, "Неверная размерность эмбеддингов!"
    print("✓ Тест пройден!")
    print()


def test_batch_processing():
    """Тест пакетной обработки"""
    print("=" * 70)
    print("ТЕСТ 3: Пакетная обработка")
    print("=" * 70)
    
    tokenizer = BPETokenizer()
    tokenizer.load("chekpoint.pkl")
    
    embedding = create_embedding_from_tokenizer(tokenizer, embedding_dim=256)
    
    # Несколько текстов
    texts = [
        "Мама мыла раму.",
        "Привет, как дела?",
        "Сегодня хорошая погода."
    ]
    
    # Кодирование
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
    print(f"Размер входного тензора: {token_tensor.shape}")
    
    # Получение эмбеддингов
    embeddings = embedding(token_tensor)
    print(f"Размер эмбеддингов: {embeddings.shape}")
    print(f"Ожидаемый размер: [{len(texts)}, {max_len}, 256]")
    
    assert embeddings.shape == (len(texts), max_len, 256), "Неверный размер!"
    print("✓ Тест пройден!")
    print()


def test_gpu_support():
    """Тест поддержки GPU"""
    print("=" * 70)
    print("ТЕСТ 4: Поддержка GPU")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")
    
    tokenizer = BPETokenizer()
    tokenizer.load("chekpoint.pkl")
    
    embedding = create_embedding_from_tokenizer(tokenizer, embedding_dim=256)
    embedding = embedding.to(device)
    
    text = "Тест на GPU"
    token_ids = tokenizer.encode(text)
    token_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
    
    embeddings = embedding(token_tensor)
    print(f"Эмбеддинги на устройстве: {embeddings.device}")
    print(f"Размер: {embeddings.shape}")
    
    assert embeddings.device.type == device.type, "Неверное устройство!"
    print("✓ Тест пройден!")
    print()


def test_sinusoidal_positional_encoding():
    """Тест синусоидального позиционного кодирования"""
    print("=" * 70)
    print("ТЕСТ 5: Синусоидальное позиционное кодирование")
    print("=" * 70)
    
    embedding_dim = 256
    max_seq_len = 512
    
    pos_encoding = SinusoidalPositionalEncoding(
        embedding_dim=embedding_dim,
        max_seq_len=max_seq_len,
        dropout=0.0  # Без dropout для теста
    )
    
    # Тестовый тензор
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, embedding_dim)
    
    # Применяем позиционное кодирование
    x_with_pos = pos_encoding(x)
    
    print(f"Входной размер: {x.shape}")
    print(f"Выходной размер: {x_with_pos.shape}")
    
    # Проверяем, что размерность сохранилась
    assert x_with_pos.shape == x.shape, "Размерность изменилась!"
    
    # Проверяем, что позиционное кодирование добавлено (значения должны отличаться)
    assert not torch.allclose(x, x_with_pos), "Позиционное кодирование не добавлено!"
    
    print("✓ Тест пройден!")
    print()


def test_learnable_positional_encoding():
    """Тест обучаемого позиционного кодирования"""
    print("=" * 70)
    print("ТЕСТ 6: Обучаемое позиционное кодирование")
    print("=" * 70)
    
    embedding_dim = 256
    max_seq_len = 512
    
    pos_encoding = LearnablePositionalEncoding(
        embedding_dim=embedding_dim,
        max_seq_len=max_seq_len,
        dropout=0.0
    )
    
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, embedding_dim)
    
    x_with_pos = pos_encoding(x)
    
    print(f"Входной размер: {x.shape}")
    print(f"Выходной размер: {x_with_pos.shape}")
    
    assert x_with_pos.shape == x.shape, "Размерность изменилась!"
    assert not torch.allclose(x, x_with_pos), "Позиционное кодирование не добавлено!"
    
    print("✓ Тест пройден!")
    print()


def test_layer_norm():
    """Тест Layer Normalization"""
    print("=" * 70)
    print("ТЕСТ 7: Layer Normalization")
    print("=" * 70)
    
    tokenizer = BPETokenizer()
    tokenizer.load("chekpoint.pkl")
    
    # С Layer Norm
    embedding_with_norm = create_embedding_from_tokenizer(
        tokenizer,
        embedding_dim=256,
        layer_norm=True
    )
    
    # Без Layer Norm
    embedding_without_norm = create_embedding_from_tokenizer(
        tokenizer,
        embedding_dim=256,
        layer_norm=False
    )
    
    text = "Тест Layer Norm"
    token_ids = tokenizer.encode(text)
    token_tensor = torch.tensor([token_ids], dtype=torch.long)
    
    embeddings_norm = embedding_with_norm(token_tensor)
    embeddings_no_norm = embedding_without_norm(token_tensor)
    
    print(f"С Layer Norm: {embeddings_norm.shape}")
    print(f"Без Layer Norm: {embeddings_no_norm.shape}")
    
    assert embeddings_norm.shape == embeddings_no_norm.shape, "Размеры не совпадают!"
    assert not torch.allclose(embeddings_norm, embeddings_no_norm), "Layer Norm не работает!"
    
    print("✓ Тест пройден!")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ЗАПУСК ТЕСТОВ EMBEDDING LAYER")
    print("=" * 70 + "\n")
    
    try:
        test_basic_embedding()
        test_with_helper_function()
        test_batch_processing()
        test_gpu_support()
        test_sinusoidal_positional_encoding()
        test_learnable_positional_encoding()
        test_layer_norm()
        
        print("=" * 70)
        print("✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("=" * 70)
    except Exception as e:
        print(f"\n✗ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()

