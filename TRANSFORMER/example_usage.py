"""
Примеры использования Transformer модуля
"""

import torch
import sys
import os

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TRANSFORMER import (
    MultiHeadSelfAttention,
    FeedForward,
    TransformerDecoderBlock,
    GPTModel
)


def example_attention():
    """Пример использования Multi-Head Self-Attention"""
    print("=" * 80)
    print("ПРИМЕР 1: Multi-Head Self-Attention")
    print("=" * 80)
    
    # Параметры
    batch_size = 2
    seq_len = 10
    embedding_dim = 256
    num_heads = 8
    
    # Создание модуля
    attention = MultiHeadSelfAttention(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        dropout=0.1,
        causal=True
    )
    
    # Входные данные
    x = torch.randn(batch_size, seq_len, embedding_dim)
    print(f"Вход: {x.shape}")
    
    # Forward pass
    output = attention(x)
    print(f"Выход: {output.shape}")
    print(f"Параметров: {sum(p.numel() for p in attention.parameters()):,}")
    print()


def example_feed_forward():
    """Пример использования Feed-Forward Network"""
    print("=" * 80)
    print("ПРИМЕР 2: Feed-Forward Network")
    print("=" * 80)
    
    # Параметры
    batch_size = 2
    seq_len = 10
    embedding_dim = 256
    ff_dim = 1024
    
    # Создание модуля
    ffn = FeedForward(
        embedding_dim=embedding_dim,
        ff_dim=ff_dim,
        dropout=0.1,
        activation='gelu'
    )
    
    # Входные данные
    x = torch.randn(batch_size, seq_len, embedding_dim)
    print(f"Вход: {x.shape}")
    
    # Forward pass
    output = ffn(x)
    print(f"Выход: {output.shape}")
    print(f"Параметров: {sum(p.numel() for p in ffn.parameters()):,}")
    print()


def example_decoder_block():
    """Пример использования Transformer Decoder Block"""
    print("=" * 80)
    print("ПРИМЕР 3: Transformer Decoder Block")
    print("=" * 80)
    
    # Параметры
    batch_size = 2
    seq_len = 10
    embedding_dim = 256
    num_heads = 8
    ff_dim = 1024
    
    # Создание модуля
    block = TransformerDecoderBlock(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=0.1
    )
    
    # Входные данные
    x = torch.randn(batch_size, seq_len, embedding_dim)
    print(f"Вход: {x.shape}")
    
    # Forward pass
    output = block(x)
    print(f"Выход: {output.shape}")
    print(f"Параметров: {sum(p.numel() for p in block.parameters()):,}")
    print()


def example_gpt_model():
    """Пример использования полной GPT модели"""
    print("=" * 80)
    print("ПРИМЕР 4: Полная GPT Model")
    print("=" * 80)
    
    # Параметры
    vocab_size = 10000
    embedding_dim = 256
    num_layers = 6
    num_heads = 8
    max_seq_len = 512
    batch_size = 2
    seq_len = 10
    
    # Создание модели
    model = GPTModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        dropout=0.1
    )
    
    # Входные данные (token IDs)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Вход (token IDs): {token_ids.shape}")
    
    # Forward pass
    logits = model(token_ids)
    print(f"Выход (logits): {logits.shape}")
    print(f"Параметров: {model.get_num_params():,} ({model.get_num_params_millions():.2f}M)")
    
    # Получение предсказаний
    probs = torch.softmax(logits, dim=-1)
    predictions = torch.argmax(logits, dim=-1)
    print(f"Предсказания: {predictions.shape}")
    print()


def example_gpt_with_tokenizer():
    """Пример использования GPT модели с токенизатором"""
    print("=" * 80)
    print("ПРИМЕР 5: GPT Model с токенизатором")
    print("=" * 80)
    
    try:
        from BPE_STUCTUR import BPETokenizer
        from EMBEDDING_LAYER import create_embedding_from_tokenizer
        
        # Загрузка токенизатора
        tokenizer = BPETokenizer()
        tokenizer.load("chekpoint.pkl")
        
        vocab_size = tokenizer.get_vocab_size()
        embedding_dim = 256
        num_layers = 6
        num_heads = 8
        
        print(f"Размер словаря: {vocab_size}")
        
        # Создание модели с токенизатором
        model = GPTModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            tokenizer=tokenizer
        )
        
        # Тестовый текст
        text = "Машинное обучение - это подраздел искусственного интеллекта."
        print(f"Входной текст: {text}")
        
        # Кодирование
        token_ids_list = tokenizer.encode(text)
        token_ids = torch.tensor([token_ids_list])
        print(f"Token IDs: {token_ids_list}")
        print(f"Длина последовательности: {len(token_ids_list)}")
        
        # Forward pass
        logits = model(token_ids)
        print(f"Выход (logits): {logits.shape}")
        print(f"Параметров: {model.get_num_params():,} ({model.get_num_params_millions():.2f}M)")
        
        # Получение предсказаний для следующего токена
        next_token_logits = logits[0, -1, :]  # Последний токен
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        top_k = 5
        top_probs, top_indices = torch.topk(next_token_probs, top_k)
        
        print(f"\nТоп-{top_k} вероятных следующих токенов:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
            token = tokenizer.vocab.get(idx.item(), f"<UNK_{idx.item()}>")
            print(f"  {i}. {token}: {prob.item():.4f}")
        
    except FileNotFoundError:
        print("⚠️  Токенизатор не найден (chekpoint.pkl)")
        print("   Сначала обучите токенизатор через tokenizer_trainer.py")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


def example_stack_blocks():
    """Пример стекирования нескольких Decoder Blocks"""
    print("=" * 80)
    print("ПРИМЕР 6: Стекирование Decoder Blocks")
    print("=" * 80)
    
    embedding_dim = 256
    num_heads = 8
    num_layers = 3
    
    # Создание стека блоков
    blocks = torch.nn.ModuleList([
        TransformerDecoderBlock(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            ff_dim=1024,
            dropout=0.1
        )
        for _ in range(num_layers)
    ])
    
    # Входные данные
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, embedding_dim)
    print(f"Вход: {x.shape}")
    
    # Проход через все блоки
    for i, block in enumerate(blocks, 1):
        x = block(x)
        print(f"После блока {i}: {x.shape}")
    
    print(f"Выход: {x.shape}")
    print(f"Всего параметров: {sum(p.numel() for p in blocks.parameters()):,}")
    print()


def main():
    """Запуск всех примеров"""
    print("\n" + "=" * 80)
    print("ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ TRANSFORMER МОДУЛЯ")
    print("=" * 80 + "\n")
    
    example_attention()
    example_feed_forward()
    example_decoder_block()
    example_gpt_model()
    example_stack_blocks()
    example_gpt_with_tokenizer()
    
    print("=" * 80)
    print("✅ ВСЕ ПРИМЕРЫ ВЫПОЛНЕНЫ")
    print("=" * 80)


if __name__ == "__main__":
    main()

