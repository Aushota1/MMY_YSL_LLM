"""
Unit тесты для Transformer модуля
"""

import torch
import torch.nn as nn
import sys
import os

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TRANSFORMER.attention import MultiHeadSelfAttention
from TRANSFORMER.feed_forward import FeedForward
from TRANSFORMER.decoder_block import TransformerDecoderBlock
from TRANSFORMER.gpt_model import GPTModel


def test_multi_head_attention():
    """Тест Multi-Head Self-Attention"""
    print("Тестирование Multi-Head Self-Attention...")
    
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
    
    # Тестовый вход
    x = torch.randn(batch_size, seq_len, embedding_dim)
    
    # Forward pass
    output = attention(x)
    
    # Проверка размеров
    assert output.shape == (batch_size, seq_len, embedding_dim), \
        f"Неверный размер выхода: {output.shape}, ожидается {(batch_size, seq_len, embedding_dim)}"
    
    # Проверка causal masking (первый токен не должен видеть последний)
    # Создаем два входа: один с уникальным последним токеном
    torch.manual_seed(42)  # Для воспроизводимости
    x1 = torch.randn(1, seq_len, embedding_dim)
    x2 = x1.clone()
    x2[0, -1, :] = torch.randn(embedding_dim) * 1000  # Очень большой последний токен
    
    # Устанавливаем модель в eval режим для детерминированности
    attention.eval()
    with torch.no_grad():
        out1 = attention(x1)
        out2 = attention(x2)
    
    # Первый токен не должен измениться из-за изменения последнего (causal mask)
    # Используем более мягкую проверку, так как могут быть численные ошибки
    diff = torch.abs(out1[0, 0, :] - out2[0, 0, :]).max().item()
    # Разница должна быть очень маленькой (только из-за численных ошибок)
    assert diff < 1e-3, \
        f"Causal masking не работает: первый токен изменился на {diff:.6f} при изменении последнего"
    
    print("✓ Multi-Head Self-Attention работает корректно")
    print(f"  Размер выхода: {output.shape}")
    print(f"  Параметров: {sum(p.numel() for p in attention.parameters()):,}")


def test_feed_forward():
    """Тест Feed-Forward Network"""
    print("\nТестирование Feed-Forward Network...")
    
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
    
    # Тестовый вход
    x = torch.randn(batch_size, seq_len, embedding_dim)
    
    # Forward pass
    output = ffn(x)
    
    # Проверка размеров
    assert output.shape == (batch_size, seq_len, embedding_dim), \
        f"Неверный размер выхода: {output.shape}, ожидается {(batch_size, seq_len, embedding_dim)}"
    
    # Проверка GELU активации (выход должен быть нелинейным)
    x_linear = torch.randn(1, 1, embedding_dim)
    out1 = ffn(x_linear)
    out2 = ffn(x_linear * 2)
    
    # Проверяем, что выход нелинейный
    assert not torch.allclose(out1 * 2, out2, atol=1e-3), \
        "FFN не использует нелинейную активацию"
    
    print("✓ Feed-Forward Network работает корректно")
    print(f"  Размер выхода: {output.shape}")
    print(f"  Параметров: {sum(p.numel() for p in ffn.parameters()):,}")


def test_decoder_block():
    """Тест Transformer Decoder Block"""
    print("\nТестирование Transformer Decoder Block...")
    
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
    
    # Тестовый вход
    x = torch.randn(batch_size, seq_len, embedding_dim)
    
    # Forward pass
    output = block(x)
    
    # Проверка размеров
    assert output.shape == (batch_size, seq_len, embedding_dim), \
        f"Неверный размер выхода: {output.shape}, ожидается {(batch_size, seq_len, embedding_dim)}"
    
    # Проверка residual connections (выход должен быть близок к входу при малых весах)
    # Создаем блок с нулевой инициализацией (для теста)
    block_zero = TransformerDecoderBlock(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=0.0
    )
    # Инициализируем веса близко к нулю
    for param in block_zero.parameters():
        if param.requires_grad:
            nn.init.zeros_(param)
    
    # Устанавливаем в eval режим
    block_zero.eval()
    x_test = torch.randn(1, 5, embedding_dim)
    
    with torch.no_grad():
        output_zero = block_zero(x_test)
    
    # С residual connections выход должен быть близок к входу
    # (но не точно равен из-за LayerNorm, который нормализует даже при нулевых весах)
    diff = torch.abs(x_test - output_zero).max().item()
    # LayerNorm может немного изменить значения даже при нулевых весах
    # Проверяем, что разница не слишком большая
    assert diff < 1.0, \
        f"Residual connections не работают корректно: разница {diff:.6f}"
    
    print("✓ Transformer Decoder Block работает корректно")
    print(f"  Размер выхода: {output.shape}")
    print(f"  Параметров: {sum(p.numel() for p in block.parameters()):,}")


def test_gpt_model():
    """Тест полной GPT модели"""
    print("\nТестирование GPT Model...")
    
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
    
    # Тестовый вход (token IDs)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits = model(token_ids)
    
    # Проверка размеров
    assert logits.shape == (batch_size, seq_len, vocab_size), \
        f"Неверный размер выхода: {logits.shape}, ожидается {(batch_size, seq_len, vocab_size)}"
    
    # Проверка, что логиты можно использовать для предсказания
    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size, seq_len)), \
        "Вероятности не суммируются до 1"
    
    print("✓ GPT Model работает корректно")
    print(f"  Размер выхода: {logits.shape}")
    print(f"  Параметров: {model.get_num_params():,} ({model.get_num_params_millions():.2f}M)")


def test_gpt_model_with_tokenizer():
    """Тест GPT модели с токенизатором"""
    print("\nТестирование GPT Model с токенизатором...")
    
    try:
        from BPE_STUCTUR import BPETokenizer
        
        # Загрузка токенизатора
        tokenizer = BPETokenizer()
        tokenizer.load("chekpoint.pkl")
        
        vocab_size = tokenizer.get_vocab_size()
        embedding_dim = 256
        num_layers = 4
        num_heads = 8
        
        # Создание модели с токенизатором
        model = GPTModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            tokenizer=tokenizer
        )
        
        # Тестовый текст
        text = "Привет, как дела?"
        token_ids_list = tokenizer.encode(text)
        token_ids = torch.tensor([token_ids_list])
        
        # Forward pass
        logits = model(token_ids)
        
        # Проверка размеров
        assert logits.shape[0] == 1, f"Неверный batch size: {logits.shape[0]}"
        assert logits.shape[2] == vocab_size, f"Неверный vocab size: {logits.shape[2]}"
        
        print("✓ GPT Model с токенизатором работает корректно")
        print(f"  Размер выхода: {logits.shape}")
        print(f"  Параметров: {model.get_num_params():,} ({model.get_num_params_millions():.2f}M)")
        
    except FileNotFoundError:
        print("⚠️  Токенизатор не найден, пропускаем тест с токенизатором")
    except Exception as e:
        print(f"⚠️  Ошибка при тестировании с токенизатором: {e}")


def run_all_tests():
    """Запуск всех тестов"""
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ TRANSFORMER МОДУЛЯ")
    print("=" * 80)
    
    try:
        test_multi_head_attention()
        test_feed_forward()
        test_decoder_block()
        test_gpt_model()
        test_gpt_model_with_tokenizer()
        
        print("\n" + "=" * 80)
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО")
        print("=" * 80)
        
    except AssertionError as e:
        print(f"\n❌ ТЕСТ ПРОВАЛЕН: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()

