"""
Тесты для TRM модулей
"""

import torch
import torch.nn as nn
import sys
import os

# Добавляем пути для импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TRM import (
    TinyRecursiveNetwork,
    OutputRefinement,
    OutputHead,
    QHead,
    StableMaxLoss,
    binary_cross_entropy_with_logits,
    TRMModel
)
from TRM.utils import RMSNorm, SwiGLU


def test_rmsnorm():
    """Тест RMSNorm"""
    print("=" * 70)
    print("ТЕСТ 1: RMSNorm")
    print("=" * 70)
    
    norm = RMSNorm(dim=512)
    x = torch.randn(2, 10, 512)
    output = norm(x)
    
    assert output.shape == x.shape, f"Форма не совпадает: {output.shape} != {x.shape}"
    print(f"✓ Форма корректна: {output.shape}")
    print(f"✓ RMSNorm работает")
    print()


def test_swiglu():
    """Тест SwiGLU"""
    print("=" * 70)
    print("ТЕСТ 2: SwiGLU")
    print("=" * 70)
    
    swiglu = SwiGLU(dim=512, hidden_dim=512)
    x = torch.randn(2, 10, 512)
    output = swiglu(x)
    
    assert output.shape == (2, 10, 512), f"Форма не совпадает: {output.shape}"
    print(f"✓ Форма корректна: {output.shape}")
    print(f"✓ SwiGLU работает")
    print()


def test_tiny_recursive_network():
    """Тест TinyRecursiveNetwork"""
    print("=" * 70)
    print("ТЕСТ 3: TinyRecursiveNetwork")
    print("=" * 70)
    
    net = TinyRecursiveNetwork(embedding_dim=512, hidden_dim=512)
    x = torch.randn(2, 10, 512)
    y = torch.randn(2, 10, 512)
    z = torch.randn(2, 10, 512)
    
    output = net(x, y, z)
    
    assert output.shape == z.shape, f"Форма не совпадает: {output.shape} != {z.shape}"
    print(f"✓ Форма корректна: {output.shape}")
    print(f"✓ TinyRecursiveNetwork работает")
    print(f"✓ Параметров: {sum(p.numel() for p in net.parameters()):,}")
    print()


def test_output_refinement():
    """Тест OutputRefinement"""
    print("=" * 70)
    print("ТЕСТ 4: OutputRefinement")
    print("=" * 70)
    
    refine = OutputRefinement(embedding_dim=512, hidden_dim=512)
    y = torch.randn(2, 10, 512)
    z = torch.randn(2, 10, 512)
    
    output = refine(y, z)
    
    assert output.shape == y.shape, f"Форма не совпадает: {output.shape} != {y.shape}"
    print(f"✓ Форма корректна: {output.shape}")
    print(f"✓ OutputRefinement работает")
    print()


def test_heads():
    """Тест OutputHead и QHead"""
    print("=" * 70)
    print("ТЕСТ 5: OutputHead и QHead")
    print("=" * 70)
    
    vocab_size = 1000
    embedding_dim = 512
    
    # OutputHead
    output_head = OutputHead(embedding_dim=embedding_dim, vocab_size=vocab_size)
    y = torch.randn(2, 10, embedding_dim)
    y_hat = output_head(y)
    
    assert y_hat.shape == (2, 10, vocab_size), f"Форма не совпадает: {y_hat.shape}"
    print(f"✓ OutputHead форма корректна: {y_hat.shape}")
    
    # QHead
    q_head = QHead(embedding_dim=embedding_dim)
    q_hat = q_head(y)
    
    assert q_hat.shape == (2, 1), f"Форма не совпадает: {q_hat.shape}"
    print(f"✓ QHead форма корректна: {q_hat.shape}")
    print()


def test_losses():
    """Тест функций потерь"""
    print("=" * 70)
    print("ТЕСТ 6: Losses")
    print("=" * 70)
    
    vocab_size = 1000
    batch_size = 2
    seq_len = 10
    
    # StableMaxLoss
    stable_loss = StableMaxLoss()
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    loss = stable_loss(logits, targets)
    assert loss.item() > 0, "Loss должен быть положительным"
    print(f"✓ StableMaxLoss работает: {loss.item():.4f}")
    
    # Binary CE
    q_hat = torch.randn(batch_size, 1)
    is_correct = torch.randint(0, 2, (batch_size,)).float()
    
    bce_loss = binary_cross_entropy_with_logits(q_hat, is_correct)
    assert bce_loss.item() > 0, "Loss должен быть положительным"
    print(f"✓ BinaryCrossEntropy работает: {bce_loss.item():.4f}")
    print()


def test_latent_recursion():
    """Тест Latent Recursion"""
    print("=" * 70)
    print("ТЕСТ 7: Latent Recursion")
    print("=" * 70)
    
    from TRM.latent_recursion import latent_recursion
    
    net = TinyRecursiveNetwork(embedding_dim=512, hidden_dim=512)
    x = torch.randn(2, 10, 512)
    y = torch.randn(2, 10, 512)
    z = torch.randn(2, 10, 512)
    
    z_updated = latent_recursion(x, y, z, net, n=6)
    
    assert z_updated.shape == z.shape, f"Форма не совпадает: {z_updated.shape} != {z.shape}"
    print(f"✓ Форма корректна: {z_updated.shape}")
    print(f"✓ Latent recursion работает")
    print()


def test_trm_model():
    """Тест полной TRMModel"""
    print("=" * 70)
    print("ТЕСТ 8: TRMModel")
    print("=" * 70)
    
    vocab_size = 1000
    embedding_dim = 512
    batch_size = 2
    seq_len = 10
    
    # Создание модели
    model = TRMModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=512,
        max_seq_len=512
    )
    
    print(f"✓ Модель создана")
    print(f"✓ Параметров: {model.get_num_params():,} ({model.get_num_params_millions():.2f}M)")
    
    # Forward pass
    x_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    y_hat, q_hat = model(x_input, n=3, T=2, N_sup=5)  # Уменьшенные параметры для теста
    
    assert y_hat.shape == (batch_size, seq_len, vocab_size), f"Форма y_hat не совпадает: {y_hat.shape}"
    assert q_hat.shape == (batch_size, 1), f"Форма q_hat не совпадает: {q_hat.shape}"
    
    print(f"✓ Forward pass работает")
    print(f"✓ y_hat форма: {y_hat.shape}")
    print(f"✓ q_hat форма: {q_hat.shape}")
    
    # Generate answer
    tokens = model.generate_answer(x_input, max_steps=5)
    assert tokens.shape == (batch_size, seq_len), f"Форма tokens не совпадает: {tokens.shape}"
    print(f"✓ Generate answer работает: {tokens.shape}")
    print()


def run_all_tests():
    """Запуск всех тестов"""
    print("\n" + "=" * 70)
    print("  ТЕСТИРОВАНИЕ TRM МОДУЛЕЙ")
    print("=" * 70 + "\n")
    
    try:
        test_rmsnorm()
        test_swiglu()
        test_tiny_recursive_network()
        test_output_refinement()
        test_heads()
        test_losses()
        test_latent_recursion()
        test_trm_model()
        
        print("=" * 70)
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("=" * 70)
        
    except Exception as e:
        print("=" * 70)
        print(f"❌ ОШИБКА В ТЕСТАХ: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()

