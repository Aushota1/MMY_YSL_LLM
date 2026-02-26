"""
Скрипт для визуализации эмбеддингов из .pth файла в 3D пространстве
Показывает векторы и кластеры с помощью PCA/t-SNE и k-means

Использование:
    python visualize_embeddings_3d.py --model path/to/model.pth
    
    python visualize_embeddings_3d.py --model model.pth --n-vectors 1000 --n-clusters 15 --method tsne
    
    python visualize_embeddings_3d.py --model model.pth --output visualization.png --show-labels

Параметры:
    --model: Путь к .pth файлу (обязательно)
    --tokenizer: Путь к токенизатору (по умолчанию: chekpoint.pkl)
    --n-vectors: Количество векторов для визуализации (по умолчанию: все)
    --n-clusters: Количество кластеров для k-means (по умолчанию: 10)
    --method: Метод уменьшения размерности: pca или tsne (по умолчанию: pca)
    --output: Путь для сохранения изображения (опционально)
    --show-labels: Показывать подписи токенов на графике
    --max-labels: Максимальное количество подписей (по умолчанию: 50)

Требования:
    - scikit-learn: pip install scikit-learn
    - matplotlib: pip install matplotlib
    - numpy: pip install numpy
    - torch: pip install torch
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
import sys
from typing import Tuple, Optional, List

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from BPE_STUCTUR import BPETokenizer

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("❌ scikit-learn не установлен. Установите: pip install scikit-learn")


def load_model_and_extract_embeddings(
    model_path: str,
    tokenizer_path: Optional[str] = None,
    device: str = 'cpu',
    max_vectors: Optional[int] = None,
    ask_tokenizer: bool = True
) -> Tuple[np.ndarray, List[str], dict]:
    """
    Загрузка модели из .pth файла и извлечение token embeddings
    
    Args:
        model_path: Путь к .pth файлу
        tokenizer_path: Путь к токенизатору (по умолчанию chekpoint.pkl)
        device: Устройство для загрузки
        max_vectors: Максимальное количество векторов для извлечения
        ask_tokenizer: Спрашивать ли путь к токенизатору интерактивно, если не найден
    
    Returns:
        (embeddings, tokens, model_info) - массив эмбеддингов, список токенов, информация о модели
    """
    print(f"\n📦 Загрузка модели из {model_path}...")
    
    # Загрузка checkpoint для определения vocab_size
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Определяем vocab_size из embedding весов
    embedding_weight_key = "embedding.token_embedding.embedding.weight"
    if embedding_weight_key in state_dict:
        vocab_size = state_dict[embedding_weight_key].shape[0]
        embedding_dim = state_dict[embedding_weight_key].shape[1]
    else:
        # Альтернативные ключи
        for key in state_dict.keys():
            if "token_embedding" in key and "weight" in key:
                vocab_size = state_dict[key].shape[0]
                embedding_dim = state_dict[key].shape[1]
                embedding_weight_key = key
                break
        else:
            # Fallback на значения из checkpoint
            embedding_dim = checkpoint.get('embedding_dim', 256)
            vocab_size = checkpoint.get('vocab_size', 30000)
    
    # Загрузка токенизатора
    if tokenizer_path is None:
        tokenizer_path = "chekpoint.pkl"
    
    tokenizer = None
    if not os.path.exists(tokenizer_path):
        if ask_tokenizer:
            print(f"\n⚠️  Токенизатор {tokenizer_path} не найден.")
            print("Введите путь к файлу токенизатора (.pkl):")
            print("(Нажмите Enter, чтобы продолжить без токенизатора)")
            user_input = input("> ").strip()
            
            if user_input:
                tokenizer_path = user_input
                if os.path.exists(tokenizer_path):
                    print(f"📖 Загрузка токенизатора из {tokenizer_path}...")
                    tokenizer = BPETokenizer()
                    try:
                        tokenizer.load(tokenizer_path)
                        print(f"✓ Токенизатор загружен. Размер словаря: {tokenizer.get_vocab_size()}")
                    except Exception as e:
                        print(f"⚠️  Ошибка при загрузке токенизатора: {e}")
                        tokenizer = None
                else:
                    print(f"⚠️  Файл {tokenizer_path} не найден. Продолжаю без токенизатора...")
                    tokenizer = None
            else:
                print("⚠️  Продолжаю без токенизатора...")
                tokenizer = None
        else:
            print(f"⚠️  Токенизатор {tokenizer_path} не найден. Продолжаю без токенизатора...")
            tokenizer = None
    else:
        print(f"📖 Загрузка токенизатора из {tokenizer_path}...")
        tokenizer = BPETokenizer()
        try:
            tokenizer.load(tokenizer_path)
            print(f"✓ Токенизатор загружен. Размер словаря: {tokenizer.get_vocab_size()}")
        except Exception as e:
            print(f"⚠️  Ошибка при загрузке токенизатора: {e}")
            tokenizer = None
    
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
    
    model_info = {
        'embedding_dim': embedding_dim,
        'vocab_size': vocab_size,
        'max_seq_len': max_seq_len,
        'learnable_pos': learnable_pos,
        'layer_norm': layer_norm
    }
    
    print(f"\n✓ Параметры модели:")
    print(f"   Embedding dim: {embedding_dim}")
    print(f"   Vocab size: {vocab_size}")
    print(f"   Max seq len: {max_seq_len}")
    print(f"   Learnable pos: {learnable_pos}")
    print(f"   Layer norm: {layer_norm}")
    
    # Извлечение token embeddings напрямую из state_dict
    print(f"\n🔍 Извлечение token embeddings...")
    
    if embedding_weight_key in state_dict:
        # Извлекаем веса напрямую из state_dict
        embeddings = state_dict[embedding_weight_key].cpu().numpy()  # [vocab_size, embedding_dim]
        
        # Применяем масштабирование, если оно было в модели
        # В TokenEmbedding используется: embeddings * sqrt(embedding_dim)
        # Но веса уже сохранены без масштабирования, так что просто используем как есть
        embeddings = embeddings * np.sqrt(embedding_dim)
    else:
        raise ValueError(f"Не удалось найти веса token embeddings в state_dict. Доступные ключи: {list(state_dict.keys())[:10]}...")
    
    # Получаем текстовые представления токенов
    tokens = []
    for token_id in range(vocab_size):
        if tokenizer and token_id in tokenizer.vocab:
            tokens.append(tokenizer.vocab[token_id])
        else:
            tokens.append(f"<TOKEN_{token_id}>")
    
    # Ограничение количества векторов
    if max_vectors and max_vectors < len(embeddings):
        print(f"📊 Ограничение до {max_vectors} векторов...")
        indices = np.random.choice(len(embeddings), size=max_vectors, replace=False)
        embeddings = embeddings[indices]
        tokens = [tokens[i] for i in indices]
    
    print(f"✓ Извлечено {len(embeddings)} векторов размерности {embeddings.shape[1]}")
    
    return embeddings, tokens, model_info


def reduce_to_3d(
    embeddings: np.ndarray,
    method: str = 'pca',
    random_state: int = 42
) -> np.ndarray:
    """
    Уменьшение размерности эмбеддингов до 3D
    
    Args:
        embeddings: Массив эмбеддингов [n_vectors, embedding_dim]
        method: Метод уменьшения ('pca' или 'tsne')
        random_state: Seed для воспроизводимости
    
    Returns:
        Массив размерности [n_vectors, 3]
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn необходим для уменьшения размерности")
    
    print(f"\n🔄 Уменьшение размерности методом {method.upper()}...")
    
    # Нормализация
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    if method.lower() == 'pca':
        reducer = PCA(n_components=3, random_state=random_state)
        embeddings_3d = reducer.fit_transform(embeddings_scaled)
        explained_variance = reducer.explained_variance_ratio_
        print(f"✓ PCA завершен. Объясненная дисперсия: {explained_variance.sum():.2%}")
        print(f"   Компоненты: {explained_variance[0]:.2%}, {explained_variance[1]:.2%}, {explained_variance[2]:.2%}")
    elif method.lower() == 'tsne':
        print("⏳ t-SNE может занять некоторое время...")
        reducer = TSNE(n_components=3, random_state=random_state, perplexity=30, n_iter=1000)
        embeddings_3d = reducer.fit_transform(embeddings_scaled)
        print(f"✓ t-SNE завершен")
    else:
        raise ValueError(f"Неизвестный метод: {method}. Используйте 'pca' или 'tsne'")
    
    return embeddings_3d


def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int = 10,
    random_state: int = 42
) -> np.ndarray:
    """
    Кластеризация эмбеддингов с помощью k-means
    
    Args:
        embeddings: Массив эмбеддингов [n_vectors, dim]
        n_clusters: Количество кластеров
        random_state: Seed для воспроизводимости
    
    Returns:
        Массив меток кластеров [n_vectors]
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn необходим для кластеризации")
    
    print(f"\n🎯 Кластеризация с {n_clusters} кластерами...")
    
    # Нормализация
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # K-means кластеризация
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_scaled)
    
    print(f"✓ Кластеризация завершена")
    print(f"   Распределение по кластерам:")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"   Кластер {cluster_id}: {count} векторов")
    
    return cluster_labels


def visualize_3d(
    embeddings_3d: np.ndarray,
    cluster_labels: np.ndarray,
    tokens: List[str],
    model_info: dict,
    output_path: Optional[str] = None,
    show_labels: bool = False,
    max_labels: int = 50
):
    """
    Визуализация эмбеддингов в 3D пространстве
    
    Args:
        embeddings_3d: Массив эмбеддингов в 3D [n_vectors, 3]
        cluster_labels: Метки кластеров [n_vectors]
        tokens: Список токенов
        model_info: Информация о модели
        output_path: Путь для сохранения изображения
        show_labels: Показывать ли подписи токенов
        max_labels: Максимальное количество подписей для отображения
    """
    print(f"\n🎨 Создание 3D визуализации...")
    
    # Создание фигуры
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Цветовая схема для кластеров
    n_clusters = len(np.unique(cluster_labels))
    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    
    # Визуализация каждого кластера
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        cluster_points = embeddings_3d[mask]
        cluster_tokens = [tokens[i] for i in range(len(tokens)) if mask[i]]
        
        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            cluster_points[:, 2],
            c=[colors[cluster_id]],
            label=f'Кластер {cluster_id} ({np.sum(mask)} векторов)',
            alpha=0.6,
            s=30,
            edgecolors='black',
            linewidths=0.5
        )
        
        # Добавление подписей для некоторых токенов
        if show_labels and len(cluster_points) > 0:
            # Выбираем случайные токены из кластера
            n_labels = min(max_labels // n_clusters, len(cluster_points))
            if n_labels > 0:
                label_indices = np.random.choice(len(cluster_points), size=n_labels, replace=False)
                for idx in label_indices:
                    ax.text(
                        cluster_points[idx, 0],
                        cluster_points[idx, 1],
                        cluster_points[idx, 2],
                        cluster_tokens[idx],
                        fontsize=6,
                        alpha=0.7
                    )
    
    # Настройка осей
    ax.set_xlabel('X (Компонента 1)', fontsize=12)
    ax.set_ylabel('Y (Компонента 2)', fontsize=12)
    ax.set_zlabel('Z (Компонента 3)', fontsize=12)
    
    # Заголовок
    title = f"3D Визуализация Эмбеддингов\n"
    title += f"Модель: {model_info['embedding_dim']}D, {model_info['vocab_size']} токенов, "
    title += f"{n_clusters} кластеров"
    ax.set_title(title, fontsize=14, pad=20)
    
    # Легенда
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Настройка вида
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    # Сохранение или отображение
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Изображение сохранено в {output_path}")
    else:
        print(f"✓ Визуализация готова. Открываю окно...")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Визуализация эмбеддингов из .pth файла в 3D пространстве'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Путь к .pth файлу с моделью'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='chekpoint.pkl',
        help='Путь к токенизатору (по умолчанию: chekpoint.pkl)'
    )
    parser.add_argument(
        '--n-vectors',
        type=int,
        default=None,
        help='Количество векторов для визуализации (по умолчанию: все)'
    )
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=10,
        help='Количество кластеров для k-means (по умолчанию: 10)'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='pca',
        choices=['pca', 'tsne'],
        help='Метод уменьшения размерности (по умолчанию: pca)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Путь для сохранения изображения (опционально)'
    )
    parser.add_argument(
        '--show-labels',
        action='store_true',
        help='Показывать подписи токенов на графике'
    )
    parser.add_argument(
        '--max-labels',
        type=int,
        default=50,
        help='Максимальное количество подписей для отображения (по умолчанию: 50)'
    )
    
    args = parser.parse_args()
    
    # Проверка наличия scikit-learn
    if not HAS_SKLEARN:
        print("❌ Ошибка: scikit-learn не установлен")
        print("   Установите: pip install scikit-learn")
        return
    
    # Проверка существования файла модели
    if not os.path.exists(args.model):
        print(f"❌ Ошибка: Файл {args.model} не найден")
        return
    
    try:
        # Загрузка модели и извлечение эмбеддингов
        # Определяем путь к токенизатору
        tokenizer_path = args.tokenizer
        if tokenizer_path and not os.path.exists(tokenizer_path):
            print(f"⚠️  Указанный токенизатор {tokenizer_path} не найден.")
            tokenizer_path = None
        
        embeddings, tokens, model_info = load_model_and_extract_embeddings(
            model_path=args.model,
            tokenizer_path=tokenizer_path,
            max_vectors=args.n_vectors,
            ask_tokenizer=True
        )
        
        # Уменьшение размерности до 3D
        embeddings_3d = reduce_to_3d(embeddings, method=args.method)
        
        # Кластеризация
        cluster_labels = cluster_embeddings(embeddings, n_clusters=args.n_clusters)
        
        # Визуализация
        visualize_3d(
            embeddings_3d=embeddings_3d,
            cluster_labels=cluster_labels,
            tokens=tokens,
            model_info=model_info,
            output_path=args.output,
            show_labels=args.show_labels,
            max_labels=args.max_labels
        )
        
        print("\n✅ Визуализация завершена!")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

