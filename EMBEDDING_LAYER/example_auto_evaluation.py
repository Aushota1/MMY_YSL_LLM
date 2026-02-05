"""
Пример использования автоматической системы обучения и оценки embeddings
"""

import sys
import os

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from EMBEDDING_LAYER.auto_evaluation_system import AutoEmbeddingSystem


def example_basic_usage():
    """Базовый пример использования"""
    print("=" * 80)
    print("ПРИМЕР 1: Базовое использование")
    print("=" * 80)
    
    # Создание системы
    system = AutoEmbeddingSystem(tokenizer_path="chekpoint.pkl")
    
    # Создание модели
    system.create_model(
        embedding_dim=256,
        max_seq_len=512,
        hidden_dim=256,
        learnable_pos=False,
        layer_norm=True
    )
    
    # Тестовые данные
    texts = [
        "Машинное обучение - это подраздел искусственного интеллекта.",
        "Нейронные сети используются для распознавания образов.",
        "Глубокое обучение требует больших вычислительных ресурсов.",
        "Токенизация - важный этап обработки текста.",
        "Embeddings представляют слова в виде векторов.",
        "Трансформеры произвели революцию в NLP.",
        "BERT и GPT - популярные языковые модели.",
        "Векторные представления кодируют семантику.",
        "Обучение с учителем использует размеченные данные.",
        "Самообучение позволяет использовать неразмеченные данные."
    ] * 50  # Увеличиваем объем данных
    
    # Обучение с автоматической оценкой
    results = system.train_with_auto_evaluation(
        texts=texts,
        num_epochs=5,
        batch_size=16,
        learning_rate=0.001,
        eval_interval=2,  # Оценка каждые 2 эпохи
        save_checkpoints=True,
        checkpoint_dir="example_checkpoints"
    )
    
    print("\n✅ Обучение завершено!")
    print(f"Лучшая эпоха: {results['best_epoch']}")
    print(f"Лучший loss: {results['best_loss']:.4f}")


def example_load_and_evaluate():
    """Пример загрузки модели и оценки"""
    print("\n" + "=" * 80)
    print("ПРИМЕР 2: Загрузка модели и оценка")
    print("=" * 80)
    
    # Создание системы
    system = AutoEmbeddingSystem(tokenizer_path="chekpoint.pkl")
    
    # Загрузка модели (если существует)
    model_path = "embedding_model.pth"
    if os.path.exists(model_path):
        system.load_model(model_path)
        
        # Оценка
        print("\n🔍 Выполнение полной оценки...")
        results = system.evaluator.evaluate_all(sample_size=1000, n_clusters=10)
        
        # Печать отчета
        system.evaluator.print_report(results)
        
        # Сохранение отчета
        system.evaluator.save_report("evaluation_report.json", results)
    else:
        print(f"⚠️  Модель {model_path} не найдена. Сначала обучите модель.")


def example_custom_evaluation():
    """Пример использования отдельных метрик"""
    print("\n" + "=" * 80)
    print("ПРИМЕР 3: Использование отдельных метрик")
    print("=" * 80)
    
    # Создание системы и модели
    system = AutoEmbeddingSystem(tokenizer_path="chekpoint.pkl")
    system.create_model(embedding_dim=256, max_seq_len=512, hidden_dim=256)
    
    # Загрузка данных и обучение (упрощенное)
    texts = ["Пример текста"] * 10
    # ... обучение ...
    
    # Использование отдельных метрик
    evaluator = system.evaluator
    
    print("\n📊 Статистика косинусного сходства:")
    cos_stats = evaluator.compute_cosine_similarity_stats(sample_size=500)
    print(f"  Среднее: {cos_stats['mean']:.4f}")
    print(f"  Стандартное отклонение: {cos_stats['std']:.4f}")
    
    print("\n📏 Статистика норм:")
    norm_stats = evaluator.compute_embedding_norms()
    print(f"  Среднее: {norm_stats['mean']:.4f}")
    print(f"  Стандартное отклонение: {norm_stats['std']:.4f}")
    
    print("\n🔗 Качество кластеризации:")
    if hasattr(evaluator, 'compute_clustering_quality'):
        clust_stats = evaluator.compute_clustering_quality(n_clusters=5, sample_size=500)
        print(f"  Silhouette Score: {clust_stats.get('silhouette_score', 0):.4f}")
    
    print("\n🔍 Поиск ближайших соседей для слова 'машина':")
    neighbors = evaluator.find_nearest_neighbors("машина", top_k=5)
    for i, (word, similarity) in enumerate(neighbors, 1):
        print(f"  {i}. {word}: {similarity:.4f}")


if __name__ == "__main__":
    print("🚀 Примеры использования автоматической системы оценки embeddings\n")
    
    # Выберите пример для запуска
    example_choice = input("Выберите пример (1-базовый, 2-загрузка и оценка, 3-отдельные метрики, all-все): ").strip().lower()
    
    if example_choice == "1" or example_choice == "all":
        try:
            example_basic_usage()
        except Exception as e:
            print(f"❌ Ошибка в примере 1: {e}")
            import traceback
            traceback.print_exc()
    
    if example_choice == "2" or example_choice == "all":
        try:
            example_load_and_evaluate()
        except Exception as e:
            print(f"❌ Ошибка в примере 2: {e}")
            import traceback
            traceback.print_exc()
    
    if example_choice == "3" or example_choice == "all":
        try:
            example_custom_evaluation()
        except Exception as e:
            print(f"❌ Ошибка в примере 3: {e}")
            import traceback
            traceback.print_exc()

