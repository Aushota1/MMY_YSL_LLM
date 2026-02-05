"""
Автоматическая система оценки качества Embeddings
Реализует метрики из EMBEDDING_TRAINING_METHODS.md и генерирует рекомендации
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import json
import os
from datetime import datetime
import sys

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BPE_STUCTUR import BPETokenizer
from EMBEDDING_LAYER.embedding_layer import EmbeddingLayer

try:
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️  scikit-learn не установлен. Некоторые метрики будут недоступны.")

try:
    from scipy.spatial.distance import cosine, euclidean
    from scipy.stats import spearmanr, pearsonr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️  scipy не установлен. Некоторые метрики будут недоступны.")


class EmbeddingEvaluator:
    """
    Автоматическая система оценки качества embeddings
    Реализует метрики из EMBEDDING_TRAINING_METHODS.md
    """
    
    def __init__(self, model: nn.Module, tokenizer: BPETokenizer, device: str = None):
        """
        Инициализация evaluator
        
        Args:
            model: Обученная модель с embedding layer
            tokenizer: BPE токенизатор
            device: Устройство для вычислений
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Извлекаем embedding layer
        if hasattr(model, 'embedding'):
            self.embedding_layer = model.embedding
        elif hasattr(model, 'embedding_layer'):
            self.embedding_layer = model.embedding_layer
        else:
            raise ValueError("Модель должна содержать embedding layer (атрибут 'embedding' или 'embedding_layer')")
        
        # Кэш для embeddings всех токенов
        self._token_embeddings_cache = None
        self._vocab_size = tokenizer.get_vocab_size()
        
        # Результаты оценки
        self.evaluation_results = {}
        self.recommendations = []
        
    def get_all_token_embeddings(self, use_positional: bool = False) -> Tuple[np.ndarray, List[str]]:
        """
        Получение embeddings для всех токенов в словаре
        
        Args:
            use_positional: Использовать ли позиционное кодирование
        
        Returns:
            (embeddings, tokens) - массив embeddings и список токенов
        """
        if self._token_embeddings_cache is not None and not use_positional:
            return self._token_embeddings_cache
        
        # Создаем тензор со всеми токенами
        all_token_ids = torch.arange(self._vocab_size, dtype=torch.long).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if use_positional:
                embeddings = self.embedding_layer(all_token_ids)  # [1, vocab_size, dim]
            else:
                # Только token embeddings без позиционного кодирования
                embeddings = self.embedding_layer.token_embedding(all_token_ids)
            
            embeddings = embeddings[0].cpu().numpy()  # [vocab_size, dim]
        
        # Получаем текстовые представления токенов
        tokens = []
        for token_id in range(self._vocab_size):
            if token_id in self.tokenizer.vocab:
                tokens.append(self.tokenizer.vocab[token_id])
            else:
                tokens.append(f"<UNK_{token_id}>")
        
        if not use_positional:
            self._token_embeddings_cache = (embeddings, tokens)
        
        return embeddings, tokens
    
    # ==================== ВНУТРЕННИЕ МЕТРИКИ ====================
    
    def compute_cosine_similarity_stats(self, sample_size: int = 1000) -> Dict[str, float]:
        """
        Статистика косинусного сходства между случайными парами токенов
        
        Returns:
            Словарь со статистикой (mean, std, min, max, median)
        """
        embeddings, _ = self.get_all_token_embeddings()
        
        # Выбираем случайные пары
        n_samples = min(sample_size, len(embeddings) // 2)
        indices = np.random.choice(len(embeddings), size=(n_samples, 2), replace=False)
        
        similarities = []
        for i, j in indices:
            # Нормализуем векторы
            vec_i = embeddings[i] / (np.linalg.norm(embeddings[i]) + 1e-8)
            vec_j = embeddings[j] / (np.linalg.norm(embeddings[j]) + 1e-8)
            similarity = np.dot(vec_i, vec_j)
            similarities.append(similarity)
        
        similarities = np.array(similarities)
        
        return {
            'mean': float(np.mean(similarities)),
            'std': float(np.std(similarities)),
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities)),
            'median': float(np.median(similarities))
        }
    
    def compute_embedding_norms(self) -> Dict[str, float]:
        """
        Статистика норм embeddings
        
        Returns:
            Словарь со статистикой норм
        """
        embeddings, _ = self.get_all_token_embeddings()
        norms = np.linalg.norm(embeddings, axis=1)
        
        return {
            'mean': float(np.mean(norms)),
            'std': float(np.std(norms)),
            'min': float(np.min(norms)),
            'max': float(np.max(norms)),
            'median': float(np.median(norms))
        }
    
    def compute_clustering_quality(self, n_clusters: int = 10, sample_size: int = 1000) -> Dict[str, float]:
        """
        Оценка качества кластеризации embeddings
        
        Args:
            n_clusters: Количество кластеров
            sample_size: Размер выборки для анализа
        
        Returns:
            Метрики качества кластеризации
        """
        if not HAS_SKLEARN:
            return {'error': 'scikit-learn не установлен'}
        
        embeddings, _ = self.get_all_token_embeddings()
        
        # Выбираем случайную выборку
        if len(embeddings) > sample_size:
            indices = np.random.choice(len(embeddings), size=sample_size, replace=False)
            sample_embeddings = embeddings[indices]
        else:
            sample_embeddings = embeddings
        
        # Нормализация
        scaler = StandardScaler()
        sample_embeddings_scaled = scaler.fit_transform(sample_embeddings)
        
        # Кластеризация
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(sample_embeddings_scaled)
        
        # Silhouette score
        silhouette = silhouette_score(sample_embeddings_scaled, clusters)
        
        # Intra-cluster и inter-cluster расстояния
        intra_distances = []
        inter_distances = []
        
        for cluster_id in range(n_clusters):
            cluster_points = sample_embeddings_scaled[clusters == cluster_id]
            if len(cluster_points) > 1:
                # Среднее расстояние внутри кластера
                cluster_center = kmeans.cluster_centers_[cluster_id]
                intra_dist = np.mean([euclidean(p, cluster_center) for p in cluster_points])
                intra_distances.append(intra_dist)
        
        # Среднее расстояние между кластерами
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                inter_dist = euclidean(kmeans.cluster_centers_[i], kmeans.cluster_centers_[j])
                inter_distances.append(inter_dist)
        
        return {
            'silhouette_score': float(silhouette),
            'intra_cluster_distance_mean': float(np.mean(intra_distances)) if intra_distances else 0.0,
            'inter_cluster_distance_mean': float(np.mean(inter_distances)) if inter_distances else 0.0,
            'n_clusters': n_clusters,
            'n_samples': len(sample_embeddings)
        }
    
    def compute_dimensionality_analysis(self) -> Dict[str, Any]:
        """
        Анализ эффективной размерности embeddings
        
        Returns:
            Информация о размерности
        """
        if not HAS_SKLEARN:
            return {'error': 'scikit-learn не установлен'}
        
        embeddings, _ = self.get_all_token_embeddings()
        
        # PCA анализ
        pca = PCA()
        pca.fit(embeddings)
        
        # Кумулятивная дисперсия
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        # Находим количество компонент для 90%, 95%, 99% дисперсии
        n_90 = np.argmax(cumulative_variance >= 0.90) + 1
        n_95 = np.argmax(cumulative_variance >= 0.95) + 1
        n_99 = np.argmax(cumulative_variance >= 0.99) + 1
        
        return {
            'original_dim': embeddings.shape[1],
            'effective_dim_90': int(n_90),
            'effective_dim_95': int(n_95),
            'effective_dim_99': int(n_99),
            'variance_explained_90': float(cumulative_variance[n_90 - 1]),
            'variance_explained_95': float(cumulative_variance[n_95 - 1]),
            'variance_explained_99': float(cumulative_variance[n_99 - 1]),
            'top_10_components_variance': float(np.sum(pca.explained_variance_ratio_[:10]))
        }
    
    def compute_alignment_uniformity(self, sample_size: int = 1000) -> Dict[str, float]:
        """
        Вычисление метрик Alignment и Uniformity (Wang & Isola, 2020)
        
        Args:
            sample_size: Размер выборки для вычисления
        
        Returns:
            Метрики Alignment и Uniformity
        """
        embeddings, _ = self.get_all_token_embeddings()
        
        # Выбираем случайную выборку
        if len(embeddings) > sample_size:
            indices = np.random.choice(len(embeddings), size=sample_size, replace=False)
            sample_embeddings = embeddings[indices]
        else:
            sample_embeddings = embeddings
        
        # Нормализуем
        norms = np.linalg.norm(sample_embeddings, axis=1, keepdims=True)
        normalized_embeddings = sample_embeddings / (norms + 1e-8)
        
        # Uniformity: log E[e^{-2||f(x) - f(y)||²}]
        # Аппроксимируем через случайные пары
        n_pairs = min(5000, len(normalized_embeddings) * (len(normalized_embeddings) - 1) // 2)
        pairs = []
        for _ in range(n_pairs):
            i, j = np.random.choice(len(normalized_embeddings), size=2, replace=False)
            pairs.append((i, j))
        
        distances_sq = []
        for i, j in pairs:
            dist_sq = np.sum((normalized_embeddings[i] - normalized_embeddings[j]) ** 2)
            distances_sq.append(dist_sq)
        
        uniformity = np.log(np.mean(np.exp(-2 * np.array(distances_sq))))
        
        # Alignment: E[||f(x) - f(x^+)||²]
        # Для упрощения используем близкие по косинусному сходству пары как "положительные"
        similarities = np.dot(normalized_embeddings, normalized_embeddings.T)
        np.fill_diagonal(similarities, -1)  # Исключаем диагональ
        
        # Берем топ-10% наиболее похожих пар как положительные
        top_k = max(10, len(normalized_embeddings) // 10)
        alignment_distances = []
        for i in range(len(normalized_embeddings)):
            top_indices = np.argsort(similarities[i])[-top_k:]
            for j in top_indices:
                dist_sq = np.sum((normalized_embeddings[i] - normalized_embeddings[j]) ** 2)
                alignment_distances.append(dist_sq)
        
        alignment = np.mean(alignment_distances) if alignment_distances else 0.0
        
        return {
            'alignment': float(alignment),
            'uniformity': float(uniformity),
            'n_samples': len(sample_embeddings)
        }
    
    def find_nearest_neighbors(self, word: str, top_k: int = 10, 
                               exclude_special: bool = True) -> List[Tuple[str, float]]:
        """
        Поиск ближайших соседей для слова
        
        Args:
            word: Входное слово
            top_k: Количество ближайших соседей
            exclude_special: Исключать ли специальные токены
        
        Returns:
            Список (токен, косинусное_сходство)
        """
        embeddings, tokens = self.get_all_token_embeddings()
        
        # Кодируем слово
        token_ids = self.tokenizer.encode(word)
        if len(token_ids) == 0:
            return []
        
        # Получаем embedding слова (среднее по токенам)
        word_embeddings = embeddings[token_ids]
        word_embedding = np.mean(word_embeddings, axis=0)
        word_embedding = word_embedding / (np.linalg.norm(word_embedding) + 1e-8)
        
        # Нормализуем все embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / (norms + 1e-8)
        
        # Косинусное сходство
        similarities = np.dot(normalized_embeddings, word_embedding)
        
        # Получаем топ-K
        top_indices = np.argsort(similarities)[-top_k-50:][::-1]  # Берем больше для фильтрации
        
        results = []
        special_token_ids = set(self.tokenizer.special_tokens.values())
        
        for idx in top_indices:
            if exclude_special and idx in special_token_ids:
                continue
            if idx in token_ids:  # Пропускаем само слово
                continue
            results.append((tokens[idx], float(similarities[idx])))
            if len(results) >= top_k:
                break
        
        return results
    
    # ==================== ПОЛНАЯ ОЦЕНКА ====================
    
    def evaluate_all(self, sample_size: int = 1000, n_clusters: int = 10) -> Dict[str, Any]:
        """
        Полная автоматическая оценка всех метрик
        
        Args:
            sample_size: Размер выборки для некоторых метрик
            n_clusters: Количество кластеров для анализа
        
        Returns:
            Полный словарь результатов оценки
        """
        print("🔍 Начало автоматической оценки embeddings...")
        print("=" * 80)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'vocab_size': self._vocab_size,
            'embedding_dim': self.embedding_layer.embedding_dim,
        }
        
        # 1. Статистика косинусного сходства
        print("\n📊 Вычисление статистики косинусного сходства...")
        results['cosine_similarity'] = self.compute_cosine_similarity_stats(sample_size)
        
        # 2. Статистика норм
        print("📏 Вычисление статистики норм embeddings...")
        results['embedding_norms'] = self.compute_embedding_norms()
        
        # 3. Качество кластеризации
        if HAS_SKLEARN:
            print("🔗 Анализ качества кластеризации...")
            results['clustering'] = self.compute_clustering_quality(n_clusters, sample_size)
        
        # 4. Анализ размерности
        if HAS_SKLEARN:
            print("📐 Анализ эффективной размерности...")
            results['dimensionality'] = self.compute_dimensionality_analysis()
        
        # 5. Alignment и Uniformity
        print("⚖️  Вычисление Alignment и Uniformity...")
        results['alignment_uniformity'] = self.compute_alignment_uniformity(sample_size)
        
        # Сохраняем результаты
        self.evaluation_results = results
        
        # Генерируем рекомендации
        print("\n💡 Генерация рекомендаций...")
        self.recommendations = self.generate_recommendations(results)
        results['recommendations'] = self.recommendations
        
        print("\n✅ Оценка завершена!")
        print("=" * 80)
        
        return results
    
    def generate_recommendations(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Генерация автоматических рекомендаций на основе метрик
        
        Args:
            results: Результаты оценки
        
        Returns:
            Список рекомендаций с приоритетами
        """
        recommendations = []
        
        # Анализ косинусного сходства
        if 'cosine_similarity' in results:
            cos_sim = results['cosine_similarity']
            mean_sim = cos_sim.get('mean', 0)
            std_sim = cos_sim.get('std', 0)
            
            if mean_sim > 0.3:
                recommendations.append({
                    'priority': 'high',
                    'category': 'quality',
                    'issue': 'Высокое среднее косинусное сходство',
                    'description': f'Среднее косинусное сходство {mean_sim:.3f} слишком высокое. Embeddings могут быть слишком похожими.',
                    'suggestions': [
                        'Увеличьте размерность embeddings',
                        'Попробуйте увеличить learning rate',
                        'Добавьте больше dropout',
                        'Увеличьте количество эпох обучения'
                    ]
                })
            elif mean_sim < -0.1:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'quality',
                    'issue': 'Низкое среднее косинусное сходство',
                    'description': f'Среднее косинусное сходство {mean_sim:.3f} очень низкое. Embeddings могут быть слишком разрозненными.',
                    'suggestions': [
                        'Уменьшите learning rate',
                        'Попробуйте уменьшить dropout',
                        'Увеличьте batch size',
                        'Проверьте инициализацию весов'
                    ]
                })
        
        # Анализ норм
        if 'embedding_norms' in results:
            norms = results['embedding_norms']
            std_norm = norms.get('std', 0)
            mean_norm = norms.get('mean', 0)
            
            if std_norm > mean_norm * 0.5:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'stability',
                    'issue': 'Большой разброс норм embeddings',
                    'description': f'Стандартное отклонение норм {std_norm:.3f} относительно велико по сравнению со средним {mean_norm:.3f}.',
                    'suggestions': [
                        'Добавьте Layer Normalization',
                        'Используйте нормализацию embeddings после обучения',
                        'Попробуйте уменьшить learning rate',
                        'Проверьте градиентный клиппинг'
                    ]
                })
        
        # Анализ кластеризации
        if 'clustering' in results and 'silhouette_score' in results['clustering']:
            silhouette = results['clustering']['silhouette_score']
            
            if silhouette < 0.2:
                recommendations.append({
                    'priority': 'high',
                    'category': 'structure',
                    'issue': 'Низкое качество кластеризации',
                    'description': f'Silhouette score {silhouette:.3f} указывает на плохую структуру кластеров.',
                    'suggestions': [
                        'Увеличьте размерность embeddings',
                        'Увеличьте количество эпох обучения',
                        'Попробуйте другой метод обучения (например, contrastive learning)',
                        'Увеличьте размер обучающего корпуса'
                    ]
                })
            elif silhouette > 0.5:
                recommendations.append({
                    'priority': 'low',
                    'category': 'structure',
                    'issue': 'Отличное качество кластеризации',
                    'description': f'Silhouette score {silhouette:.3f} указывает на хорошую структуру кластеров.',
                    'suggestions': [
                        'Продолжайте обучение с текущими параметрами',
                        'Можно попробовать уменьшить размерность для ускорения'
                    ]
                })
        
        # Анализ размерности
        if 'dimensionality' in results:
            dim_info = results['dimensionality']
            eff_dim_95 = dim_info.get('effective_dim_95', dim_info.get('original_dim', 0))
            orig_dim = dim_info.get('original_dim', 0)
            
            if eff_dim_95 < orig_dim * 0.5:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'efficiency',
                    'issue': 'Низкая эффективная размерность',
                    'description': f'Эффективная размерность {eff_dim_95} значительно меньше исходной {orig_dim}.',
                    'suggestions': [
                        'Можно уменьшить размерность embeddings без потери качества',
                        'Это улучшит скорость обучения и использования',
                        'Попробуйте уменьшить embedding_dim до эффективной размерности'
                    ]
                })
        
        # Анализ Alignment и Uniformity
        if 'alignment_uniformity' in results:
            au = results['alignment_uniformity']
            alignment = au.get('alignment', 0)
            uniformity = au.get('uniformity', 0)
            
            if alignment > 0.5:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'quality',
                    'issue': 'Высокий Alignment (хорошо)',
                    'description': f'Alignment {alignment:.3f} указывает на хорошую близость похожих примеров.',
                    'suggestions': [
                        'Продолжайте обучение в том же направлении',
                        'Можно попробовать увеличить batch size'
                    ]
                })
            
            if uniformity < -1.0:
                recommendations.append({
                    'priority': 'high',
                    'category': 'quality',
                    'issue': 'Низкая Uniformity',
                    'description': f'Uniformity {uniformity:.3f} слишком низкая. Embeddings могут быть сконцентрированы в небольшой области.',
                    'suggestions': [
                        'Увеличьте размерность embeddings',
                        'Попробуйте добавить регуляризацию',
                        'Используйте более разнообразный обучающий корпус',
                        'Попробуйте contrastive learning'
                    ]
                })
        
        # Сортируем по приоритету
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return recommendations
    
    def print_report(self, results: Optional[Dict[str, Any]] = None):
        """
        Печать подробного отчета об оценке
        
        Args:
            results: Результаты оценки (если None, используются сохраненные)
        """
        if results is None:
            results = self.evaluation_results
        
        if not results:
            print("❌ Нет результатов для отчета. Сначала выполните evaluate_all()")
            return
        
        print("\n" + "=" * 80)
        print("📊 ОТЧЕТ О КАЧЕСТВЕ EMBEDDINGS")
        print("=" * 80)
        
        print(f"\n📅 Дата оценки: {results.get('timestamp', 'N/A')}")
        print(f"📏 Размер словаря: {results.get('vocab_size', 'N/A')}")
        print(f"📐 Размерность embeddings: {results.get('embedding_dim', 'N/A')}")
        
        # Косинусное сходство
        if 'cosine_similarity' in results:
            print("\n" + "-" * 80)
            print("📊 СТАТИСТИКА КОСИНУСНОГО СХОДСТВА")
            print("-" * 80)
            cs = results['cosine_similarity']
            print(f"  Среднее:     {cs.get('mean', 0):.4f}")
            print(f"  Стандартное отклонение: {cs.get('std', 0):.4f}")
            print(f"  Минимум:     {cs.get('min', 0):.4f}")
            print(f"  Максимум:    {cs.get('max', 0):.4f}")
            print(f"  Медиана:     {cs.get('median', 0):.4f}")
        
        # Нормы
        if 'embedding_norms' in results:
            print("\n" + "-" * 80)
            print("📏 СТАТИСТИКА НОРМ EMBEDDINGS")
            print("-" * 80)
            norms = results['embedding_norms']
            print(f"  Среднее:     {norms.get('mean', 0):.4f}")
            print(f"  Стандартное отклонение: {norms.get('std', 0):.4f}")
            print(f"  Минимум:     {norms.get('min', 0):.4f}")
            print(f"  Максимум:    {norms.get('max', 0):.4f}")
        
        # Кластеризация
        if 'clustering' in results:
            print("\n" + "-" * 80)
            print("🔗 КАЧЕСТВО КЛАСТЕРИЗАЦИИ")
            print("-" * 80)
            clust = results['clustering']
            print(f"  Silhouette Score:        {clust.get('silhouette_score', 0):.4f}")
            print(f"  Среднее внутрикластерное расстояние: {clust.get('intra_cluster_distance_mean', 0):.4f}")
            print(f"  Среднее межкластерное расстояние:    {clust.get('inter_cluster_distance_mean', 0):.4f}")
            print(f"  Количество кластеров:    {clust.get('n_clusters', 0)}")
        
        # Размерность
        if 'dimensionality' in results:
            print("\n" + "-" * 80)
            print("📐 АНАЛИЗ РАЗМЕРНОСТИ")
            print("-" * 80)
            dim = results['dimensionality']
            print(f"  Исходная размерность:    {dim.get('original_dim', 0)}")
            print(f"  Эффективная (90%):       {dim.get('effective_dim_90', 0)}")
            print(f"  Эффективная (95%):      {dim.get('effective_dim_95', 0)}")
            print(f"  Эффективная (99%):      {dim.get('effective_dim_99', 0)}")
            print(f"  Дисперсия топ-10 компонент: {dim.get('top_10_components_variance', 0):.4f}")
        
        # Alignment и Uniformity
        if 'alignment_uniformity' in results:
            print("\n" + "-" * 80)
            print("⚖️  ALIGNMENT И UNIFORMITY")
            print("-" * 80)
            au = results['alignment_uniformity']
            print(f"  Alignment:   {au.get('alignment', 0):.4f}")
            print(f"  Uniformity:  {au.get('uniformity', 0):.4f}")
        
        # Рекомендации
        if 'recommendations' in results and results['recommendations']:
            print("\n" + "=" * 80)
            print("💡 РЕКОМЕНДАЦИИ")
            print("=" * 80)
            
            for i, rec in enumerate(results['recommendations'], 1):
                priority_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(rec['priority'], '⚪')
                print(f"\n{priority_emoji} [{rec['priority'].upper()}] {rec['issue']}")
                print(f"   Категория: {rec['category']}")
                print(f"   Описание: {rec['description']}")
                print(f"   Предложения:")
                for suggestion in rec['suggestions']:
                    print(f"     • {suggestion}")
        
        print("\n" + "=" * 80)
    
    def save_report(self, filepath: str, results: Optional[Dict[str, Any]] = None):
        """
        Сохранение отчета в JSON файл
        
        Args:
            filepath: Путь к файлу для сохранения
            results: Результаты оценки (если None, используются сохраненные)
        """
        if results is None:
            results = self.evaluation_results
        
        if not results:
            print("❌ Нет результатов для сохранения")
            return
        
        # Конвертируем numpy типы в Python типы для JSON
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Отчет сохранен: {filepath}")

