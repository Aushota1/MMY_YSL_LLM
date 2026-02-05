"""
Скрипт для проверки работы embeddings
Находит ближайшие по контексту слова для заданного слова или предложения
"""

import torch
import torch.nn as nn
import sys
import os
from typing import List, Tuple
import numpy as np

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from BPE_STUCTUR import BPETokenizer
from EMBEDDING_LAYER.embedding_layer import EmbeddingLayer, create_embedding_from_tokenizer


class SimpleLanguageModel(nn.Module):
    """Простая модель для обучения эмбеддингов (копия из embedding_trainer)"""
    
    def __init__(self, embedding_layer: EmbeddingLayer, vocab_size: int, hidden_dim: int = 256):
        super().__init__()
        self.embedding = embedding_layer
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_layer.embedding_dim
        
        # Простой feed-forward слой
        self.ff = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, vocab_size)
        )
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(token_ids)
        logits = self.ff(embeddings)
        return logits


class EmbeddingSimilarity:
    """Класс для поиска ближайших слов по embeddings"""
    
    def __init__(self, model_path: str, tokenizer_path: str = "chekpoint.pkl", device: str = None):
        """
        Инициализация
        
        Args:
            model_path: Путь к файлу .pth с обученной моделью
            tokenizer_path: Путь к файлу .pkl с токенизатором
            device: Устройство для вычислений ('cuda' или 'cpu')
        """
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Загрузка токенизатора
        print(f"📖 Загрузка токенизатора из {tokenizer_path}...")
        self.tokenizer = BPETokenizer()
        self.tokenizer.load(tokenizer_path)
        print(f"✓ Токенизатор загружен. Размер словаря: {self.tokenizer.get_vocab_size()}")
        
        # Загрузка модели
        print(f"📦 Загрузка модели из {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Получение параметров из checkpoint
        embedding_dim = checkpoint.get('embedding_dim', 256)
        vocab_size = checkpoint.get('vocab_size', self.tokenizer.get_vocab_size())
        hidden_dim = checkpoint.get('hidden_dim', 256)
        
        # Определяем параметры из state_dict
        state_dict = checkpoint['model_state_dict']
        
        # Определяем hidden_dim из структуры checkpoint, если возможно
        if 'ff.0.weight' in state_dict:
            # hidden_dim можно определить из размера выходного слоя первого Linear
            # ff.0.weight имеет размер [hidden_dim, embedding_dim]
            hidden_dim = state_dict['ff.0.weight'].shape[0]
        
        # Определяем max_seq_len из структуры state_dict
        max_seq_len = checkpoint.get('max_seq_len', 512)
        learnable_pos_key = "embedding.positional_encoding.pos_encoding.position_embedding.weight"
        sinusoidal_key = "embedding.positional_encoding.pos_encoding.pe"
        
        # Определяем learnable_pos и max_seq_len из структуры state_dict
        if learnable_pos_key in state_dict:
            learnable_pos = True
            # Для обучаемого кодирования: position_embedding.weight имеет размер [max_seq_len, embedding_dim]
            max_seq_len = state_dict[learnable_pos_key].shape[0]
            print(f"✓ Определено: обучаемое позиционное кодирование, max_seq_len={max_seq_len}")
        elif sinusoidal_key in state_dict:
            learnable_pos = False
            # Для синусоидального: pe buffer имеет размер [1, max_seq_len, embedding_dim]
            pe_buffer = state_dict[sinusoidal_key]
            if len(pe_buffer.shape) >= 2:
                max_seq_len = pe_buffer.shape[1]
            print(f"✓ Определено: синусоидальное позиционное кодирование, max_seq_len={max_seq_len}")
        else:
            # Пробуем из checkpoint метаданных
            learnable_pos = checkpoint.get('learnable_pos', False)
            max_seq_len = checkpoint.get('max_seq_len', 512)
            print(f"⚠ Используется значение по умолчанию: learnable_pos={learnable_pos}, max_seq_len={max_seq_len}")
        
        # Определяем layer_norm
        layer_norm_key = "embedding.layer_norm.weight"
        if layer_norm_key in state_dict:
            layer_norm = True
        else:
            layer_norm = checkpoint.get('layer_norm', True)
        
        # Создание embedding layer с правильными параметрами
        self.embedding_layer = create_embedding_from_tokenizer(
            self.tokenizer,
            embedding_dim=embedding_dim,
            max_seq_len=max_seq_len,  # Используем определенный max_seq_len
            learnable_pos=learnable_pos,
            layer_norm=layer_norm
        )
        
        # Создание полной модели для загрузки весов
        self.model = SimpleLanguageModel(
            embedding_layer=self.embedding_layer,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim
        )
        
        # Загрузка весов
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print(f"✓ Веса модели загружены успешно")
        except RuntimeError as e:
            # Если строгая загрузка не удалась, пробуем нестрогую
            print(f"⚠ Предупреждение при загрузке весов: {e}")
            print(f"⚠ Попытка нестрогой загрузки...")
            missing_keys, unexpected_keys = self.model.load_state_dict(
                checkpoint['model_state_dict'], strict=False
            )
            if missing_keys:
                print(f"⚠ Отсутствующие ключи (игнорируются): {missing_keys[:5]}...")
            if unexpected_keys:
                print(f"⚠ Неожиданные ключи (игнорируются): {unexpected_keys[:5]}...")
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Модель загружена. Embedding dim: {embedding_dim}, Vocab size: {vocab_size}")
        
        # Получение всех token embeddings (без позиционного кодирования)
        # Это веса из token_embedding слоя
        with torch.no_grad():
            self.all_token_embeddings = self.embedding_layer.token_embedding.embedding.weight
            # Нормализуем для косинусного расстояния
            self.all_token_embeddings_norm = torch.nn.functional.normalize(
                self.all_token_embeddings, p=2, dim=1
            )
        
        print(f"✓ Embeddings подготовлены для поиска")
    
    def get_embedding(self, text: str, use_positional: bool = False) -> torch.Tensor:
        """
        Получение embedding для текста
        
        Args:
            text: Входной текст (слово или предложение)
            use_positional: Использовать ли позиционное кодирование
        
        Returns:
            Tensor с embedding [embedding_dim]
        """
        # Кодирование текста
        token_ids = self.tokenizer.encode(text)
        
        if len(token_ids) == 0:
            raise ValueError(f"Не удалось закодировать текст: '{text}'")
        
        # Преобразование в tensor
        token_ids_tensor = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            if use_positional:
                # Используем полный embedding layer (с позиционным кодированием)
                embeddings = self.embedding_layer(token_ids_tensor)  # [1, seq_len, dim]
            else:
                # Используем только token embedding (без позиционного кодирования)
                # Это лучше для поиска похожих слов по семантике
                embeddings = self.embedding_layer.token_embedding(token_ids_tensor)  # [1, seq_len, dim]
            
            # Усредняем по всем токенам (для слов из нескольких токенов или предложений)
            # Или берем первый токен для одиночных слов
            if len(token_ids) == 1:
                embedding = embeddings[0, 0]  # [dim]
            else:
                # Усредняем, игнорируя padding
                mask = token_ids_tensor != self.tokenizer.special_tokens.get('<PAD>', 0)
                mask = mask.float().unsqueeze(-1)  # [1, seq_len, 1]
                masked_embeddings = embeddings * mask
                embedding = masked_embeddings.sum(dim=1) / mask.sum(dim=1)  # [1, dim]
                embedding = embedding.squeeze(0)  # [dim]
            
            # Нормализуем для косинусного расстояния
            embedding = torch.nn.functional.normalize(embedding.unsqueeze(0), p=2, dim=1)
        
        return embedding.squeeze(0)  # [dim]
    
    def find_similar_words(
        self, 
        text: str, 
        top_k: int = 10, 
        use_positional: bool = False,
        exclude_special: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Поиск ближайших слов по косинусному расстоянию
        
        Args:
            text: Входное слово или предложение
            top_k: Количество ближайших слов для вывода
            use_positional: Использовать ли позиционное кодирование
            exclude_special: Исключать ли специальные токены из результатов
        
        Returns:
            Список кортежей (слово, косинусное_сходство)
        """
        # Получаем embedding для входного текста
        query_embedding = self.get_embedding(text, use_positional=use_positional)
        query_embedding = query_embedding.unsqueeze(0)  # [1, dim]
        
        # Вычисляем косинусное сходство со всеми токенами
        # Косинусное сходство = dot product (так как векторы нормализованы)
        similarities = torch.mm(query_embedding, self.all_token_embeddings_norm.t())  # [1, vocab_size]
        similarities = similarities.squeeze(0)  # [vocab_size]
        
            # Получаем топ-K индексов (берем больше для фильтрации специальных токенов)
        top_k = min(top_k, len(similarities))
        top_indices = torch.topk(similarities, min(top_k * 3, len(similarities)), largest=True).indices
        
        # Фильтруем специальные токены и собираем результаты
        results = []
        special_token_ids = set(self.tokenizer.special_tokens.values())
        
        for idx in top_indices:
            token_id = idx.item()
            
            # Пропускаем специальные токены, если нужно
            if exclude_special and token_id in special_token_ids:
                continue
            
            # Получаем токен
            if token_id in self.tokenizer.vocab:
                token = self.tokenizer.vocab[token_id]
                similarity = similarities[idx].item()
                
                # Пропускаем сам запрос, если он есть в словаре
                # (для этого нужно сравнить декодированные токены)
                results.append((token, similarity))
                
                if len(results) >= top_k:
                    break
        
        return results
    
    def print_similar_words(
        self, 
        text: str, 
        top_k: int = 10, 
        use_positional: bool = False,
        exclude_special: bool = True
    ):
        """
        Вывод ближайших слов в красивом формате
        
        Args:
            text: Входное слово или предложение
            top_k: Количество ближайших слов для вывода
            use_positional: Использовать ли позиционное кодирование
            exclude_special: Исключать ли специальные токены из результатов
        """
        print(f"\n{'='*80}")
        print(f"🔍 Поиск похожих слов для: '{text}'")
        print(f"{'='*80}\n")
        
        try:
            similar_words = self.find_similar_words(
                text, 
                top_k=top_k, 
                use_positional=use_positional,
                exclude_special=exclude_special
            )
            
            if not similar_words:
                print("❌ Не найдено похожих слов")
                return
            
            print(f"📊 Топ-{len(similar_words)} ближайших слов:\n")
            print(f"{'Ранг':<6} {'Слово':<30} {'Косинусное сходство':<20}")
            print("-" * 60)
            
            for rank, (word, similarity) in enumerate(similar_words, 1):
                # Форматируем слово (убираем </w> и показываем читабельно)
                display_word = word.replace('</w>', '').replace('<', '&lt;').replace('>', '&gt;')
                if len(display_word) > 28:
                    display_word = display_word[:25] + "..."
                
                print(f"{rank:<6} {display_word:<30} {similarity:>19.4f}")
            
            print(f"\n{'='*80}\n")
            
        except Exception as e:
            print(f"❌ Ошибка при поиске похожих слов: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Интерактивный режим работы"""
    print("="*80)
    print("  🔍 ПРОВЕРКА РАБОТЫ EMBEDDINGS")
    print("  Поиск ближайших слов по контексту")
    print("="*80)
    print()
    
    # Запрос путей
    model_path = input("Путь к файлу модели (.pth) [по умолчанию: embedding_model.pth]: ").strip()
    if not model_path:
        model_path = "embedding_model.pth"
    
    tokenizer_path = input("Путь к файлу токенизатора (.pkl) [по умолчанию: chekpoint.pkl]: ").strip()
    if not tokenizer_path:
        tokenizer_path = "chekpoint.pkl"
    
    # Проверка существования файлов
    if not os.path.exists(model_path):
        print(f"❌ Файл модели не найден: {model_path}")
        return
    
    if not os.path.exists(tokenizer_path):
        print(f"❌ Файл токенизатора не найден: {tokenizer_path}")
        return
    
    try:
        # Инициализация
        similarity_finder = EmbeddingSimilarity(model_path, tokenizer_path)
        
        print("\n" + "="*80)
        print("✓ Система готова к работе!")
        print("="*80)
        print("\nВведите слово или предложение для поиска похожих слов.")
        print("Для выхода введите 'quit' или 'exit'.\n")
        
        # Интерактивный цикл
        while True:
            query = input("🔍 Введите запрос: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q', 'выход']:
                print("\n👋 До свидания!")
                break
            
            # Запрос количества результатов
            try:
                top_k_input = input("Количество результатов [по умолчанию: 10]: ").strip()
                top_k = int(top_k_input) if top_k_input else 10
            except ValueError:
                top_k = 10
            
            # Запрос использования позиционного кодирования
            use_pos_input = input("Использовать позиционное кодирование? [y/N]: ").strip().lower()
            use_positional = use_pos_input in ['y', 'yes', 'да', 'д']
            
            # Поиск и вывод
            similarity_finder.print_similar_words(
                query, 
                top_k=top_k,
                use_positional=use_positional
            )
    
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Можно использовать как скрипт с аргументами командной строки
    import argparse
    
    parser = argparse.ArgumentParser(description="Поиск похожих слов по embeddings")
    parser.add_argument("--model", type=str, default="embedding_model.pth",
                       help="Путь к файлу модели (.pth)")
    parser.add_argument("--tokenizer", type=str, default="chekpoint.pkl",
                       help="Путь к файлу токенизатора (.pkl)")
    parser.add_argument("--text", type=str, default=None,
                       help="Текст для поиска (если не указан, запускается интерактивный режим)")
    parser.add_argument("--top-k", type=int, default=10,
                       help="Количество ближайших слов")
    parser.add_argument("--positional", action="store_true",
                       help="Использовать позиционное кодирование")
    
    args = parser.parse_args()
    
    if args.text:
        # Режим командной строки
        try:
            similarity_finder = EmbeddingSimilarity(args.model, args.tokenizer)
            similarity_finder.print_similar_words(
                args.text,
                top_k=args.top_k,
                use_positional=args.positional
            )
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Интерактивный режим
        main()

