"""
Скрипт для создания большого датасета для обучения embeddings
Объединяет существующие файлы, очищает текст, разбивает на предложения
"""

import os
import re
import sys
from typing import List, Set
from collections import Counter
import random


class DatasetBuilder:
    """Класс для создания датасета для обучения embeddings"""
    
    def __init__(self):
        self.texts = []
        self.sentences = []
        self.stats = {
            'total_files': 0,
            'total_texts': 0,
            'total_sentences': 0,
            'total_words': 0,
            'unique_words': 0,
            'avg_sentence_length': 0
        }
    
    def clean_text(self, text: str) -> str:
        """
        Очистка текста от лишних символов и нормализация
        
        Args:
            text: Исходный текст
        
        Returns:
            Очищенный текст
        """
        # Удаляем множественные пробелы
        text = re.sub(r'\s+', ' ', text)
        
        # Удаляем специальные символы, но оставляем пунктуацию
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\"\'\«\»]', ' ', text)
        
        # Нормализуем кавычки
        text = text.replace('«', '"').replace('»', '"')
        text = text.replace('„', '"').replace('"', '"')
        text = text.replace('‚', "'").replace(''', "'")
        
        # Удаляем множественные пробелы снова
        text = re.sub(r'\s+', ' ', text)
        
        # Удаляем пробелы перед пунктуацией
        text = re.sub(r'\s+([\.\,\!\?\;\:])', r'\1', text)
        
        # Добавляем пробелы после пунктуации, если их нет
        text = re.sub(r'([\.\,\!\?\;\:])([^\s])', r'\1 \2', text)
        
        return text.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Разбиение текста на предложения
        
        Args:
            text: Текст для разбиения
        
        Returns:
            Список предложений
        """
        # Простое разбиение по знакам препинания
        # Улучшенная версия, учитывающая сокращения
        sentences = []
        
        # Временная замена сокращений
        abbreviations = ['т.д.', 'т.п.', 'т.е.', 'т.к.', 'и т.д.', 'и т.п.', 
                        'др.', 'пр.', 'стр.', 'г.', 'гг.', 'в.', 'вв.',
                        'н.э.', 'до н.э.', 'ст.', 'кв.', 'м.', 'км.',
                        'руб.', 'коп.', 'кг.', 'гр.', 'л.', 'мл.',
                        'мин.', 'сек.', 'ч.', 'сут.', 'нед.', 'мес.',
                        'др.', 'проч.', 'см.', 'рис.', 'табл.', 'стр.',
                        'им.', 'род.', 'дат.', 'вин.', 'твор.', 'предл.',
                        'ед.', 'мн.', 'муж.', 'жен.', 'ср.', 'род.',
                        'напр.', 'т.н.', 'т.о.', 'т.с.', 'т.ч.', 'т.е.',
                        'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Inc.',
                        'Ltd.', 'Co.', 'Corp.', 'etc.', 'vs.', 'e.g.',
                        'i.e.', 'a.m.', 'p.m.', 'U.S.', 'U.K.', 'U.N.']
        
        # Временная замена
        abbr_map = {}
        for i, abbr in enumerate(abbreviations):
            placeholder = f"__ABBR_{i}__"
            text = text.replace(abbr, placeholder)
            abbr_map[placeholder] = abbr
        
        # Разбиение по знакам конца предложения
        # Используем более сложный паттерн
        pattern = r'(?<=[.!?])\s+(?=[А-ЯЁA-Z])'
        parts = re.split(pattern, text)
        
        # Восстанавливаем сокращения
        for part in parts:
            for placeholder, abbr in abbr_map.items():
                part = part.replace(placeholder, abbr)
            
            part = part.strip()
            if part and len(part) > 5:  # Минимальная длина предложения
                sentences.append(part)
        
        # Если разбиение не сработало, используем простое
        if not sentences:
            simple_sentences = re.split(r'[.!?]+\s+', text)
            for sent in simple_sentences:
                sent = sent.strip()
                if sent and len(sent) > 5:
                    sentences.append(sent)
        
        return sentences
    
    def load_from_file(self, filepath: str, encoding: str = 'utf-8') -> List[str]:
        """
        Загрузка текста из файла
        
        Args:
            filepath: Путь к файлу
            encoding: Кодировка файла
        
        Returns:
            Список предложений
        """
        if not os.path.exists(filepath):
            print(f"⚠️  Файл не найден: {filepath}")
            return []
        
        print(f"📂 Загрузка: {filepath}")
        
        try:
            with open(filepath, 'r', encoding=encoding, errors='ignore') as f:
                content = f.read()
            
            # Очистка
            cleaned = self.clean_text(content)
            
            # Разбиение на предложения
            sentences = self.split_into_sentences(cleaned)
            
            print(f"   ✓ Загружено {len(sentences)} предложений")
            
            return sentences
        
        except Exception as e:
            print(f"   ✗ Ошибка при загрузке: {e}")
            return []
    
    def load_from_directory(self, directory: str, extensions: List[str] = None) -> List[str]:
        """
        Загрузка всех текстовых файлов из директории
        
        Args:
            directory: Путь к директории
            extensions: Список расширений файлов (по умолчанию: .txt, .md)
        
        Returns:
            Список всех предложений
        """
        if extensions is None:
            extensions = ['.txt', '.md']
        
        all_sentences = []
        
        if not os.path.exists(directory):
            print(f"⚠️  Директория не найдена: {directory}")
            return all_sentences
        
        print(f"📁 Сканирование директории: {directory}")
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    filepath = os.path.join(root, file)
                    sentences = self.load_from_file(filepath)
                    all_sentences.extend(sentences)
                    self.stats['total_files'] += 1
        
        return all_sentences
    
    def generate_synthetic_texts(self, base_texts: List[str], multiplier: int = 2) -> List[str]:
        """
        Генерация дополнительных текстов на основе существующих
        (перемешивание предложений, создание вариаций)
        
        Args:
            base_texts: Базовые тексты
            multiplier: Во сколько раз увеличить датасет
        
        Returns:
            Расширенный список текстов
        """
        if not base_texts:
            return []
        
        print(f"🔄 Генерация синтетических текстов (x{multiplier})...")
        
        synthetic = []
        
        # Метод 1: Перемешивание предложений
        for _ in range(multiplier - 1):
            shuffled = base_texts.copy()
            random.shuffle(shuffled)
            synthetic.extend(shuffled)
        
        # Метод 2: Объединение случайных предложений
        for _ in range(len(base_texts) // 2):
            # Берем 2-4 случайных предложения и объединяем
            n_sentences = random.randint(2, 4)
            selected = random.sample(base_texts, min(n_sentences, len(base_texts)))
            combined = ' '.join(selected)
            synthetic.append(combined)
        
        print(f"   ✓ Сгенерировано {len(synthetic)} дополнительных текстов")
        
        return synthetic
    
    def add_sample_texts(self) -> List[str]:
        """
        Добавление примеров текстов разных типов
        (если нужно расширить датасет)
        
        Returns:
            Список примеров текстов
        """
        samples = [
            # Научные тексты
            "Машинное обучение представляет собой подраздел искусственного интеллекта.",
            "Нейронные сети используются для решения сложных задач распознавания образов.",
            "Глубокое обучение требует больших вычислительных ресурсов и данных.",
            "Токенизация является важным этапом предобработки текстовых данных.",
            "Embeddings представляют слова и фразы в виде числовых векторов.",
            
            # Технические тексты
            "Трансформеры произвели революцию в области обработки естественного языка.",
            "BERT и GPT являются популярными архитектурами языковых моделей.",
            "Векторные представления кодируют семантическую информацию о словах.",
            "Обучение с учителем использует размеченные данные для тренировки моделей.",
            "Самообучение позволяет использовать неразмеченные данные для обучения.",
            
            # Общие тексты
            "Программирование требует логического мышления и внимательности к деталям.",
            "Алгоритмы определяют последовательность действий для решения задачи.",
            "Структуры данных организуют информацию для эффективного доступа.",
            "Базы данных хранят и управляют большими объемами информации.",
            "Сети обеспечивают связь между различными устройствами и системами.",
            
            # Философские/абстрактные
            "Знание является основой понимания окружающего мира.",
            "Обучение представляет собой процесс приобретения новых навыков.",
            "Коммуникация позволяет обмениваться информацией между людьми.",
            "Творчество требует воображения и способности мыслить нестандартно.",
            "Исследование расширяет границы человеческого понимания.",
        ]
        
        return samples
    
    def filter_sentences(self, sentences: List[str], 
                        min_length: int = 10,
                        max_length: int = 500) -> List[str]:
        """
        Фильтрация предложений по длине
        
        Args:
            sentences: Список предложений
            min_length: Минимальная длина
            max_length: Максимальная длина
        
        Returns:
            Отфильтрованный список
        """
        filtered = []
        for sent in sentences:
            length = len(sent.split())
            if min_length <= length <= max_length:
                filtered.append(sent)
        
        return filtered
    
    def calculate_stats(self, sentences: List[str]):
        """Вычисление статистики датасета"""
        self.stats['total_sentences'] = len(sentences)
        
        # Подсчет слов
        all_words = []
        for sent in sentences:
            words = sent.split()
            all_words.extend(words)
        
        self.stats['total_words'] = len(all_words)
        self.stats['unique_words'] = len(set(all_words))
        
        if sentences:
            total_chars = sum(len(s) for s in sentences)
            self.stats['avg_sentence_length'] = total_chars / len(sentences)
    
    def build_dataset(self, 
                     input_files: List[str] = None,
                     input_dirs: List[str] = None,
                     output_file: str = "embedding_dataset.txt",
                     min_sentence_length: int = 10,
                     max_sentence_length: int = 500,
                     use_synthetic: bool = False,
                     synthetic_multiplier: int = 2,
                     add_samples: bool = False,
                     shuffle: bool = True,
                     deduplicate: bool = True) -> str:
        """
        Построение датасета
        
        Args:
            input_files: Список путей к файлам
            input_dirs: Список путей к директориям
            output_file: Путь к выходному файлу
            min_sentence_length: Минимальная длина предложения (в словах)
            max_sentence_length: Максимальная длина предложения (в словах)
            use_synthetic: Использовать ли синтетическую генерацию
            synthetic_multiplier: Множитель для синтетических текстов
            add_samples: Добавить ли примеры текстов
            shuffle: Перемешать ли предложения
            deduplicate: Удалить ли дубликаты
        
        Returns:
            Путь к созданному файлу
        """
        print("=" * 80)
        print("🔨 СОЗДАНИЕ ДАТАСЕТА ДЛЯ ОБУЧЕНИЯ EMBEDDINGS")
        print("=" * 80)
        print()
        
        all_sentences = []
        
        # Загрузка из файлов
        if input_files:
            print("📄 Загрузка из файлов:")
            for filepath in input_files:
                sentences = self.load_from_file(filepath)
                all_sentences.extend(sentences)
        
        # Загрузка из директорий
        if input_dirs:
            print("\n📁 Загрузка из директорий:")
            for directory in input_dirs:
                sentences = self.load_from_directory(directory)
                all_sentences.extend(sentences)
        
        # Добавление примеров
        if add_samples:
            print("\n📝 Добавление примеров текстов...")
            samples = self.add_sample_texts()
            all_sentences.extend(samples)
            print(f"   ✓ Добавлено {len(samples)} примеров")
        
        # Фильтрация
        print(f"\n🔍 Фильтрация предложений...")
        print(f"   До фильтрации: {len(all_sentences)} предложений")
        all_sentences = self.filter_sentences(
            all_sentences, 
            min_sentence_length, 
            max_sentence_length
        )
        print(f"   После фильтрации: {len(all_sentences)} предложений")
        
        # Удаление дубликатов
        if deduplicate:
            print(f"\n🔄 Удаление дубликатов...")
            print(f"   До удаления: {len(all_sentences)} предложений")
            unique_sentences = list(dict.fromkeys(all_sentences))  # Сохраняет порядок
            print(f"   После удаления: {len(unique_sentences)} предложений")
            all_sentences = unique_sentences
        
        # Синтетическая генерация
        if use_synthetic and all_sentences:
            synthetic = self.generate_synthetic_texts(all_sentences, synthetic_multiplier)
            all_sentences.extend(synthetic)
        
        # Перемешивание
        if shuffle:
            print(f"\n🔀 Перемешивание предложений...")
            random.shuffle(all_sentences)
        
        # Вычисление статистики
        self.calculate_stats(all_sentences)
        
        # Сохранение
        print(f"\n💾 Сохранение датасета в {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for sentence in all_sentences:
                f.write(sentence + '\n')
        
        # Печать статистики
        print("\n" + "=" * 80)
        print("📊 СТАТИСТИКА ДАТАСЕТА")
        print("=" * 80)
        print(f"   Файлов обработано:     {self.stats['total_files']}")
        print(f"   Всего предложений:     {self.stats['total_sentences']:,}")
        print(f"   Всего слов:            {self.stats['total_words']:,}")
        print(f"   Уникальных слов:       {self.stats['unique_words']:,}")
        print(f"   Средняя длина предложения: {self.stats['avg_sentence_length']:.1f} символов")
        print(f"   Размер файла:          {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
        print("=" * 80)
        
        return output_file


def main():
    """Главная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Создание датасета для обучения embeddings")
    parser.add_argument("--files", nargs='+', help="Список файлов для загрузки")
    parser.add_argument("--dirs", nargs='+', help="Список директорий для загрузки")
    parser.add_argument("--output", type=str, default="embedding_dataset.txt", 
                       help="Выходной файл (по умолчанию: embedding_dataset.txt)")
    parser.add_argument("--min-length", type=int, default=10, 
                       help="Минимальная длина предложения в словах")
    parser.add_argument("--max-length", type=int, default=500, 
                       help="Максимальная длина предложения в словах")
    parser.add_argument("--synthetic", action="store_true", 
                       help="Использовать синтетическую генерацию")
    parser.add_argument("--synthetic-multiplier", type=int, default=2, 
                       help="Множитель для синтетических текстов")
    parser.add_argument("--add-samples", action="store_true", 
                       help="Добавить примеры текстов")
    parser.add_argument("--no-shuffle", action="store_true", 
                       help="Не перемешивать предложения")
    parser.add_argument("--no-deduplicate", action="store_true", 
                       help="Не удалять дубликаты")
    
    args = parser.parse_args()
    
    # По умолчанию используем war_and_peace.ru.txt, если он существует
    default_files = []
    if os.path.exists("war_and_peace.ru.txt"):
        default_files.append("war_and_peace.ru.txt")
    
    input_files = args.files if args.files else default_files
    input_dirs = args.dirs if args.dirs else []
    
    # Создание билдера
    builder = DatasetBuilder()
    
    # Построение датасета
    output_file = builder.build_dataset(
        input_files=input_files,
        input_dirs=input_dirs,
        output_file=args.output,
        min_sentence_length=args.min_length,
        max_sentence_length=args.max_length,
        use_synthetic=args.synthetic,
        synthetic_multiplier=args.synthetic_multiplier,
        add_samples=args.add_samples,
        shuffle=not args.no_shuffle,
        deduplicate=not args.no_deduplicate
    )
    
    print(f"\n✅ Датасет создан: {output_file}")
    print(f"📝 Используйте его для обучения: python EMBEDDING_LAYER/auto_evaluation_system.py --data {output_file}")


if __name__ == "__main__":
    main()

