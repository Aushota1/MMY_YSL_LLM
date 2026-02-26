"""
Консольное приложение для обучения и работы с BPE токенизатором
================================================================
Удобный интерфейс для:
- Обучения токенизатора с нуля
- Дообучения существующей модели
- Тестирования токенизатора
- Работы с файлами
"""

import os
import pickle
import re
import sys
from typing import List, Optional
from BPE_STUCTUR import BPETokenizer


def word_tokenize(text: str) -> List[str]:
    """
    Разбивает текст на токены по словам (без BPE).
    Каждое слово — отдельный токен. Разделитель — пробелы и переводы строк.
    Пустые токены не возвращаются.
    """
    if not text or not text.strip():
        return []
    return [t for t in re.split(r'\s+', text.strip()) if t]


class WordTokenizer:
    """
    Словный токенизатор: каждое слово — один токен.
    Сначала текст разбивается на слова, затем слова сопоставляются с числовыми id
    для использования в embedding layer и обучении LLM.
    """
    TOKENIZER_TYPE = 'word'

    def __init__(self):
        self.vocab = {}           # id -> token (слово)
        self.inverse_vocab = {}   # token -> id
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
            '<SEP>': 4,
        }
        self._build_initial_vocab()

    def _build_initial_vocab(self):
        """Инициализация словаря специальными токенами"""
        self.vocab = {i: t for t, i in self.special_tokens.items()}
        self.inverse_vocab = {t: i for i, t in self.vocab.items()}

    def train(self, corpus: List[str], verbose: bool = False) -> None:
        """
        Обучение на корпусе: собираем все уникальные слова и присваиваем им id.
        Сначала весь текст разбивается на слова (алгоритм 12), затем строится словарь.
        """
        self._build_initial_vocab()
        word_set = set()
        for text in corpus:
            if not text or not text.strip():
                continue
            for w in word_tokenize(text):
                word_set.add(w)
        sorted_words = sorted(word_set)
        next_id = len(self.special_tokens)
        for w in sorted_words:
            self.vocab[next_id] = w
            self.inverse_vocab[w] = next_id
            next_id += 1
        if verbose:
            print(f"   Уникальных слов: {len(word_set)}, размер словаря: {self.get_vocab_size()}")

    def encode(self, text: str) -> List[int]:
        """Текст -> список id (для embedding / LLM). Неизвестные слова -> <UNK>."""
        if not text or not text.strip():
            return []
        unk_id = self.special_tokens.get('<UNK>', 1)
        ids = []
        for w in word_tokenize(text):
            ids.append(self.inverse_vocab.get(w, unk_id))
        return ids

    def decode(self, token_ids: List[int]) -> str:
        """Список id -> текст (для вывода из LLM)."""
        tokens = []
        for i in token_ids:
            tokens.append(self.vocab.get(i, '<UNK>'))
        return ' '.join(tokens)

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, path: str) -> None:
        """Сохранение в том же формате, что и BPE: vocab_size, vocab, inverse_vocab, merges, special_tokens, pattern_string, has_regex, merge_order."""
        data = {
            'vocab_size': len(self.vocab),
            'vocab': self.vocab,
            'inverse_vocab': self.inverse_vocab,
            'merges': {},
            'special_tokens': self.special_tokens,
            'pattern_string': None,
            'has_regex': False,
            'merge_order': [],
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        """Загрузка из pkl в формате BPE (те же ключи). merges/pattern/merge_order игнорируются."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.vocab = data['vocab']
        self.inverse_vocab = data['inverse_vocab']
        self.special_tokens = data.get('special_tokens', self.special_tokens)


class TokenizerTrainerApp:
    """Консольное приложение для работы с токенизатором"""
    
    def __init__(self):
        self.tokenizer: Optional[BPETokenizer] = None
        self.current_model_path: Optional[str] = None
        self.word_tokenizer: Optional[WordTokenizer] = None
        self.word_model_path: Optional[str] = None
    
    def clear_screen(self):
        """Очистка экрана"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self, title: str):
        """Печать заголовка"""
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70 + "\n")
    
    def print_menu(self):
        """Главное меню"""
        self.clear_screen()
        print("=" * 70)
        print("  BPE ТОКЕНИЗАТОР - ГЛАВНОЕ МЕНЮ")
        print("=" * 70)
        print("\n1. Создать новый токенизатор")
        print("2. Загрузить существующую модель")
        print("3. Обучить токенизатор (с нуля)")
        print("4. Дообучить токенизатор (расширить словарь)")
        print("5. Обучить на файле")
        print("6. Дообучить на файле")
        print("7. Найти новые пары в словаре")
        print("8. Тестировать токенизатор")
        print("9. Показать информацию о модели")
        print("10. Сохранить модель")
        print("11. Интерактивное тестирование")
        print("12. Словный токенизатор (word-level для LLM/embedding)")
        print("0. Выход")
        print("\n" + "=" * 70)
    
    def create_tokenizer(self):
        """Создание нового токенизатора"""
        self.print_header("СОЗДАНИЕ НОВОГО ТОКЕНИЗАТОРА")
        
        try:
            vocab_size = input("Введите размер словаря (по умолчанию 30000): ").strip()
            vocab_size = int(vocab_size) if vocab_size else 30000
            
            self.tokenizer = BPETokenizer(vocab_size=vocab_size)
            self.current_model_path = None
            
            print(f"\n✓ Токенизатор создан с размером словаря: {vocab_size}")
            print(f"  Текущий размер словаря: {self.tokenizer.get_vocab_size()}")
            input("\nНажмите Enter для продолжения...")
        except ValueError:
            print("\n✗ Ошибка: введите корректное число")
            input("Нажмите Enter для продолжения...")
    
    def load_model(self):
        """Загрузка существующей модели"""
        self.print_header("ЗАГРУЗКА МОДЕЛИ")
        
        model_path = input("Введите путь к файлу модели (.pkl): ").strip()
        
        if not model_path:
            print("\n✗ Ошибка: путь не указан")
            input("Нажмите Enter для продолжения...")
            return
        
        if not os.path.exists(model_path):
            print(f"\n✗ Ошибка: файл {model_path} не найден")
            input("Нажмите Enter для продолжения...")
            return
        
        try:
            self.tokenizer = BPETokenizer()
            self.tokenizer.load(model_path)
            self.current_model_path = model_path
            
            print(f"\n✓ Модель успешно загружена из: {model_path}")
            print(f"  Размер словаря: {self.tokenizer.get_vocab_size()}")
            print(f"  Количество слияний: {len(self.tokenizer.merges)}")
            input("\nНажмите Enter для продолжения...")
        except Exception as e:
            print(f"\n✗ Ошибка при загрузке: {e}")
            input("Нажмите Enter для продолжения...")
    
    def train_from_input(self):
        """Обучение токенизатора с вводом текста"""
        if not self.tokenizer:
            print("\n✗ Ошибка: сначала создайте или загрузите токенизатор")
            input("Нажмите Enter для продолжения...")
            return
        
        self.print_header("ОБУЧЕНИЕ ТОКЕНИЗАТОРА")
        
        print("Введите тексты для обучения (каждый текст с новой строки).")
        print("Для завершения ввода введите пустую строку или 'END':")
        print("-" * 70)
        
        corpus = []
        line_num = 1
        
        while True:
            text = input(f"Текст {line_num}: ").strip()
            if not text or text.upper() == 'END':
                break
            if text:
                corpus.append(text)
                line_num += 1
        
        if not corpus:
            print("\n✗ Ошибка: не введено ни одного текста")
            input("Нажмите Enter для продолжения...")
            return
        
        verbose = input("\nПоказывать прогресс обучения? (y/n, по умолчанию n): ").strip().lower() == 'y'
        
        checkpoint_path = input("Путь для сохранения чекпоинтов (Enter = не сохранять): ").strip()
        checkpoint_interval = 100
        if checkpoint_path:
            if not checkpoint_path.endswith('.pkl'):
                checkpoint_path += '.pkl'
            interval_input = input("Сохранять чекпоинт каждые N итераций (по умолчанию 100): ").strip()
            checkpoint_interval = int(interval_input) if interval_input else 100
            print(f"💾 Чекпоинты будут сохраняться каждые {checkpoint_interval} итераций")
        
        print(f"\nНачинаем обучение на {len(corpus)} текстах...")
        if checkpoint_path:
            print(f"💡 Для прерывания нажмите Ctrl+C - прогресс будет сохранен")
        print("-" * 70)
        
        try:
            self.tokenizer.train(
                corpus, 
                verbose=verbose,
                checkpoint_path=checkpoint_path if checkpoint_path else None,
                checkpoint_interval=checkpoint_interval
            )
            print("-" * 70)
            print(f"\n✓ Обучение завершено!")
            print(f"  Размер словаря: {self.tokenizer.get_vocab_size()}")
            print(f"  Количество слияний: {len(self.tokenizer.merges)}")
            if checkpoint_path:
                self.current_model_path = checkpoint_path
            else:
                self.current_model_path = None
            input("\nНажмите Enter для продолжения...")
        except KeyboardInterrupt:
            print("\n\n⚠️  Обучение прервано")
            if checkpoint_path:
                print(f"✅ Последний чекпоинт сохранен в: {checkpoint_path}")
                print("💡 Вы можете продолжить обучение, загрузив этот файл")
                self.current_model_path = checkpoint_path
            input("Нажмите Enter для продолжения...")
        except Exception as e:
            print(f"\n✗ Ошибка при обучении: {e}")
            input("Нажмите Enter для продолжения...")
    
    def continue_training_from_input(self):
        """Дообучение токенизатора с вводом текста"""
        if not self.tokenizer:
            print("\n✗ Ошибка: сначала создайте или загрузите токенизатор")
            input("Нажмите Enter для продолжения...")
            return
        
        if not hasattr(self.tokenizer, 'vocab') or not self.tokenizer.vocab or len(self.tokenizer.vocab) <= len(self.tokenizer.special_tokens):
            print("\n✗ Ошибка: токенизатор должен быть обучен перед дообучением")
            input("Нажмите Enter для продолжения...")
            return
        
        self.print_header("ДООБУЧЕНИЕ ТОКЕНИЗАТОРА")
        
        print(f"Текущий размер словаря: {self.tokenizer.get_vocab_size()}")
        print(f"Максимальный размер словаря: {self.tokenizer.vocab_size}")
        print(f"Можно добавить до {self.tokenizer.vocab_size - self.tokenizer.get_vocab_size()} новых токенов")
        print("\nВведите новые тексты для дообучения (каждый текст с новой строки).")
        print("Для завершения ввода введите пустую строку или 'END':")
        print("-" * 70)
        
        corpus = []
        line_num = 1
        
        while True:
            text = input(f"Текст {line_num}: ").strip()
            if not text or text.upper() == 'END':
                break
            if text:
                corpus.append(text)
                line_num += 1
        
        if not corpus:
            print("\n✗ Ошибка: не введено ни одного текста")
            input("Нажмите Enter для продолжения...")
            return
        
        max_new_merges_input = input("\nМаксимальное количество новых слияний (Enter = без ограничений): ").strip()
        max_new_merges = int(max_new_merges_input) if max_new_merges_input else None
        
        verbose = input("Показывать прогресс обучения? (y/n, по умолчанию n): ").strip().lower() == 'y'
        
        checkpoint_path = input("Путь для сохранения чекпоинтов (Enter = не сохранять): ").strip()
        checkpoint_interval = 100
        if checkpoint_path:
            if not checkpoint_path.endswith('.pkl'):
                checkpoint_path += '.pkl'
            interval_input = input("Сохранять чекпоинт каждые N итераций (по умолчанию 100): ").strip()
            checkpoint_interval = int(interval_input) if interval_input else 100
            print(f"💾 Чекпоинты будут сохраняться каждые {checkpoint_interval} итераций")
        
        print(f"\nНачинаем дообучение на {len(corpus)} новых текстах...")
        if checkpoint_path:
            print(f"💡 Для прерывания нажмите Ctrl+C - прогресс будет сохранен")
        print("-" * 70)
        
        try:
            old_vocab_size = self.tokenizer.get_vocab_size()
            self.tokenizer.continue_training(
                corpus, 
                verbose=verbose, 
                max_new_merges=max_new_merges,
                checkpoint_path=checkpoint_path if checkpoint_path else None,
                checkpoint_interval=checkpoint_interval
            )
            new_vocab_size = self.tokenizer.get_vocab_size()
            print("-" * 70)
            print(f"\n✓ Дообучение завершено!")
            print(f"  Старый размер словаря: {old_vocab_size}")
            print(f"  Новый размер словаря: {new_vocab_size}")
            print(f"  Добавлено токенов: {new_vocab_size - old_vocab_size}")
            print(f"  Общее количество слияний: {len(self.tokenizer.merges)}")
            if checkpoint_path:
                self.current_model_path = checkpoint_path
            else:
                self.current_model_path = None
            input("\nНажмите Enter для продолжения...")
        except KeyboardInterrupt:
            print("\n\n⚠️  Дообучение прервано")
            if checkpoint_path:
                print(f"✅ Последний чекпоинт сохранен в: {checkpoint_path}")
                print("💡 Вы можете продолжить дообучение, загрузив этот файл")
                self.current_model_path = checkpoint_path
            input("Нажмите Enter для продолжения...")
        except Exception as e:
            print(f"\n✗ Ошибка при дообучении: {e}")
            input("Нажмите Enter для продолжения...")
    
    def train_from_file(self):
        """Обучение токенизатора на файле"""
        if not self.tokenizer:
            print("\n✗ Ошибка: сначала создайте или загрузите токенизатор")
            input("Нажмите Enter для продолжения...")
            return
        
        self.print_header("ОБУЧЕНИЕ НА ФАЙЛЕ")
        
        file_path = input("Введите путь к файлу с корпусом: ").strip()
        
        if not file_path:
            print("\n✗ Ошибка: путь не указан")
            input("Нажмите Enter для продолжения...")
            return
        
        if not os.path.exists(file_path):
            print(f"\n✗ Ошибка: файл {file_path} не найден")
            input("Нажмите Enter для продолжения...")
            return
        
        encoding = input("Кодировка файла (по умолчанию utf-8): ").strip() or 'utf-8'
        verbose = input("Показывать прогресс обучения? (y/n, по умолчанию n): ").strip().lower() == 'y'
        
        checkpoint_path = input("Путь для сохранения чекпоинтов (Enter = не сохранять): ").strip()
        checkpoint_interval = 100
        if checkpoint_path:
            if not checkpoint_path.endswith('.pkl'):
                checkpoint_path += '.pkl'
            interval_input = input("Сохранять чекпоинт каждые N итераций (по умолчанию 100): ").strip()
            checkpoint_interval = int(interval_input) if interval_input else 100
            print(f"💾 Чекпоинты будут сохраняться каждые {checkpoint_interval} итераций")
        
        print(f"\nНачинаем обучение на файле {file_path}...")
        if checkpoint_path:
            print(f"💡 Для прерывания нажмите Ctrl+C - прогресс будет сохранен")
        print("-" * 70)
        
        try:
            # Читаем файл
            with open(file_path, 'r', encoding=encoding) as f:
                corpus = f.readlines()
            
            print(f"Загружено {len(corpus)} строк из файла")
            self.tokenizer.train(
                corpus, 
                verbose=verbose,
                checkpoint_path=checkpoint_path if checkpoint_path else None,
                checkpoint_interval=checkpoint_interval
            )
            print("-" * 70)
            print(f"\n✓ Обучение завершено!")
            print(f"  Размер словаря: {self.tokenizer.get_vocab_size()}")
            print(f"  Количество слияний: {len(self.tokenizer.merges)}")
            if checkpoint_path:
                self.current_model_path = checkpoint_path
            else:
                self.current_model_path = None
            input("\nНажмите Enter для продолжения...")
        except KeyboardInterrupt:
            print("\n\n⚠️  Обучение прервано")
            if checkpoint_path:
                print(f"✅ Последний чекпоинт сохранен в: {checkpoint_path}")
                print("💡 Вы можете продолжить обучение, загрузив этот файл")
                self.current_model_path = checkpoint_path
            input("Нажмите Enter для продолжения...")
        except Exception as e:
            print(f"\n✗ Ошибка при обучении: {e}")
            input("Нажмите Enter для продолжения...")
    
    def continue_training_from_file(self):
        """Дообучение токенизатора на файле"""
        if not self.tokenizer:
            print("\n✗ Ошибка: сначала создайте или загрузите токенизатор")
            input("Нажмите Enter для продолжения...")
            return
        
        if not hasattr(self.tokenizer, 'vocab') or not self.tokenizer.vocab or len(self.tokenizer.vocab) <= len(self.tokenizer.special_tokens):
            print("\n✗ Ошибка: токенизатор должен быть обучен перед дообучением")
            input("Нажмите Enter для продолжения...")
            return
        
        self.print_header("ДООБУЧЕНИЕ НА ФАЙЛЕ")
        
        file_path = input("Введите путь к файлу с новым корпусом: ").strip()
        
        if not file_path:
            print("\n✗ Ошибка: путь не указан")
            input("Нажмите Enter для продолжения...")
            return
        
        if not os.path.exists(file_path):
            print(f"\n✗ Ошибка: файл {file_path} не найден")
            input("Нажмите Enter для продолжения...")
            return
        
        encoding = input("Кодировка файла (по умолчанию utf-8): ").strip() or 'utf-8'
        max_new_merges_input = input("Максимальное количество новых слияний (Enter = без ограничений): ").strip()
        max_new_merges = int(max_new_merges_input) if max_new_merges_input else None
        verbose = input("Показывать прогресс обучения? (y/n, по умолчанию n): ").strip().lower() == 'y'
        
        checkpoint_path = input("Путь для сохранения чекпоинтов (Enter = не сохранять): ").strip()
        checkpoint_interval = 100
        if checkpoint_path:
            if not checkpoint_path.endswith('.pkl'):
                checkpoint_path += '.pkl'
            interval_input = input("Сохранять чекпоинт каждые N итераций (по умолчанию 100): ").strip()
            checkpoint_interval = int(interval_input) if interval_input else 100
            print(f"💾 Чекпоинты будут сохраняться каждые {checkpoint_interval} итераций")
        
        print(f"\nНачинаем дообучение на файле {file_path}...")
        if checkpoint_path:
            print(f"💡 Для прерывания нажмите Ctrl+C - прогресс будет сохранен")
        print("-" * 70)
        
        try:
            # Читаем файл
            with open(file_path, 'r', encoding=encoding) as f:
                corpus = f.readlines()
            
            print(f"Загружено {len(corpus)} строк из файла")
            old_vocab_size = self.tokenizer.get_vocab_size()
            self.tokenizer.continue_training(
                corpus, 
                verbose=verbose, 
                max_new_merges=max_new_merges,
                checkpoint_path=checkpoint_path if checkpoint_path else None,
                checkpoint_interval=checkpoint_interval
            )
            new_vocab_size = self.tokenizer.get_vocab_size()
            print("-" * 70)
            print(f"\n✓ Дообучение завершено!")
            print(f"  Старый размер словаря: {old_vocab_size}")
            print(f"  Новый размер словаря: {new_vocab_size}")
            print(f"  Добавлено токенов: {new_vocab_size - old_vocab_size}")
            print(f"  Общее количество слияний: {len(self.tokenizer.merges)}")
            if checkpoint_path:
                self.current_model_path = checkpoint_path
            else:
                self.current_model_path = None
            input("\nНажмите Enter для продолжения...")
        except KeyboardInterrupt:
            print("\n\n⚠️  Дообучение прервано")
            if checkpoint_path:
                print(f"✅ Последний чекпоинт сохранен в: {checkpoint_path}")
                print("💡 Вы можете продолжить дообучение, загрузив этот файл")
                self.current_model_path = checkpoint_path
            input("Нажмите Enter для продолжения...")
        except Exception as e:
            print(f"\n✗ Ошибка при дообучении: {e}")
            input("Нажмите Enter для продолжения...")
    
    def find_new_pairs(self):
        """Поиск новых пар в существующем словаре"""
        if not self.tokenizer:
            print("\n✗ Ошибка: сначала создайте или загрузите токенизатор")
            input("Нажмите Enter для продолжения...")
            return
        
        self.print_header("ПОИСК НОВЫХ ПАР В СЛОВАРЕ")
        
        print("Введите тексты для анализа (каждый текст с новой строки).")
        print("Для завершения ввода введите пустую строку или 'END':")
        print("-" * 70)
        
        corpus = []
        line_num = 1
        
        while True:
            text = input(f"Текст {line_num}: ").strip()
            if not text or text.upper() == 'END':
                break
            if text:
                corpus.append(text)
                line_num += 1
        
        if not corpus:
            print("\n✗ Ошибка: не введено ни одного текста")
            input("Нажмите Enter для продолжения...")
            return
        
        max_new_merges_input = input("\nМаксимальное количество новых слияний (Enter = без ограничений): ").strip()
        max_new_merges = int(max_new_merges_input) if max_new_merges_input else None
        
        verbose = input("Показывать прогресс поиска? (y/n, по умолчанию n): ").strip().lower() == 'y'
        
        checkpoint_path = input("Путь для сохранения чекпоинтов (Enter = не сохранять): ").strip()
        checkpoint_interval = 100
        if checkpoint_path:
            if not checkpoint_path.endswith('.pkl'):
                checkpoint_path += '.pkl'
            interval_input = input("Сохранять чекпоинт каждые N итераций (по умолчанию 100): ").strip()
            checkpoint_interval = int(interval_input) if interval_input else 100
            print(f"💾 Чекпоинты будут сохраняться каждые {checkpoint_interval} итераций")
        
        print(f"\nНачинаем поиск новых пар на {len(corpus)} текстах...")
        print(f"Текущий размер словаря: {self.tokenizer.get_vocab_size()}")
        print(f"Текущее количество слияний: {len(self.tokenizer.merges)}")
        if checkpoint_path:
            print(f"💡 Для прерывания нажмите Ctrl+C - прогресс будет сохранен")
        print("-" * 70)
        
        try:
            old_merges_count = len(self.tokenizer.merges)
            new_merges_count = self.tokenizer.find_new_pairs_in_vocab(
                corpus, 
                verbose=verbose, 
                max_new_merges=max_new_merges,
                checkpoint_path=checkpoint_path if checkpoint_path else None,
                checkpoint_interval=checkpoint_interval
            )
            new_merges_count_total = len(self.tokenizer.merges)
            
            print("-" * 70)
            print(f"\n✓ Поиск новых пар завершен!")
            print(f"  Старое количество слияний: {old_merges_count}")
            print(f"  Новое количество слияний: {new_merges_count_total}")
            print(f"  Найдено новых слияний: {new_merges_count}")
            print(f"  Новый размер словаря: {self.tokenizer.get_vocab_size()}")
            if checkpoint_path:
                self.current_model_path = checkpoint_path
            else:
                self.current_model_path = None
            input("\nНажмите Enter для продолжения...")
        except KeyboardInterrupt:
            print("\n\n⚠️  Поиск новых пар прерван")
            if checkpoint_path:
                print(f"✅ Последний чекпоинт сохранен в: {checkpoint_path}")
                print("💡 Вы можете продолжить поиск, загрузив этот файл")
                self.current_model_path = checkpoint_path
            input("Нажмите Enter для продолжения...")
        except Exception as e:
            print(f"\n✗ Ошибка при поиске новых пар: {e}")
            input("Нажмите Enter для продолжения...")
    
    def find_new_pairs_from_file(self):
        """Поиск новых пар в словаре на основе файла"""
        if not self.tokenizer:
            print("\n✗ Ошибка: сначала создайте или загрузите токенизатор")
            input("Нажмите Enter для продолжения...")
            return
        
        self.print_header("ПОИСК НОВЫХ ПАР В СЛОВАРЕ (ИЗ ФАЙЛА)")
        
        file_path = input("Введите путь к файлу с корпусом: ").strip()
        
        if not file_path:
            print("\n✗ Ошибка: путь не указан")
            input("Нажмите Enter для продолжения...")
            return
        
        if not os.path.exists(file_path):
            print(f"\n✗ Ошибка: файл {file_path} не найден")
            input("Нажмите Enter для продолжения...")
            return
        
        encoding = input("Кодировка файла (по умолчанию utf-8): ").strip() or 'utf-8'
        
        max_new_merges_input = input("\nМаксимальное количество новых слияний (Enter = без ограничений): ").strip()
        max_new_merges = int(max_new_merges_input) if max_new_merges_input else None
        
        verbose = input("Показывать прогресс поиска? (y/n, по умолчанию n): ").strip().lower() == 'y'
        
        checkpoint_path = input("Путь для сохранения чекпоинтов (Enter = не сохранять): ").strip()
        checkpoint_interval = 100
        if checkpoint_path:
            if not checkpoint_path.endswith('.pkl'):
                checkpoint_path += '.pkl'
            interval_input = input("Сохранять чекпоинт каждые N итераций (по умолчанию 100): ").strip()
            checkpoint_interval = int(interval_input) if interval_input else 100
            print(f"💾 Чекпоинты будут сохраняться каждые {checkpoint_interval} итераций")
        
        print(f"\nНачинаем поиск новых пар на файле {file_path}...")
        print(f"Текущий размер словаря: {self.tokenizer.get_vocab_size()}")
        print(f"Текущее количество слияний: {len(self.tokenizer.merges)}")
        if checkpoint_path:
            print(f"💡 Для прерывания нажмите Ctrl+C - прогресс будет сохранен")
        print("-" * 70)
        
        try:
            # Читаем файл
            with open(file_path, 'r', encoding=encoding) as f:
                corpus = [line.strip() for line in f if line.strip()]
            
            if not corpus:
                print("\n✗ Ошибка: файл пуст или не содержит текста")
                input("Нажмите Enter для продолжения...")
                return
            
            old_merges_count = len(self.tokenizer.merges)
            new_merges_count = self.tokenizer.find_new_pairs_in_vocab(
                corpus,
                verbose=verbose,
                max_new_merges=max_new_merges,
                checkpoint_path=checkpoint_path if checkpoint_path else None,
                checkpoint_interval=checkpoint_interval
            )
            new_merges_count_total = len(self.tokenizer.merges)
            
            print("-" * 70)
            print(f"\n✓ Поиск новых пар завершен!")
            print(f"  Обработано строк: {len(corpus)}")
            print(f"  Старое количество слияний: {old_merges_count}")
            print(f"  Новое количество слияний: {new_merges_count_total}")
            print(f"  Найдено новых слияний: {new_merges_count}")
            print(f"  Новый размер словаря: {self.tokenizer.get_vocab_size()}")
            if checkpoint_path:
                self.current_model_path = checkpoint_path
            else:
                self.current_model_path = None
            input("\nНажмите Enter для продолжения...")
        except KeyboardInterrupt:
            print("\n\n⚠️  Поиск новых пар прерван")
            if checkpoint_path:
                print(f"✅ Последний чекпоинт сохранен в: {checkpoint_path}")
                print("💡 Вы можете продолжить поиск, загрузив этот файл")
                self.current_model_path = checkpoint_path
            input("Нажмите Enter для продолжения...")
        except Exception as e:
            print(f"\n✗ Ошибка при поиске новых пар: {e}")
            input("Нажмите Enter для продолжения...")
    
    def test_tokenizer(self):
        """Тестирование токенизатора"""
        if not self.tokenizer:
            print("\n✗ Ошибка: сначала создайте или загрузите токенизатор")
            input("Нажмите Enter для продолжения...")
            return
        
        if not hasattr(self.tokenizer, 'vocab') or not self.tokenizer.vocab or len(self.tokenizer.vocab) <= len(self.tokenizer.special_tokens):
            print("\n✗ Ошибка: токенизатор должен быть обучен")
            input("Нажмите Enter для продолжения...")
            return
        
        self.print_header("ТЕСТИРОВАНИЕ ТОКЕНИЗАТОРА")
        
        test_text = input("Введите текст для тестирования: ").strip()
        
        if not test_text:
            print("\n✗ Ошибка: текст не введен")
            input("Нажмите Enter для продолжения...")
            return
        
        try:
            # Кодирование
            token_ids = self.tokenizer.encode(test_text)
            tokens = [self.tokenizer.vocab[id_] for id_ in token_ids]
            
            # Декодирование
            decoded_text = self.tokenizer.decode(token_ids)
            
            print("\n" + "-" * 70)
            print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
            print("-" * 70)
            print(f"Исходный текст:     '{test_text}'")
            print(f"Токены (IDs):        {token_ids}")
            print(f"Токены (текст):      {tokens}")
            print(f"Количество токенов:  {len(token_ids)}")
            print(f"Декодированный текст: '{decoded_text}'")
            print(f"Совпадает:           {test_text.lower() == decoded_text}")
            print("-" * 70)
            
            input("\nНажмите Enter для продолжения...")
        except Exception as e:
            print(f"\n✗ Ошибка при тестировании: {e}")
            input("Нажмите Enter для продолжения...")
    
    def show_model_info(self):
        """Показать информацию о модели"""
        if not self.tokenizer:
            print("\n✗ Ошибка: токенизатор не создан")
            input("Нажмите Enter для продолжения...")
            return
        
        self.print_header("ИНФОРМАЦИЯ О МОДЕЛИ")
        
        print(f"Максимальный размер словаря: {self.tokenizer.vocab_size}")
        print(f"Текущий размер словаря:      {self.tokenizer.get_vocab_size()}")
        print(f"Свободных слотов:            {self.tokenizer.vocab_size - self.tokenizer.get_vocab_size()}")
        
        if hasattr(self.tokenizer, 'merges'):
            print(f"Количество слияний:        {len(self.tokenizer.merges)}")
        
        print(f"\nСпециальные токены:")
        for token, token_id in self.tokenizer.special_tokens.items():
            print(f"  {token}: {token_id}")
        
        if hasattr(self.tokenizer, 'vocab') and self.tokenizer.vocab:
            print(f"\nПримеры токенов (первые 10):")
            for i, (token_id, token) in enumerate(list(self.tokenizer.vocab.items())[:10]):
                print(f"  ID {token_id}: {repr(token)}")
        
        if self.current_model_path:
            print(f"\nТекущий файл модели: {self.current_model_path}")
        else:
            print(f"\nМодель не сохранена или была изменена")
        
        input("\nНажмите Enter для продолжения...")
    
    def save_model(self):
        """Сохранение модели"""
        if not self.tokenizer:
            print("\n✗ Ошибка: токенизатор не создан")
            input("Нажмите Enter для продолжения...")
            return
        
        if not hasattr(self.tokenizer, 'vocab') or not self.tokenizer.vocab or len(self.tokenizer.vocab) <= len(self.tokenizer.special_tokens):
            print("\n✗ Ошибка: токенизатор должен быть обучен перед сохранением")
            input("Нажмите Enter для продолжения...")
            return
        
        self.print_header("СОХРАНЕНИЕ МОДЕЛИ")
        
        if self.current_model_path:
            print(f"Текущий файл: {self.current_model_path}")
            use_current = input("Использовать текущий файл? (y/n, по умолчанию n): ").strip().lower() == 'y'
            if use_current:
                model_path = self.current_model_path
            else:
                model_path = input("Введите путь для сохранения (.pkl): ").strip()
        else:
            model_path = input("Введите путь для сохранения (.pkl): ").strip()
        
        if not model_path:
            print("\n✗ Ошибка: путь не указан")
            input("Нажмите Enter для продолжения...")
            return
        
        if not model_path.endswith('.pkl'):
            model_path += '.pkl'
        
        try:
            self.tokenizer.save(model_path)
            self.current_model_path = model_path
            print(f"\n✓ Модель успешно сохранена в: {model_path}")
            print(f"  Размер словаря: {self.tokenizer.get_vocab_size()}")
            input("\nНажмите Enter для продолжения...")
        except Exception as e:
            print(f"\n✗ Ошибка при сохранении: {e}")
            input("Нажмите Enter для продолжения...")
    
    def interactive_test(self):
        """Интерактивное тестирование"""
        if not self.tokenizer:
            print("\n✗ Ошибка: сначала создайте или загрузите токенизатор")
            input("Нажмите Enter для продолжения...")
            return
        
        if not hasattr(self.tokenizer, 'vocab') or not self.tokenizer.vocab or len(self.tokenizer.vocab) <= len(self.tokenizer.special_tokens):
            print("\n✗ Ошибка: токенизатор должен быть обучен")
            input("Нажмите Enter для продолжения...")
            return
        
        self.print_header("ИНТЕРАКТИВНОЕ ТЕСТИРОВАНИЕ")
        print("Введите тексты для тестирования (пустая строка для выхода)")
        print("-" * 70)
        
        while True:
            test_text = input("\nТекст для тестирования: ").strip()
            
            if not test_text:
                break
            
            try:
                token_ids = self.tokenizer.encode(test_text)
                tokens = [self.tokenizer.vocab[id_] for id_ in token_ids]
                decoded_text = self.tokenizer.decode(token_ids)
                
                print(f"  Токены: {tokens}")
                print(f"  Количество: {len(token_ids)}")
                print(f"  Декодировано: '{decoded_text}'")
            except Exception as e:
                print(f"  ✗ Ошибка: {e}")
        
        print("\nВыход из интерактивного режима...")
        input("Нажмите Enter для продолжения...")
    
    def word_tokenizer_menu(self):
        """Подменю словного токенизатора (алгоритм 12: слова -> токены -> числа для LLM/embedding)"""
        while True:
            self.clear_screen()
            self.print_header("СЛОВНЫЙ ТОКЕНИЗАТОР (WORD-LEVEL ДЛЯ LLM/EMBEDDING)")
            print("Сначала текст разбивается на слова (каждое слово = токен), затем слова")
            print("сопоставляются с числовыми id для embedding layer и обучения LLM.")
            print("-" * 70)
            if self.word_tokenizer:
                print(f"  Модель загружена. Размер словаря: {self.word_tokenizer.get_vocab_size()}")
                if self.word_model_path:
                    print(f"  Файл: {self.word_model_path}")
            else:
                print("  Модель не создана.")
            print("-" * 70)
            print("1. Создать и обучить на введённых текстах")
            print("2. Обучить на файле")
            print("3. Загрузить модель (.pkl)")
            print("4. Сохранить модель")
            print("5. Показать токены текста (только разбиение на слова)")
            print("6. Кодировать/декодировать (текст -> числа -> текст)")
            print("0. Назад в главное меню")
            print("-" * 70)
            choice = input("Выберите действие: ").strip()

            if choice == '0':
                break
            elif choice == '1':
                self.train_word_from_input()
            elif choice == '2':
                self.train_word_from_file()
            elif choice == '3':
                self.load_word_model()
            elif choice == '4':
                self.save_word_model()
            elif choice == '5':
                self.word_show_tokens_console()
            elif choice == '6':
                self.word_encode_decode_console()
            else:
                print("\n✗ Неверный выбор.")
                input("Нажмите Enter...")

    def train_word_from_input(self):
        """Обучение словного токенизатора на введённых текстах"""
        self.print_header("ОБУЧЕНИЕ СЛОВНОГО ТОКЕНИЗАТОРА (ВВОД)")
        print("Введите тексты. Каждая строка — один документ. Пустая строка или END — конец.")
        print("-" * 70)
        corpus = []
        n = 1
        while True:
            line = input(f"Текст {n}: ").strip()
            if not line or line.upper() == 'END':
                break
            corpus.append(line)
            n += 1
        if not corpus:
            print("\n✗ Нет текстов.")
            input("Нажмите Enter...")
            return
        verbose = input("Показывать прогресс? (y/n): ").strip().lower() == 'y'
        self.word_tokenizer = WordTokenizer()
        self.word_tokenizer.train(corpus, verbose=verbose)
        self.word_model_path = None
        print(f"\n✓ Обучено. Размер словаря: {self.word_tokenizer.get_vocab_size()} (слова → id для LLM/embedding)")
        input("Нажмите Enter...")

    def train_word_from_file(self):
        """Обучение словного токенизатора на файле"""
        self.print_header("ОБУЧЕНИЕ СЛОВНОГО ТОКЕНИЗАТОРА (ФАЙЛ)")
        path = input("Путь к файлу с корпусом: ").strip()
        if not path or not os.path.exists(path):
            print("\n✗ Файл не указан или не найден.")
            input("Нажмите Enter...")
            return
        enc = input("Кодировка (Enter = utf-8): ").strip() or 'utf-8'
        try:
            with open(path, 'r', encoding=enc) as f:
                corpus = f.readlines()
        except Exception as e:
            print(f"\n✗ Ошибка чтения: {e}")
            input("Нажмите Enter...")
            return
        verbose = input("Показывать прогресс? (y/n): ").strip().lower() == 'y'
        self.word_tokenizer = WordTokenizer()
        self.word_tokenizer.train(corpus, verbose=verbose)
        self.word_model_path = None
        print(f"\n✓ Обучено. Строк: {len(corpus)}, размер словаря: {self.word_tokenizer.get_vocab_size()}")
        input("Нажмите Enter...")

    def load_word_model(self):
        """Загрузка словной модели"""
        self.print_header("ЗАГРУЗКА СЛОВНОЙ МОДЕЛИ")
        path = input("Путь к файлу (.pkl): ").strip()
        if not path or not os.path.exists(path):
            print("\n✗ Файл не указан или не найден.")
            input("Нажмите Enter...")
            return
        try:
            self.word_tokenizer = WordTokenizer()
            self.word_tokenizer.load(path)
            self.word_model_path = path
            print(f"\n✓ Модель загружена. Размер словаря: {self.word_tokenizer.get_vocab_size()}")
        except Exception as e:
            print(f"\n✗ Ошибка: {e}")
        input("Нажмите Enter...")

    def save_word_model(self):
        """Сохранение словной модели"""
        if not self.word_tokenizer:
            print("\n✗ Сначала создайте или загрузите словный токенизатор.")
            input("Нажмите Enter...")
            return
        self.print_header("СОХРАНЕНИЕ СЛОВНОЙ МОДЕЛИ")
        path = input("Путь для сохранения (.pkl): ").strip()
        if not path:
            print("\n✗ Путь не указан.")
            input("Нажмите Enter...")
            return
        if not path.endswith('.pkl'):
            path += '.pkl'
        try:
            self.word_tokenizer.save(path)
            self.word_model_path = path
            print(f"\n✓ Сохранено: {path}")
        except Exception as e:
            print(f"\n✗ Ошибка: {e}")
        input("Нажмите Enter...")

    def word_show_tokens_console(self):
        """Показать разбиение текста на слова (без модели)"""
        self.print_header("ТОКЕНЫ ПО СЛОВАМ (БЕЗ BPE)")
        print("Введите текст — будет показано разбиение на слова. Пустая строка — выход.")
        print("-" * 70)
        while True:
            text = input("\nТекст: ").strip()
            if not text:
                break
            tokens = word_tokenize(text)
            print(f"  Токены: {tokens}")
            print(f"  Количество: {len(tokens)}")
        input("Нажмите Enter...")

    def word_encode_decode_console(self):
        """Кодирование текста в числа и обратно (для проверки LLM/embedding)"""
        if not self.word_tokenizer:
            print("\n✗ Сначала создайте или загрузите словный токенизатор (п. 1–3).")
            input("Нажмите Enter...")
            return
        self.print_header("ТЕКСТ -> ЧИСЛА -> ТЕКСТ")
        print("Введите текст. Будет показана последовательность id и декодированный текст.")
        print("Пустая строка — выход.")
        print("-" * 70)
        while True:
            text = input("\nТекст: ").strip()
            if not text:
                break
            ids = self.word_tokenizer.encode(text)
            decoded = self.word_tokenizer.decode(ids)
            print(f"  ID:    {ids}")
            print(f"  Декод: {decoded}")
        input("Нажмите Enter...")

    def run(self):
        """Запуск приложения"""
        while True:
            self.print_menu()
            
            choice = input("Выберите действие: ").strip()
            
            if choice == '0':
                if self.tokenizer and self.current_model_path:
                    save = input("\nСохранить BPE-модель перед выходом? (y/n): ").strip().lower() == 'y'
                    if save:
                        self.save_model()
                if self.word_tokenizer:
                    save_w = input("Сохранить словную модель (word-level) перед выходом? (y/n): ").strip().lower() == 'y'
                    if save_w:
                        self.save_word_model()
                print("\nДо свидания!")
                break
            elif choice == '1':
                self.create_tokenizer()
            elif choice == '2':
                self.load_model()
            elif choice == '3':
                self.train_from_input()
            elif choice == '4':
                self.continue_training_from_input()
            elif choice == '5':
                self.train_from_file()
            elif choice == '6':
                self.continue_training_from_file()
            elif choice == '7':
                self.find_new_pairs()
            elif choice == '8':
                self.test_tokenizer()
            elif choice == '9':
                self.show_model_info()
            elif choice == '10':
                self.save_model()
            elif choice == '11':
                self.interactive_test()
            elif choice == '12':
                self.word_tokenizer_menu()
            else:
                print("\n✗ Неверный выбор. Попробуйте снова.")
                input("Нажмите Enter для продолжения...")


if __name__ == "__main__":
    app = TokenizerTrainerApp()
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n\nПрервано пользователем. Выход...")
        sys.exit(0)

