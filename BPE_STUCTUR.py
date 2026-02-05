import re 
import pickle
import unicodedata
try:
    import regex
    HAS_REGEX = True
except ImportError:
    import re as regex
    HAS_REGEX = False
from collections import defaultdict,Counter
from typing import List,Tuple,Dict,Set,Optional

class BPETokenizer:
    def __init__(self,vocab_size:int=30000):
        if vocab_size <= 0:
            raise ValueError("vocab_size должен быть больше 0")
        
        self.vocab_size = vocab_size
        self.vocab = {} #id -> token
        self.inverse_vocab = {} #token -> id
        self.merges = {} #token,token -> token
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
            '<SEP>': 4,
        }
        # Регулярное выражение для разбиения текста на элементы перед токенизацией
        # Использует Unicode свойства \p{L} (буквы) и \p{N} (цифры)
        if HAS_REGEX:
            self.pattern_string = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            self.pattern = regex.compile(self.pattern_string)
        else:
            # Fallback для стандартного re (эквиваленты Unicode свойств)
            self.pattern_string = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"""
            self.pattern = re.compile(self.pattern_string, re.UNICODE)
        
        # Порядок слияний для правильного применения при encode
        self.merge_order = []  # список пар в порядке их создания

    def add_special_tokens(self, tokens: Dict[str, int]):
        """
        Добавление специальных токенов
        
        Args:
            tokens: Словарь {токен: id}
        """
        for token, token_id in tokens.items():
            self.special_tokens[token] = token_id
    
    def preprocess_text(self, text: str) -> str:
        """
        Предобработка текста перед токенизацией
        
        Args:
            text: Входной текст
            
        Returns:
            Обработанный текст
        """
        # Unicode нормализация (NFD - каноническая декомпозиция)
        try:
            text = unicodedata.normalize('NFD', text)
        except Exception:
            # Если нормализация не удалась, продолжаем без неё
            pass
        
        # Приведение к нижнему регистру (опционально)
        text = text.lower()
        
        # Нормализация пробелов
        text = re.sub(r'\s+', ' ', text)
        
        # Добавление пробелов вокруг знаков препинания
        text = re.sub(r'([.,!?;:()"«»])', r' \1 ', text)
        
        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def split_text_with_pattern(self, text: str) -> List[str]:
        """
        Разбиение текста на элементы с помощью регулярного выражения
        
        Args:
            text: Входной текст
            
        Returns:
            Список элементов текста после разбиения
        """
        # Используем регулярное выражение для разбиения
        matches = self.pattern.findall(text)
        # Фильтруем пустые строки
        elements = [match for match in matches if match.strip()]
        return elements
    
    def get_stats(self, words: List[List[str]]) -> Dict[Tuple[str, str], int]:
        """
        Подсчет частот пар символов
        
        Args:
            words: Список слов, разбитых на символы
            
        Returns:
            Словарь с частотами пар
        """
        stats = defaultdict(int)
        
        for word in words:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                stats[pair] += 1
                
        return stats
    def merge_pair(self, pair: Tuple[str, str], words: List[List[str]]) -> List[List[str]]:
        """
        Объединение наиболее частой пары токенов
        
        Args:
            pair: Пара для объединения
            words: Список слов, разбитых на токены
            
        Returns:
            Обновленный список слов
        """
        new_words = []
        
        for word in words:
            new_word = []
            i = 0
            
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
                    
            new_words.append(new_word)
            
        return new_words
    def train(self, corpus: List[str], verbose: bool = False, 
              checkpoint_path: Optional[str] = None, 
              checkpoint_interval: int = 100):
        """
        Обучение BPE-токенизатора на корпусе текстов с поддержкой чекпоинтов
        
        Args:
            corpus: Список текстов для обучения
            verbose: Вывод информации о процессе обучения
            checkpoint_path: Путь для сохранения чекпоинтов (None = не сохранять)
            checkpoint_interval: Сохранять чекпоинт каждые N итераций
        """
        # Валидация входных данных
        if not corpus:
            raise ValueError("Корпус не может быть пустым")
        
        # Шаг 1: Предобработка текстов
        processed_texts = [self.preprocess_text(text) for text in corpus]
        
        # Шаг 2: Разбиение текста на элементы с помощью регулярного выражения
        words = []
        for text in processed_texts:
            elements = self.split_text_with_pattern(text)
            words.extend(elements)
        
        # Шаг 3: Начальный словарь - все символы
        vocab = set()
        for word in words:
            vocab.update(word)
        
        # Добавляем символ конца слова
        vocab.add('</w>')
        
        # Шаг 4: Разбиваем слова на символы
        word_tokens = []
        for word in words:
            tokenized_word = list(word) + ['</w>']
            word_tokens.append(tokenized_word)
            vocab.update(tokenized_word)
        
        # Шаг 5: Итеративное обучение BPE
        merges = {}
        merge_order = []  # Сохраняем порядок слияний
        # Используем sorted() для детерминированного порядка токенов
        current_vocab = sorted(list(vocab))
        
        max_iterations = self.vocab_size - len(vocab)
        iteration = 0
        
        try:
            for iteration in range(max_iterations):
                # Подсчет статистики
                stats = self.get_stats(word_tokens)
                
                if not stats:
                    break
                    
                # Находим наиболее частую пару
                best_pair = max(stats, key=stats.get)
                best_freq = stats[best_pair]
                
                if best_freq < 2:
                    break
                    
                # Создаем новый токен
                new_token = best_pair[0] + best_pair[1]
                
                # Обновляем словари
                token_id = len(current_vocab)
                current_vocab.append(new_token)
                merges[best_pair] = new_token
                merge_order.append(best_pair)  # Сохраняем порядок
                
                # Объединяем пару в словах
                word_tokens = self.merge_pair(best_pair, word_tokens)
                
                if verbose and iteration % 50 == 0:
                    print(f"Iteration {iteration}: Merged {best_pair} -> {new_token}")
                
                # Автоматическое сохранение чекпоинта
                if checkpoint_path and iteration > 0 and iteration % checkpoint_interval == 0:
                    temp_merges = dict(merges)
                    temp_merge_order = list(merge_order)
                    temp_vocab = list(current_vocab)
                    self._build_vocabulary(temp_vocab, temp_merges, temp_merge_order)
                    self.save(checkpoint_path)
                    if verbose:
                        print(f"💾 Чекпоинт сохранен (итерация {iteration})")
            
            # Шаг 6: Создаем финальный словарь
            self._build_vocabulary(current_vocab, merges, merge_order)
            
            if checkpoint_path:
                self.save(checkpoint_path)
                if verbose:
                    print(f"✅ Обучение завершено. Модель сохранена: {checkpoint_path}")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Обучение прервано пользователем (Ctrl+C)")
            
            # Сохраняем текущий прогресс
            if merges and merge_order:
                temp_merges = dict(merges)
                temp_merge_order = list(merge_order)
                temp_vocab = list(current_vocab)
                self._build_vocabulary(temp_vocab, temp_merges, temp_merge_order)
                
                if checkpoint_path:
                    self.save(checkpoint_path)
                    print(f"✅ Прогресс сохранен в: {checkpoint_path}")
                    print(f"   Завершено итераций: {iteration}")
                    print(f"   Токенов: {len(temp_vocab)}, Слияний: {len(temp_merges)}")
                    print(f"💡 Вы можете продолжить обучение, загрузив этот файл")
                else:
                    print("⚠️  Укажите checkpoint_path для сохранения прогресса")
                    print("💡 Используйте: tokenizer.train(corpus, checkpoint_path='model.pkl')")
            
            raise  # Пробрасываем исключение дальше
    
    def continue_training(self, corpus: List[str], verbose: bool = False, 
                          max_new_merges: Optional[int] = None,
                          checkpoint_path: Optional[str] = None, 
                          checkpoint_interval: int = 100):
        """
        Дообучение токенизатора на новом корпусе с расширением словаря
        
        Args:
            corpus: Список новых текстов для дообучения
            verbose: Вывод информации о процессе обучения
            max_new_merges: Максимальное количество новых слияний (None = до vocab_size)
            checkpoint_path: Путь для сохранения чекпоинтов (None = не сохранять)
            checkpoint_interval: Сохранять чекпоинт каждые N итераций
        """
        if not hasattr(self, 'vocab') or not self.vocab:
            raise ValueError("Токенизатор должен быть обучен перед дообучением. Используйте train() сначала.")
        
        # Шаг 1: Предобработка новых текстов
        processed_texts = [self.preprocess_text(text) for text in corpus]
        
        # Шаг 2: Разбиение текста на элементы
        words = []
        for text in processed_texts:
            elements = self.split_text_with_pattern(text)
            words.extend(elements)
        
        # Шаг 3: Начинаем с существующего словаря
        existing_tokens = set(self.inverse_vocab.keys())
        existing_tokens.discard('</w>')  # Убираем служебный токен из проверки
        
        # Собираем все символы из новых слов
        new_chars = set()
        for word in words:
            new_chars.update(word)
        
        # Добавляем новые символы в словарь
        vocab = set(existing_tokens)
        vocab.update(new_chars)
        vocab.add('</w>')
        
        # Шаг 4: Разбиваем слова на токены, используя существующие слияния
        word_tokens = []
        for word in words:
            # Начинаем с символов
            tokens = list(word) + ['</w>']
            
            # Применяем существующие слияния
            if hasattr(self, 'merge_order') and self.merge_order:
                for pair in self.merge_order:
                    if pair not in self.merges:
                        continue
                    new_tokens = []
                    i = 0
                    while i < len(tokens):
                        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                            new_tokens.append(self.merges[pair])
                            i += 2
                        else:
                            new_tokens.append(tokens[i])
                            i += 1
                    tokens = new_tokens
                    if len(tokens) == 1:
                        break
            
            word_tokens.append(tokens)
            # Добавляем все токены в словарь
            vocab.update(tokens)
        
        # Шаг 5: Продолжаем обучение с существующими слияниями
        merges = dict(self.merges) if hasattr(self, 'merges') and self.merges else {}
        merge_order = list(self.merge_order) if hasattr(self, 'merge_order') and self.merge_order else []
        # Используем sorted() для детерминированного порядка токенов
        current_vocab = sorted(list(vocab))
        
        # Определяем сколько новых слияний можно добавить
        current_vocab_size = len(current_vocab)
        if max_new_merges is None:
            max_new_merges = max(0, self.vocab_size - current_vocab_size)
        
        iterations = min(max_new_merges, self.vocab_size - current_vocab_size)
        
        try:
            for iteration in range(iterations):
                # Подсчет статистики
                stats = self.get_stats(word_tokens)
                
                if not stats:
                    break
                
                # Фильтруем пары, которые уже были объединены
                new_stats = {pair: freq for pair, freq in stats.items() if pair not in merges}
                
                if not new_stats:
                    break  # Все пары уже объединены
                
                # Находим наиболее частую пару из новых
                best_pair = max(new_stats, key=new_stats.get)
                best_freq = new_stats[best_pair]
                
                if best_freq < 2:
                    break
                
                # Создаем новый токен
                new_token = best_pair[0] + best_pair[1]
                
                # Обновляем словари
                if new_token not in current_vocab:
                    current_vocab.append(new_token)
                merges[best_pair] = new_token
                merge_order.append(best_pair)  # Добавляем в конец порядка
                
                # Объединяем пару в словах
                word_tokens = self.merge_pair(best_pair, word_tokens)
                
                if verbose and iteration % 50 == 0:
                    print(f"Continue training iteration {iteration}: Merged {best_pair} -> {new_token}")
                
                # Автоматическое сохранение чекпоинта
                if checkpoint_path and iteration > 0 and iteration % checkpoint_interval == 0:
                    temp_merges = dict(merges)
                    temp_merge_order = list(merge_order)
                    temp_vocab = list(current_vocab)
                    self._build_vocabulary(temp_vocab, temp_merges, temp_merge_order)
                    self.save(checkpoint_path)
                    if verbose:
                        print(f"💾 Чекпоинт сохранен (итерация {iteration})")
            
            # Шаг 6: Обновляем словарь
            self._build_vocabulary(current_vocab, merges, merge_order)
            
            if checkpoint_path:
                self.save(checkpoint_path)
                if verbose:
                    print(f"✅ Дообучение завершено. Модель сохранена: {checkpoint_path}")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Дообучение прервано пользователем (Ctrl+C)")
            
            # Сохраняем текущий прогресс
            if merges and merge_order:
                temp_merges = dict(merges)
                temp_merge_order = list(merge_order)
                temp_vocab = list(current_vocab)
                self._build_vocabulary(temp_vocab, temp_merges, temp_merge_order)
                
                if checkpoint_path:
                    self.save(checkpoint_path)
                    print(f"✅ Прогресс сохранен в: {checkpoint_path}")
                    print(f"   Завершено итераций: {iteration}")
                    print(f"   Токенов: {len(temp_vocab)}, Слияний: {len(temp_merges)}")
                    print(f"💡 Вы можете продолжить дообучение, загрузив этот файл")
                else:
                    print("⚠️  Укажите checkpoint_path для сохранения прогресса")
            
            raise  # Пробрасываем исключение дальше
    
    def find_new_pairs_in_vocab(self, corpus: List[str], max_new_merges: Optional[int] = None,
                                 verbose: bool = False, checkpoint_path: Optional[str] = None,
                                 checkpoint_interval: int = 100) -> int:
        """
        Поиск новых пар для слияния в существующем словаре на основе корпуса
        
        Анализирует корпус текстов и ищет пары токенов из существующего словаря,
        которые часто встречаются вместе и могут быть объединены.
        
        Args:
            corpus: Список текстов для анализа
            max_new_merges: Максимальное количество новых слияний (None = до vocab_size)
            verbose: Вывод информации о процессе
            checkpoint_path: Путь для сохранения чекпоинтов (None = не сохранять)
            checkpoint_interval: Сохранять чекпоинт каждые N итераций
        
        Returns:
            Количество найденных и добавленных новых слияний
        """
        if not hasattr(self, 'vocab') or not self.vocab:
            raise ValueError("Токенизатор должен быть обучен. Используйте train() или load() сначала.")
        
        if not corpus:
            raise ValueError("Корпус не может быть пустым")
        
        # Шаг 1: Предобработка текстов
        processed_texts = [self.preprocess_text(text) for text in corpus]
        
        # Шаг 2: Разбиение текста на элементы
        words = []
        for text in processed_texts:
            elements = self.split_text_with_pattern(text)
            words.extend(elements)
        
        # Шаг 3: Токенизация слов с использованием существующих слияний
        word_tokens = []
        for word in words:
            # Начинаем с символов
            tokens = list(word) + ['</w>']
            
            # Применяем существующие слияния
            if hasattr(self, 'merge_order') and self.merge_order:
                for pair in self.merge_order:
                    if pair not in self.merges:
                        continue
                    new_tokens = []
                    i = 0
                    while i < len(tokens):
                        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                            new_tokens.append(self.merges[pair])
                            i += 2
                        else:
                            new_tokens.append(tokens[i])
                            i += 1
                    tokens = new_tokens
                    if len(tokens) == 1:
                        break
            
            word_tokens.append(tokens)
        
        # Шаг 4: Инициализация существующих слияний
        merges = dict(self.merges) if hasattr(self, 'merges') and self.merges else {}
        merge_order = list(self.merge_order) if hasattr(self, 'merge_order') and self.merge_order else []
        
        # Шаг 5: Определяем сколько новых слияний можно добавить
        current_vocab_size = len(self.vocab)
        if max_new_merges is None:
            max_new_merges = max(0, self.vocab_size - current_vocab_size)
        
        iterations = min(max_new_merges, self.vocab_size - current_vocab_size)
        
        new_merges_count = 0
        
        try:
            for iteration in range(iterations):
                # Подсчет статистики пар в токенизированных словах
                stats = self.get_stats(word_tokens)
                
                if not stats:
                    break
                
                # Фильтруем пары, которые уже были объединены
                new_stats = {pair: freq for pair, freq in stats.items() if pair not in merges}
                
                if not new_stats:
                    break  # Все пары уже объединены
                
                # Находим наиболее частую пару из новых
                best_pair = max(new_stats, key=new_stats.get)
                best_freq = new_stats[best_pair]
                
                if best_freq < 2:
                    break
                
                # Проверяем, что оба токена из пары существуют в словаре
                token1, token2 = best_pair
                if token1 not in self.inverse_vocab or token2 not in self.inverse_vocab:
                    # Пропускаем пары с несуществующими токенами
                    continue
                
                # Создаем новый токен
                new_token = token1 + token2
                
                # Проверяем, не превысит ли это размер словаря
                if len(self.vocab) >= self.vocab_size:
                    break
                
                # Обновляем словари
                merges[best_pair] = new_token
                merge_order.append(best_pair)
                
                # Обновляем word_tokens
                word_tokens = self.merge_pair(best_pair, word_tokens)
                
                new_merges_count += 1
                
                if verbose:
                    print(f"Итерация {iteration + 1}: Объединена пара {best_pair} (частота: {best_freq}) → {new_token}")
                
                # Чекпоинт
                if checkpoint_path and (iteration + 1) % checkpoint_interval == 0:
                    # Временно обновляем словарь для сохранения
                    temp_merges = dict(merges)
                    temp_merge_order = list(merge_order)
                    # Собираем все существующие токены
                    temp_vocab = sorted(list(set(self.inverse_vocab.keys())))
                    # Добавляем новые токены из слияний
                    for merge_token in temp_merges.values():
                        if merge_token not in temp_vocab:
                            temp_vocab.append(merge_token)
                    temp_vocab = sorted(list(set(temp_vocab)))
                    
                    self._build_vocabulary(temp_vocab, temp_merges, temp_merge_order)
                    self.save(checkpoint_path)
                    if verbose:
                        print(f"💾 Чекпоинт сохранен: {checkpoint_path}")
                        print(f"   Токенов: {len(self.vocab)}, Слияний: {len(merges)}")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Поиск новых пар прерван")
            if checkpoint_path and merges:
                # Сохраняем текущее состояние
                temp_merges = dict(merges)
                temp_merge_order = list(merge_order)
                # Собираем все существующие токены
                temp_vocab = sorted(list(set(self.inverse_vocab.keys())))
                # Добавляем новые токены из слияний
                for merge_token in temp_merges.values():
                    if merge_token not in temp_vocab:
                        temp_vocab.append(merge_token)
                temp_vocab = sorted(list(set(temp_vocab)))
                
                self._build_vocabulary(temp_vocab, temp_merges, temp_merge_order)
                self.save(checkpoint_path)
                print(f"✅ Последний чекпоинт сохранен в: {checkpoint_path}")
                print(f"   Найдено новых слияний: {new_merges_count}")
                print(f"💡 Вы можете продолжить поиск, загрузив этот файл")
            raise
        
        # Шаг 6: Обновляем словарь с новыми слияниями
        if merges and merge_order:
            # Собираем все существующие токены
            all_tokens = sorted(list(set(self.inverse_vocab.keys())))
            # Добавляем новые токены из слияний
            for merge_token in merges.values():
                if merge_token not in all_tokens:
                    all_tokens.append(merge_token)
            all_tokens = sorted(list(set(all_tokens)))
            
            self._build_vocabulary(all_tokens, merges, merge_order)
            
            if checkpoint_path:
                self.save(checkpoint_path)
                if verbose:
                    print(f"💾 Финальный чекпоинт сохранен: {checkpoint_path}")
        
        return new_merges_count
    
    def _build_vocabulary(self, tokens: List[str], merges: Dict[Tuple[str, str], str], merge_order: List[Tuple[str, str]]):
        """
        Построение словаря токенов
        
        Args:
            tokens: Список всех токенов
            merges: Словарь слияний
            merge_order: Порядок слияний
        """
        # Начинаем с базового словаря
        self.vocab = {}
        self.inverse_vocab = {}
        self.merges = merges
        self.merge_order = merge_order  # Сохраняем порядок слияний
        
        # Добавляем специальные токены
        for token, token_id in self.special_tokens.items():
            self.vocab[token_id] = token
            self.inverse_vocab[token] = token_id
        
        # Добавляем обычные токены
        next_id = max(self.special_tokens.values(), default=-1) + 1
        
        # Удаляем дубликаты для эффективности
        unique_tokens = set(tokens)
        for token in sorted(unique_tokens):  # Сортируем для детерминированности
            if token not in self.inverse_vocab:
                self.vocab[next_id] = token
                self.inverse_vocab[token] = next_id
                next_id += 1
    def encode(self, text: str) -> List[int]:
        """
        Кодирование текста в список токенов
        
        Args:
            text: Входной текст
            
        Returns:
            Список идентификаторов токенов
        """
        try:
            # Предобработка
            text = self.preprocess_text(text)
            
            # Разбиение текста на элементы с помощью регулярного выражения
            elements = self.split_text_with_pattern(text)
            
            # Кодирование каждого элемента
            token_ids = []
            
            for word in elements:
                # Начинаем с символов
                tokens = list(word) + ['</w>']
                
                # Применяем слияния в правильном порядке (как при обучении)
                if hasattr(self, 'merge_order') and self.merge_order:
                    # Применяем слияния в порядке их создания
                    for pair in self.merge_order:
                        if pair not in self.merges:
                            continue
                        
                        new_tokens = []
                        i = 0
                        while i < len(tokens):
                            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                                new_tokens.append(self.merges[pair])
                                i += 2
                            else:
                                new_tokens.append(tokens[i])
                                i += 1
                        tokens = new_tokens
                        
                        # Если больше нет пар для слияния, выходим
                        if len(tokens) == 1:
                            break
                else:
                    # Fallback: старый алгоритм (если merge_order не задан)
                    while len(tokens) > 1:
                        pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
                        found_pair = None
                        for pair in pairs:
                            if pair in self.merges:
                                found_pair = pair
                                break
                        
                        if not found_pair:
                            break
                        
                        new_tokens = []
                        i = 0
                        while i < len(tokens):
                            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == found_pair:
                                new_tokens.append(self.merges[found_pair])
                                i += 2
                            else:
                                new_tokens.append(tokens[i])
                                i += 1
                        tokens = new_tokens
                
                # Добавляем токены слова к общему списку
                for token in tokens:
                    if token in self.inverse_vocab:
                        token_ids.append(self.inverse_vocab[token])
                    else:
                        # Если токен не найден, разбиваем на символы
                        for char in token:
                            if char in self.inverse_vocab:
                                token_ids.append(self.inverse_vocab[char])
                            else:
                                # Обработка неизвестных символов
                                token_ids.append(self.inverse_vocab.get('<UNK>', 0))
            
            return token_ids
        except Exception as e:
            # Обработка ошибок при кодировании
            print(f"Ошибка при кодировании текста: {e}")
            return [self.inverse_vocab.get('<UNK>', 0)]
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Декодирование списка токенов в текст
        
        Args:
            token_ids: Список идентификаторов токенов
            
        Returns:
            Декодированный текст
        """
        try:
            tokens = []
            
            for token_id in token_ids:
                if token_id in self.vocab:
                    tokens.append(self.vocab[token_id])
                else:
                    tokens.append('<UNK>')
            
            # Объединяем токены
            text = ''.join(tokens)
            
            # Убираем маркеры конца слова и восстанавливаем пробелы
            # Сохраняем информацию о пробелах более аккуратно
            text = text.replace('</w>', ' ')
            
            # Восстанавливаем знаки препинания (убираем пробелы перед ними)
            text = re.sub(r'\s+([.,!?;:()"«»])', r'\1', text)
            
            # Нормализуем множественные пробелы, но сохраняем одиночные
            text = re.sub(r' {2,}', ' ', text)
            text = text.strip()
            
            return text
        except Exception as e:
            # Обработка ошибок при декодировании
            print(f"Ошибка при декодировании токенов: {e}")
            return ""
    def train_on_file(self, file_path: str, encoding: str = 'utf-8'):
        """
        Обучение на файле
        
        Args:
            file_path: Путь к файлу с корпусом
            encoding: Кодировка файла
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                corpus = f.readlines()
            
            self.train(corpus)
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл {file_path} не найден")
        except Exception as e:
            raise Exception(f"Ошибка при чтении файла: {e}")
    
    def save(self, path: str):
        """
        Сохранение токенизатора
        
        Args:
            path: Путь для сохранения
        """
        try:
            data = {
                'vocab_size': self.vocab_size,
                'vocab': self.vocab,
                'inverse_vocab': self.inverse_vocab,
                'merges': self.merges,
                'special_tokens': self.special_tokens,
                'pattern_string': getattr(self, 'pattern_string', None),
                'has_regex': HAS_REGEX,
                'merge_order': getattr(self, 'merge_order', []),
            }
            
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            raise Exception(f"Ошибка при сохранении токенизатора: {e}")
    
    def load(self, path: str):
        """
        Загрузка токенизатора
        
        Args:
            path: Путь к файлу с сохраненным токенизатором
        """
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            self.vocab_size = data['vocab_size']
            self.vocab = data['vocab']
            self.inverse_vocab = data['inverse_vocab']
            self.merges = data['merges']
            self.special_tokens = data['special_tokens']
            
            # Восстанавливаем паттерн regex
            pattern_string = data.get('pattern_string')
            if pattern_string:
                self.pattern_string = pattern_string
                if data.get('has_regex', HAS_REGEX) and HAS_REGEX:
                    self.pattern = regex.compile(pattern_string)
                else:
                    self.pattern = re.compile(pattern_string, re.UNICODE)
            
            # Восстанавливаем порядок слияний
            self.merge_order = data.get('merge_order', [])
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл {path} не найден")
        except Exception as e:
            raise Exception(f"Ошибка при загрузке токенизатора: {e}")
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Пакетное кодирование
        """
        return [self.encode(text) for text in texts]
    
    def decode_batch(self, batch_token_ids: List[List[int]]) -> List[str]:
        """
        Пакетное декодирование
        """
        return [self.decode(token_ids) for token_ids in batch_token_ids]
    
    def get_vocab_size(self) -> int:
        """
        Получение размера словаря
        """
        return len(self.vocab)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Токенизация без кодирования в IDs
        """
        token_ids = self.encode(text)
        return [self.vocab[token_id] for token_id in token_ids]