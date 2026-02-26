"""
Интерфейс для тестирования и оценки GPT модели
Включает генерацию текста, оценку качества и интерактивное тестирование
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys
import os
import math
from typing import List, Tuple, Optional, Dict
import json
from collections import Counter

# Добавляем родительскую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BPE_STUCTUR import BPETokenizer
from TRANSFORMER import GPTModel


class GPTTester:
    """Класс для тестирования и оценки GPT модели"""
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str = "chekpoint.pkl",
        device: str = None
    ):
        """
        Инициализация тестера
        
        Args:
            model_path: Путь к чекпоинту модели (.pth)
            tokenizer_path: Путь к токенизатору (.pkl)
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
        # ВАЖНО: всегда используем размер словаря токенизатора, а не из чекпоинта
        # так как модель должна работать с текущим токенизатором
        vocab_size = self.tokenizer.get_vocab_size()
        embedding_dim = checkpoint.get('embedding_dim', 256)
        num_layers = checkpoint.get('num_layers', 6)
        num_heads = checkpoint.get('num_heads', 8)
        max_seq_len = checkpoint.get('max_seq_len', 512)
        
        # Определяем параметры из state_dict
        state_dict = checkpoint['model_state_dict']
        
        # Определяем learnable_pos и max_seq_len из структуры state_dict
        learnable_pos_key = "embedding.positional_encoding.pos_encoding.position_embedding.weight"
        sinusoidal_key = "embedding.positional_encoding.pos_encoding.pe"
        
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
            print(f"⚠️  Используется значение по умолчанию: learnable_pos={learnable_pos}, max_seq_len={max_seq_len}")
        
        # Определяем layer_norm
        layer_norm_key = "embedding.layer_norm.weight"
        if layer_norm_key in state_dict:
            layer_norm = True
        else:
            layer_norm = checkpoint.get('layer_norm', True)
        
        print(f"✓ Параметры модели:")
        print(f"   vocab_size: {vocab_size}")
        print(f"   embedding_dim: {embedding_dim}")
        print(f"   num_layers: {num_layers}")
        print(f"   num_heads: {num_heads}")
        print(f"   max_seq_len: {max_seq_len}")
        print(f"   learnable_pos: {learnable_pos}")
        print(f"   layer_norm: {layer_norm}")
        
        # Создание EmbeddingLayer с правильными параметрами
        from EMBEDDING_LAYER.embedding_layer import create_embedding_from_tokenizer
        
        embedding_layer = create_embedding_from_tokenizer(
            self.tokenizer,
            embedding_dim=embedding_dim,
            max_seq_len=max_seq_len,
            learnable_pos=learnable_pos,
            layer_norm=layer_norm
        )
        
        # Создание модели с правильным EmbeddingLayer
        self.model = GPTModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            embedding_layer=embedding_layer  # Передаем созданный embedding_layer
        )
        
        # Загрузка весов
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print(f"✓ Веса модели загружены успешно")
        except RuntimeError as e:
            # Если строгая загрузка не удалась, пробуем нестрогую с фильтрацией
            print(f"⚠️  Предупреждение при загрузке весов: {e}")
            print(f"⚠️  Попытка нестрогой загрузки с фильтрацией несовпадающих размеров...")
            
            # Фильтруем state_dict, удаляя ключи с несовпадающими размерами
            model_state_dict = self.model.state_dict()
            filtered_state_dict = {}
            
            for key, value in checkpoint['model_state_dict'].items():
                if key in model_state_dict:
                    # Проверяем совпадение размеров
                    if model_state_dict[key].shape == value.shape:
                        filtered_state_dict[key] = value
                    else:
                        print(f"⚠️  Пропущен ключ '{key}': размеры не совпадают "
                              f"(чекпоинт: {value.shape}, модель: {model_state_dict[key].shape})")
                else:
                    # Ключ отсутствует в модели - пропускаем
                    pass
            
            # Загружаем отфильтрованный state_dict
            missing_keys, unexpected_keys = self.model.load_state_dict(
                filtered_state_dict, strict=False
            )
            if missing_keys:
                print(f"⚠️  Отсутствующие ключи (игнорируются): {len(missing_keys)} ключей")
                if len(missing_keys) <= 10:
                    for key in missing_keys:
                        print(f"      - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"      - {key}")
                    print(f"      ... и еще {len(missing_keys) - 5} ключей")
            if unexpected_keys:
                print(f"⚠️  Неожиданные ключи (игнорируются): {len(unexpected_keys)} ключей")
                if len(unexpected_keys) <= 10:
                    for key in unexpected_keys:
                        print(f"      - {key}")
                else:
                    for key in unexpected_keys[:5]:
                        print(f"      - {key}")
                    print(f"      ... и еще {len(unexpected_keys) - 5} ключей")
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Модель загружена и переведена в режим оценки")
        print(f"   Параметров: {self.model.get_num_params():,} ({self.model.get_num_params_millions():.2f}M)")
        print(f"   Устройство: {self.device}")
        
        # Специальные токены
        self.pad_id = self.tokenizer.special_tokens.get('<PAD>', 0)
        self.bos_id = self.tokenizer.special_tokens.get('<BOS>', None)
        self.eos_id = self.tokenizer.special_tokens.get('<EOS>', None)
        self.unk_id = self.tokenizer.special_tokens.get('<UNK>', None)
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        stop_tokens: Optional[List[int]] = None
    ) -> str:
        """
        Генерация текста из промпта
        
        Args:
            prompt: Начальный текст (промпт)
            max_length: Максимальная длина генерируемого текста
            temperature: Температура для сэмплирования (выше = более случайно)
            top_k: Количество топ токенов для рассмотрения
            top_p: Nucleus sampling (cumulative probability)
            repetition_penalty: Штраф за повторения (>1.0 уменьшает повторения)
            stop_tokens: Список токенов для остановки генерации
        
        Returns:
            Сгенерированный текст
        """
        self.model.eval()
        
        # Кодирование промпта
        token_ids = self.tokenizer.encode(prompt)
        if len(token_ids) == 0:
            return prompt
        
        # Конвертация в тензор
        input_ids = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        generated = token_ids.copy()
        
        # Словарь для отслеживания повторений
        token_counts = Counter(generated)
        
        stop_tokens = stop_tokens or []
        if self.eos_id is not None:
            stop_tokens.append(self.eos_id)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits = self.model(input_ids)  # [1, seq_len, vocab_size]
                
                # Берем логиты для последнего токена
                next_token_logits = logits[0, -1, :]  # [vocab_size]
                
                # Применяем repetition penalty
                if repetition_penalty != 1.0:
                    for token_id, count in token_counts.items():
                        if next_token_logits[token_id] > 0:
                            next_token_logits[token_id] /= (repetition_penalty ** count)
                
                # Применяем temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Top-k фильтрация
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) фильтрация
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Удаляем токены с cumulative probability выше threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Сэмплирование
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                # Проверка на стоп-токены
                if next_token in stop_tokens:
                    break
                
                # Добавляем токен
                generated.append(next_token)
                token_counts[next_token] = token_counts.get(next_token, 0) + 1
                
                # Обновляем input_ids для следующей итерации
                input_ids = torch.tensor([generated], dtype=torch.long).to(self.device)
                
                # Проверка на максимальную длину последовательности
                if len(generated) >= self.model.embedding.max_seq_len:
                    break
        
        # Декодирование
        generated_text = self.tokenizer.decode(generated)
        return generated_text
    
    def calculate_perplexity(
        self,
        texts: List[str],
        batch_size: int = 8,
        max_length: int = 512
    ) -> float:
        """
        Вычисление perplexity на наборе текстов
        
        Args:
            texts: Список текстов для оценки
            batch_size: Размер батча
            max_length: Максимальная длина последовательности
        
        Returns:
            Perplexity (чем меньше, тем лучше)
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id, reduction='sum')
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_inputs = []
                batch_targets = []
                
                for text in batch_texts:
                    token_ids = self.tokenizer.encode(text)
                    
                    if len(token_ids) > max_length:
                        token_ids = token_ids[:max_length]
                    
                    if len(token_ids) < 2:
                        continue
                    
                    # Input: все кроме последнего, Target: все кроме первого
                    input_ids = token_ids[:-1]
                    target_ids = token_ids[1:]
                    
                    # Паддинг
                    pad_length = max_length - 1 - len(input_ids)
                    if pad_length > 0:
                        input_ids = input_ids + [self.pad_id] * pad_length
                        target_ids = target_ids + [self.pad_id] * pad_length
                    
                    batch_inputs.append(input_ids)
                    batch_targets.append(target_ids)
                
                if not batch_inputs:
                    continue
                
                # Конвертация в тензоры
                input_tensor = torch.tensor(batch_inputs, dtype=torch.long).to(self.device)
                target_tensor = torch.tensor(batch_targets, dtype=torch.long).to(self.device)
                
                # Forward pass
                logits = self.model(input_tensor)  # [batch, seq_len, vocab_size]
                
                # Reshape для loss
                logits_flat = logits.view(-1, logits.size(-1))
                target_flat = target_tensor.view(-1)
                
                # Loss
                loss = criterion(logits_flat, target_flat)
                
                # Подсчет не-pad токенов
                non_pad_mask = (target_flat != self.pad_id)
                num_tokens = non_pad_mask.sum().item()
                
                total_loss += loss.item()
                total_tokens += num_tokens
        
        if total_tokens == 0:
            return float('inf')
        
        # Perplexity = exp(average_loss)
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return perplexity
    
    def calculate_accuracy(
        self,
        texts: List[str],
        batch_size: int = 8,
        max_length: int = 512
    ) -> Dict[str, float]:
        """
        Вычисление accuracy на наборе текстов
        
        Args:
            texts: Список текстов для оценки
            batch_size: Размер батча
            max_length: Максимальная длина последовательности
        
        Returns:
            Словарь с метриками accuracy
        """
        self.model.eval()
        total_correct = 0
        total_tokens = 0
        total_sequences = 0
        correct_sequences = 0
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_inputs = []
                batch_targets = []
                
                for text in batch_texts:
                    token_ids = self.tokenizer.encode(text)
                    
                    if len(token_ids) > max_length:
                        token_ids = token_ids[:max_length]
                    
                    if len(token_ids) < 2:
                        continue
                    
                    input_ids = token_ids[:-1]
                    target_ids = token_ids[1:]
                    
                    pad_length = max_length - 1 - len(input_ids)
                    if pad_length > 0:
                        input_ids = input_ids + [self.pad_id] * pad_length
                        target_ids = target_ids + [self.pad_id] * pad_length
                    
                    batch_inputs.append(input_ids)
                    batch_targets.append(target_ids)
                
                if not batch_inputs:
                    continue
                
                input_tensor = torch.tensor(batch_inputs, dtype=torch.long).to(self.device)
                target_tensor = torch.tensor(batch_targets, dtype=torch.long).to(self.device)
                
                # Forward pass
                logits = self.model(input_tensor)
                
                # Предсказания
                predictions = torch.argmax(logits, dim=-1)
                
                # Подсчет правильных предсказаний
                non_pad_mask = (target_tensor != self.pad_id)
                correct = (predictions == target_tensor) & non_pad_mask
                
                total_correct += correct.sum().item()
                total_tokens += non_pad_mask.sum().item()
                
                # Accuracy по последовательностям
                for seq_idx in range(len(batch_texts)):
                    seq_correct = (predictions[seq_idx] == target_tensor[seq_idx]) & non_pad_mask[seq_idx]
                    if seq_correct.sum().item() == non_pad_mask[seq_idx].sum().item():
                        correct_sequences += 1
                    total_sequences += 1
        
        token_accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        sequence_accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0.0
        
        return {
            'token_accuracy': token_accuracy,
            'sequence_accuracy': sequence_accuracy
        }
    
    def evaluate_on_file(
        self,
        file_path: str,
        max_lines: int = None,
        save_report: str = None
    ) -> Dict:
        """
        Полная оценка модели на файле с текстами
        
        Args:
            file_path: Путь к файлу с текстами
            max_lines: Максимум строк для оценки
            save_report: Путь для сохранения отчета (JSON)
        
        Returns:
            Словарь с метриками
        """
        print(f"\n📊 Оценка модели на файле: {file_path}")
        
        # Загрузка текстов
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                line = line.strip()
                if line and len(line) > 10:
                    texts.append(line)
        
        print(f"✓ Загружено {len(texts)} текстов для оценки")
        
        # Вычисление метрик
        print("\n🔍 Вычисление Perplexity...")
        perplexity = self.calculate_perplexity(texts)
        print(f"   Perplexity: {perplexity:.2f}")
        
        print("\n🔍 Вычисление Accuracy...")
        accuracy = self.calculate_accuracy(texts)
        print(f"   Token Accuracy: {accuracy['token_accuracy']:.4f} ({accuracy['token_accuracy']*100:.2f}%)")
        print(f"   Sequence Accuracy: {accuracy['sequence_accuracy']:.4f} ({accuracy['sequence_accuracy']*100:.2f}%)")
        
        # Формирование отчета
        report = {
            'perplexity': perplexity,
            'token_accuracy': accuracy['token_accuracy'],
            'sequence_accuracy': accuracy['sequence_accuracy'],
            'num_texts': len(texts),
            'vocab_size': self.tokenizer.get_vocab_size(),
            'model_params': self.model.get_num_params()
        }
        
        # Сохранение отчета
        if save_report:
            with open(save_report, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Отчет сохранен: {save_report}")
        
        return report
    
    def interactive_mode(self):
        """Интерактивный режим для тестирования генерации"""
        print("\n" + "=" * 60)
        print("🎮 ИНТЕРАКТИВНЫЙ РЕЖИМ ГЕНЕРАЦИИ")
        print("=" * 60)
        print("\nВведите промпт для генерации текста.")
        print("Команды:")
        print("  'exit' или 'quit' - выход")
        print("  'params' - показать параметры генерации")
        print("  'set <param> <value>' - изменить параметр (temperature, top_k, top_p, max_length)")
        print("-" * 60)
        
        # Параметры по умолчанию
        params = {
            'temperature': 1.0,
            'top_k': 50,
            'top_p': 0.9,
            'max_length': 100,
            'repetition_penalty': 1.0
        }
        
        while True:
            try:
                prompt = input("\n💬 Промпт: ").strip()
                
                if not prompt:
                    continue
                
                if prompt.lower() in ['exit', 'quit', 'q']:
                    print("👋 Выход из интерактивного режима")
                    break
                
                if prompt.lower() == 'params':
                    print("\n📋 Текущие параметры:")
                    for key, value in params.items():
                        print(f"   {key}: {value}")
                    continue
                
                if prompt.lower().startswith('set '):
                    parts = prompt.split()
                    if len(parts) == 3:
                        param_name = parts[1]
                        param_value = float(parts[2]) if '.' in parts[2] else int(parts[2])
                        if param_name in params:
                            params[param_name] = param_value
                            print(f"✓ Параметр {param_name} установлен в {param_value}")
                        else:
                            print(f"❌ Неизвестный параметр: {param_name}")
                    continue
                
                # Генерация
                print("\n🤖 Генерация...")
                generated = self.generate(
                    prompt=prompt,
                    max_length=params['max_length'],
                    temperature=params['temperature'],
                    top_k=params['top_k'],
                    top_p=params['top_p'],
                    repetition_penalty=params['repetition_penalty']
                )
                
                print(f"\n📝 Результат:")
                print(f"   {generated}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Выход из интерактивного режима")
                break
            except Exception as e:
                print(f"\n❌ Ошибка: {e}")


def main():
    """Главная функция с меню"""
    print("=" * 60)
    print("🧪 ТЕСТЕР GPT МОДЕЛИ")
    print("=" * 60)
    
    # Параметры
    model_path = input("\nВведите путь к чекпоинту модели (.pth): ").strip()
    if not model_path:
        model_path = "gpt_model_checkpoint.pth"
    
    tokenizer_path = input("Введите путь к токенизатору (.pkl) [chekpoint.pkl]: ").strip()
    if not tokenizer_path:
        tokenizer_path = "chekpoint.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ Ошибка: файл {model_path} не найден!")
        return
    
    if not os.path.exists(tokenizer_path):
        print(f"❌ Ошибка: файл {tokenizer_path} не найден!")
        return
    
    # Создание тестера
    tester = GPTTester(model_path, tokenizer_path)
    
    # Меню
    while True:
        print("\n" + "=" * 60)
        print("МЕНЮ")
        print("=" * 60)
        print("1. Интерактивная генерация текста")
        print("2. Оценка на файле (Perplexity + Accuracy)")
        print("3. Быстрый тест генерации")
        print("4. Выход")
        print("-" * 60)
        
        choice = input("\nВыберите опцию (1-4): ").strip()
        
        if choice == '1':
            tester.interactive_mode()
        
        elif choice == '2':
            file_path = input("\nВведите путь к файлу с текстами: ").strip()
            if not file_path or not os.path.exists(file_path):
                print("❌ Файл не найден!")
                continue
            
            max_lines = input("Максимум строк (Enter для всех): ").strip()
            max_lines = int(max_lines) if max_lines else None
            
            save_report = input("Сохранить отчет? (путь или Enter для пропуска): ").strip()
            save_report = save_report if save_report else None
            
            tester.evaluate_on_file(file_path, max_lines=max_lines, save_report=save_report)
        
        elif choice == '3':
            prompt = input("\nВведите промпт: ").strip()
            if prompt:
                print("\n🤖 Генерация...")
                generated = tester.generate(prompt, max_length=100)
                print(f"\n📝 Результат:")
                print(f"   {generated}")
            else:
                print("❌ Промпт не может быть пустым!")
        
        elif choice == '4':
            print("👋 До свидания!")
            break
        
        else:
            print("❌ Неверный выбор!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Программа прервана пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

