"""
Получение эмбеддингов для текста из файла с использованием обученной модели (.pth).

Вход:
  - файл с текстом (.txt);
  - обученная модель эмбеддингов (.pth);
  - токенизатор (.pkl), совместимый с моделью (BPE или словный).

Выход:
  - массив эмбеддингов формы (число_токенов, embedding_dim) — по одному вектору на каждый токен в тексте.
"""

import argparse
import os
import pickle
import sys

import numpy as np
import torch

# Родительская директория проекта для импортов
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BPE_STUCTUR import BPETokenizer
from tokenizer_trainer import WordTokenizer
from EMBEDDING_LAYER.embedding_layer import EmbeddingLayer


def load_tokenizer(pkl_path: str):
    """
    Загружает токенизатор из .pkl (BPE или словный по формату файла).
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError("Файл .pkl должен содержать словарь (формат BPE/word).")
    # BPE: есть 'merges' (и обычно 'vocab', 'inverse_vocab')
    if "merges" in data:
        tok = BPETokenizer()
        tok.load(pkl_path)
        return tok
    # Словный: vocab + inverse_vocab (и опционально tokenizer_type)
    if "vocab" in data and "inverse_vocab" in data:
        tok = WordTokenizer()
        tok.load(pkl_path)
        return tok
    raise ValueError("Неизвестный формат .pkl: нужны 'merges' (BPE) или 'vocab'/'inverse_vocab' (word).")


def load_embedding_layer_from_checkpoint(
    pth_path: str,
    vocab_size: int,
    device: torch.device,
    max_seq_len: int = 512,
) -> torch.nn.Module:
    """
    Загружает чекпоинт .pth и восстанавливает только EmbeddingLayer.
    """
    checkpoint = torch.load(pth_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("Чекпоинт должен быть словарём (model_state_dict, embedding_dim, ...).")

    state = checkpoint.get("model_state_dict")
    if state is None:
        raise ValueError("В чекпоинте отсутствует 'model_state_dict'.")

    embedding_dim = checkpoint.get("embedding_dim")
    if embedding_dim is None:
        raise ValueError("В чекпоинте отсутствует 'embedding_dim'.")

    # Собираем только ключи, относящиеся к embedding (префикс 'embedding.')
    embedding_state = {}
    for k, v in state.items():
        if k.startswith("embedding."):
            new_key = k[len("embedding.") :]
            embedding_state[new_key] = v

    if not embedding_state:
        raise ValueError("В model_state_dict нет ключей с префиксом 'embedding.'.")

    learnable_pos = any("position_embedding" in k for k in embedding_state)
    layer = EmbeddingLayer(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        max_seq_len=max_seq_len,
        dropout=0.0,
        learnable_pos=learnable_pos,
        layer_norm=True,
    )
    layer.load_state_dict(embedding_state, strict=False)
    layer = layer.to(device)
    layer.eval()
    return layer


def text_to_embeddings(
    text_path: str,
    pth_path: str,
    tokenizer_path: str,
    encoding: str = "utf-8",
    device: str = "cpu",
    max_seq_len: int = 512,
    output_npy: str = None,
) -> np.ndarray:
    """
    Читает текст из файла, кодирует токенизатором и возвращает массив эмбеддингов.

    Returns:
        embeddings: np.ndarray формы (num_tokens, embedding_dim), dtype float32.
    """
    device_t = torch.device(device)

    # Токенизатор
    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    # Модель эмбеддингов из .pth
    embedding_layer = load_embedding_layer_from_checkpoint(
        pth_path, vocab_size=vocab_size, device=device_t, max_seq_len=max_seq_len
    )
    embedding_dim = embedding_layer.embedding_dim

    # Текст из файла
    with open(text_path, "r", encoding=encoding) as f:
        text = f.read()

    # Кодирование в id
    token_ids = tokenizer.encode(text)
    if not token_ids:
        return np.zeros((0, embedding_dim), dtype=np.float32)

    # Ограничение длины (как при обучении)
    token_ids = token_ids[:max_seq_len]

    # Эмбеддинги без градиентов
    ids_tensor = torch.tensor([token_ids], dtype=torch.long, device=device_t)
    with torch.no_grad():
        emb = embedding_layer(ids_tensor)

    # [1, seq_len, dim] -> [seq_len, dim]
    emb = emb[0].cpu().numpy()

    if output_npy:
        os.makedirs(os.path.dirname(os.path.abspath(output_npy)) or ".", exist_ok=True)
        np.save(output_npy, emb)
        print(f"Эмбеддинги сохранены: {output_npy} shape={emb.shape}")

    return emb


def main():
    parser = argparse.ArgumentParser(
        description="Получение массива эмбеддингов для текста из файла по обученной модели .pth"
    )
    parser.add_argument("text", help="Путь к файлу с текстом (.txt)")
    parser.add_argument("model", help="Путь к обученной модели (.pth)")
    parser.add_argument(
        "--tokenizer",
        "-t",
        required=True,
        help="Путь к токенизатору (.pkl), совместимому с моделью",
    )
    parser.add_argument("--encoding", default="utf-8", help="Кодировка текстового файла")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"), help="Устройство для модели")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Максимальная длина последовательности")
    parser.add_argument("--output", "-o", default=None, help="Путь для сохранения массива .npy (опционально)")
    args = parser.parse_args()

    if not os.path.isfile(args.text):
        print(f"Ошибка: файл с текстом не найден: {args.text}")
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"Ошибка: файл модели не найден: {args.model}")
        sys.exit(1)
    if not os.path.isfile(args.tokenizer):
        print(f"Ошибка: файл токенизатора не найден: {args.tokenizer}")
        sys.exit(1)

    embeddings = text_to_embeddings(
        text_path=args.text,
        pth_path=args.model,
        tokenizer_path=args.tokenizer,
        encoding=args.encoding,
        device=args.device,
        max_seq_len=args.max_seq_len,
        output_npy=args.output,
    )
    print(f"Токенов: {embeddings.shape[0]}, размерность: {embeddings.shape[1]}")
    if args.output is None:
        print("Массив эмбеддингов возвращён в переменную (для сохранения укажите --output path.npy)")


if __name__ == "__main__":
    main()
