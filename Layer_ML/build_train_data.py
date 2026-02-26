"""
Пайплайн: все .txt из lerning_data → text_to_embeddings → embedding_to_fixed_vector → train_data/<класс>_<номер>.npy

По умолчанию method=mean, output_dim=256 (как при ручном вызове), чтобы векторы хорошо различались при визуализации.
Для сохранения уникальности по токенам используйте --method sequence --output-dim 2048.

Использование (из корня проекта):
  python Layer_ML/build_train_data.py --model path/to/model.pth --tokenizer path/to/tokenizer.pkl
  python Layer_ML/build_train_data.py -m final_model.pth -t verylog.pkl --output-dim 256 --method mean
"""

import argparse
import os
import sys

import numpy as np

# Корень проекта для импортов
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Layer_ML.text_to_embeddings import text_to_embeddings
from Layer_ML.embedding_to_fixed_vector import embeddings_to_fixed_vector


def collect_txt_by_class(input_dir: str):
    """
    Собирает пути к .txt файлам, сгруппированные по классу (имя листовой папки).
    Возвращает: { "mux": ["path/to/mux_000.txt", ...], "decoder": [...], ... }
    """
    by_class = {}
    input_dir = os.path.abspath(input_dir)
    for root, _dirs, files in os.walk(input_dir):
        for f in files:
            if not f.lower().endswith(".txt"):
                continue
            path = os.path.join(root, f)
            # Класс = имя листовой папки (последняя компонента пути перед файлом)
            rel = os.path.relpath(path, input_dir)
            parts = rel.split(os.sep)
            if len(parts) >= 2:
                class_name = parts[-2]  # e.g. lerning_data/combinational/mux/file.txt -> mux
            else:
                class_name = "other"
            by_class.setdefault(class_name, []).append(path)
    for k in by_class:
        by_class[k].sort()
    return by_class


def main():
    parser = argparse.ArgumentParser(
        description="Обработка .txt из lerning_data: text_to_embeddings → fixed vector → train_data/<class>_<num>.npy"
    )
    parser.add_argument("--model", "-m", required=True, help="Путь к модели .pth")
    parser.add_argument("--tokenizer", "-t", required=True, help="Путь к токенизатору .pkl")
    parser.add_argument(
        "--input-dir",
        "-i",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "lerning_data"),
        help="Папка с .txt (по умолчанию: Layer_ML/lerning_data)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train_data"),
        help="Папка для сохранения .npy (по умолчанию: train_data)",
    )
    parser.add_argument(
        "--output-dim", "-d", type=int, default=256,
        help="Размерность выходного вектора (для mean/max лучше = embedding_dim модели, напр. 256)",
    )
    parser.add_argument(
        "--method",
        choices=["sequence", "mean", "max", "first", "last", "mean_max"],
        default="mean",
        help="Метод приведения к фиксированной размерности (mean даёт хорошую визуализацию)",
    )
    parser.add_argument("--encoding", default="utf-8", help="Кодировка .txt файлов")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--max-seq-len", type=int, default=512)
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Ошибка: папка не найдена: {args.input_dir}")
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"Ошибка: модель не найдена: {args.model}")
        sys.exit(1)
    if not os.path.isfile(args.tokenizer):
        print(f"Ошибка: токенизатор не найден: {args.tokenizer}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    by_class = collect_txt_by_class(args.input_dir)
    total = sum(len(v) for v in by_class.values())
    print(f"Найдено классов: {len(by_class)}, файлов .txt: {total}")
    print(f"Выход: {args.output_dir}, output_dim={args.output_dim}, method={args.method}")
    print("-" * 60)

    done = 0
    err = 0
    for class_name, paths in sorted(by_class.items()):
        for num, txt_path in enumerate(paths):
            out_name = f"{class_name}_{num:04d}.npy"
            out_path = os.path.join(args.output_dir, out_name)
            try:
                emb = text_to_embeddings(
                    text_path=txt_path,
                    pth_path=args.model,
                    tokenizer_path=args.tokenizer,
                    encoding=args.encoding,
                    device=args.device,
                    max_seq_len=args.max_seq_len,
                )
                vec = embeddings_to_fixed_vector(
                    emb,
                    output_dim=args.output_dim,
                    method=args.method,
                )
                np.save(out_path, vec)
                done += 1
                if done % 100 == 0:
                    print(f"  обработано: {done}/{total}")
            except Exception as e:
                err += 1
                print(f"  ошибка {txt_path}: {e}")
    print("-" * 60)
    print(f"Готово. Успешно: {done}, ошибок: {err}. Файлы в {args.output_dir}")


if __name__ == "__main__":
    main()
