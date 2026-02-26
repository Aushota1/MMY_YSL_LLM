"""
Подготовка датасета для обучения классификаторов из папки train_data.

Каждый .npy файл — один объект; координаты вектора — фичи, класс берётся из имени файла
(например alu_0001.npy -> класс alu).

Использование:
  python Learn_ML/create_dataset.py --input-dir train_data
  python Learn_ML/create_dataset.py -i train_data --output Learn_ML/dataset.csv
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _parse_class_from_filename(basename: str):
    """Из имени файла вида class_0001.npy извлекает class."""
    m = re.match(r"^(.+)_\d+\.npy$", basename, re.IGNORECASE)
    return m.group(1) if m else None


def create_dataset(train_data_dir: str, output_path: str | None = None) -> pd.DataFrame:
    """
    Собирает датасет из .npy файлов в папке train_data_dir.

    Строки = объекты (файлы), столбцы = f0, f1, ..., f_{d-1}, target.
    Класс (target) извлекается из имени файла (часть до _XXXX.npy).

    Параметры
    ---------
    train_data_dir : str
        Путь к папке с .npy файлами.
    output_path : str, optional
        Если задан, датасет сохраняется в CSV или parquet (по расширению).

    Возвращает
    ----------
    pd.DataFrame
        Таблица с признаками f0..f_{d-1} и столбцом target.
    """
    train_data_dir = os.path.abspath(train_data_dir)
    if not os.path.isdir(train_data_dir):
        raise FileNotFoundError(f"Папка не найдена: {train_data_dir}")

    npy_files = [
        f
        for f in os.listdir(train_data_dir)
        if f.lower().endswith(".npy")
    ]
    npy_files.sort()

    if not npy_files:
        return pd.DataFrame()

    rows = []
    feature_dim: int | None = None
    skipped = 0

    for fname in npy_files:
        class_name = _parse_class_from_filename(fname)
        if class_name is None:
            skipped += 1
            continue

        path = os.path.join(train_data_dir, fname)
        try:
            arr = np.load(path)
            vec = np.asarray(arr).flatten()
        except Exception as e:
            skipped += 1
            print(f"  пропуск {fname}: {e}", file=sys.stderr)
            continue

        if feature_dim is None:
            feature_dim = len(vec)
        elif len(vec) != feature_dim:
            skipped += 1
            print(f"  пропуск {fname}: размерность {len(vec)}, ожидается {feature_dim}", file=sys.stderr)
            continue

        row = {f"f{i}": float(vec[i]) for i in range(len(vec))}
        row["target"] = class_name
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c != "target"]
    df = df[feature_cols + ["target"]]

    if output_path:
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if output_path.lower().endswith(".parquet"):
            df.to_parquet(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)
        print(f"Сохранено: {output_path}")

    return df


def main():
    root = _project_root()
    default_input = os.path.join(root, "train_data")

    parser = argparse.ArgumentParser(
        description="Подготовка датасета из .npy в train_data: векторы -> pandas с целевой переменной из имени файла"
    )
    parser.add_argument(
        "--input-dir", "-i",
        default=default_input,
        help=f"Папка с .npy (по умолчанию: {default_input})",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Путь для сохранения датасета (CSV или parquet)",
    )
    args = parser.parse_args()

    df = create_dataset(args.input_dir, args.output)

    if df.empty:
        print("Нет данных. Проверьте папку и имена файлов (ожидается вид class_0001.npy).")
        sys.exit(1)

    n_rows = len(df)
    n_classes = df["target"].nunique()
    n_features = len([c for c in df.columns if c != "target"])
    print(f"Строк: {n_rows}, признаков: {n_features}, классов: {n_classes}")
    print("Классы:", sorted(df["target"].unique().tolist()))


if __name__ == "__main__":
    main()
