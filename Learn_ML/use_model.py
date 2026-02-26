"""
Тестирование обученной модели классификатора по вводу текста.

Консольное приложение: ввод текста (например, код RTL-модуля) → тот же пайплайн,
что в build_train_data (text_to_embeddings → embedding_to_fixed_vector) → вектор
→ модель из model.joblib → вывод предсказанного класса.

Запуск из корня проекта:
  python Learn_ML/use_model.py
  python Learn_ML/use_model.py path/to/module.txt
  python Learn_ML/use_model.py --model Learn_ML/models/model.joblib --pth Layer_ML/final_model.pth -t Layer_ML/verylog.pkl
"""

import argparse
import os
import sys
import tempfile

import numpy as np

# Корень проекта для импортов Layer_ML и Learn_ML
_learn_ml_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_learn_ml_dir)
sys.path.insert(0, _project_root)

from Layer_ML.text_to_embeddings import text_to_embeddings
from Layer_ML.embedding_to_fixed_vector import embeddings_to_fixed_vector
from Learn_ML.train_classifiers import load_model

# Параметры, совместимые с обучением (как в build_train_data по умолчанию)
OUTPUT_DIM = 256
METHOD = "mean"

# Пути по умолчанию
DEFAULT_JOBLIB = os.path.join(_learn_ml_dir, "models", "model.joblib")
DEFAULT_PTH = os.path.join(_project_root, "Layer_ML", "final_model.pth")
DEFAULT_TOKENIZER = os.path.join(_project_root, "Layer_ML", "verylog.pkl")


def text_to_vector(
    text: str,
    pth_path: str,
    tokenizer_path: str,
    output_dim: int = OUTPUT_DIM,
    method: str = METHOD,
    encoding: str = "utf-8",
    device: str = "cpu",
    max_seq_len: int = 512,
) -> np.ndarray:
    """
    Преобразует текст в один вектор фиксированной размерности (как в build_train_data).
    Текст записывается во временный файл, затем text_to_embeddings → embeddings_to_fixed_vector.
    """
    fd, temp_path = tempfile.mkstemp(suffix=".txt", prefix="use_model_", text=True)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
        emb = text_to_embeddings(
            text_path=temp_path,
            pth_path=pth_path,
            tokenizer_path=tokenizer_path,
            encoding=encoding,
            device=device,
            max_seq_len=max_seq_len,
        )
        vec = embeddings_to_fixed_vector(emb, output_dim=output_dim, method=method)
        return vec
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def predict_class(vector: np.ndarray, bundle: dict) -> str:
    """
    Применяет scaler (если есть), предсказывает класс, возвращает название класса.
    """
    X = vector.reshape(1, -1)
    scaler = bundle.get("scaler")
    if scaler is not None:
        X = scaler.transform(X)
    model = bundle["model"]
    le = bundle.get("label_encoder")
    y_pred = model.predict(X)
    if le is not None:
        return le.inverse_transform(y_pred)[0]
    return str(y_pred[0])


def main():
    parser = argparse.ArgumentParser(
        description="Тестирование модели: текст → вектор → предсказанный класс"
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=None,
        help="Путь к файлу с текстом (опционально; без него — интерактивный ввод)",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_JOBLIB,
        help=f"Путь к модели классификатора .joblib (по умолчанию: {DEFAULT_JOBLIB})",
    )
    parser.add_argument(
        "--pth", "-p",
        default=DEFAULT_PTH,
        help=f"Путь к модели эмбеддингов .pth (по умолчанию: Layer_ML/final_model.pth)",
    )
    parser.add_argument(
        "--tokenizer", "-t",
        default=DEFAULT_TOKENIZER,
        help=f"Путь к токенизатору .pkl (по умолчанию: Layer_ML/verylog.pkl)",
    )
    parser.add_argument("--encoding", default="utf-8", help="Кодировка файла/ввода")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()

    for path, name in [
        (args.model, "модель классификатора"),
        (args.pth, "модель эмбеддингов (.pth)"),
        (args.tokenizer, "токенизатор (.pkl)"),
    ]:
        path_abs = os.path.abspath(path)
        if not os.path.isfile(path_abs):
            print(f"Ошибка: не найден файл {name}: {path_abs}")
            sys.exit(1)

    print("Загрузка модели...")
    bundle = load_model(os.path.abspath(args.model))
    pth_abs = os.path.abspath(args.pth)
    tok_abs = os.path.abspath(args.tokenizer)

    def run_prediction(text: str) -> None:
        if not text.strip():
            return
        try:
            vec = text_to_vector(
                text,
                pth_path=pth_abs,
                tokenizer_path=tok_abs,
                output_dim=OUTPUT_DIM,
                method=METHOD,
                encoding=args.encoding,
                device=args.device,
            )
            class_name = predict_class(vec, bundle)
            print(f"Предсказанный класс: {class_name}")
        except Exception as e:
            print(f"Ошибка: {e}")

    if args.input_file is not None:
        input_path = os.path.abspath(args.input_file)
        if not os.path.isfile(input_path):
            print(f"Ошибка: файл не найден: {input_path}")
            sys.exit(1)
        with open(input_path, "r", encoding=args.encoding) as f:
            text = f.read()
        run_prediction(text)
        return

    # Интерактивный режим
    print("Текст можно вводить несколькими строками. Пустая строка — выполнить предсказание, 'quit' — выход.")
    print("-" * 50)
    while True:
        print("Текст (пустая строка = предсказание, quit = выход):")
        lines = []
        while True:
            line = input()
            if line.strip().lower() == "quit":
                sys.exit(0)
            if line == "":
                break
            lines.append(line)
        text = "\n".join(lines)
        if not text.strip():
            continue
        run_prediction(text)
        print("-" * 50)


if __name__ == "__main__":
    main()
