"""
Консольное приложение для подготовки датасета и обучения классификаторов.

Интерактивное меню или CLI:
  python Learn_ML/console_train.py
  python Learn_ML/console_train.py --prepare-data --train svm --save-model Learn_ML/models/svm.joblib

Зависимости: pip install scikit-learn xgboost pandas numpy joblib
"""

import argparse
import os
import sys

# Корень проекта и Learn_ML в пути
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from create_dataset import create_dataset
from train_classifiers import (
    load_dataset,
    prepare_splits,
    save_model,
    train_classifier,
    evaluate_model,
    check_overfitting,
    plot_confusion_matrix_heatmap,
    cross_validate_classifier,
)


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_train_data_dir():
    return os.path.join(_project_root(), "train_data")


def _default_dataset_cache():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.csv")


def run_prepare_data(input_dir: str | None = None, output_path: str | None = None) -> None:
    """Подготовить датасет из train_data и опционально сохранить в CSV."""
    input_dir = input_dir or _default_train_data_dir()
    output_path = output_path or _default_dataset_cache()
    df = create_dataset(input_dir, output_path=output_path)
    if df.empty:
        print("Нет данных.")
        return
    print(f"Строк: {len(df)}, признаков: {len(df.columns) - 1}, классов: {df['target'].nunique()}")


def run_train(
    classifier_name: str,
    dataset_path: str | None = None,
    train_data_dir: str | None = None,
    test_size: float = 0.2,
    scale: bool = True,
):
    """
    Загружает датасет, разбивает на train/test, обучает выбранный классификатор.
    Возвращает (model, scaler, label_encoder, metrics) для последующего сохранения.
    """
    cache = dataset_path or _default_dataset_cache()
    df = load_dataset(cache_path=cache, train_data_dir=train_data_dir or _default_train_data_dir())
    if df.empty:
        print("Датасет пуст. Сначала подготовьте данные (п. 1).")
        return None

    X_train, X_test, y_train, y_test, scaler, le = prepare_splits(
        df, test_size=test_size, scale=scale
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    model, metrics = train_classifier(
        classifier_name, X_train, y_train, X_test, y_test
    )
    print(f"Accuracy: {metrics['accuracy']:.4f}, F1 (weighted): {metrics['f1_weighted']:.4f}")
    return {
        "model": model,
        "scaler": scaler,
        "label_encoder": le,
        "metrics": metrics,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def run_save(result: dict | None, path: str) -> None:
    """Сохраняет модель из result (после run_train) в path."""
    if not result:
        print("Сначала обучите модель (п. 3).")
        return
    save_model(
        model=result["model"],
        path=path,
        scaler=result.get("scaler"),
        label_encoder=result.get("label_encoder"),
    )


def interactive_menu():
    """Интерактивное меню в консоли."""
    train_data_dir = _default_train_data_dir()
    dataset_cache = _default_dataset_cache()
    result = None

    while True:
        print()
        print("--- Learn_ML: обучение классификаторов ---")
        print("1. Подготовить/загрузить датасет")
        print("2. Выбрать классификатор и обучить")
        print("3. Сохранить модель")
        print("4. Оценка модели (метрики train/test, отчёт)")
        print("5. Проверка на переобучение")
        print("6. Тепловая карта (confusion matrix)")
        print("7. Кросс-валидация")
        print("8. Выход")
        choice = input("Выбор: ").strip()

        if choice == "1":
            run_prepare_data(input_dir=train_data_dir, output_path=dataset_cache)

        elif choice == "2":
            print("Классификаторы: svm, xgboost, rf, logistic, knn, gb")
            name = input("Имя классификатора: ").strip() or "svm"
            result = run_train(
                classifier_name=name,
                dataset_path=dataset_cache,
                train_data_dir=train_data_dir,
            )

        elif choice == "3":
            if result:
                path = input(
                    "Путь для сохранения (Enter = Learn_ML/models/model.joblib): "
                ).strip()
                path = path or os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "models", "model.joblib"
                )
                run_save(result, path)
            else:
                print("Сначала обучите модель (п. 2).")

        elif choice == "4":
            if result and "X_train" in result:
                ev = evaluate_model(
                    result["model"],
                    result["X_train"], result["y_train"],
                    result["X_test"], result["y_test"],
                    result.get("label_encoder"),
                )
                print("\n--- Train ---")
                print("Accuracy:", ev["train_metrics"]["accuracy"], "F1:", ev["train_metrics"]["f1_weighted"])
                print("\n--- Test ---")
                print("Accuracy:", ev["test_metrics"]["accuracy"], "F1:", ev["test_metrics"]["f1_weighted"])
                print("\n--- Classification report ---")
                print(ev["classification_report"])
            else:
                print("Сначала обучите модель (п. 2).")

        elif choice == "5":
            if result and "X_train" in result:
                of = check_overfitting(
                    result["model"],
                    result["X_train"], result["y_train"],
                    result["X_test"], result["y_test"],
                )
                print("\n--- Переобучение ---")
                print("Train  Accuracy:", f"{of['train_accuracy']:.4f}", " F1:", f"{of['train_f1']:.4f}")
                print("Test   Accuracy:", f"{of['test_accuracy']:.4f}", " F1:", f"{of['test_f1']:.4f}")
                print("Разница (train - test): Accuracy:", f"{of['gap_accuracy']:.4f}", " F1:", f"{of['gap_f1']:.4f}")
                print("Признаки переобучения:", "да" if of["likely_overfitting"] else "нет")
            else:
                print("Сначала обучите модель (п. 2).")

        elif choice == "6":
            if result and "X_test" in result and "y_test" in result:
                save_path = input("Сохранить в файл (Enter = показать в окне): ").strip()
                y_pred = result["model"].predict(result["X_test"])
                plot_confusion_matrix_heatmap(
                    result["y_test"], y_pred,
                    result.get("label_encoder"),
                    save_path=save_path if save_path else None,
                )
            else:
                print("Сначала обучите модель (п. 2).")

        elif choice == "7":
            cache = dataset_cache
            df = load_dataset(cache_path=cache, train_data_dir=train_data_dir)
            if df.empty:
                print("Датасет пуст. Сначала подготовьте данные (п. 1).")
            else:
                name = input("Имя классификатора (svm, xgboost, rf, logistic, knn, gb): ").strip() or "rf"
                n_splits_str = input("Число фолдов (Enter = 5): ").strip() or "5"
                try:
                    n_splits = int(n_splits_str)
                except ValueError:
                    n_splits = 5
                cv = cross_validate_classifier(name, df, n_splits=n_splits, scale=True)
                print(f"\nКросс-валидация ({n_splits} фолдов):")
                print(f"  Accuracy:       {cv['accuracy_mean']:.4f} ± {cv['accuracy_std']:.4f}")
                print(f"  F1 (weighted):  {cv['f1_weighted_mean']:.4f} ± {cv['f1_weighted_std']:.4f}")

        elif choice == "8":
            break
        else:
            print("Неверный выбор.")


def main():
    root = _project_root()
    default_train = os.path.join(root, "train_data")
    default_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.csv")

    parser = argparse.ArgumentParser(
        description="Подготовка датасета и обучение классификаторов (SVM, XGBoost, RF, Logistic, KNN)"
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Подготовить датасет из train_data и сохранить в CSV",
    )
    parser.add_argument(
        "--input-dir",
        default=default_train,
        help="Папка с .npy (по умолчанию: train_data в корне проекта)",
    )
    parser.add_argument(
        "--dataset",
        default=default_cache,
        help="Путь к кэшу датасета (CSV/parquet)",
    )
    parser.add_argument(
        "--train",
        choices=["svm", "xgboost", "rf", "logistic", "knn", "gb"],
        help="Обучить указанный классификатор",
    )
    parser.add_argument(
        "--save-model",
        metavar="PATH",
        help="Путь для сохранения модели (после --train)",
    )
    parser.add_argument(
        "--no-menu",
        action="store_true",
        help="Не запускать интерактивное меню (только CLI)",
    )
    args = parser.parse_args()

    if args.prepare_data:
        run_prepare_data(input_dir=args.input_dir, output_path=args.dataset)

    if args.train:
        result = run_train(
            classifier_name=args.train,
            dataset_path=args.dataset,
            train_data_dir=args.input_dir,
        )
        if result and args.save_model:
            run_save(result, args.save_model)

    if not args.prepare_data and not args.train:
        if args.no_menu:
            print("Укажите --prepare-data и/или --train или запустите без --no-menu для меню.")
            sys.exit(0)
        interactive_menu()


if __name__ == "__main__":
    main()
