"""
Подсчёт параметров модели по final_model.pth (и опционально verylog.pkl для справки).
Запуск из корня проекта: python Layer_ML/count_params.py
"""
import os
import sys

# корень проекта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    pth_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_model.pth")
    pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verylog.pkl")

    if not os.path.isfile(pth_path):
        print(f"Файл не найден: {pth_path}")
        return

    import torch
    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]
    total = sum(p.numel() for p in state.values())

    print("=" * 60)
    print("Параметры модели (final_model.pth)")
    print("=" * 60)
    print(f"Всего параметров: {total:,}  ({total/1e6:.2f} M)")
    print()
    print("По слоям (от больших к меньшим):")
    for k, v in sorted(state.items(), key=lambda x: -x[1].numel()):
        print(f"  {k}: {v.numel():,}")
    print("=" * 60)

    # Метаданные из чекпоинта
    for key in ("embedding_dim", "num_layers", "num_heads", "vocab_size"):
        if key in checkpoint:
            print(f"  {key}: {checkpoint[key]}")

    if os.path.isfile(pkl_path):
        import pickle
        with open(pkl_path, "rb") as f:
            tok_data = pickle.load(f)
        if isinstance(tok_data, dict):
            if "merges" in tok_data:
                n_merges = len(tok_data.get("merges", []))
                vocab = tok_data.get("vocab", tok_data.get("inverse_vocab", {}))
                n_vocab = len(vocab) if isinstance(vocab, dict) else 0
                print()
                print("Токенизатор (verylog.pkl): BPE, размер словаря:", n_vocab, ", пар слияний:", n_merges)
            elif "vocab" in tok_data:
                print()
                print("Токенизатор (verylog.pkl): словный, размер словаря:", len(tok_data.get("vocab", {})))


if __name__ == "__main__":
    main()
