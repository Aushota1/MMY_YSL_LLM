"""
Приложение для загрузки нескольких .npy файлов с эмбеддингами и отрисовки векторов.
Поддержка косинусного сходства: L2-нормализация перед снижением размерности,
чтобы векторы с высоким косинусным сходством (0.8, 0.9) отображались рядом.
Методы: PCA, t-SNE, UMAP. Визуализация: matplotlib (в окне) + Plotly (интерактивно в браузере).
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

# Родительская директория для импортов
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Стиль matplotlib: современный и читаемый
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    try:
        plt.style.use("seaborn-whitegrid")
    except Exception:
        plt.style.use("ggplot")
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def load_npy_safe(path: str):
    """Загружает .npy, возвращает массив формы (n, d) или None при ошибке."""
    try:
        arr = np.load(path)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            return None
        return arr
    except Exception:
        return None


def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    """L2-нормализация по строкам. После неё евклидово расстояние отражает косинусное сходство."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return (X / norms).astype(np.float32)


def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = "pca",
    n_components: int = 2,
    max_points: int = 2000,
    random_state: int = 42,
    use_cosine_norm: bool = True,
):
    """
    Уменьшение размерности. Если use_cosine_norm=True, сначала L2-нормализация по строкам,
    чтобы близкие по косинусному сходству векторы оказались рядом на графике.
    """
    if not HAS_SKLEARN:
        raise ImportError("Установите scikit-learn: pip install scikit-learn")
    n, d = embeddings.shape
    if n == 0:
        return np.zeros((0, n_components), dtype=np.float32)
    if n > max_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=max_points, replace=False)
        embeddings = embeddings[idx]
        n = max_points

    if use_cosine_norm:
        X = l2_normalize_rows(embeddings)
    else:
        X = StandardScaler().fit_transform(embeddings)

    # Один вектор: PCA/UMAP центрируют данные → одна точка уходит в (0,0). Используем первые компоненты.
    if n == 1:
        row = np.asarray(X[0], dtype=np.float32)
        out = np.zeros((1, n_components), dtype=np.float32)
        for j in range(min(n_components, len(row))):
            out[0, j] = row[j]
        return out

    if method.lower() == "pca":
        reducer = PCA(n_components=min(n_components, n, d), random_state=random_state)
        out = reducer.fit_transform(X)
    elif method.lower() == "tsne":
        perplexity = min(30, max(5, n // 4))
        reducer = TSNE(
            n_components=min(n_components, 3),
            random_state=random_state,
            perplexity=perplexity,
            n_iter=1000,
            metric="euclidean",
        )
        out = reducer.fit_transform(X)
    elif method.lower() == "umap" and HAS_UMAP:
        n_comp = min(n_components, 3)
        reducer = umap.UMAP(
            n_components=n_comp,
            random_state=random_state,
            metric="cosine" if use_cosine_norm else "euclidean",
            n_neighbors=min(15, n - 1) if n > 1 else 1,
            min_dist=0.1,
        )
        out = reducer.fit_transform(X)
        if out.shape[1] < n_components:
            pad = np.zeros((out.shape[0], n_components - out.shape[1]), dtype=np.float32)
            out = np.hstack([out, pad])
    else:
        if method.lower() == "umap" and not HAS_UMAP:
            raise ImportError("UMAP: установите umap-learn: pip install umap-learn")
        reducer = PCA(n_components=min(n_components, n, d), random_state=random_state)
        out = reducer.fit_transform(X)

    if out.shape[1] < n_components:
        pad = np.zeros((out.shape[0], n_components - out.shape[1]), dtype=np.float32)
        out = np.hstack([out, pad])
    return out.astype(np.float32)


class VisualizeNpyApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Layer_ML — Визуализация .npy эмбеддингов")
        self.root.minsize(900, 650)
        self.file_paths = []
        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Файлы .npy:").pack(anchor=tk.W)
        list_frame = ttk.Frame(top)
        list_frame.pack(fill=tk.BOTH, expand=True)
        self.listbox = tk.Listbox(list_frame, height=5, selectmode=tk.EXTENDED, font=("Consolas", 10))
        scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=scroll.set)

        btn_frame = ttk.Frame(top)
        btn_frame.pack(fill=tk.X, pady=6)
        ttk.Button(btn_frame, text="Добавить файлы…", command=self._add_files).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Удалить выбранные", command=self._remove_selected).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Очистить список", command=self._clear_list).pack(side=tk.LEFT)

        opts = ttk.Frame(top)
        opts.pack(fill=tk.X, pady=6)
        ttk.Label(opts, text="Метод:").pack(side=tk.LEFT, padx=4)
        methods = ["pca", "tsne"]
        if HAS_UMAP:
            methods.append("umap")
        self.method_var = tk.StringVar(value="umap" if HAS_UMAP else "pca")
        method_combo = ttk.Combobox(opts, textvariable=self.method_var, values=methods, state="readonly", width=6)
        method_combo.pack(side=tk.LEFT, padx=6)
        ttk.Label(opts, text="Ось:").pack(side=tk.LEFT, padx=4)
        self.dim_var = tk.StringVar(value="2D")
        dim_combo = ttk.Combobox(opts, textvariable=self.dim_var, values=["2D", "3D"], state="readonly", width=4)
        dim_combo.pack(side=tk.LEFT, padx=6)
        self.cosine_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Учёт косинусного сходства (L2-norm)", variable=self.cosine_var).pack(side=tk.LEFT, padx=8)
        ttk.Label(opts, text="Макс. точек:").pack(side=tk.LEFT, padx=4)
        self.max_points_var = tk.IntVar(value=2000)
        ttk.Spinbox(opts, from_=100, to=20000, increment=500, textvariable=self.max_points_var, width=7).pack(side=tk.LEFT, padx=4)
        ttk.Button(opts, text="Построить в окне", command=self._plot).pack(side=tk.LEFT, padx=6)
        ttk.Button(opts, text="Интерактивно в браузере (Plotly)", command=self._plot_plotly).pack(side=tk.LEFT, padx=4)
        ttk.Button(opts, text="Сохранить PNG…", command=self._save_figure).pack(side=tk.LEFT)
        self.last_fig = None

        canvas_frame = ttk.Frame(self.root, padding=8)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = None
        self.status_var = tk.StringVar(
            value="Добавьте .npy. Включите «Учёт косинусного сходства» — похожие векторы (cos≈0.8–0.9) будут рядом."
        )
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X, side=tk.BOTTOM)

        self._last_datasets = []
        self._last_labels = []
        self._last_use_3d = False

    def _add_files(self):
        paths = filedialog.askopenfilenames(
            title="Выберите файлы .npy",
            filetypes=[("NumPy arrays", "*.npy"), ("All files", "*.*")],
        )
        for p in paths:
            p = p.strip()
            if p and p not in self.file_paths:
                self.file_paths.append(p)
                self.listbox.insert(tk.END, os.path.basename(p))

    def _remove_selected(self):
        sel = list(self.listbox.curselection())
        for i in reversed(sel):
            self.listbox.delete(i)
            del self.file_paths[i]

    def _clear_list(self):
        self.listbox.delete(0, tk.END)
        self.file_paths.clear()

    def _get_full_paths(self):
        return list(self.file_paths)

    def _load_and_reduce(self):
        paths = self._get_full_paths()
        if not paths:
            return None, None, None, None
        if not HAS_SKLEARN:
            return None, None, None, "Установите scikit-learn"
        try:
            max_pts = max(100, int(self.max_points_var.get()))
        except Exception:
            max_pts = 2000
        method = self.method_var.get().strip() or ("umap" if HAS_UMAP else "pca")
        use_3d = self.dim_var.get().strip().upper() == "3D"
        n_components = 3 if use_3d else 2
        use_cosine = self.cosine_var.get()

        datasets = []
        labels = []
        for path in paths:
            arr = load_npy_safe(path)
            if arr is None:
                messagebox.showwarning("Ошибка файла", f"Не удалось загрузить: {path}")
                continue
            if arr.size == 0:
                continue
            try:
                reduced = reduce_dimensions(
                    arr,
                    method=method,
                    n_components=n_components,
                    max_points=max_pts,
                    use_cosine_norm=use_cosine,
                )
            except Exception as e:
                messagebox.showerror("Ошибка", f"{os.path.basename(path)}: {e}")
                continue
            datasets.append(reduced)
            labels.append(os.path.basename(path))
        if not datasets:
            return None, None, None, "Нет данных"
        return datasets, labels, use_3d, None

    def _plot(self):
        result = self._load_and_reduce()
        if result[3] is not None:
            if result[3] != "Нет данных":
                messagebox.showerror("Ошибка", result[3])
            self.status_var.set(result[3])
            return
        datasets, labels, use_3d, _ = result
        self._last_datasets = datasets
        self._last_labels = labels
        self._last_use_3d = use_3d

        method = self.method_var.get().strip() or "pca"
        n_components = 3 if use_3d else 2

        self.status_var.set("Построение графика…")
        self.root.update()

        fig = Figure(figsize=(9, 7), dpi=110)
        if use_3d:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)

        for i, (data, label) in enumerate(zip(datasets, labels)):
            if data.size == 0:
                continue
            c = COLORS[i % len(COLORS)]
            if use_3d and data.shape[1] >= 3:
                ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=c, label=label, alpha=0.7, s=28, edgecolors="white", linewidths=0.3)
            else:
                ax.scatter(data[:, 0], data[:, 1], c=c, label=label, alpha=0.7, s=28, edgecolors="white", linewidths=0.3)

        ax.set_title(f"Векторы .npy — {method.upper()} (учёт косинусного сходства: {'да' if self.cosine_var.get() else 'нет'})", fontsize=11)
        ax.legend(loc="best", fontsize=9)
        if not use_3d:
            ax.set_xlabel("Компонента 1")
            ax.set_ylabel("Компонента 2")
        ax.set_facecolor("#f8f9fa")
        fig.patch.set_facecolor("white")
        fig.tight_layout()

        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.last_fig = fig
        self.status_var.set(f"Файлов: {len(datasets)}, точек: {sum(d.shape[0] for d in datasets)}. Похожие векторы (cos) отображаются рядом.")

    def _plot_plotly(self):
        if not HAS_PLOTLY:
            messagebox.showinfo("Plotly", "Установите: pip install plotly")
            return
        result = self._load_and_reduce()
        if result[3] is not None:
            if result[3] != "Нет данных":
                messagebox.showerror("Ошибка", result[3])
            return
        datasets, labels, use_3d, _ = result
        method = self.method_var.get().strip() or "pca"

        xs, ys, zs, file_labels, point_idx = [], [], [], [], []
        for fi, (data, label) in enumerate(zip(datasets, labels)):
            for j in range(len(data)):
                xs.append(data[j, 0])
                ys.append(data[j, 1])
                zs.append(data[j, 2] if use_3d and data.shape[1] >= 3 else 0)
                file_labels.append(label)
                point_idx.append(j)

        if use_3d and any(z != 0 for z in zs):
            fig = go.Figure()
            for i, label in enumerate(labels):
                mask = [l == label for l in file_labels]
                fig.add_trace(go.Scatter3d(
                    x=[xs[j] for j in range(len(xs)) if mask[j]],
                    y=[ys[j] for j in range(len(ys)) if mask[j]],
                    z=[zs[j] for j in range(len(zs)) if mask[j]],
                    mode="markers",
                    name=label,
                    marker=dict(size=6, opacity=0.8, color=COLORS[i % len(COLORS)]),
                ))
            fig.update_layout(
                title=f"Визуализация .npy — {method.upper()} (косинусное сходство учтено)",
                scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", bgcolor="#f8f9fa"),
                template="plotly_white",
                font=dict(size=11),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            )
        else:
            fig = go.Figure()
            for i, label in enumerate(labels):
                mask = [l == label for l in file_labels]
                fig.add_trace(go.Scatter(
                    x=[xs[j] for j in range(len(xs)) if mask[j]],
                    y=[ys[j] for j in range(len(ys)) if mask[j]],
                    mode="markers",
                    name=label,
                    marker=dict(size=8, opacity=0.75, color=COLORS[i % len(COLORS)], line=dict(width=0.5, color="white")),
                ))
            fig.update_layout(
                title=f"Визуализация .npy — {method.upper()} (похожие векторы рядом)",
                xaxis_title="Компонента 1",
                yaxis_title="Компонента 2",
                template="plotly_white",
                font=dict(size=11),
                plot_bgcolor="#f8f9fa",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                hovermode="closest",
            )
        try:
            fig.show()
            self.status_var.set("Интерактивный график открыт в браузере.")
        except Exception as e:
            messagebox.showerror("Plotly", str(e))

    def _save_figure(self):
        if self.last_fig is None:
            messagebox.showinfo("Нет графика", "Сначала нажмите «Построить в окне».")
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить рисунок",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("All files", "*.*")],
        )
        if path:
            try:
                self.last_fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
                self.status_var.set(f"Сохранено: {path}")
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

    def run(self):
        self.root.mainloop()


def main():
    app = VisualizeNpyApp()
    app.run()


if __name__ == "__main__":
    main()
