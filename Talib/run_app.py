"""
Точка входа GUI приложения Talib.
Запуск: из папки Talib выполнить python run_app.py
"""

import os
import sys

# Добавляем папку Talib в путь, чтобы импорты config, data, model и т.д. работали
_TALIB_DIR = os.path.dirname(os.path.abspath(__file__))
if _TALIB_DIR not in sys.path:
    sys.path.insert(0, _TALIB_DIR)

import tkinter as tk

from ui.main_window import MainWindow


def main():
    root = tk.Tk()
    root.title("Talib — обучение и тест торговой модели")
    root.minsize(800, 600)
    app = MainWindow(root)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()


if __name__ == "__main__":
    main()
