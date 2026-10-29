"""Widgets comunes: pestaña con figura de Matplotlib y utilidades de exportación."""
from __future__ import annotations

import csv
import datetime as _dt
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


def output_dir() -> Path:
    folder = Path(__file__).resolve().parents[1] / "salidas"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def timestamp(prefix: str, ext: str) -> str:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}.{ext}"


def save_rows_csv(parent: tk.Misc, default_stem: str, header: list[str],
                  rows: list[tuple[str, ...]]) -> str | None:
    path = filedialog.asksaveasfilename(
        parent=parent,
        title="Exportar tabla como CSV",
        initialdir=str(output_dir()),
        initialfile=timestamp(default_stem, "csv"),
        defaultextension=".csv",
        filetypes=[("CSV", "*.csv"), ("Todos los archivos", "*.*")],
    )
    if not path:
        return None
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(header)
        writer.writerows(rows)
    return path


class PlotTab(ttk.Frame):
    def __init__(self, master: tk.Misc, app):
        super().__init__(master)
        self.app = app
        self._dirty = True

        self.controls = ttk.Frame(self, padding=(10, 8, 10, 4))
        self.figure = Figure(figsize=(10.5, 5.2), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        self.toolbar.update()
        self.footer = ttk.Frame(self)
        self.plot_frame = ttk.Frame(self)

        self.controls.pack(side="top", fill="x")
        self.toolbar.pack(side="bottom", fill="x")
        self.footer.pack(side="bottom", fill="x")
        self.plot_frame.pack(side="top", fill="both", expand=True)
        self.canvas.get_tk_widget().pack(in_=self.plot_frame, fill="both", expand=True)

    def mark_dirty(self) -> None:
        self._dirty = True

    def on_show(self) -> None:
        if self._dirty:
            self._dirty = False
            self.refresh()

    def on_audio_changed(self) -> None:
        self.mark_dirty()
        if self.winfo_ismapped():
            self.on_show()

    def on_params_changed(self) -> None:
        self.mark_dirty()
        if self.winfo_ismapped():
            self.on_show()

    def refresh(self) -> None:
        raise NotImplementedError

    def shutdown(self) -> None:
        pass

    def clear(self) -> None:
        self.figure.clear()

    def axes(self, nrows: int = 1, height_ratios: list[float] | None = None):
        self.figure.clear()
        gs = self.figure.add_gridspec(nrows, 1, height_ratios=height_ratios)
        axes = []
        for i in range(nrows):
            axes.append(self.figure.add_subplot(gs[i, 0]))
        return axes[0] if nrows == 1 else axes

    def message(self, text: str) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=11, wrap=True)
        self.draw()

    def draw(self) -> None:
        self.canvas.draw_idle()


def info_label(master: tk.Misc, columns: int = 1) -> ttk.Label:
    return ttk.Label(master, text="", justify="left", style="Info.TLabel", wraplength=1200)