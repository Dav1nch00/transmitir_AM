"""Widgets comunes: pestaña con figura de Matplotlib y utilidades de exportación."""
from __future__ import annotations

import csv
import datetime as _dt
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib import rcParams
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# Tipografía y espaciado compactos: con ventanas pequeñas o escalado de
# Windows, los títulos y etiquetas grandes dejan muy poco alto para las
# trazas.  Estos valores son globales para todas las pestañas.
rcParams.update({
    "axes.titlesize": 10,
    "axes.titlepad": 2.5,
    "axes.labelsize": 8,
    "axes.labelpad": 1.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.borderpad": 0.3,
    "legend.labelspacing": 0.25,
    "figure.titlesize": 10,
})


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
    """Pestaña con controles arriba, figura incrustada y barra de guardado."""

    def __init__(self, master: tk.Misc, app):
        super().__init__(master)
        self.app = app
        self._dirty = True

        self.controls = ttk.Frame(self, padding=(10, 8, 10, 4))
        self.footer = ttk.Frame(self)
        self.plot_frame = ttk.Frame(self)
        self.save_strip = ttk.Frame(self)
        self.save_info = tk.StringVar(value="")
        # El lienzo debe ser hijo de ``plot_frame``: si se crea como hijo de la
        # pestaña y solo se empaqueta con ``in_``, ``plot_frame`` (creado
        # después) queda por encima en el orden de apilado y tapa la gráfica.
        self.figure = Figure(figsize=(10.5, 5.2), dpi=100, constrained_layout=True)
        self.figure.set_constrained_layout_pads(
            w_pad=0.03, h_pad=0.015, wspace=0.0, hspace=0.0)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        self.toolbar.update()

        self.controls.pack(side="top", fill="x")
        # Orden visual de arriba a abajo: controles, gráfica, barra de
        # Matplotlib, barra de guardado y pie (tabla).  El primer widget
        # empaquetado abajo queda en el borde inferior.
        self.footer.pack(side="bottom", fill="x")
        self.save_strip.pack(side="bottom", fill="x")
        self.toolbar.pack(side="bottom", fill="x")
        self.plot_frame.pack(side="top", fill="both", expand=True)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        ttk.Button(self.save_strip, text="💾 Guardar gráfica",
                   command=self.save_figure).pack(side="left", padx=(10, 6), pady=3)
        ttk.Label(self.save_strip, textvariable=self.save_info,
                  style="Subtitle.TLabel").pack(side="left")

    # -- ciclo de vida -----------------------------------------------------
    def mark_dirty(self) -> None:
        self._dirty = True

    def show_footer(self, visible: bool) -> None:
        """Muestra u oculta el pie de la pestaña (tablas, notas, etc.).

        Al ocultarlo, ``plot_frame`` recupera todo el alto disponible en lugar
        de dejar una banda vacía.  El pie se vuelve a empaquetar en su sitio
        original, debajo de la barra de guardado.
        """
        if visible:
            if not self.footer.winfo_manager():
                self.footer.pack(side="bottom", fill="x",
                                 before=self.save_strip)
        elif self.footer.winfo_manager():
            self.footer.pack_forget()

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

    def refresh(self) -> None:  # pragma: no cover - lo implementan las pestañas
        raise NotImplementedError

    def shutdown(self) -> None:
        pass

    # -- figura ------------------------------------------------------------
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
        # Dibujado sincrónico: draw_idle() puede dejar el canvas en blanco al
        # cambiar de pestaña o redimensionar la ventana.
        self.canvas.draw()
        self.canvas.flush_events()

    def save_figure(self) -> None:
        """Guarda en un archivo la figura completa de la pestaña actual."""
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Guardar gráfica de la pestaña",
            initialdir=str(output_dir()),
            initialfile=timestamp("grafica", "png"),
            defaultextension=".png",
            filetypes=[("Imagen PNG", "*.png"), ("PDF", "*.pdf"),
                       ("SVG", "*.svg"), ("Todos los archivos", "*.*")],
        )
        if not path:
            return
        try:
            self.figure.savefig(path, dpi=200, facecolor="white")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Guardar gráfica",
                                 f"No se pudo guardar la imagen:\n{exc}",
                                 parent=self)
            self.app.set_status(f"Error al guardar la gráfica: {exc}", error=True)
            return
        self.save_info.set(f"Guardada: {Path(path).name}")
        self.app.set_status(f"Gráfica guardada en {path}")


def info_label(master: tk.Misc, columns: int = 1) -> ttk.Label:
    return ttk.Label(master, text="", justify="left", style="Info.TLabel", wraplength=1200)
