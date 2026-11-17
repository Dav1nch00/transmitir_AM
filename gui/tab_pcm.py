"""Pestaña 9: PCM, palabra binaria, trama serie y código de línea."""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from core.pcm import (LINE_CODES, frame_words, line_waveform, pcm_bit_stream,
                      pcm_metrics)

from .widgets import PlotTab, save_rows_csv

_LINE_NOTES = {
    "NRZ unipolar": "1 → +1 durante todo el bit; 0 → 0 V.",
    "NRZ bipolar": "1 → +1; 0 → −1 durante todo el bit.",
    "RZ unipolar": "1 → +1 solo en la primera mitad del bit; 0 → 0.",
    "Manchester": "1 → +1 y luego −1; 0 → −1 y luego +1 (transición en el centro).",
}


class TabPCM(PlotTab):
    """Muestreo/cuantización, palabra PCM y trama serie con código de línea."""

    def __init__(self, master, app):
        super().__init__(master, app)
        self.window_ms_var = tk.DoubleVar(value=20.0)
        self.line_var = tk.StringVar(value="NRZ bipolar")
        self.words_var = tk.IntVar(value=8)
        self.info_var = tk.StringVar(value="Sin datos.")
        self._build_controls()
        self._build_table()
        self.message("Cargue o grabe un audio para estudiar su trama PCM.")

    # ------------------------------------------------------------------
    def _build_controls(self) -> None:
        c = self.controls
        ttk.Label(c, text="Ventana visible (ms)").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(c, from_=2, to=200, increment=5,
                    textvariable=self.window_ms_var, width=7,
                    command=self.refresh).grid(row=0, column=1, padx=(0, 12))

        ttk.Label(c, text="Código de línea").grid(row=0, column=2, sticky="w")
        line_combo = ttk.Combobox(c, textvariable=self.line_var, width=14,
                                  state="readonly", values=list(LINE_CODES))
        line_combo.grid(row=0, column=3, padx=(4, 12))
        line_combo.bind("<<ComboboxSelected>>", self._on_control_changed)

        ttk.Label(c, text="Palabras visibles").grid(row=0, column=4, sticky="w")
        ttk.Spinbox(c, from_=1, to=32, increment=1,
                    textvariable=self.words_var, width=5,
                    command=self.refresh).grid(row=0, column=5, padx=(4, 12))

        ttk.Button(c, text="Actualizar", command=self.refresh).grid(
            row=0, column=6, padx=(0, 14))

        ttk.Label(c, text="Bits MSB primero · niveles centrados en la palabra",
                  style="Subtitle.TLabel").grid(row=0, column=7, sticky="w")

        ttk.Label(c, textvariable=self.info_var, style="Info.TLabel",
                  wraplength=1240, justify="left").grid(
            row=1, column=0, columnspan=8, sticky="w", pady=(6, 2))

    def _build_table(self) -> None:
        self.table_frame = ttk.Frame(self.footer, padding=(10, 2, 10, 4))
        self.table_frame.pack(fill="x")
        left = ttk.Frame(self.table_frame)
        left.pack(side="left", fill="both", expand=True)

        columns = ("n", "t", "x", "q", "code", "vdac", "word")
        headings = {
            "n": "n", "t": "t [ms]", "x": "x(n)", "q": "q",
            "code": "código", "vdac": "V_DAC [V]", "word": "palabra PCM",
        }
        widths = {"n": 55, "t": 75, "x": 95, "q": 60, "code": 70,
                  "vdac": 100, "word": 130}
        self.table = ttk.Treeview(left, columns=columns, show="headings", height=5)
        for col in columns:
            self.table.heading(col, text=headings[col])
            self.table.column(col, width=widths[col], anchor="center", stretch=False)
        scroll = ttk.Scrollbar(left, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.pack(side="left", fill="both", expand=True)
        scroll.pack(side="left", fill="y")

        right = ttk.Frame(self.table_frame)
        right.pack(side="right", fill="y", padx=(8, 0))
        ttk.Button(right, text="Exportar CSV",
                   command=self._export_csv).pack(fill="x", pady=1)
        ttk.Label(right, text="Primeras 40 muestras\nde la ventana y su palabra.",
                  style="Subtitle.TLabel", justify="left").pack(anchor="w", pady=(6, 0))

    # ------------------------------------------------------------------
    def _on_control_changed(self, _event=None) -> None:
        self.refresh()
        self.app.set_status("Trama PCM actualizada.")

    def _values(self) -> tuple[float, int] | None:
        try:
            window_ms = float(self.window_ms_var.get())
        except (tk.TclError, ValueError):
            window_ms = 20.0
        window_ms = min(200.0, max(2.0, window_ms))
        try:
            words = int(self.words_var.get())
        except (tk.TclError, ValueError):
            words = 8
        return window_ms, min(32, max(1, words))

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        d = self.app.get_digitized()
        values = self._values()
        if d is None or values is None:
            self.info_var.set("Sin datos. Grabe o cargue un audio y revise los parámetros.")
            self.message("Cargue o grabe un audio para estudiar su trama PCM.")
            self._clear_table()
            return

        window_ms, word_count = values
        q = d.quant
        bits = int(q.cfg.bits)
        n = max(2, min(len(d.x_n), int(round(window_ms / 1000.0 * d.fs_eff))))
        t_n = d.t_n[:n]
        x_n = d.x_n[:n]
        vq = q.volts_q[:n]
        t_o, x_o = d.window_original(n / d.fs_eff)
        metrics = pcm_metrics(d.fs_eff, bits)

        stream = pcm_bit_stream(q.codes_u[:n], bits, word_count)
        t_line, y_line = line_waveform(stream, samples_per_bit=20,
                                       mode=self.line_var.get())

        ax1, ax2 = self.axes(2, height_ratios=[1.0, 1.15])
        ax1.plot(t_o * 1e3, x_o, color="#1f77b4", lw=1.1, label="x(t) original")
        ax1.plot(t_n * 1e3, x_n, "o", ms=3.0, color="#1f77b4",
                 label="x(n) muestras")
        ax1.step(t_n * 1e3, vq, where="post", color="#ff7f0e", lw=1.2,
                 label="V_DAC (escalera)")
        ax1.set_xlim(0.0, max(window_ms, 0.1))
        ax1.set_ylabel("Amplitud / tensión")
        ax1.set_title(
            f"Muestreo y cuantización · fs = {d.fs_eff:.0f} Hz · "
            f"{bits} bits/muestra · Δ = {q.cfg.lsb * 1e3:.4f} mV")
        ax1.grid(alpha=0.25)
        ax1.legend(loc="upper right", fontsize=8, ncol=3)

        nbits = int(stream.size)
        if t_line.size:
            ax2.plot(t_line, y_line, color="#2ca02c", lw=1.4)
            for w in range(max(0, nbits // bits)):
                ax2.axvspan(w * bits, (w + 1) * bits,
                            color="#2ca02c" if w % 2 == 0 else "#9467bd",
                            alpha=0.06)
                ax2.axvline(w * bits, color="#999999", lw=0.6, ls=":")
            ax2.axhline(0.0, color="#888888", lw=0.5)
            ax2.set_xlim(0.0, max(1.0, float(nbits)))
            ax2.set_xticks([w * bits + bits / 2.0
                            for w in range(max(0, nbits // bits))])
            ax2.set_xticklabels([str(w) for w in range(max(0, nbits // bits))])
            # Etiquetar los bits de la primera palabra para leer la trama.
            if 0 < bits <= 12 and nbits >= bits:
                for b in range(bits):
                    ax2.text(b + 0.5, 1.15, str(int(stream[b])),
                             ha="center", va="bottom", fontsize=8,
                             color="#333333")
            ax2.set_ylim(-1.35, 1.45)
            ax2.set_ylabel("Nivel de línea")
            ax2.set_xlabel("Tiempo [bits] · número de palabra en las marcas")
            ax2.set_title(
                f"Trama serie PCM · {self.line_var.get()} · "
                f"{nbits} bits = {nbits // bits} palabra(s) de {bits} bits")
            ax2.grid(axis="y", alpha=0.2)
        else:
            ax2.axis("off")
        self.draw()

        words = frame_words(q.codes_u[:n], bits, word_count)
        word_text = " · ".join(f"{i}:{text}" for i, (_code, text) in enumerate(words[:4]))
        self.info_var.set(
            f"fs de x(n) = {d.fs_eff:.0f} Hz · Ts = {1e6 / d.fs_eff:.2f} µs · "
            f"{bits} bits/muestra · niveles = {q.cfg.levels} · "
            f"Δ = {q.cfg.lsb:.6f} V\n"
            f"Tasa PCM = fs·N = {metrics.bitrate:.0f} bit/s "
            f"({metrics.bytes_per_second:.1f} B/s) · "
            f"Tb = {metrics.bit_period * 1e6:.2f} µs · "
            f"palabra = {metrics.word_period * 1e6:.2f} µs · "
            f"{n} muestras en {window_ms:.0f} ms\n"
            f"Código de línea «{self.line_var.get()}»: "
            f"{_LINE_NOTES.get(self.line_var.get(), '')} "
            f"Primeras palabras (código:binario) → {word_text}"
        )
        self._fill_table(t_n, x_n, q)

    # ------------------------------------------------------------------
    def _clear_table(self) -> None:
        for item in self.table.get_children():
            self.table.delete(item)

    def _fill_table(self, t_n, x_n, q) -> None:
        self._clear_table()
        bits = int(q.cfg.bits)
        end = min(len(x_n), 40)
        for i in range(end):
            self.table.insert("", "end", values=(
                str(i),
                f"{t_n[i] * 1e3:.3f}",
                f"{x_n[i]:+.6f}",
                f"{int(q.codes[i]):+d}",
                str(int(q.codes_u[i])),
                f"{q.volts_q[i]:+.6f}",
                format(int(q.codes_u[i]), f"0{bits}b"),
            ))

    def _export_csv(self) -> None:
        d = self.app.get_digitized()
        if d is None:
            messagebox.showinfo("Exportar CSV", "Primero cargue o grabe un audio.")
            return
        q = d.quant
        bits = int(q.cfg.bits)
        total = min(len(d.x_n), 100_000)
        rows = []
        for i in range(total):
            rows.append((
                str(i),
                f"{d.t_n[i] * 1e3:.4f}",
                f"{d.x_n[i]:+.8f}",
                f"{int(q.codes[i]):+d}",
                str(int(q.codes_u[i])),
                f"{q.volts_q[i]:+.8f}",
                format(int(q.codes_u[i]), f"0{bits}b"),
            ))
        header = ["n", "t_ms", "x_n", "q", "codigo", "V_DAC_V", "palabra_pcm"]
        path = save_rows_csv(self, "trama_pcm", header, rows)
        if path:
            extra = "" if total == len(d.x_n) else f" (limitado a {total} de {len(d.x_n)})"
            self.app.set_status(f"CSV PCM exportado: {path}{extra}")
