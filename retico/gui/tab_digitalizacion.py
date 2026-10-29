"""Pestaña 2: digitalización, ADC/DAC, error y tabla de códigos."""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from core.quantization import level_table, pcm_stream, sample_table

from .widgets import PlotTab, save_rows_csv


class TabDigitalizacion(PlotTab):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.window_ms_var = tk.DoubleVar(value=20.0)
        self.info_var = tk.StringVar(value="Sin datos.")
        self._build_controls()
        self._build_table()
        self.message("Cargue o grabe un audio para digitalizarlo.")

    # ------------------------------------------------------------------
    def _build_controls(self) -> None:
        c = self.controls
        ttk.Label(c, text="Ventana visible (ms)").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(c, from_=2, to=200, increment=5, textvariable=self.window_ms_var,
                    width=7, command=self.refresh).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(c, text="Actualizar", command=self.refresh).grid(row=0, column=2, padx=(0, 10))
        ttk.Label(c, text="La escalera naranja es la salida del DAC ideal.",
                  style="Subtitle.TLabel").grid(row=0, column=3, sticky="w")
        ttk.Label(c, textvariable=self.info_var, style="Info.TLabel", wraplength=1240,
                  justify="left").grid(row=1, column=0, columnspan=8, sticky="w", pady=(6, 2))

    def _build_table(self) -> None:
        frame = ttk.Frame(self.footer, padding=(10, 2, 10, 4))
        frame.pack(fill="x")
        left = ttk.Frame(frame)
        left.pack(side="left", fill="both", expand=True)

        columns = ("n", "t", "x", "v", "q", "code", "bin", "vdac", "err")
        headings = {
            "n": "n", "t": "t [ms]", "x": "x(n)", "v": "v = x·Vref [V]",
            "q": "q", "code": "código", "bin": "binario", "vdac": "V_DAC [V]",
            "err": "error [mV]",
        }
        widths = {"n": 55, "t": 75, "x": 85, "v": 95, "q": 60, "code": 70,
                  "bin": 95, "vdac": 90, "err": 85}
        self.table = ttk.Treeview(left, columns=columns, show="headings", height=7)
        for col in columns:
            self.table.heading(col, text=headings[col])
            self.table.column(col, width=widths[col], anchor="center", stretch=False)
        scroll = ttk.Scrollbar(left, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.pack(side="left", fill="both", expand=True)
        scroll.pack(side="left", fill="y")

        right = ttk.Frame(frame)
        right.pack(side="right", fill="y", padx=(8, 0))
        ttk.Button(right, text="Exportar CSV", command=self._export_csv).pack(fill="x", pady=1)
        ttk.Button(right, text="Niveles del ADC", command=self._show_levels).pack(fill="x", pady=1)
        ttk.Label(right, text="Primeras 40 muestras\nde la ventana mostrada.",
                  style="Subtitle.TLabel", justify="left").pack(anchor="w", pady=(6, 0))

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        d = self.app.get_digitized()
        if d is None:
            self.info_var.set("Sin datos. Grabe o cargue un audio y revise los parámetros.")
            self.message("Cargue o grabe un audio para digitalizarlo.")
            self._clear_table()
            return

        try:
            window_ms = float(self.window_ms_var.get())
        except ValueError:
            window_ms = 20.0
        window_ms = min(200.0, max(2.0, window_ms))
        window_s = window_ms / 1000.0

        n = max(2, min(len(d.x_n), int(round(window_s * d.fs_eff))))
        t_n = d.t_n[:n]
        x_n = d.x_n[:n]
        vq = d.quant.volts_q[:n]
        err = d.quant.error[:n]
        t_o, x_o = d.window_original(n / d.fs_eff)

        ax1, ax2 = self.axes(2, height_ratios=[2.2, 1.0])
        ax1.plot(t_o * 1e3, x_o, color="#1f77b4", lw=1.1, label="x(t) original")
        ax1.plot(t_n * 1e3, x_n, "o", ms=3.2, color="#1f77b4", label="x(n) muestras")
        ax1.step(t_n * 1e3, vq, where="post", color="#ff7f0e", lw=1.2,
                 label="V_DAC (escalera)")
        ax1.set_xlim(0.0, max(window_ms, 0.1))
        ax1.set_ylabel("Amplitud / tensión")
        ax1.set_title(f"Digitalización: {d.params.fs_target} Hz · {d.params.bits} bits · "
                      f"Δ = {d.quant.cfg.lsb * 1e3:.4f} mV")
        ax1.grid(alpha=0.25)
        ax1.legend(loc="upper right", fontsize=8, ncol=3)

        ax2.plot(t_n * 1e3, err * 1e3, "o-", ms=3, lw=0.8, color="#d62728",
                 label="error de cuantización")
        half = d.quant.cfg.lsb * 0.5 * 1e3
        ax2.axhline(half, color="#555555", ls="--", lw=0.8, label="±Δ/2")
        ax2.axhline(-half, color="#555555", ls="--", lw=0.8)
        ax2.axhline(0.0, color="#999999", lw=0.5)
        ax2.set_xlim(0.0, max(window_ms, 0.1))
        ax2.set_xlabel("Tiempo [ms]")
        ax2.set_ylabel("Error [mV]")
        ax2.grid(alpha=0.25)
        ax2.legend(loc="upper right", fontsize=8)
        self.draw()

        q = d.quant
        pcm = pcm_stream(q.codes_u, q.cfg, 12)
        self.info_var.set(
            f"fs original {d.audio.fs} Hz → fs de x(n) {d.fs_eff:.0f} Hz · "
            f"Ts = {1e6 / d.fs_eff:.2f} µs · N = {d.n_samples} muestras · "
            f"ganancia de normalización = {d.gain:.3f}\n"
            f"{q.cfg.bits} bits · niveles = {q.cfg.levels} · Δ = {q.cfg.lsb:.6f} V · "
            f"error máx = {q.error_max * 1e3:.4f} mV (cota Δ/2 = {half:.4f} mV) · "
            f"error RMS = {q.error_rms * 1e3:.4f} mV · saturadas = {q.clipped}\n"
            f"SQNR medida = {q.sqnr_db:.2f} dB · esperada = {q.sqnr_expected_db:.2f} dB · "
            f"senoide a plena escala = 6.02·N + 1.76 = {q.sqnr_sine_db:.2f} dB · "
            f"ENOB = {q.enob:.2f} bits · bitrate = {q.bitrate / 1000:.1f} kbit/s\n"
            f"Trama PCM (primeras 12 muestras): {pcm}"
        )
        self._fill_table(t_n, x_n, d, min(40, n))

    # ------------------------------------------------------------------
    def _clear_table(self) -> None:
        for item in self.table.get_children():
            self.table.delete(item)

    def _fill_table(self, t_n, x_n, d, rows: int) -> None:
        self._clear_table()
        for row in sample_table(t_n, x_n, d.quant, 0, rows):
            self.table.insert("", "end", values=row)

    def _export_csv(self) -> None:
        d = self.app.get_digitized()
        if d is None:
            messagebox.showinfo("Exportar CSV", "Primero cargue o grabe un audio.")
            return
        total = min(len(d.x_n), 100_000)
        rows = sample_table(d.t_n, d.x_n, d.quant, 0, total)
        header = ["n", "t_ms", "x_n", "v_V", "q", "codigo", "binario", "V_DAC_V", "error_mV"]
        path = save_rows_csv(self, "muestras_adc", header, rows)
        if path:
            extra = "" if total == len(d.x_n) else f" (limitado a {total} de {len(d.x_n)})"
            self.app.set_status(f"CSV exportado: {path}{extra}")

    def _show_levels(self) -> None:
        d = self.app.get_digitized()
        if d is None:
            messagebox.showinfo("Niveles del ADC", "Primero cargue o grabe un audio.")
            return
        rows, note = level_table(d.quant.cfg, d.quant.codes_u, max_rows=256)
        win = tk.Toplevel(self)
        win.title("Niveles del ADC")
        win.geometry("430x520")
        ttk.Label(win, text=note, style="Subtitle.TLabel", wraplength=400).pack(
            padx=10, pady=(8, 4), anchor="w")
        frame = ttk.Frame(win)
        frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        tv = ttk.Treeview(frame, columns=("code", "volt", "bin"), show="headings")
        for col, text, width in (("code", "Código", 100), ("volt", "Tensión [V]", 150),
                                 ("bin", "Binario", 150)):
            tv.heading(col, text=text)
            tv.column(col, width=width, anchor="center")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=tv.yview)
        tv.configure(yscrollcommand=scroll.set)
        tv.pack(side="left", fill="both", expand=True)
        scroll.pack(side="left", fill="y")
        for row in rows:
            tv.insert("", "end", values=row)

    def shutdown(self) -> None:
        pass
