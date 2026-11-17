"""Pestaña 2: digitalización, ADC/DAC, error y tabla de códigos."""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from core.dsp import bandwidth_99, display_slice, spectrum
from core.quantization import level_table, pcm_stream, sample_table

from .widgets import PlotTab, save_rows_csv


def _byte_word(n: int) -> str:
    return f"{int(n)} byte" if int(n) == 1 else f"{int(n)} bytes"


class TabDigitalizacion(PlotTab):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.window_ms_var = tk.DoubleVar(value=20.0)
        self.view_var = tk.StringVar(value="Digitalización")
        self.unit_var = tk.StringVar(value="Bit")
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
        ttk.Button(c, text="Actualizar", command=self.refresh).grid(row=0, column=2, padx=(0, 14))

        ttk.Label(c, text="Vista").grid(row=0, column=3, sticky="w")
        self.view_combo = ttk.Combobox(c, textvariable=self.view_var, width=20,
                                       state="readonly",
                                       values=["Digitalización", "Muestreo Nyquist"])
        self.view_combo.grid(row=0, column=4, padx=(4, 14))

        ttk.Label(c, text="Mostrar resolución en").grid(row=0, column=5, sticky="w")
        self.unit_combo = ttk.Combobox(c, textvariable=self.unit_var, width=7,
                                       state="readonly", values=["Bit", "Byte"])
        self.unit_combo.grid(row=0, column=6, padx=(4, 12))

        ttk.Label(c, text="1 byte = 8 bits · el cambio es solo de visualización",
                  style="Subtitle.TLabel").grid(row=0, column=7, sticky="w")

        ttk.Label(c, textvariable=self.info_var, style="Info.TLabel", wraplength=1240,
                  justify="left").grid(row=1, column=0, columnspan=8, sticky="w",
                                       pady=(6, 2))

        self.view_combo.bind("<<ComboboxSelected>>", self._on_display_changed)
        self.unit_combo.bind("<<ComboboxSelected>>", self._on_display_changed)

    def _build_table(self) -> None:
        self.table_frame = ttk.Frame(self.footer, padding=(10, 2, 10, 4))
        self.table_frame.pack(fill="x")
        left = ttk.Frame(self.table_frame)
        left.pack(side="left", fill="both", expand=True)

        columns = ("n", "t", "x", "v", "q", "code", "bin", "vdac", "err")
        headings = {
            "n": "n", "t": "t [ms]", "x": "x(n)", "v": "v = x·Vref [V]",
            "q": "q", "code": "código", "bin": "binario", "vdac": "V_DAC [V]",
            "err": "error [mV]",
        }
        widths = {"n": 55, "t": 75, "x": 85, "v": 95, "q": 60, "code": 70,
                  "bin": 95, "vdac": 90, "err": 85}
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
        ttk.Button(right, text="Exportar CSV", command=self._export_csv).pack(fill="x", pady=1)
        ttk.Button(right, text="Niveles del ADC", command=self._show_levels).pack(fill="x", pady=1)
        ttk.Label(right, text="Primeras 40 muestras\nde la ventana mostrada.",
                  style="Subtitle.TLabel", justify="left").pack(anchor="w", pady=(6, 0))

    # ------------------------------------------------------------------
    def _on_display_changed(self, _event=None) -> None:
        self.refresh()
        self.app.set_status("Vista actualizada sin volver a digitalizar.")

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        d = self.app.get_digitized()
        if d is None:
            self.info_var.set("Sin datos. Grabe o cargue un audio y revise los parámetros.")
            self.message("Cargue o grabe un audio para digitalizarlo.")
            self._clear_table()
            return

        if self.view_var.get().startswith("Muestreo"):
            self._refresh_nyquist(d)
        else:
            self._refresh_digitalization(d)

    # ------------------------------------------------------------------
    def _refresh_digitalization(self, d) -> None:
        self.show_footer(True)
        if not self.table_frame.winfo_manager():
            self.table_frame.pack(fill="x")

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

        q = d.quant
        byte_mode = self.unit_var.get().lower().startswith("byte")
        bits = int(q.cfg.bits)
        nbytes = max(1, (bits + 7) // 8)
        unit = "byte" if byte_mode else "bit"
        unidad_titulo = _byte_word(nbytes) if byte_mode else f"{bits} bits"

        self.table.heading("bin", text=(f"hex ({_byte_word(nbytes)})" if byte_mode
                                        else f"binario ({bits} bits)"))
        self.table.column("bin", width=140 if byte_mode else 95, anchor="center",
                          stretch=False)

        ax1, ax2 = self.axes(2, height_ratios=[2.2, 1.0])
        ax1.plot(t_o * 1e3, x_o, color="#1f77b4", lw=1.1, label="x(t) original")
        ax1.plot(t_n * 1e3, x_n, "o", ms=3.2, color="#1f77b4", label="x(n) muestras")
        ax1.step(t_n * 1e3, vq, where="post", color="#ff7f0e", lw=1.2,
                 label="V_DAC (escalera)")
        ax1.set_xlim(0.0, max(window_ms, 0.1))
        ax1.set_ylabel("Amplitud / tensión")
        ax1.set_title(f"Digitalización: {d.params.fs_target} Hz · {unidad_titulo} · "
                      f"Δ = {q.cfg.lsb * 1e3:.4f} mV")
        ax1.grid(alpha=0.25)
        ax1.legend(loc="upper right", fontsize=8, ncol=3)

        ax2.plot(t_n * 1e3, err * 1e3, "o-", ms=3, lw=0.8, color="#d62728",
                 label="error de cuantización")
        half = q.cfg.lsb * 0.5 * 1e3
        ax2.axhline(half, color="#555555", ls="--", lw=0.8, label="±Δ/2")
        ax2.axhline(-half, color="#555555", ls="--", lw=0.8)
        ax2.axhline(0.0, color="#999999", lw=0.5)
        ax2.set_xlim(0.0, max(window_ms, 0.1))
        ax2.set_xlabel("Tiempo [ms]")
        ax2.set_ylabel("Error [mV]")
        ax2.grid(alpha=0.25)
        ax2.legend(loc="upper right", fontsize=8)
        self.draw()

        pcm = pcm_stream(q.codes_u, q.cfg, 12, unit=unit)
        if byte_mode:
            resolucion = (
                f"Resolución mostrada: {_byte_word(nbytes)} = {bits} bits · "
                f"niveles = {q.cfg.levels} · Δ = {q.cfg.lsb:.6f} V\n"
                f"Tasa = {q.bitrate / 8 / 1000:.1f} kB/s · "
                f"tamaño = {len(d.x_n) * nbytes / 1024:.1f} kB "
                f"({_byte_word(nbytes)} por muestra) · "
                f"bitrate equivalente = {q.bitrate / 1000:.1f} kbit/s"
            )
            pcm_label = "Trama PCM en bytes (hex)"
        else:
            resolucion = (
                f"Resolución mostrada: {bits} bits · niveles = {q.cfg.levels} · "
                f"Δ = {q.cfg.lsb:.6f} V · bitrate = {q.bitrate / 1000:.1f} kbit/s"
            )
            pcm_label = "Trama PCM en bits (binario)"

        self.info_var.set(
            f"fs original {d.audio.fs} Hz → fs de x(n) {d.fs_eff:.0f} Hz · "
            f"Ts = {1e6 / d.fs_eff:.2f} µs · N = {d.n_samples} muestras · "
            f"ganancia de normalización = {d.gain:.3f}\n"
            f"{resolucion}\n"
            f"error máx = {q.error_max * 1e3:.4f} mV (cota Δ/2 = {half:.4f} mV) · "
            f"error RMS = {q.error_rms * 1e3:.4f} mV · saturadas = {q.clipped}\n"
            f"SQNR medida = {q.sqnr_db:.2f} dB · esperada = {q.sqnr_expected_db:.2f} dB · "
            f"senoide a plena escala = 6.02·N + 1.76 = {q.sqnr_sine_db:.2f} dB · "
            f"ENOB = {q.enob:.2f} bits\n"
            f"{pcm_label} (primeras 12 muestras): {pcm}"
        )
        self._fill_table(t_n, x_n, d, min(40, n), unit=unit)

    # ------------------------------------------------------------------
    def _refresh_nyquist(self, d) -> None:
        if self.table_frame.winfo_manager():
            self.table_frame.pack_forget()
        self.show_footer(False)

        p = d.params
        fs = float(d.fs_eff)
        f_o, m_o = spectrum(d.x_orig, d.audio.fs, p.window)
        f_n, m_n = spectrum(d.x_n, fs, p.window)
        f_max = bandwidth_99(f_o, m_o) if f_o.size else 0.0
        nyq = fs / 2.0
        fs_min = 2.0 * f_max
        ok = fs >= fs_min

        try:
            window_ms = float(self.window_ms_var.get())
        except ValueError:
            window_ms = 20.0
        window_ms = min(200.0, max(2.0, window_ms))
        n = max(2, min(len(d.x_n), int(round(window_ms / 1000.0 * fs))))
        t_n = d.t_n[:n] * 1e3
        x_n = d.x_n[:n]
        t_o, x_o = d.window_original(n / fs)
        t_od, x_od = display_slice(t_o, x_o, 6000)

        step = max(1, n // 4000)
        t_plot = t_n[::step]
        x_plot = x_n[::step]

        ax1, ax2 = self.axes(2, height_ratios=[1.0, 1.0])
        ax1.plot(t_od * 1e3, x_od, color="#1f77b4", lw=1.1, label="x(t) original")
        ax1.vlines(t_plot, 0.0, x_plot, color="#ff7f0e", lw=0.6, alpha=0.85)
        ax1.plot(t_plot, x_plot, "o", ms=3.0, color="#d62728",
                 label="muestras x[n]")
        ax1.axhline(0.0, color="#888888", lw=0.5)
        ax1.set_xlim(0.0, max(window_ms, 0.1))
        ax1.set_xlabel("Tiempo [ms]")
        ax1.set_ylabel("Amplitud")
        ax1.set_title(f"Muestreo NQS · fs = {fs:.0f} Hz · Ts = {1e6 / fs:.2f} µs · "
                      f"{n} muestras en {window_ms:.0f} ms")
        ax1.grid(alpha=0.25)
        ax1.legend(loc="upper right", fontsize=8, ncol=2)

        if f_o.size:
            ax2.plot(f_o, m_o * p.vref, color="#333333", lw=0.8,
                     label="espectro original")
            ax2.plot(f_n, m_n * p.vref, color="#2ca02c", lw=0.9,
                     label="espectro de x[n]")
            ax2.axvline(nyq, color="#d62728", ls="--", lw=1.0,
                        label=f"Nyquist fs/2 = {nyq:.0f} Hz")
            ax2.axvline(f_max, color="#ff7f0e", ls=":", lw=1.0,
                        label=f"f_max 99 % = {f_max:.0f} Hz")
            xmax = min(float(f_o[-1]), max(f_max * 1.25, nyq * 1.05, 1.0))
            if nyq < xmax:
                ax2.axvspan(nyq, xmax, color="#d62728", alpha=0.08)
            ax2.set_xlim(0.0, max(xmax, 1.0))
        ax2.set_xlabel("Frecuencia [Hz]")
        ax2.set_ylabel("Amplitud [V]")
        ax2.set_title("Criterio NQS: la energía por encima de fs/2 se plegaría")
        ax2.grid(alpha=0.25)
        ax2.legend(fontsize=8, loc="upper right")
        self.draw()

        bits = int(d.quant.cfg.bits)
        nbytes = max(1, (bits + 7) // 8)
        byte_mode = self.unit_var.get().lower().startswith("byte")
        unidad = f"{_byte_word(nbytes)} ({bits} bits)" if byte_mode else f"{bits} bits"
        estado = "CUMPLE" if ok else "NO CUMPLE"
        filtro = "activado" if p.antialias else "desactivado"
        self.info_var.set(
            f"Criterio NQS (Nyquist–Shannon): fs ≥ 2·f_max.\n"
            f"f_max (99 % de la energía) = {f_max:.1f} Hz · fs = {fs:.0f} Hz · "
            f"fs/2 = {nyq:.0f} Hz · 2·f_max = {fs_min:.0f} Hz → {estado}.\n"
            f"{n} muestras mostradas · Ts = {1e6 / fs:.2f} µs · "
            f"resolución {unidad} · filtro antialiasing {filtro}. "
            + ("La señal se puede reconstruir sin aliasing." if ok else
               "Hay componentes por encima de fs/2: aparecerá aliasing.")
        )

    # ------------------------------------------------------------------
    def _clear_table(self) -> None:
        for item in self.table.get_children():
            self.table.delete(item)

    def _fill_table(self, t_n, x_n, d, rows: int, unit: str = "bit") -> None:
        self._clear_table()
        for row in sample_table(t_n, x_n, d.quant, 0, rows, unit=unit):
            self.table.insert("", "end", values=row)

    def _export_csv(self) -> None:
        d = self.app.get_digitized()
        if d is None:
            messagebox.showinfo("Exportar CSV", "Primero cargue o grabe un audio.")
            return
        total = min(len(d.x_n), 100_000)
        unit = "byte" if self.unit_var.get().lower().startswith("byte") else "bit"
        rows = sample_table(d.t_n, d.x_n, d.quant, 0, total, unit=unit)
        header = ["n", "t_ms", "x_n", "v_V", "q", "codigo",
                  "hex_bytes" if unit == "byte" else "binario",
                  "V_DAC_V", "error_mV"]
        path = save_rows_csv(self, "muestras_adc", header, rows)
        if path:
            extra = "" if total == len(d.x_n) else f" (limitado a {total} de {len(d.x_n)})"
            self.app.set_status(f"CSV exportado: {path}{extra}")

    def _show_levels(self) -> None:
        d = self.app.get_digitized()
        if d is None:
            messagebox.showinfo("Niveles del ADC", "Primero cargue o grabe un audio.")
            return
        rows, note = level_table(d.quant.cfg, d.quant.codes_u, max_rows=256,
                                 unit="byte" if self.unit_var.get().lower().startswith("byte")
                                 else "bit")
        win = tk.Toplevel(self)
        win.title("Niveles del ADC")
        win.geometry("430x520")
        ttk.Label(win, text=note, style="Subtitle.TLabel", wraplength=400).pack(
            padx=10, pady=(8, 4), anchor="w")
        frame = ttk.Frame(win)
        frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        tv = ttk.Treeview(frame, columns=("code", "volt", "bin"), show="headings")
        code_label = ("Hex (bytes)" if self.unit_var.get().lower().startswith("byte")
                      else "Binario")
        for col, text, width in (("code", "Código", 100), ("volt", "Tensión [V]", 150),
                                 ("bin", code_label, 150)):
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
