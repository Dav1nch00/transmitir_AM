"""Pestaña 3: FFT, espectro unilateral y espectrograma."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

from core.dsp import (bandwidth_99, db, peak_frequency, resolution,
                      spectral_centroid, spectrogram, spectrum)

from .widgets import PlotTab


class TabEspectro(PlotTab):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.db_var = tk.BooleanVar(value=True)
        self.info_var = tk.StringVar(value="Sin datos.")
        self._build_controls()
        self.message("Reciba una señal para calcular su FFT.")

    def _build_controls(self) -> None:
        c = self.controls
        ttk.Checkbutton(c, text="Espectro en dBV", variable=self.db_var,
                        command=self.refresh).grid(row=0, column=0, padx=(0, 12))
        ttk.Label(c, text="El espectrograma usa la señal digitalizada.",
                  style="Subtitle.TLabel").grid(row=0, column=1, sticky="w")
        ttk.Label(c, textvariable=self.info_var, style="Info.TLabel", wraplength=1240,
                  justify="left").grid(row=1, column=0, columnspan=4, sticky="w", pady=(6, 2))

    def refresh(self) -> None:
        d = self.app.get_digitized()
        if d is None:
            self.info_var.set("Sin datos. Reciba una señal y revise los parámetros.")
            self.message("Reciba una señal para calcular su FFT.")
            return

        p = d.params
        f_o, m_o = spectrum(d.x_orig, d.audio.fs, p.window)
        f_n, m_n = spectrum(d.x_n, d.fs_eff, p.window)
        if f_o.size == 0 or f_n.size == 0:
            self.message("El audio es demasiado corto para calcular la FFT.")
            return

        use_db = bool(self.db_var.get())
        y_o = db(m_o * p.vref) if use_db else m_o * p.vref
        y_n = db(m_n * p.vref) if use_db else m_n * p.vref
        ylabel = "Amplitud [dBV]" if use_db else "Amplitud [V]"

        ax1, ax2, ax3 = self.axes(3, height_ratios=[1.0, 1.0, 1.15])

        self._plot_spectrum(ax1, f_o, y_o, m_o, d.audio.fs, ylabel, "original")
        self._plot_spectrum(ax2, f_n, y_n, m_n, d.fs_eff, ylabel, "digitalizada")

        f_s, t_s, sxx = spectrogram(d.x_n, d.fs_eff, p.window,
                                    nperseg=min(1024, max(64, len(d.x_n) // 8)),
                                    noverlap=min(768, max(32, len(d.x_n) // 16)))
        vmax = float(np.max(sxx)) if sxx.size else 0.0
        mesh = ax3.pcolormesh(t_s, f_s, sxx, shading="auto",
                              vmin=vmax - 80.0, vmax=vmax, cmap="magma")
        ax3.set_xlabel("Tiempo [s]")
        ax3.set_ylabel("Frecuencia [Hz]")
        ax3.set_title("Espectrograma de x(n)")
        ax3.set_ylim(0.0, d.fs_eff / 2.0)
        self.figure.colorbar(mesh, ax=ax3, label="Magnitud [dB]")
        self.draw()

        peak = peak_frequency(f_n, m_n)
        bw = bandwidth_99(f_n, m_n)
        centroid = spectral_centroid(f_n, m_n, fmax=d.fs_eff / 2.0)
        df_o = resolution(d.audio.fs, len(d.x_orig))
        df_n = resolution(d.fs_eff, len(d.x_n))
        nyq = d.fs_eff / 2.0
        verdict = "cumple" if d.fs_eff >= 2 * bw else "no cumple"
        self.info_var.set(
            f"Ventana {p.window} · N original = {len(d.x_orig)} (Δf = {df_o:.3f} Hz) · "
            f"N digitalizada = {len(d.x_n)} (Δf = {df_n:.3f} Hz)\n"
            f"Pico espectral = {peak:.1f} Hz · ancho de banda al 99 % = {bw:.1f} Hz · "
            f"centroide = {centroid:.1f} Hz · Nyquist de x(n) = {nyq:.1f} Hz → {verdict}\n"
            f"La resolución frecuencial es Δf = fs/N; la ventana {p.window} reduce la fuga "
            f"espectral a costa de ensanchar el lóbulo principal."
        )

    def _plot_spectrum(self, ax, f: np.ndarray, y: np.ndarray, mag: np.ndarray,
                       fs: float, ylabel: str, nombre: str) -> None:
        ax.plot(f, y, color="#1f77b4" if nombre == "original" else "#2ca02c", lw=0.8)
        peak = peak_frequency(f, mag)
        ax.axvline(peak, color="#d62728", ls="--", lw=0.9)
        ax.annotate(f"{peak:.1f} Hz", xy=(peak, ax.get_ylim()[1]),
                    xytext=(4, -12), textcoords="offset points", color="#d62728",
                    fontsize=8)
        ax.axvline(fs / 2.0, color="#777777", ls=":", lw=0.9)
        ax.set_xlim(0.0, fs / 2.0)
        ax.set_xlabel("Frecuencia [Hz]")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Espectro de la señal {nombre} · fs = {fs:.0f} Hz · "
                     f"Nyquist = {fs / 2:.0f} Hz")
        ax.grid(alpha=0.25)