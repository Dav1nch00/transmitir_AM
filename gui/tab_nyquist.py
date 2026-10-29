"""Pestaña 4: Nyquist, aliasing, demostración con tonos y reconstrucción."""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from core.audio_io import Player
from core.dsp import (alias_frequency, bandwidth_99, energy_above, pick_window_start,
                      resample_to, sinc_interp, spectrum, zero_order_hold)
from core.pipeline import fs_key

from .widgets import PlotTab

FS_CHOICES = [8000, 16000, 22050, 44100]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


class TabNyquist(PlotTab):
    def __init__(self, master, app, mode=None):
        super().__init__(master, app)
        self.fixed_mode = mode if mode in ("aliasing", "recon") else None
        self.mode_var = tk.StringVar(value=self.fixed_mode or "aliasing")
        self.info_var = tk.StringVar(value="Sin datos.")
        self.tone_f = tk.StringVar(value="7000")
        self.tone_fs = tk.StringVar(value="8000")
        self.recon_window_ms = tk.DoubleVar(value=5.0)
        self.play_info = tk.StringVar(value="")
        self._fs_vars: dict[int, tk.BooleanVar] = {}
        self._player: Player | None = None
        self._player_poll_id: str | None = None

        self._build_controls()
        self._build_table()
        self._switch_mode()
        self.message("Cargue o grabe un audio para analizar Nyquist y aliasing.")

    # ------------------------------------------------------------------
    def _build_controls(self) -> None:
        c = self.controls
        if self.fixed_mode is None:
            mode = ttk.Frame(c)
            mode.grid(row=0, column=0, columnspan=10, sticky="w", pady=(0, 4))
            ttk.Label(mode, text="Vista:").pack(side="left", padx=(0, 6))
            ttk.Radiobutton(mode, text="Muestreo y aliasing", value="aliasing",
                            variable=self.mode_var, command=self._switch_mode).pack(side="left")
            ttk.Radiobutton(mode, text="Reconstrucción", value="recon",
                            variable=self.mode_var, command=self._switch_mode).pack(
                side="left", padx=(10, 0))
            ttk.Label(mode, textvariable=self.play_info, style="Metric.TLabel").pack(
                side="left", padx=(16, 0))
        elif self.fixed_mode == "recon":
            ttk.Label(c, textvariable=self.play_info, style="Metric.TLabel").grid(
                row=0, column=0, sticky="w", pady=(0, 4))

        self.alias_ctrl = ttk.Frame(c)
        self.alias_ctrl.grid(row=1, column=0, columnspan=10, sticky="w")
        ttk.Label(self.alias_ctrl, text="Comparar fs:").grid(row=0, column=0, padx=(0, 6))
        for i, fs in enumerate(FS_CHOICES):
            var = tk.BooleanVar(value=fs in (8000, 16000, 44100))
            self._fs_vars[fs] = var
            ttk.Checkbutton(self.alias_ctrl, text=fs_key(fs), variable=var,
                            command=self.refresh).grid(row=0, column=i + 1, padx=(0, 8))

        ttk.Label(self.alias_ctrl, text="Demo de tono: f (Hz)").grid(
            row=0, column=5, padx=(16, 4))
        ttk.Entry(self.alias_ctrl, textvariable=self.tone_f, width=7).grid(row=0, column=6)
        ttk.Label(self.alias_ctrl, text="fs (Hz)").grid(row=0, column=7, padx=(8, 4))
        ttk.Entry(self.alias_ctrl, textvariable=self.tone_fs, width=7).grid(row=0, column=8)
        ttk.Button(self.alias_ctrl, text="Demostrar", command=self._tone_demo).grid(
            row=0, column=9, padx=(8, 0))
        ttk.Label(self.alias_ctrl, text="(la comparación usa los primeros 10 s)",
                  style="Subtitle.TLabel").grid(row=0, column=10, padx=(12, 0))

        self.recon_ctrl = ttk.Frame(c)
        self.recon_ctrl.grid(row=1, column=0, columnspan=10, sticky="w")
        ttk.Label(self.recon_ctrl, text="Ventana visible (ms)").grid(row=0, column=0)
        ttk.Spinbox(self.recon_ctrl, from_=1, to=50, increment=1,
                    textvariable=self.recon_window_ms, width=6,
                    command=self.refresh).grid(row=0, column=1, padx=(4, 14))
        ttk.Button(self.recon_ctrl, text="▶ Original", command=lambda: self._play("original")).grid(
            row=0, column=2, padx=2)
        ttk.Button(self.recon_ctrl, text="▶ Cuantizada", command=lambda: self._play("cuantizada")).grid(
            row=0, column=3, padx=2)
        ttk.Button(self.recon_ctrl, text="▶ Reconstruida", command=lambda: self._play("recon")).grid(
            row=0, column=4, padx=2)
        ttk.Button(self.recon_ctrl, text="■ Detener", command=self._stop_play).grid(
            row=0, column=5, padx=(8, 12))
        ttk.Label(self.recon_ctrl, text="La reconstruida se remuestrea a la fs original "
                                        "para poder reproducirla.",
                  style="Subtitle.TLabel").grid(row=0, column=6)

        ttk.Label(c, textvariable=self.info_var, style="Info.TLabel", wraplength=1240,
                  justify="left").grid(row=2, column=0, columnspan=10, sticky="w", pady=(6, 2))

    def _build_table(self) -> None:
        self.table_frame = ttk.Frame(self.footer, padding=(10, 0, 10, 4))
        columns = ("fs", "nyq", "fmax", "energia", "veredicto")
        headings = {"fs": "fs [Hz]", "nyq": "Nyquist [Hz]", "fmax": "f_max 99 % [Hz]",
                    "energia": "Energía > Nyquist", "veredicto": "¿Cumple Nyquist?"}
        widths = {"fs": 110, "nyq": 120, "fmax": 140, "energia": 160, "veredicto": 240}
        self.alias_table = ttk.Treeview(self.table_frame, columns=columns,
                                        show="headings", height=4)
        for col in columns:
            self.alias_table.heading(col, text=headings[col])
            self.alias_table.column(col, width=widths[col], anchor="center")
        self.alias_table.pack(fill="x")

    def _switch_mode(self) -> None:
        if self.fixed_mode is not None:
            self.mode_var.set(self.fixed_mode)
        if self.mode_var.get() == "aliasing":
            self.recon_ctrl.grid_remove()
            self.alias_ctrl.grid()
        else:
            self.alias_ctrl.grid_remove()
            self.recon_ctrl.grid()
        self.mark_dirty()
        if self.winfo_ismapped():
            self.on_show()

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        if self.mode_var.get() == "aliasing":
            self._refresh_aliasing()
        else:
            self._refresh_reconstruction()

    def _selected_fs(self) -> list[int]:
        return [fs for fs, var in self._fs_vars.items() if var.get()]

    def _refresh_aliasing(self) -> None:
        if not self.table_frame.winfo_manager():
            self.table_frame.pack(fill="x", padx=0, pady=0)
        d = self.app.get_digitized()
        if d is None:
            self.info_var.set("Sin datos. Grabe o cargue un audio y revise los parámetros.")
            self.message("Cargue o grabe un audio para analizar Nyquist y aliasing.")
            for item in self.alias_table.get_children():
                self.alias_table.delete(item)
            return

        p = d.params
        fs_orig = d.audio.fs
        limit = min(len(d.x_orig), max(2048, int(10.0 * fs_orig)))
        x_cmp = d.x_orig[:limit]
        f0, m0 = spectrum(x_cmp, fs_orig, p.window)
        amp0 = m0 * p.vref
        f_max = bandwidth_99(f0, m0)
        fs_min = 2.0 * f_max

        selected = self._selected_fs()
        if not selected:
            selected = [8000, 16000]

        ax1, ax2 = self.axes(2, height_ratios=[1.0, 1.0])
        ax1.plot(f0, amp0, color="#333333", lw=0.8, label="original")
        for i, fs in enumerate(selected):
            ax1.axvline(fs / 2.0, color=COLORS[i % len(COLORS)], ls=":", lw=1.0,
                        label=f"Nyquist {fs_key(fs)}")
        ax1.set_xlim(0.0, fs_orig / 2.0)
        ax1.set_xlabel("Frecuencia [Hz]")
        ax1.set_ylabel("Amplitud [V]")
        ax1.set_title(f"Espectro original y límites de Nyquist · f_max (99 %) = {f_max:.1f} Hz")
        ax1.grid(alpha=0.25)
        ax1.legend(fontsize=8, loc="upper right")

        rows = []
        for i, fs in enumerate(selected):
            y, fs_eff = resample_to(x_cmp, fs_orig, fs, p.antialias)
            f_s, m_s = spectrum(y, fs_eff, p.window)
            ax2.plot(f_s, m_s * p.vref, color=COLORS[i % len(COLORS)], lw=0.9,
                     label=f"x(n) a {fs_key(fs)}")
            frac = energy_above(f0, m0, fs / 2.0)
            ok = fs >= fs_min
            rows.append((str(fs), f"{fs / 2:.0f}", f"{f_max:.1f}",
                         f"{frac * 100:.2f} %", "SÍ" if ok else "NO"))
        ax2.set_xlim(0.0, max(selected) / 2.0)
        ax2.set_xlabel("Frecuencia [Hz]")
        ax2.set_ylabel("Amplitud [V]")
        ax2.set_title("Espectro de la señal muestreada: las componentes plegadas aparecen aquí")
        ax2.grid(alpha=0.25)
        ax2.legend(fontsize=8, loc="upper right")
        self.draw()

        for item in self.alias_table.get_children():
            self.alias_table.delete(item)
        for row in rows:
            self.alias_table.insert("", "end", values=row)

        actual_ok = d.fs_eff >= fs_min
        estado = "CUMPLE" if actual_ok else "NO CUMPLE"
        filtro = "activado (resample_poly)" if p.antialias else "desactivado (muestreo directo)"
        self.info_var.set(
            f"f_max (99 % de la energía) = {f_max:.1f} Hz → Nyquist-Shannon exige "
            f"fs ≥ 2·f_max = {fs_min:.1f} Hz.\n"
            f"fs actual = {d.fs_eff:.0f} Hz → {estado}. Filtro antialiasing {filtro}. "
            f"Ventana {p.window}.\n"
            f"La columna «Energía > Nyquist» mide, en el espectro original, cuánta energía "
            f"quedaría por encima de fs/2 y se plegaría si no se filtra."
        )

    # ------------------------------------------------------------------
    def _refresh_reconstruction(self) -> None:
        if self.table_frame.winfo_manager():
            self.table_frame.pack_forget()
        d = self.app.get_digitized()
        if d is None:
            self.info_var.set("Sin datos. Grabe o cargue un audio y revise los parámetros.")
            self.message("Cargue o grabe un audio para reconstruir la señal.")
            return

        p = d.params
        try:
            win_ms = float(self.recon_window_ms.get())
        except ValueError:
            win_ms = 5.0
        win_ms = min(50.0, max(1.0, win_ms))
        win_s = win_ms / 1000.0
        n = max(2, min(len(d.x_n), int(round(win_s * d.fs_eff))))
        start = pick_window_start(d.x_n, d.fs_eff, win_s)
        start = min(start, max(0, len(d.x_n) - n))
        idx = np.arange(start, start + n)
        t_s = d.t_n[idx]

        oversample = 20
        n_out = max(2, n * oversample)
        t_out = (start + np.arange(n_out) / oversample) / d.fs_eff
        sinc = sinc_interp(d.x_n, d.fs_eff, t_out, half_width=32)
        zoh = zero_order_hold(d.x_n, d.fs_eff, t_out)

        scale = p.vref * 1000.0  # a milivoltios
        ax1, ax2 = self.axes(2, height_ratios=[2.2, 1.0])
        ax1.step(t_out * 1e3, zoh * scale, where="post", color="#ff7f0e", lw=1.1,
                 label="DAC: retención de orden cero")
        ax1.plot(t_out * 1e3, sinc * scale, color="#1f77b4", lw=1.2,
                 label="reconstrucción sinc (65 muestras)")
        ax1.plot(t_s * 1e3, d.x_n[idx] * scale, "o", ms=4, color="#2ca02c",
                 label="muestras x[n]")
        ax1.set_xlim(t_s[0] * 1e3, t_s[-1] * 1e3)
        ax1.set_xlabel("Tiempo [ms]")
        ax1.set_ylabel("Tensión [mV]")
        ax1.set_title(f"Reconstrucción · fs = {d.fs_eff:.0f} Hz · Ts = {1e6 / d.fs_eff:.2f} µs")
        ax1.grid(alpha=0.25)
        ax1.legend(fontsize=8, loc="upper right")

        err = (zoh - sinc) * scale
        ax2.plot(t_out * 1e3, err, color="#d62728", lw=0.9)
        ax2.axhline(0.0, color="#999999", lw=0.5)
        ax2.set_xlim(t_s[0] * 1e3, t_s[-1] * 1e3)
        ax2.set_xlabel("Tiempo [ms]")
        ax2.set_ylabel("ZOH − sinc [mV]")
        ax2.set_title("Diferencia entre la retención de orden cero y la interpolación sinc")
        ax2.grid(alpha=0.25)
        self.draw()

        e_rms = float(np.sqrt(np.mean(err ** 2))) if len(err) else 0.0
        e_max = float(np.max(np.abs(err))) if len(err) else 0.0
        self.info_var.set(
            f"Se reconstruye una ventana de {win_ms:.1f} ms con {n} muestras "
            f"(inicio en t = {t_s[0]:.4f} s).\n"
            f"Error ZOH−sinc: RMS = {e_rms:.2f} mV · máximo = {e_max:.2f} mV. "
            f"Cuanto menor sea fs, más visible es la escalera del DAC.\n"
            f"«Reconstruida» remuestrea la señal cuantizada a {d.audio.fs} Hz con un filtro "
            f"polifásico; el sinc de la gráfica usa 65 muestras por punto."
        )

    # ------------------------------------------------------------------
    def _tone_demo(self) -> None:
        try:
            f_tone = float(self.tone_f.get().replace(",", "."))
            fs_tone = float(self.tone_fs.get().replace(",", "."))
        except ValueError:
            messagebox.showerror("Demo de tono", "f y fs deben ser números.")
            return
        if f_tone <= 0 or fs_tone < 1000:
            messagebox.showerror("Demo de tono", "Use f > 0 y fs ≥ 1000 Hz.")
            return
        f_alias = alias_frequency(f_tone, fs_tone)
        nyq = fs_tone / 2.0
        folded = f_tone > nyq

        win = tk.Toplevel(self)
        win.title(f"Demo de aliasing · {f_tone:.0f} Hz a {fs_tone:.0f} Hz")
        win.geometry("900x640")
        text = (f"Tono de {f_tone:.1f} Hz muestreado a fs = {fs_tone:.1f} Hz. "
                f"Nyquist = {nyq:.1f} Hz → "
                + (f"NO cumple: la frecuencia aparente es {f_alias:.1f} Hz."
                   if folded else "cumple: la frecuencia se conserva."))
        ttk.Label(win, text=text, style="Metric.TLabel", wraplength=860).pack(
            padx=10, pady=(8, 2), anchor="w")

        fig = Figure(figsize=(8.6, 5.0), dpi=100, constrained_layout=True)
        canvas = FigureCanvasTkAgg(fig, master=win)
        toolbar = NavigationToolbar2Tk(canvas, win, pack_toolbar=False)
        toolbar.update()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=4)
        toolbar.pack(side="bottom", fill="x")

        t_end = max(8.0 / fs_tone, min(0.006, 8.0 / f_tone))
        t_cont = np.linspace(0.0, t_end, 1200)
        t_s = np.arange(0.0, t_end, 1.0 / fs_tone)
        x_cont = np.sin(2 * np.pi * f_tone * t_cont)
        x_s = np.sin(2 * np.pi * f_tone * t_s)

        ax1, ax2 = fig.subplots(2, 1, height_ratios=[1.0, 1.2])
        ax1.plot(t_cont * 1e3, x_cont, color="#999999", lw=0.8, label="tono continuo")
        ax1.stem(t_s * 1e3, x_s, linefmt="#1f77b4", markerfmt="o", basefmt=" ",
                 label="muestras")
        ax1.set_xlabel("Tiempo [ms]")
        ax1.set_ylabel("Amplitud")
        ax1.set_title(f"Tono de {f_tone:.0f} Hz y sus muestras a {fs_tone:.0f} Hz")
        ax1.grid(alpha=0.25)
        ax1.legend(fontsize=8)

        n = int(max(2048, round(fs_tone)))
        t_long = np.arange(n) / fs_tone
        x_long = np.sin(2 * np.pi * f_tone * t_long)
        f_fft, m_fft = spectrum(x_long, fs_tone, "Rectangular")
        ax2.plot(f_fft, m_fft, color="#1f77b4", lw=0.9)
        ax2.axvline(f_alias, color="#d62728", ls="--", lw=1.0,
                    label=f"pico aparente = {f_alias:.1f} Hz")
        ax2.set_xlim(0.0, nyq)
        ax2.set_xlabel("Frecuencia [Hz]")
        ax2.set_ylabel("Amplitud")
        ax2.set_title("Espectro de la señal muestreada (solo se ve la frecuencia aparente)")
        ax2.grid(alpha=0.25)
        ax2.legend(fontsize=8)
        canvas.draw_idle()

    # ------------------------------------------------------------------
    def _dsp_for_playback(self):
        d = self.app.get_digitized()
        if d is None:
            messagebox.showinfo("Reproducir", "Primero cargue o grabe un audio.")
            return None
        return d

    def _play(self, kind: str) -> None:
        d = self._dsp_for_playback()
        if d is None:
            return
        p = d.params
        try:
            if kind == "original":
                x = self.app.audio.x
                fs = self.app.audio.fs
                name = "original"
            elif kind == "cuantizada":
                x = d.quant.volts_q / max(p.vref, 1e-12)
                fs = d.fs_eff
                name = "cuantizada"
            else:
                base = d.quant.volts_q / max(p.vref, 1e-12)
                x, fs = resample_to(base, d.fs_eff, d.audio.fs, antialias=True)
                name = "reconstruida"
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Reproducir", str(exc))
            return

        self._stop_play()
        self._player = Player(x, int(round(fs)))
        self._player.start()
        self.play_info.set(f"Reproduciendo: {name} @ {fs:.0f} Hz")
        self.app.set_status(f"Reproduciendo señal {name} a {fs:.0f} Hz…")
        self._poll_player()

    def _poll_player(self) -> None:
        self._player_poll_id = None
        p = self._player
        if p is None:
            return
        if p.error:
            self.play_info.set("")
            self.app.set_status(p.error, error=True)
            self._player = None
            return
        if p.is_playing:
            try:
                self._player_poll_id = self.after(150, self._poll_player)
            except tk.TclError:
                self._player_poll_id = None
            return
        self.play_info.set("")
        self.app.set_status("Reproducción terminada.")

    def _stop_play(self) -> None:
        if self._player_poll_id is not None:
            try:
                self.after_cancel(self._player_poll_id)
            except tk.TclError:
                pass
            self._player_poll_id = None
        if self._player is not None:
            self._player.stop()
            self._player = None
        self.play_info.set("")

    def shutdown(self) -> None:
        self._stop_play()
