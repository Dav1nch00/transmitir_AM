"""Pestaña 6: envío de la señal por modulación AM (lógica de Emisor.py)."""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from core.am import modulate_am, prepare_baseband
from core.audio_io import Player

from .widgets import PlotTab


class TabEnviar(PlotTab):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.carrier_var = tk.StringVar(value="10000")
        self.index_var = tk.DoubleVar(value=0.95)
        self.source_var = tk.StringVar(value="Señal digitalizada (DAC)")
        self.status_tx = tk.StringVar(value="Sin transmitir.")
        self._player: Player | None = None
        self._poll_id: str | None = None
        self._base_signal: np.ndarray | None = None
        self._am_signal: np.ndarray | None = None
        self._am_fs: float = 44100.0
        self._build_controls()
        self.message("Cargue un audio, digitalícelo y pulse «Enviar señal».")

    # ------------------------------------------------------------------
    def _build_controls(self) -> None:
        c = self.controls
        row1 = ttk.Frame(c)
        row1.pack(fill="x")
        ttk.Label(row1, text="Fuente").pack(side="left", padx=(0, 4))
        ttk.Combobox(row1, textvariable=self.source_var, state="readonly", width=28,
                     values=["Señal digitalizada (DAC)", "Audio original"]).pack(
            side="left", padx=(0, 12))

        ttk.Label(row1, text="Portadora (Hz)").pack(side="left", padx=(0, 4))
        ttk.Spinbox(row1, from_=1000, to=22000, increment=100,
                    textvariable=self.carrier_var, width=8).pack(side="left", padx=(0, 12))

        ttk.Label(row1, text="Índice m").pack(side="left", padx=(0, 4))
        ttk.Spinbox(row1, from_=0.1, to=1.0, increment=0.1,
                    textvariable=self.index_var, width=6).pack(side="left", padx=(0, 12))

        self.send_btn = ttk.Button(row1, text="📡 Enviar señal", command=self._send)
        self.send_btn.pack(side="left", padx=(0, 4))
        self.stop_btn = ttk.Button(row1, text="■ Detener", command=self._stop,
                                  state="disabled")
        self.stop_btn.pack(side="left", padx=(0, 12))
        ttk.Label(row1, textvariable=self.status_tx, style="Metric.TLabel").pack(
            side="left", padx=(8, 0))

        ttk.Label(c, text="La señal se modula en AM (portadora ~10 kHz) y se reproduce "
                          "por el altavoz, como en Emisor.py.",
                  style="Subtitle.TLabel").pack(anchor="w", pady=(4, 0))

    # ------------------------------------------------------------------
    def _send(self) -> None:
        audio = self.app.audio
        if audio is None:
            messagebox.showinfo("Enviar señal", "Primero grabe o cargue un audio.")
            return

        source = self.source_var.get()
        d = None
        if source.startswith("Señal digitalizada"):
            d = self.app.get_digitized()
            if d is None:
                messagebox.showinfo("Enviar señal",
                                    "Primero digitalice el audio (pestaña 2).")
                return

        try:
            carrier = float(self.carrier_var.get().replace(",", "."))
            index_mod = float(self.index_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Enviar señal", "Portadora e índice deben ser numéricos.")
            return
        if carrier <= 0:
            messagebox.showerror("Enviar señal", "La portadora debe ser > 0 Hz.")
            return

        if d is not None:
            vref = max(float(d.params.vref), 1e-12)
            base = np.asarray(d.quant.volts_q, dtype=np.float64) / vref
            fs_in = float(d.fs_eff)
        else:
            base = np.asarray(audio.x, dtype=np.float64)
            fs_in = float(audio.fs)

        base, fs_tx = prepare_baseband(base, fs_in, 44100.0)
        am = modulate_am(base, fs_tx, carrier_freq=carrier, index_mod=index_mod)

        self._base_signal = base
        self._am_signal = am
        self._am_fs = fs_tx

        self._stop()
        self._player = Player(am, int(round(fs_tx)))
        self._player.start()
        self.send_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_tx.set(f"Transmitiendo AM a {carrier:.0f} Hz por el altavoz…")
        self.app.set_status(f"Enviando señal AM a {carrier:.0f} Hz…")
        self.refresh()
        self._poll_player()

    def _poll_player(self) -> None:
        self._poll_id = None
        p = self._player
        if p is None:
            return
        if p.error:
            self.status_tx.set(p.error)
            self.app.set_status(p.error, error=True)
            self._player = None
            self.send_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            return
        if p.is_playing:
            try:
                self._poll_id = self.after(150, self._poll_player)
            except tk.TclError:
                self._poll_id = None
            return
        self.send_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_tx.set("Transmisión finalizada.")
        self.app.set_status("Transmisión finalizada.")

    def _stop(self) -> None:
        if self._poll_id is not None:
            try:
                self.after_cancel(self._poll_id)
            except tk.TclError:
                pass
            self._poll_id = None
        if self._player is not None:
            self._player.stop()
            self._player = None
        self.send_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        if self._am_signal is None or self._base_signal is None:
            self.message("Pulse «Enviar señal» para modular y transmitir.")
            return
        am = self._am_signal
        base = self._base_signal
        fs = self._am_fs
        n = min(len(am), int(round(0.05 * fs)))
        t = np.arange(n) / fs
        ax1, ax2 = self.axes(2, height_ratios=[1.0, 1.0])
        ax1.plot(t * 1e3, base[:n], color="#1f77b4", lw=1.0, label="banda base (digitalizada)")
        ax1.set_xlim(0.0, t[-1] * 1e3)
        ax1.set_xlabel("Tiempo [ms]")
        ax1.set_ylabel("Amplitud")
        ax1.set_title("Señal de banda base")
        ax1.grid(alpha=0.25)
        ax1.legend(fontsize=8)

        ax2.plot(t * 1e3, am[:n], color="#d62728", lw=0.8, label="señal AM transmitida")
        ax2.set_xlim(0.0, t[-1] * 1e3)
        ax2.set_xlabel("Tiempo [ms]")
        ax2.set_ylabel("Amplitud")
        ax2.set_title(f"Señal modulada AM · fs = {fs:.0f} Hz")
        ax2.grid(alpha=0.25)
        ax2.legend(fontsize=8)
        self.draw()

    def shutdown(self) -> None:
        self._stop()