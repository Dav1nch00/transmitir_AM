"""Pestaña 1: recepción de la señal AM transmitida (lógica de Receptor.py)."""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from core import receptor as rx
from core.audio_io import (Audio, AudioError, MicRecorder, Player,
                           list_input_devices, default_input_index)
from core.dsp import display_slice, rms

from .widgets import PlotTab


class TabRecibir(PlotTab):
    def __init__(self, master, app):
        super().__init__(master, app)
        self._recorder: MicRecorder | None = None
        self._player: Player | None = None
        self._poll_id: str | None = None
        self._player_poll_id: str | None = None
        self._duration = 7.0

        self.device_var = tk.StringVar()
        self.fs_var = tk.StringVar(value="44100")
        self.duration_var = tk.DoubleVar(value=7.0)
        self.level_var = tk.DoubleVar(value=0.0)
        self.record_info = tk.StringVar(value="Sin recibir.")
        self.info_var = tk.StringVar(value="Pulse «Recibir» para capturar la señal transmitida.")
        self.cutoff_var = tk.StringVar(value="4000")
        self.carrier_var = tk.StringVar(value="17000")

        self._build_controls()
        self._load_devices()
        self.message("Pulse «Recibir» para capturar la señal AM del aire\n"
                    "y demodularla (Receptor.py).")

    def _build_controls(self) -> None:
        c = self.controls
        row1 = ttk.Frame(c)
        row1.pack(fill="x")
        ttk.Label(row1, text="Dispositivo").pack(side="left", padx=(0, 4))
        self.device_combo = ttk.Combobox(row1, textvariable=self.device_var, width=36,
                                        state="readonly")
        self.device_combo.pack(side="left", padx=(0, 4))
        ttk.Button(row1, text="Buscar", width=8, command=self._load_devices).pack(
            side="left", padx=(0, 12))

        ttk.Label(row1, text="fs").pack(side="left", padx=(0, 4))
        ttk.Combobox(row1, textvariable=self.fs_var, width=7,
                     values=["44100"]).pack(side="left", padx=(0, 12))

        ttk.Label(row1, text="Duración (s)").pack(side="left", padx=(0, 4))
        ttk.Spinbox(row1, from_=3, to=15, increment=1, textvariable=self.duration_var,
                    width=5).pack(side="left", padx=(0, 12))

        self.recv_btn = ttk.Button(row1, text="🎧 Recibir", command=self._start_recv)
        self.recv_btn.pack(side="left", padx=(0, 4))
        self.stop_btn = ttk.Button(row1, text="■ Detener", command=self._stop_recv,
                                  state="disabled")
        self.stop_btn.pack(side="left", padx=(0, 12))

        ttk.Label(row1, text="Nivel").pack(side="left", padx=(0, 4))
        ttk.Progressbar(row1, variable=self.level_var, maximum=100.0,
                        length=120).pack(side="left", padx=(0, 8))
        ttk.Label(row1, textvariable=self.record_info, style="Metric.TLabel").pack(
            side="left")

        row2 = ttk.Frame(c)
        row2.pack(fill="x", pady=(4, 0))
        ttk.Label(row2, text="Portadora (Hz)").pack(side="left", padx=(0, 4))
        ttk.Entry(row2, textvariable=self.carrier_var, width=8).pack(side="left", padx=(0, 12))
        ttk.Label(row2, text="Corte (Hz)").pack(side="left", padx=(0, 4))
        ttk.Entry(row2, textvariable=self.cutoff_var, width=8).pack(side="left", padx=(0, 12))
        self.play_btn = ttk.Button(row2, text="▶ Reproducir", command=self._play)
        self.play_btn.pack(side="left", padx=(0, 4))
        self.stop_play_btn = ttk.Button(row2, text="■ Silencio", command=self._stop_play,
                                       state="disabled")
        self.stop_play_btn.pack(side="left", padx=(0, 12))

        ttk.Label(c, textvariable=self.info_var, style="Info.TLabel", wraplength=1240,
                  justify="left").pack(anchor="w", pady=(4, 0))

    def _load_devices(self) -> None:
        if self._recorder is not None and self._recorder.is_recording:
            return
        try:
            self._devices = list_input_devices()
        except Exception as exc:
            self._devices = []
            self.device_combo["values"] = ["(sin micrófono)"]
            self.device_var.set("(sin micrófono)")
            self.app.set_status(str(exc), error=True)
            return
        labels = [f"{d['index']}: {d['name']}" for d in self._devices]
        self.device_combo["values"] = labels
        default = default_input_index()
        selected = 0
        for i, d in enumerate(self._devices):
            if d["index"] == default:
                selected = i
                break
        if labels:
            self.device_var.set(labels[selected])
        self.app.set_status(f"{len(labels)} dispositivo(s) de entrada encontrados.")

    def _selected_device(self) -> int | None:
        text = self.device_var.get()
        if not text or text.startswith("("):
            return None
        try:
            return int(text.split(":", 1)[0])
        except ValueError:
            return None

    def _start_recv(self) -> None:
        if self._recorder is not None and self._recorder.is_recording:
            return
        self._stop_play()
        try:
            fs = int(round(float(self.fs_var.get())))
            duration = float(self.duration_var.get())
        except ValueError:
            messagebox.showerror("Recepción", "La fs y la duración deben ser numéricas.")
            return
        if not 3.0 <= duration <= 15.0:
            messagebox.showerror("Recepción", "La duración debe estar entre 3 y 15 s.")
            return
        self._duration = duration
        self._recorder = MicRecorder(fs=fs, device=self._selected_device())
        try:
            self._recorder.start()
        except AudioError as exc:
            self._recorder = None
            messagebox.showerror("Micrófono", str(exc))
            self.app.set_status(str(exc), error=True)
            return

        self.recv_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.level_var.set(0.0)
        self.record_info.set(f"Recibiendo: 0.0 / {duration:.1f} s")
        self.app.set_status(f"Recibiendo señal AM a {fs} Hz…")
        self._poll_recording()

    def _poll_recording(self) -> None:
        self._poll_id = None
        rec = self._recorder
        if rec is None or not rec.is_recording:
            return
        elapsed = rec.elapsed
        self.level_var.set(min(100.0, rec.level * 100.0))
        self.record_info.set(f"Recibiendo: {elapsed:.1f} / {self._duration:.1f} s")
        if elapsed >= self._duration:
            self._stop_recv()
            return
        try:
            self._poll_id = self.after(100, self._poll_recording)
        except tk.TclError:
            self._poll_id = None

    def _stop_recv(self) -> None:
        if self._poll_id is not None:
            try:
                self.after_cancel(self._poll_id)
            except tk.TclError:
                pass
            self._poll_id = None
        rec = self._recorder
        self._recorder = None
        self.recv_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.level_var.set(0.0)
        if rec is None:
            return
        try:
            rec_audio = rec.stop()
        except AudioError as exc:
            self.record_info.set("Sin recibir.")
            messagebox.showerror("Micrófono", str(exc))
            self.app.set_status(str(exc), error=True)
            return

        try:
            carrier = float(self.carrier_var.get().replace(",", "."))
            cutoff = float(self.cutoff_var.get().replace(",", "."))
        except ValueError:
            carrier = 17000.0
            cutoff = 4000.0

        demod, fs = rx.demodulate_am(rec_audio.x, rec_audio.fs, carrier_freq=carrier,
                                     cutoff_freq=cutoff)
        audio = Audio(x=demod, fs=int(round(fs)), source="receptor", path="",
                      channels=1, bits=16, is_float=True)
        self.record_info.set(f"Recibido: {audio.duration:.2f} s")
        self.app.set_audio(audio)
        self.app.set_status(f"Señal demodulada: {audio.duration:.2f} s a {audio.fs} Hz.")
        self.refresh()

    def _play(self) -> None:
        audio = self.app.audio
        if audio is None:
            messagebox.showinfo("Reproducir", "Primero reciba una señal.")
            return
        self._stop_play()
        self._player = Player(audio.x, audio.fs)
        self._player.start()
        self.play_btn.configure(state="disabled")
        self.stop_play_btn.configure(state="normal")
        self.app.set_status(f"Reproduciendo audio recibido a {audio.fs} Hz…")
        self._poll_player()

    def _poll_player(self) -> None:
        self._player_poll_id = None
        p = self._player
        if p is None:
            return
        if p.error:
            self.app.set_status(p.error, error=True)
            self._player = None
            self.play_btn.configure(state="normal")
            self.stop_play_btn.configure(state="disabled")
            return
        if p.is_playing:
            try:
                self._player_poll_id = self.after(150, self._poll_player)
            except tk.TclError:
                self._player_poll_id = None
            return
        self.play_btn.configure(state="normal")
        self.stop_play_btn.configure(state="disabled")
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
        self.play_btn.configure(state="normal")
        self.stop_play_btn.configure(state="disabled")

    def refresh(self) -> None:
        audio = self.app.audio
        if audio is None:
            self.info_var.set("Sin señal recibida.")
            self.message("Todavía no hay señal recibida.\nUse «Recibir».")
            return
        x = np.asarray(audio.x, dtype=np.float64)
        t = np.arange(len(x), dtype=np.float64) / audio.fs
        td, xd = display_slice(t, x, 12000)
        ax = self.axes(1)
        ax.plot(td, xd, color="#1f77b4", lw=0.6, label="señal demodulada")
        r = rms(x)
        ax.axhline(r, color="#d62728", ls="--", lw=0.8, label=f"RMS = {r:.4f}")
        ax.axhline(-r, color="#d62728", ls="--", lw=0.8)
        ax.axhline(0.0, color="#888888", lw=0.5)
        ax.set_xlim(0.0, max(audio.duration, 0.01))
        ax.set_xlabel("Tiempo [s]")
        ax.set_ylabel("Amplitud")
        ax.set_title(f"Señal recibida y demodulada · {audio.name}")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
        self.draw()
        self.info_var.set(
            f"{audio.duration:.3f} s · fs = {audio.fs} Hz · "
            f"{len(x)} muestras · pico = {np.max(np.abs(x)):.4f} · RMS = {r:.4f}"
        )

    def shutdown(self) -> None:
        if self._poll_id is not None:
            try:
                self.after_cancel(self._poll_id)
            except tk.TclError:
                pass
        self._stop_play()
        if self._recorder is not None:
            self._recorder.cancel()
            self._recorder = None