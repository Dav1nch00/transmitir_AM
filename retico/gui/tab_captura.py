"""Pestaña 1: captura desde micrófono, carga/guardado de WAV y reproducción."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

from core import audio_io
from core.audio_io import AudioError, MicRecorder, Player
from core.dsp import display_slice, rms

from .widgets import PlotTab, output_dir, timestamp


class TabCaptura(PlotTab):
    def __init__(self, master, app):
        super().__init__(master, app)
        self._recorder: MicRecorder | None = None
        self._player: Player | None = None
        self._poll_id: str | None = None
        self._player_poll_id: str | None = None
        self._duration = 5.0
        self._devices: list[dict] = []

        self.device_var = tk.StringVar()
        self.fs_var = tk.StringVar(value="44100")
        self.duration_var = tk.DoubleVar(value=5.0)
        self.level_var = tk.DoubleVar(value=0.0)
        self.record_info = tk.StringVar(value="Sin grabar.")
        self.info_var = tk.StringVar(value="Cargue un WAV o grabe desde el micrófono.")

        self._build_controls()
        self._load_devices()
        self.message(
            "Pulse «Grabar» para capturar audio del micrófono\n"
            "o «Cargar WAV» para analizar un archivo existente."
        )

    # ------------------------------------------------------------------
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
                     values=["8000", "16000", "22050", "32000", "44100", "48000"]).pack(
            side="left", padx=(0, 12))

        ttk.Label(row1, text="Duración (s)").pack(side="left", padx=(0, 4))
        ttk.Spinbox(row1, from_=1, to=60, increment=1, textvariable=self.duration_var,
                    width=5).pack(side="left", padx=(0, 12))

        self.record_btn = ttk.Button(row1, text="● Grabar", command=self._start_record)
        self.record_btn.pack(side="left", padx=(0, 4))
        self.stop_btn = ttk.Button(row1, text="■ Detener", command=self._stop_record,
                                   state="disabled")
        self.stop_btn.pack(side="left", padx=(0, 12))

        ttk.Label(row1, text="Nivel").pack(side="left", padx=(0, 4))
        ttk.Progressbar(row1, variable=self.level_var, maximum=100.0,
                        length=120).pack(side="left", padx=(0, 8))
        ttk.Label(row1, textvariable=self.record_info, style="Metric.TLabel").pack(
            side="left")

        row2 = ttk.Frame(c)
        row2.pack(fill="x", pady=(4, 0))
        self.play_btn = ttk.Button(row2, text="▶ Reproducir", command=self._play)
        self.play_btn.pack(side="left", padx=(0, 4))
        self.stop_play_btn = ttk.Button(row2, text="■ Silencio", command=self._stop_play,
                                        state="disabled")
        self.stop_play_btn.pack(side="left", padx=(0, 12))
        ttk.Button(row2, text="Cargar WAV", command=self._load_wav).pack(
            side="left", padx=(0, 4))
        ttk.Button(row2, text="Guardar WAV", command=self._save_wav).pack(
            side="left", padx=(0, 12))

        ttk.Label(c, textvariable=self.info_var, style="Info.TLabel", wraplength=1240,
                  justify="left").pack(anchor="w", pady=(4, 0))

    # ------------------------------------------------------------------
    def _load_devices(self) -> None:
        if self._recorder is not None and self._recorder.is_recording:
            return
        try:
            self._devices = audio_io.list_input_devices()
        except AudioError as exc:
            self._devices = []
            self.device_combo["values"] = ["(sin micrófono)"]
            self.device_var.set("(sin micrófono)")
            self.app.set_status(str(exc), error=True)
            return
        labels = [f"{d['index']}: {d['name']}" for d in self._devices]
        self.device_combo["values"] = labels
        default = audio_io.default_input_index()
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

    # ------------------------------------------------------------------
    def _start_record(self) -> None:
        if self._recorder is not None and self._recorder.is_recording:
            return
        self._stop_play()
        try:
            fs = int(round(float(self.fs_var.get())))
            duration = float(self.duration_var.get())
        except ValueError:
            messagebox.showerror("Grabación", "La fs y la duración deben ser numéricas.")
            return
        if not 1.0 <= duration <= 60.0:
            messagebox.showerror("Grabación", "La duración debe estar entre 1 y 60 s.")
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

        self.record_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.level_var.set(0.0)
        self.record_info.set(f"Grabando: 0.0 / {duration:.1f} s")
        self.app.set_status(f"Grabando a {fs} Hz…")
        self._poll_recording()

    def _poll_recording(self) -> None:
        self._poll_id = None
        rec = self._recorder
        if rec is None or not rec.is_recording:
            return
        elapsed = rec.elapsed
        self.level_var.set(min(100.0, rec.level * 100.0))
        self.record_info.set(f"Grabando: {elapsed:.1f} / {self._duration:.1f} s")
        if elapsed >= self._duration:
            self._stop_record()
            return
        try:
            self._poll_id = self.after(100, self._poll_recording)
        except tk.TclError:
            self._poll_id = None

    def _stop_record(self) -> None:
        if self._poll_id is not None:
            try:
                self.after_cancel(self._poll_id)
            except tk.TclError:
                pass
            self._poll_id = None
        rec = self._recorder
        self._recorder = None
        self.record_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.level_var.set(0.0)
        if rec is None:
            return
        try:
            audio = rec.stop()
        except AudioError as exc:
            self.record_info.set("Sin grabar.")
            messagebox.showerror("Micrófono", str(exc))
            self.app.set_status(str(exc), error=True)
            return
        self.record_info.set(f"Grabado: {audio.duration:.2f} s")
        self.app.set_audio(audio)
        self.app.set_status(f"Grabación lista: {audio.duration:.2f} s a {audio.fs} Hz.")

    # ------------------------------------------------------------------
    def _play(self) -> None:
        audio = self.app.audio
        if audio is None:
            messagebox.showinfo("Reproducir", "Primero grabe o cargue un audio.")
            return
        self._stop_play()
        self._player = Player(audio.x, audio.fs)
        self._player.start()
        self.play_btn.configure(state="disabled")
        self.stop_play_btn.configure(state="normal")
        self.app.set_status(f"Reproduciendo {audio.name} a {audio.fs} Hz…")
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

    # ------------------------------------------------------------------
    def _load_wav(self) -> None:
        initial = Path(__file__).resolve().parents[2] / "music tes"
        if not initial.exists():
            initial = Path.home()
        path = filedialog.askopenfilename(
            parent=self, title="Abrir archivo WAV", initialdir=str(initial),
            filetypes=[("Audio WAV", "*.wav *.WAV"), ("Todos los archivos", "*.*")])
        if not path:
            return
        try:
            audio = audio_io.read_wav(path)
        except AudioError as exc:
            messagebox.showerror("WAV", str(exc))
            self.app.set_status(str(exc), error=True)
            return
        self.record_info.set("Archivo cargado.")
        self.app.set_audio(audio)
        self.app.set_status(f"WAV cargado: {Path(path).name} ({audio.duration:.2f} s).")

    def _save_wav(self) -> None:
        audio = self.app.audio
        if audio is None:
            messagebox.showinfo("Guardar WAV", "Primero grabe o cargue un audio.")
            return
        path = filedialog.asksaveasfilename(
            parent=self, title="Guardar grabación", initialdir=str(output_dir()),
            initialfile=timestamp("audio", "wav"), defaultextension=".wav",
            filetypes=[("WAV 16 bits", "*.wav")])
        if not path:
            return
        try:
            audio_io.save_wav(path, audio.x, audio.fs, bits=16)
        except AudioError as exc:
            messagebox.showerror("Guardar WAV", str(exc))
            return
        self.app.set_status(f"WAV guardado en {path}")

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        audio = self.app.audio
        if audio is None:
            self.info_var.set("Sin audio. Grabe desde el micrófono o cargue un WAV.")
            self.message("Todavía no hay audio.\nUse «Grabar» o «Cargar WAV».")
            return
        x = np.asarray(audio.x, dtype=np.float64)
        t = np.arange(len(x), dtype=np.float64) / audio.fs
        td, xd = display_slice(t, x, 12000)
        ax = self.axes(1)
        ax.plot(td, xd, color="#1f77b4", lw=0.6, label="x(t)")
        r = rms(x)
        ax.axhline(r, color="#d62728", ls="--", lw=0.8, label=f"RMS = {r:.4f}")
        ax.axhline(-r, color="#d62728", ls="--", lw=0.8)
        ax.axhline(0.0, color="#888888", lw=0.5)
        ax.set_xlim(0.0, max(audio.duration, 0.01))
        ax.set_xlabel("Tiempo [s]")
        ax.set_ylabel("Amplitud (escala completa = 1)")
        ax.set_title(f"Forma de onda · {audio.name}")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
        self.draw()

        tipo = "float" if audio.is_float else f"PCM {audio.bits} bits"
        self.info_var.set(
            f"{audio.duration:.3f} s · fs = {audio.fs} Hz · Ts = {1e6 / audio.fs:.2f} µs · "
            f"{audio.channels} canal(es) · {tipo} · "
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
