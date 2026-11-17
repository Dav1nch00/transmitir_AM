"""Pestaña 7: recepción y demodulación de AM, FM y técnicas digitales.

Flujo de trabajo
----------------
1. Elegir la técnica de la señal recibida (AM, FM, ASK, FSK, PSK, QAM,
   GFSK o MFSK).
2. Elegir el origen: escuchar por micrófono o cargar un WAV modulado.
3. Ajustar los parámetros de demodulación.
4. Pulsar «Demodular».
5. Ver la gráfica de la señal demodulada y reproducirla o guardarla.
"""
from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from core.am import demodulate_am
from core.audio_io import (Audio, AudioError, MicRecorder, Player,
                           default_input_index, list_input_devices, read_wav,
                           save_wav)
from core.dsp import display_slice, rms
from core.modulation import (ANALOG_TECHNIQUES, DIGITAL_TECHNIQUES,
                             ModulationError, demodulate_digital, demodulate_fm)

from .widgets import PlotTab, output_dir, timestamp

TECHNIQUES = ANALOG_TECHNIQUES + DIGITAL_TECHNIQUES
SRC_MIC = "Micrófono (escuchar)"
SRC_FILE = "Archivo WAV modulado"
FSK_TECHNIQUES = ("FSK", "GFSK", "MFSK")


class TabRecibir(PlotTab):
    def __init__(self, master, app):
        super().__init__(master, app)
        self._recorder: MicRecorder | None = None
        self._player: Player | None = None
        self._poll_id: str | None = None
        self._player_poll_id: str | None = None
        self._duration = 20.0

        self._received: Audio | None = None
        self._demod: Audio | None = None
        self._received_name = ""
        self._demod_info = ""
        self._seen_tx_serial = 0

        self.device_var = tk.StringVar()
        self.fs_var = tk.StringVar(value="44100")
        self.duration_var = tk.DoubleVar(value=20.0)
        self.level_var = tk.DoubleVar(value=0.0)
        self.tech_var = tk.StringVar(value="AM")
        self.source_var = tk.StringVar(value=SRC_MIC)
        self.carrier_var = tk.StringVar(value="10000")
        self.cutoff_var = tk.StringVar(value="4000")
        self.gain_var = tk.StringVar(value="0")
        self.fm_dev_var = tk.StringVar(value="3000")
        self.baud_var = tk.StringVar(value="4000")
        self.bps_var = tk.StringVar(value="2")
        self.pcm_bits_var = tk.StringVar(
            value=self._app_int("bits_var", 8))
        self.audio_fs_var = tk.StringVar(
            value=self._app_int("fs_var", 16000))
        self.dig_dev_var = tk.StringVar(value="6000")
        self.bt_var = tk.StringVar(value="0.5")
        self.find_start_var = tk.BooleanVar(value=True)
        self.record_info = tk.StringVar(value="Sin recibir.")
        self.info_var = tk.StringVar(value="")

        self._build_controls()
        self._load_devices()
        self._on_technique_changed()
        self._on_source_changed()
        self.refresh()

    # ------------------------------------------------------------------
    # Construcción de la interfaz
    # ------------------------------------------------------------------
    def _build_controls(self) -> None:
        c = self.controls

        # -- fila 1: técnica, origen y acciones --------------------------
        row1 = ttk.Frame(c)
        row1.pack(fill="x")

        ttk.Label(row1, text="Técnica").pack(side="left", padx=(0, 4))
        self.tech_combo = ttk.Combobox(
            row1, textvariable=self.tech_var, state="readonly", width=8,
            values=list(TECHNIQUES))
        self.tech_combo.pack(side="left", padx=(0, 12))
        self.tech_combo.bind("<<ComboboxSelected>>", self._on_technique_changed)

        ttk.Label(row1, text="Origen").pack(side="left", padx=(0, 4))
        self.source_combo = ttk.Combobox(
            row1, textvariable=self.source_var, state="readonly", width=23,
            values=[SRC_MIC, SRC_FILE])
        self.source_combo.pack(side="left", padx=(0, 8))
        self.source_combo.bind("<<ComboboxSelected>>", self._on_source_changed)

        self.load_btn = ttk.Button(row1, text="📂 Cargar WAV modulado",
                                   command=self._load_file)
        self.load_btn.pack(side="left", padx=(0, 8))

        self.demod_btn = ttk.Button(row1, text="🔊 Demodular",
                                    command=self._demodulate, state="disabled")
        self.demod_btn.pack(side="left", padx=(0, 8))

        self.play_btn = ttk.Button(row1, text="▶ Reproducir",
                                   command=self._play, state="disabled")
        self.play_btn.pack(side="left", padx=(0, 4))

        self.stop_play_btn = ttk.Button(row1, text="■ Silencio",
                                        command=self._stop_play,
                                        state="disabled")
        self.stop_play_btn.pack(side="left", padx=(0, 8))

        self.save_audio_btn = ttk.Button(row1, text="💾 Guardar audio",
                                         command=self._save_audio,
                                         state="disabled")
        self.save_audio_btn.pack(side="left", padx=(0, 4))

        # -- fila 2: captura por micrófono ------------------------------
        self.mic_frame = ttk.Frame(c)
        mic = ttk.Frame(self.mic_frame)
        mic.pack(fill="x")

        ttk.Label(mic, text="Micrófono").pack(side="left", padx=(0, 4))
        self.device_combo = ttk.Combobox(mic, textvariable=self.device_var,
                                         width=30, state="readonly")
        self.device_combo.pack(side="left", padx=(0, 4))
        ttk.Button(mic, text="Buscar", width=8,
                   command=self._load_devices).pack(side="left", padx=(0, 12))

        ttk.Label(mic, text="fs captura").pack(side="left", padx=(0, 4))
        ttk.Combobox(mic, textvariable=self.fs_var, width=7,
                     values=["44100", "48000"]).pack(side="left", padx=(0, 12))

        ttk.Label(mic, text="Duración (s)").pack(side="left", padx=(0, 4))
        ttk.Spinbox(mic, from_=3, to=60, increment=1,
                    textvariable=self.duration_var, width=5).pack(
            side="left", padx=(0, 12))

        self.recv_btn = ttk.Button(mic, text="🎧 Escuchar / Recibir",
                                   command=self._start_recv)
        self.recv_btn.pack(side="left", padx=(0, 4))
        self.stop_btn = ttk.Button(mic, text="■ Detener",
                                   command=self._stop_recv, state="disabled")
        self.stop_btn.pack(side="left", padx=(0, 12))

        ttk.Label(mic, text="Nivel").pack(side="left", padx=(0, 4))
        ttk.Progressbar(mic, variable=self.level_var, maximum=100.0,
                        length=110).pack(side="left", padx=(0, 8))
        ttk.Label(mic, textvariable=self.record_info,
                  style="Metric.TLabel").pack(side="left")

        # -- fila 3: parámetros de demodulación -------------------------
        self.params_row = ttk.Frame(c)
        self.params_row.pack(fill="x", pady=(6, 0))

        self.frame_carrier = ttk.Frame(self.params_row)
        ttk.Label(self.frame_carrier, text="Portadora (Hz)").pack(
            side="left", padx=(0, 4))
        ttk.Spinbox(self.frame_carrier, from_=500, to=22050, increment=100,
                    textvariable=self.carrier_var, width=8).pack(
            side="left", padx=(0, 16))

        self.frame_cutoff = ttk.Frame(self.params_row)
        ttk.Label(self.frame_cutoff, text="Corte de audio (Hz)").pack(
            side="left", padx=(0, 4))
        ttk.Spinbox(self.frame_cutoff, from_=500, to=15000, increment=100,
                    textvariable=self.cutoff_var, width=8).pack(
            side="left", padx=(0, 16))

        self.frame_fmdev = ttk.Frame(self.params_row)
        ttk.Label(self.frame_fmdev, text="Desviación Δf (Hz)").pack(
            side="left", padx=(0, 4))
        ttk.Spinbox(self.frame_fmdev, from_=100, to=15000, increment=100,
                    textvariable=self.fm_dev_var, width=8).pack(
            side="left", padx=(0, 16))

        self.frame_dig = ttk.Frame(self.params_row)
        ttk.Label(self.frame_dig, text="Baudios").pack(side="left", padx=(0, 4))
        ttk.Spinbox(self.frame_dig, from_=100, to=11000, increment=100,
                    textvariable=self.baud_var, width=7).pack(
            side="left", padx=(0, 12))
        ttk.Label(self.frame_dig, text="Bits/símbolo").pack(
            side="left", padx=(0, 4))
        self.bps_combo = ttk.Combobox(self.frame_dig, textvariable=self.bps_var,
                                      state="readonly", width=3,
                                      values=["1", "2", "3", "4", "5", "6"])
        self.bps_combo.pack(side="left", padx=(0, 12))
        ttk.Label(self.frame_dig, text="Bits PCM").pack(side="left",
                                                        padx=(0, 4))
        ttk.Spinbox(self.frame_dig, from_=2, to=24, increment=1,
                    textvariable=self.pcm_bits_var, width=5).pack(
            side="left", padx=(0, 12))
        ttk.Label(self.frame_dig, text="fs audio (Hz)").pack(
            side="left", padx=(0, 4))
        ttk.Spinbox(self.frame_dig, from_=4000, to=48000, increment=100,
                    textvariable=self.audio_fs_var, width=8).pack(
            side="left", padx=(0, 16))

        self.frame_digdev = ttk.Frame(self.params_row)
        ttk.Label(self.frame_digdev, text="Desviación Δf (Hz)").pack(
            side="left", padx=(0, 4))
        ttk.Spinbox(self.frame_digdev, from_=100, to=15000, increment=100,
                    textvariable=self.dig_dev_var, width=8).pack(
            side="left", padx=(0, 16))

        self.frame_digbt = ttk.Frame(self.params_row)
        ttk.Label(self.frame_digbt, text="Producto BT").pack(
            side="left", padx=(0, 4))
        ttk.Spinbox(self.frame_digbt, from_=0.1, to=1.0, increment=0.05,
                    textvariable=self.bt_var, width=6).pack(
            side="left", padx=(0, 16))

        self.frame_gain = ttk.Frame(self.params_row)
        ttk.Label(self.frame_gain, text="Ganancia (dB)").pack(
            side="left", padx=(0, 4))
        ttk.Spinbox(self.frame_gain, from_=-20, to=20, increment=1,
                    textvariable=self.gain_var, width=5).pack(
            side="left", padx=(0, 16))

        self.frame_find = ttk.Frame(self.params_row)
        ttk.Checkbutton(self.frame_find, text="Buscar inicio de la ráfaga",
                        variable=self.find_start_var).pack(
            side="left", padx=(0, 16))

        # -- información y ayuda ---------------------------------------
        ttk.Label(c, textvariable=self.info_var, style="Info.TLabel",
                  wraplength=1240, justify="left").pack(anchor="w",
                                                        pady=(6, 0))
        ttk.Label(
            c,
            text="Elija la técnica y el origen, ajuste los parámetros y pulse "
                 "«Demodular». En el micrófono se busca automáticamente el "
                 "inicio de la ráfaga; para un WAV generado por la aplicación "
                 "la señal empieza en la muestra 0. Para recibir una señal "
                 "digital larga por micrófono, use una «Duración (s)» mayor "
                 "que la duración de la señal generada en la pestaña 6.",
            style="Subtitle.TLabel", wraplength=1240, justify="left").pack(
            anchor="w", pady=(2, 0))

        self.message("Elija la técnica y el origen de la señal modulada.\n"
                     "Después pulse «Demodular».")

    # ------------------------------------------------------------------
    # Cambio de técnica / origen
    # ------------------------------------------------------------------
    def _pack_param_frames(self) -> None:
        frames = (self.frame_carrier, self.frame_cutoff, self.frame_fmdev,
                  self.frame_dig, self.frame_digdev, self.frame_digbt,
                  self.frame_gain, self.frame_find)
        for frame in frames:
            frame.pack_forget()

        tech = self.tech_var.get()
        self.frame_carrier.pack(side="left")
        if tech == "AM":
            self.frame_cutoff.pack(side="left")
        elif tech == "FM":
            self.frame_fmdev.pack(side="left")
            self.frame_cutoff.pack(side="left")
        else:
            self.frame_dig.pack(side="left")
            if tech in FSK_TECHNIQUES:
                self.frame_digdev.pack(side="left")
            if tech == "GFSK":
                self.frame_digbt.pack(side="left")
            self.frame_find.pack(side="left")
        self.frame_gain.pack(side="left")

    def _on_technique_changed(self, _event=None) -> None:
        tech = self.tech_var.get()
        values = ["2", "4", "6"] if tech == "QAM" else \
            ["1", "2", "3", "4", "5", "6"]
        self.bps_combo.configure(values=values)
        if self.bps_var.get() not in values:
            self.bps_var.set(values[0])
        self._pack_param_frames()
        self._invalidate_demod()
        self.mark_dirty()
        self.refresh()

    def _on_source_changed(self, _event=None) -> None:
        mic = self.source_var.get() == SRC_MIC
        if mic:
            self.mic_frame.pack(fill="x", pady=(6, 0), before=self.params_row)
            self.find_start_var.set(True)
        else:
            self.mic_frame.pack_forget()
            self.find_start_var.set(False)
        self._received = None
        self._received_name = ""
        self._invalidate_demod()
        self.mark_dirty()
        self.refresh()

    # ------------------------------------------------------------------
    # Micrófono
    # ------------------------------------------------------------------
    def _load_devices(self) -> None:
        if self._recorder is not None and self._recorder.is_recording:
            return
        try:
            self._devices = list_input_devices()
        except Exception as exc:  # noqa: BLE001
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
            messagebox.showerror("Recepción",
                                 "La fs y la duración deben ser numéricas.")
            return
        if not 3.0 <= duration <= 60.0:
            messagebox.showerror("Recepción",
                                 "La duración debe estar entre 3 y 60 s.")
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
        self.app.set_status(
            f"Escuchando la señal {self.tech_var.get()} a {fs} Hz…")
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

        self._received = rec_audio
        self._received_name = f"micrófono ({rec_audio.duration:.2f} s)"
        self.record_info.set(f"Recibido: {rec_audio.duration:.2f} s")
        self._invalidate_demod()
        self.app.set_status(
            f"Señal capturada ({rec_audio.duration:.2f} s). Pulse «Demodular».")
        self.mark_dirty()
        self.refresh()

    # ------------------------------------------------------------------
    # Archivo
    # ------------------------------------------------------------------
    def _load_file(self) -> None:
        path = filedialog.askopenfilename(
            parent=self,
            title="Cargar una señal modulada",
            initialdir=str(output_dir()),
            filetypes=[("Audio WAV", "*.wav"), ("Todos los archivos", "*.*")],
        )
        if not path:
            return
        try:
            audio = read_wav(path)
        except AudioError as exc:
            messagebox.showerror("Cargar WAV", str(exc))
            self.app.set_status(str(exc), error=True)
            return

        self.source_var.set(SRC_FILE)
        self.mic_frame.pack_forget()
        self.find_start_var.set(False)
        self._received = audio
        self._received_name = Path(path).name
        self.record_info.set(f"Cargado: {Path(path).name} · {audio.duration:.2f} s")
        self._invalidate_demod()
        self.app.set_status(
            f"WAV cargado: {Path(path).name}. Pulse «Demodular».")
        self.mark_dirty()
        self.refresh()

    # ------------------------------------------------------------------
    # Demodulación
    # ------------------------------------------------------------------
    def _num(self, var: tk.StringVar, name: str) -> float:
        try:
            return float(str(var.get()).replace(",", "."))
        except (tk.TclError, ValueError) as exc:
            raise ModulationError(f"«{name}» debe ser un número.") from exc

    def _int(self, var: tk.StringVar, name: str) -> int:
        value = self._num(var, name)
        if abs(value - round(value)) > 1e-9:
            raise ModulationError(f"«{name}» debe ser un número entero.")
        return int(round(value))

    def _demodulate(self) -> None:
        received = self._received
        if received is None:
            messagebox.showinfo(
                "Demodular",
                "Primero capture la señal con el micrófono o cargue un WAV "
                "modulado.")
            return
        self._stop_play()
        tech = self.tech_var.get()

        try:
            carrier = self._num(self.carrier_var, "Portadora")
            gain_db = self._num(self.gain_var, "Ganancia")
            if tech == "AM":
                cutoff = self._num(self.cutoff_var, "Corte de audio")
                demod, fs_out = demodulate_am(
                    received.x, received.fs, carrier_freq=carrier,
                    cutoff_freq=cutoff)
                details = (f"AM · fc = {carrier:.0f} Hz · corte = "
                           f"{cutoff:.0f} Hz · fs de captura = {received.fs} Hz")
            elif tech == "FM":
                deviation = self._num(self.fm_dev_var, "Desviación Δf")
                cutoff = self._num(self.cutoff_var, "Corte de audio")
                demod, fs_out = demodulate_fm(
                    received.x, received.fs, carrier_freq=carrier,
                    deviation=deviation, cutoff_freq=cutoff)
                details = (f"FM · fc = {carrier:.0f} Hz · Δf = "
                           f"{deviation:.0f} Hz · corte = {cutoff:.0f} Hz · "
                           f"fs de captura = {received.fs} Hz")
            elif tech in DIGITAL_TECHNIQUES:
                baud = self._num(self.baud_var, "Baudios")
                bps = self._int(self.bps_var, "Bits/símbolo")
                pcm_bits = self._int(self.pcm_bits_var, "Bits PCM")
                audio_fs = self._num(self.audio_fs_var, "fs del audio")
                deviation = (self._num(self.dig_dev_var, "Desviación Δf")
                             if tech in FSK_TECHNIQUES else 0.0)
                bt = self._num(self.bt_var, "Producto BT") if tech == "GFSK" else 0.5
                result = demodulate_digital(
                    tech, received.x, received.fs, carrier, baud, bps,
                    pcm_bits, audio_fs=audio_fs, deviation=deviation, bt=bt,
                    find_start=bool(self.find_start_var.get()))
                demod = result.signal
                fs_out = result.fs
                details = result.info
            else:
                raise ModulationError(f"Técnica no soportada: {tech}")
        except ModulationError as exc:
            messagebox.showerror("Demodulación", str(exc))
            self.app.set_status(str(exc), error=True)
            return
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Demodulación",
                                 f"No se pudo demodular la señal:\n{exc}")
            self.app.set_status(f"Error de demodulación: {exc}", error=True)
            return

        demod = np.asarray(demod, dtype=np.float64).ravel()
        if demod.size == 0:
            messagebox.showwarning(
                "Demodulación",
                "La demodulación no produjo muestras de audio. Revise los "
                "parámetros y la duración de la señal recibida.")
            return
        if gain_db != 0.0:
            demod = np.clip(demod * (10.0 ** (gain_db / 20.0)), -1.0, 1.0)

        self._demod = Audio(
            x=demod, fs=int(round(fs_out)), source="receptor", path="",
            channels=1, bits=16, is_float=True)
        self._demod_info = details
        self.play_btn.configure(state="normal")
        self.save_audio_btn.configure(state="normal")
        # Guardar la última demodulación para la comparación original vs.
        # recuperado antes de reemplazar el audio compartido.
        self.app.last_demodulation = self._demod
        # El audio recuperado pasa a ser el audio compartido de la aplicación:
        # puede analizarse con FFT, digitalizarse de nuevo o reconstruirse.
        self.app.set_audio(self._demod)
        self.app.set_status(
            f"{tech} demodulada: {self._demod.duration:.2f} s a "
            f"{self._demod.fs} Hz.")
        self.mark_dirty()
        self.refresh()

    def _invalidate_demod(self) -> None:
        self._stop_play()
        self._demod = None
        self._demod_info = ""
        self.play_btn.configure(state="disabled")
        self.stop_play_btn.configure(state="disabled")
        self.save_audio_btn.configure(state="disabled")
        self.demod_btn.configure(
            state="normal" if self._received is not None else "disabled")

    # ------------------------------------------------------------------
    # Reproducción y guardado
    # ------------------------------------------------------------------
    def _play(self) -> None:
        audio = self._demod
        if audio is None:
            messagebox.showinfo("Reproducir",
                                "Primero demodule una señal.")
            return
        self._stop_play()
        self._player = Player(audio.x, audio.fs)
        self._player.start()
        self.play_btn.configure(state="disabled")
        self.stop_play_btn.configure(state="normal")
        self.app.set_status(
            f"Reproduciendo el audio demodulado a {audio.fs} Hz…")
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
        if self._demod is not None:
            self.play_btn.configure(state="normal")
        self.stop_play_btn.configure(state="disabled")

    def _save_audio(self) -> None:
        audio = self._demod
        if audio is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Guardar el audio demodulado",
            initialdir=str(output_dir()),
            initialfile=timestamp(
                f"demodulado_{self.tech_var.get().lower()}", "wav"),
            defaultextension=".wav",
            filetypes=[("Audio WAV", "*.wav"), ("Todos los archivos", "*.*")],
        )
        if not path:
            return
        try:
            save_wav(path, audio.x, audio.fs, bits=16)
        except (AudioError, ValueError) as exc:
            messagebox.showerror("Guardar audio", f"No se pudo guardar:\n{exc}")
            self.app.set_status(f"Error al guardar el audio: {exc}", error=True)
            return
        self.app.set_status(f"Audio demodulado guardado en {path}")

    # ------------------------------------------------------------------
    # Autocompletar con la última transmisión
    # ------------------------------------------------------------------
    def on_show(self) -> None:
        self._apply_last_transmission()
        super().on_show()

    def _apply_last_transmission(self) -> None:
        cfg = getattr(self.app, "last_transmission", None)
        serial = int(getattr(self.app, "last_transmission_serial", 0))
        if not cfg or serial == self._seen_tx_serial:
            return
        self._seen_tx_serial = serial
        tech = str(cfg.get("technique", "")).upper()
        if tech not in TECHNIQUES:
            return

        self.tech_var.set(tech)
        self.carrier_var.set(self._fmt(cfg.get("carrier"), 10000.0))
        if tech == "FM":
            self.fm_dev_var.set(self._fmt(cfg.get("fm_dev"), 3000.0))
        if tech in DIGITAL_TECHNIQUES:
            self.baud_var.set(self._fmt(cfg.get("baud"), 4000.0))
            bps = cfg.get("bps")
            if bps is not None:
                self.bps_var.set(str(int(bps)))
            self.pcm_bits_var.set(self._fmt(cfg.get("pcm_bits"), 8))
            self.audio_fs_var.set(self._fmt(cfg.get("audio_fs"), 16000.0))
            self.dig_dev_var.set(self._fmt(cfg.get("dig_dev"), 6000.0))
            self.bt_var.set(self._fmt(cfg.get("bt"), 0.5))
        self._on_technique_changed()
        self.app.set_status(
            f"Parámetros de la última transmisión {tech} cargados en la "
            "pestaña de recepción.")

    @staticmethod
    def _fmt(value, default: float) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float(default)
        return f"{number:g}"

    def _app_int(self, name: str, default: int) -> str:
        """Valor entero de una variable global de la aplicación."""
        try:
            var = getattr(self.app, name)
            return str(int(round(float(str(var.get()).replace(",", ".")))))
        except (AttributeError, TypeError, ValueError, tk.TclError):
            return str(int(default))

    # ------------------------------------------------------------------
    # Gráfica
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        received = self._received
        demod = self._demod
        tech = self.tech_var.get()

        if received is None and demod is None:
            self.info_var.set(
                "Sin señal recibida. Elija la técnica, capture con el "
                "micrófono o cargue un WAV modulado.")
            self.message("Elija la técnica y el origen de la señal modulada.\n"
                         "Después pulse «Demodular».")
            return

        if demod is not None and received is not None:
            ax1, ax2 = self.axes(2, height_ratios=[1.0, 1.25])
            x = np.asarray(received.x, dtype=np.float64)
            t = np.arange(len(x), dtype=np.float64) / received.fs
            td, xd = display_slice(t, x, 12000)
            ax1.plot(td, xd, color="#ff7f0e", lw=0.6,
                     label="señal modulada recibida")
            ax1.set_xlim(0.0, max(received.duration, 0.01))
            ax1.set_xlabel("Tiempo [s]")
            ax1.set_ylabel("Amplitud")
            ax1.set_title(f"Señal modulada recibida · {tech} · "
                          f"fs = {received.fs} Hz")
            ax1.grid(alpha=0.25)
            ax1.legend(loc="upper right", fontsize=8)

            y = np.asarray(demod.x, dtype=np.float64)
            ty = np.arange(len(y), dtype=np.float64) / demod.fs
            tdy, xdy = display_slice(ty, y, 12000)
            ax2.plot(tdy, xdy, color="#1f77b4", lw=0.6,
                     label="señal demodulada")
            r = rms(y)
            ax2.axhline(r, color="#d62728", ls="--", lw=0.8,
                        label=f"RMS = {r:.4f}")
            ax2.axhline(-r, color="#d62728", ls="--", lw=0.8)
            ax2.axhline(0.0, color="#888888", lw=0.5)
            ax2.set_xlim(0.0, max(demod.duration, 0.01))
            ax2.set_xlabel("Tiempo [s]")
            ax2.set_ylabel("Amplitud")
            ax2.set_title(f"Señal demodulada · fs = {demod.fs} Hz")
            ax2.grid(alpha=0.25)
            ax2.legend(loc="upper right", fontsize=8)
            self.draw()
            self.info_var.set(
                f"{self._demod_info}\nAudio demodulado: "
                f"{demod.duration:.3f} s · fs = {demod.fs} Hz · "
                f"{len(y)} muestras · pico = {np.max(np.abs(y)):.4f} · "
                f"RMS = {r:.4f}")
            return

        # Solo hay señal recibida (todavía sin demodular).
        ax = self.axes(1)
        x = np.asarray(received.x, dtype=np.float64)
        t = np.arange(len(x), dtype=np.float64) / received.fs
        td, xd = display_slice(t, x, 12000)
        ax.plot(td, xd, color="#ff7f0e", lw=0.6,
                label="señal modulada recibida")
        ax.set_xlim(0.0, max(received.duration, 0.01))
        ax.set_xlabel("Tiempo [s]")
        ax.set_ylabel("Amplitud")
        ax.set_title(f"Señal modulada recibida · {tech} · "
                     f"pulse «Demodular»")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
        self.draw()
        name = self._received_name or received.source
        self.info_var.set(
            f"Señal recibida: {name} · {received.duration:.3f} s · "
            f"fs = {received.fs} Hz · {len(x)} muestras.")
        self.demod_btn.configure(state="normal")

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
