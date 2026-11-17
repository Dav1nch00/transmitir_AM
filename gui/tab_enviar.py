"""Pestaña 6: generación y transmisión de la señal modulada.

Permite elegir la técnica (AM, FM o digitales ASK/FSK/PSK/QAM/GFSK/MFSK),
ajustar sus parámetros, generar la señal y después guardarla en WAV o
transmitirla por el altavoz.
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

from core.am import modulate_am, prepare_baseband
from core.audio_io import Audio, AudioError, Player, save_wav
from core.modulation import (ANALOG_TECHNIQUES, DIGITAL_TECHNIQUES,
                             ModulationError, bits_from_codes, modulate_digital,
                             modulate_fm)
from core.quantization import AdcConfig, quantize

from .widgets import PlotTab, output_dir, timestamp

TECHNIQUES = ANALOG_TECHNIQUES + DIGITAL_TECHNIQUES
FS_TX = 44100.0
MAX_MOD_DURATION = 120.0  # duración máxima de la señal modulada, en segundos


class TabEnviar(PlotTab):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.tech_var = tk.StringVar(value="AM")
        self.carrier_var = tk.StringVar(value="10000")
        self.index_var = tk.StringVar(value="0.95")
        self.fm_dev_var = tk.StringVar(value="3000")
        self.baud_var = tk.StringVar(value="4000")
        self.bps_var = tk.StringVar(value="2")
        self.dig_dev_var = tk.StringVar(value="6000")
        self.bt_var = tk.StringVar(value="0.5")
        self.audio_seconds_var = tk.StringVar(value="2.0")
        self.dig_est_var = tk.StringVar(value="")
        self.source_var = tk.StringVar(value="Señal digitalizada (DAC)")
        self.status_tx = tk.StringVar(value="Sin generar.")
        self.info_var = tk.StringVar(value="")
        self.mod_label_var = tk.StringVar(value="")

        self._player: Player | None = None
        self._poll_id: str | None = None
        self._base_signal: np.ndarray | None = None
        self._mod_signal: np.ndarray | None = None
        self._fs_tx: float = FS_TX
        self._tech: str = ""
        self._mod_info: str = ""
        self._base_label: str = "banda base"
        self._is_digital: bool = False
        self._baud: float = 0.0
        self._last_pcm_bits: int | None = None
        self._last_audio_fs: float | None = None
        self._reference_audio: Audio | None = None

        self._build_controls()
        self._on_technique_changed()
        self.message("Cargue un audio, elija la técnica y pulse «Generar modulada».")

    # ------------------------------------------------------------------
    def _build_controls(self) -> None:
        c = self.controls
        row1 = ttk.Frame(c)
        row1.pack(fill="x")

        ttk.Label(row1, text="Fuente").pack(side="left", padx=(0, 4))
        self.source_combo = ttk.Combobox(
            row1, textvariable=self.source_var, state="readonly", width=24,
            values=["Señal digitalizada (DAC)", "Audio original"])
        self.source_combo.pack(side="left", padx=(0, 12))
        self.source_combo.bind("<<ComboboxSelected>>", self._invalidate_generated)

        ttk.Label(row1, text="Técnica").pack(side="left", padx=(0, 4))
        self.tech_combo = ttk.Combobox(
            row1, textvariable=self.tech_var, state="readonly", width=8,
            values=list(TECHNIQUES))
        self.tech_combo.pack(side="left", padx=(0, 8))
        self.tech_combo.bind("<<ComboboxSelected>>", self._on_technique_changed)

        ttk.Label(row1, text="Portadora (Hz)").pack(side="left", padx=(0, 4))
        ttk.Spinbox(row1, from_=500, to=22000, increment=100,
                    textvariable=self.carrier_var, width=8).pack(
            side="left", padx=(0, 12))

        self.gen_btn = ttk.Button(row1, text="🎛 Generar modulada",
                                  command=self._generate)
        self.gen_btn.pack(side="left", padx=(0, 6))

        self.save_btn = ttk.Button(row1, text="💾 Guardar WAV modulada",
                                   command=self._save_modulated, state="disabled")
        self.save_btn.pack(side="left", padx=(0, 4))

        self.tx_btn = ttk.Button(row1, text="📡 Transmitir por el aire",
                                 command=self._transmit, state="disabled")
        self.tx_btn.pack(side="left", padx=(0, 4))

        self.stop_btn = ttk.Button(row1, text="■ Detener", command=self._stop,
                                   state="disabled")
        self.stop_btn.pack(side="left", padx=(0, 10))

        # -- parámetros dinámicos de cada técnica ----------------------
        self.dyn_frame = ttk.Frame(c)
        self.dyn_frame.pack(fill="x", pady=(6, 0))

        self.frame_am = ttk.Frame(self.dyn_frame)
        ttk.Label(self.frame_am, text="Índice de modulación m").pack(
            side="left", padx=(0, 4))
        ttk.Spinbox(self.frame_am, from_=0.1, to=1.0, increment=0.05,
                    textvariable=self.index_var, width=6).pack(side="left",
                                                               padx=(0, 16))

        self.frame_fm = ttk.Frame(self.dyn_frame)
        ttk.Label(self.frame_fm, text="Desviación Δf (Hz)").pack(
            side="left", padx=(0, 4))
        ttk.Spinbox(self.frame_fm, from_=100, to=15000, increment=100,
                    textvariable=self.fm_dev_var, width=8).pack(side="left",
                                                                padx=(0, 16))

        self.frame_dig = ttk.Frame(self.dyn_frame)
        ttk.Label(self.frame_dig, text="Baudios").pack(side="left", padx=(0, 4))
        ttk.Spinbox(self.frame_dig, from_=100, to=11000, increment=100,
                    textvariable=self.baud_var, width=7).pack(side="left",
                                                               padx=(0, 12))
        ttk.Label(self.frame_dig, text="Bits/símbolo").pack(side="left",
                                                            padx=(0, 4))
        self.bps_combo = ttk.Combobox(self.frame_dig, textvariable=self.bps_var,
                                      state="readonly", width=3,
                                      values=["1", "2", "3", "4", "5", "6"])
        self.bps_combo.pack(side="left", padx=(0, 4))
        self.bps_combo.bind("<<ComboboxSelected>>", self._update_mod_label)
        ttk.Label(self.frame_dig, textvariable=self.mod_label_var,
                  style="Subtitle.TLabel").pack(side="left", padx=(0, 14))

        ttk.Label(self.frame_dig, text="Audio (s)").pack(side="left",
                                                         padx=(0, 4))
        self.audio_sec_spin = ttk.Spinbox(
            self.frame_dig, from_=0.2, to=8.0, increment=0.5,
            textvariable=self.audio_seconds_var, width=6)
        self.audio_sec_spin.pack(side="left", padx=(0, 4))
        ttk.Label(self.frame_dig, textvariable=self.dig_est_var,
                  style="Subtitle.TLabel").pack(side="left", padx=(0, 14))

        self.frame_dev = ttk.Frame(self.dyn_frame)
        ttk.Label(self.frame_dev, text="Desviación Δf (Hz)").pack(
            side="left", padx=(0, 4))
        ttk.Spinbox(self.frame_dev, from_=100, to=15000, increment=100,
                    textvariable=self.dig_dev_var, width=8).pack(side="left",
                                                                 padx=(0, 16))

        self.frame_bt = ttk.Frame(self.dyn_frame)
        ttk.Label(self.frame_bt, text="Producto BT").pack(side="left",
                                                          padx=(0, 4))
        ttk.Spinbox(self.frame_bt, from_=0.1, to=1.0, increment=0.05,
                    textvariable=self.bt_var, width=6).pack(side="left",
                                                            padx=(0, 16))

        ttk.Label(c, textvariable=self.status_tx, style="Metric.TLabel").pack(
            anchor="w", pady=(6, 0))
        ttk.Label(c, textvariable=self.info_var, style="Info.TLabel",
                  wraplength=1150, justify="left").pack(anchor="w", pady=(2, 0))

        ttk.Label(
            c,
            text="AM y FM usan toda la banda base. Las técnicas digitales "
                 "convierten en bits los segundos de audio indicados en "
                 "«Audio (s)» y el resto del audio no se transmite. La señal "
                 "modulada ocupa más tiempo cuanto más larga sea esa duración, "
                 "cuantos más bits PCM tenga y menos baudios se usen. Tras "
                 "generar puede guardar el WAV o transmitir la señal por el "
                 "altavoz.",
            style="Subtitle.TLabel", wraplength=1150, justify="left").pack(
            anchor="w", pady=(4, 0))

        ttk.Label(
            c,
            text="Después de generar, la pestaña 7 (Recibir) adopta "
                 "automáticamente la técnica y los parámetros de esta "
                 "transmisión para demodularla. Si va a recibir la señal por "
                 "micrófono, ajuste allí una «Duración (s)» mayor que la "
                 "duración de la señal modulada.",
            style="Subtitle.TLabel", wraplength=1150, justify="left").pack(
            anchor="w", pady=(2, 0))

        for var in (self.baud_var, self.bps_var, self.audio_seconds_var,
                    self.source_var):
            var.trace_add("write", lambda *_args: self._update_dig_estimate())

    # ------------------------------------------------------------------
    def _on_technique_changed(self, _event=None) -> None:
        tech = self.tech_var.get()
        for frame in (self.frame_am, self.frame_fm, self.frame_dig,
                      self.frame_dev, self.frame_bt):
            frame.pack_forget()

        if tech == "AM":
            self.frame_am.pack(side="left")
        elif tech == "FM":
            self.frame_fm.pack(side="left")
        else:
            self.frame_dig.pack(side="left")
            if tech in ("FSK", "GFSK", "MFSK"):
                self.frame_dev.pack(side="left")
            if tech == "GFSK":
                self.frame_bt.pack(side="left")

        if tech == "QAM":
            values = ["2", "4", "6"]
        else:
            values = ["1", "2", "3", "4", "5", "6"]
        self.bps_combo.configure(values=values)
        if self.bps_var.get() not in values:
            self.bps_var.set(values[0])

        self._update_mod_label()
        self._invalidate_generated()

    def _update_mod_label(self, _event=None) -> None:
        try:
            bps = int(self.bps_var.get())
        except ValueError:
            bps = 1
        m = 1 << bps
        tech = self.tech_var.get()
        if tech in DIGITAL_TECHNIQUES:
            self.mod_label_var.set(f"M = {m}")
        else:
            self.mod_label_var.set("")
        self._update_dig_estimate()

    # ------------------------------------------------------------------
    def _source_fs(self) -> float | None:
        """Frecuencia de muestreo del audio que entraría al modulador."""
        if self.source_var.get().startswith("Señal digitalizada"):
            try:
                return float(self.app.current_params().fs_target)
            except Exception:  # noqa: BLE001
                pass
        audio = getattr(self.app, "audio", None)
        if audio is not None:
            return float(audio.fs)
        return None

    def _source_len(self) -> int | None:
        """Cantidad de muestras disponibles en la fuente elegida."""
        audio = getattr(self.app, "audio", None)
        if audio is None:
            return None
        if self.source_var.get().startswith("Señal digitalizada"):
            try:
                p = self.app.current_params()
                return max(1, int(round(audio.duration * p.fs_target)))
            except Exception:  # noqa: BLE001
                return None
        return len(audio.x)

    def _update_dig_estimate(self, _event=None) -> None:
        """Muestra cuánto durará el audio útil y la señal modulada."""
        tech = self.tech_var.get()
        if tech not in DIGITAL_TECHNIQUES:
            self.dig_est_var.set("")
            return
        try:
            seconds = self._num(self.audio_seconds_var, "Audio")
            baud = self._num(self.baud_var, "Baudios")
            bps = max(1, int(self.bps_var.get()))
            pcm_bits = int(self.app.bits_var.get())
        except (ModulationError, tk.TclError, ValueError):
            self.dig_est_var.set("")
            return
        fs_in = self._source_fs()
        if fs_in is None or fs_in <= 0.0 or baud <= 0.0 or pcm_bits < 1:
            self.dig_est_var.set("")
            return
        available = self._source_len()
        requested = max(1, int(round(seconds * fs_in)))
        used = requested if available is None else min(requested, available)
        used_seconds = used / fs_in
        symbols = -(-(used * pcm_bits) // bps)
        duration = symbols / baud
        text = f"≈ {used_seconds:.2f} s de audio → {duration:.1f} s de señal"
        if available is not None and requested > available:
            text = (f"≈ {used_seconds:.2f} s de audio (máx. "
                    f"{available / fs_in:.2f} s) → {duration:.1f} s de señal")
        if duration > MAX_MOD_DURATION:
            text += f" · máx. {MAX_MOD_DURATION:.0f} s"
        self.dig_est_var.set(text)

    # ------------------------------------------------------------------
    def _invalidate_generated(self, _event=None) -> None:
        self._stop()
        self._mod_signal = None
        self._base_signal = None
        self._tech = ""
        self._mod_info = ""
        self._reference_audio = None
        self.save_btn.configure(state="disabled")
        self.tx_btn.configure(state="disabled")
        self.info_var.set("")
        self.status_tx.set("Sin generar.")
        self.message("Elija la técnica y sus parámetros, y pulse «Generar modulada».")

    def mark_dirty(self) -> None:
        super().mark_dirty()
        self._invalidate_generated()
        self._update_dig_estimate()

    # ------------------------------------------------------------------
    def _num(self, var: tk.StringVar, name: str) -> float:
        try:
            return float(var.get().replace(",", "."))
        except (tk.TclError, ValueError) as exc:
            raise ModulationError(f"«{name}» debe ser un número.") from exc

    @staticmethod
    def _try_num(var: tk.StringVar, default: float) -> float:
        try:
            return float(var.get().replace(",", "."))
        except (tk.TclError, ValueError):
            return float(default)

    @classmethod
    def _try_int(cls, var: tk.StringVar, default: int) -> int:
        return int(round(cls._try_num(var, float(default))))

    def _generate(self) -> None:
        audio = self.app.audio
        if audio is None:
            messagebox.showinfo("Generar modulada",
                                "Primero grabe o cargue un audio.")
            return

        tech = self.tech_var.get()
        self._stop()
        self._reference_audio = None
        try:
            carrier = self._num(self.carrier_var, "Portadora")
            if tech in ANALOG_TECHNIQUES:
                base, fs_tx, signal, info, base_label = self._generate_analog(
                    audio, tech, carrier)
                is_digital = False
                baud = 0.0
            else:
                base, fs_tx, signal, info, baud = self._generate_digital(
                    audio, tech, carrier)
                base_label = "bits modulantes (NRZ)"
                is_digital = True
        except ModulationError as exc:
            messagebox.showerror("Modulación", str(exc))
            self.app.set_status(str(exc), error=True)
            return
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Modulación", f"No se pudo modular:\n{exc}")
            self.app.set_status(f"Error de modulación: {exc}", error=True)
            return

        self._base_signal = base
        self._mod_signal = signal
        self._fs_tx = float(fs_tx)
        self._tech = tech
        self._mod_info = info
        self._base_label = base_label
        self._is_digital = is_digital
        self._baud = float(baud)

        # Publicar la configuración para que la pestaña de recepción pueda
        # demodular exactamente esta transmisión.
        self.app.last_transmission = {
            "technique": tech,
            "carrier": float(carrier),
            "fm_dev": self._try_num(self.fm_dev_var, 3000.0),
            "baud": self._try_num(self.baud_var, 4000.0),
            "bps": self._try_int(self.bps_var, 2),
            "dig_dev": self._try_num(self.dig_dev_var, 6000.0),
            "bt": self._try_num(self.bt_var, 0.5),
            "pcm_bits": self._last_pcm_bits,
            "audio_fs": self._last_audio_fs,
        }
        self.app.last_transmission_reference = self._reference_audio
        self.app.last_transmission_serial = int(
            getattr(self.app, "last_transmission_serial", 0)) + 1

        self.save_btn.configure(state="normal")
        self.tx_btn.configure(state="normal")
        if is_digital and self._reference_audio is not None:
            audio_duration = self._reference_audio.duration
            self.status_tx.set(
                f"{tech} generada · señal {len(signal) / fs_tx:.2f} s · "
                f"audio {audio_duration:.2f} s")
        else:
            self.status_tx.set(f"{tech} generada · {len(signal) / fs_tx:.2f} s")
        self.app.set_status(f"Señal {tech} generada. Guárdela o transmítala.")
        self.refresh()

    def _generate_analog(self, audio, tech: str, carrier: float):
        self._last_pcm_bits = None
        self._last_audio_fs = None
        source = self.source_var.get()
        if source.startswith("Señal digitalizada"):
            d = self.app.get_digitized()
            if d is None:
                raise ModulationError("Primero digitalice el audio (pestaña 2).")
            vref = max(float(d.params.vref), 1e-12)
            ref = np.asarray(d.quant.volts_q, dtype=np.float64) / vref
            fs_in = float(d.fs_eff)
            ref_label = "señal digitalizada (DAC)"
        else:
            ref = np.asarray(audio.x, dtype=np.float64)
            fs_in = float(audio.fs)
            ref_label = "audio original"

        self._reference_audio = Audio(
            x=ref, fs=int(round(fs_in)), source=f"referencia TX · {ref_label}",
            path="", channels=1, bits=16, is_float=True)
        base, fs_tx = prepare_baseband(ref, fs_in, FS_TX)
        if tech == "AM":
            index = self._num(self.index_var, "Índice de modulación")
            if not 0.0 < index <= 1.0:
                raise ModulationError("El índice de modulación debe estar en (0, 1].")
            signal = modulate_am(base, fs_tx, carrier_freq=carrier,
                                 index_mod=index)
            info = (f"AM-DSB-FC · fc = {carrier:.0f} Hz · m = {index:.2f} · "
                    f"fs = {fs_tx:.0f} Hz · duración = {len(signal) / fs_tx:.2f} s")
            label = "banda base analógica"
        else:
            dev = self._num(self.fm_dev_var, "Desviación Δf")
            signal = modulate_fm(base, fs_tx, carrier_freq=carrier,
                                 deviation=dev)
            info = (f"FM · fc = {carrier:.0f} Hz · Δf = {dev:.0f} Hz · "
                    f"fs = {fs_tx:.0f} Hz · duración = {len(signal) / fs_tx:.2f} s")
            label = "banda base analógica"
        return base, fs_tx, signal, info, label

    def _generate_digital(self, audio, tech: str, carrier: float):
        source = self.source_var.get()
        if source.startswith("Señal digitalizada"):
            d = self.app.get_digitized()
            if d is None:
                raise ModulationError("Primero digitalice el audio (pestaña 2).")
            codes = d.quant.codes_u
            pcm_bits = int(d.params.bits)
            fs_in = float(d.fs_eff)
            ref_full = (np.asarray(d.quant.volts_q, dtype=np.float64)
                        / max(float(d.params.vref), 1e-12))
            ref_label = "señal digitalizada (DAC)"
        else:
            try:
                pcm_bits = int(self.app.bits_var.get())
            except (tk.TclError, ValueError):
                pcm_bits = 8
            qr = quantize(np.asarray(audio.x, dtype=np.float64),
                          AdcConfig(pcm_bits, 1.0), fs=float(audio.fs))
            codes = qr.codes_u
            fs_in = float(audio.fs)
            ref_full = np.asarray(audio.x, dtype=np.float64)
            ref_label = "audio original"

        seconds = self._num(self.audio_seconds_var, "Audio")
        if not 0.05 <= seconds <= 60.0:
            raise ModulationError(
                "La duración de audio debe estar entre 0.05 y 60 s.")
        samples = max(1, int(round(seconds * fs_in)))
        samples = min(samples, len(codes))
        used_seconds = samples / fs_in
        bits = bits_from_codes(codes[:samples], pcm_bits)
        self._reference_audio = Audio(
            x=ref_full[:samples], fs=int(round(fs_in)),
            source=f"referencia TX · {ref_label}",
            path="", channels=1, bits=16, is_float=True)

        baud = self._num(self.baud_var, "Baudios")
        bps = max(1, int(self.bps_var.get()))
        dev = self._num(self.dig_dev_var, "Desviación Δf")
        bt = self._num(self.bt_var, "Producto BT")

        symbols = -(-bits.size // bps)
        est_duration = symbols / baud if baud > 0 else float("inf")
        if est_duration > MAX_MOD_DURATION:
            raise ModulationError(
                f"La señal modulada duraría {est_duration:.0f} s "
                f"(máximo {MAX_MOD_DURATION:.0f} s). Reduzca «Audio (s)» "
                "o aumente los baudios.")

        result = modulate_digital(tech, bits, FS_TX, carrier_freq=carrier,
                                  symbol_rate=baud, bits_per_symbol=bps,
                                  deviation=dev, bt=bt)
        self._last_pcm_bits = int(pcm_bits)
        self._last_audio_fs = float(fs_in)
        info = (f"{result.info} · audio útil = {used_seconds:.2f} s "
                f"({samples} muestras PCM de {pcm_bits} bits, fs original "
                f"{fs_in:.0f} Hz) · salida a {result.fs} Hz")
        return result.base, result.fs, result.signal, info, baud

    # ------------------------------------------------------------------
    def _save_modulated(self) -> None:
        signal = self._mod_signal
        if signal is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Guardar la señal modulada como WAV",
            initialdir=str(output_dir()),
            initialfile=timestamp(f"modulada_{self._tech.lower()}", "wav"),
            defaultextension=".wav",
            filetypes=[("Audio WAV", "*.wav"), ("Todos los archivos", "*.*")],
        )
        if not path:
            return
        try:
            save_wav(path, signal, int(round(self._fs_tx)), bits=16)
        except (AudioError, ValueError) as exc:
            messagebox.showerror("Guardar WAV", f"No se pudo guardar:\n{exc}")
            self.app.set_status(f"Error al guardar el WAV: {exc}", error=True)
            return
        self.status_tx.set(f"Guardada: {Path(path).name}")
        self.app.set_status(f"Señal modulada guardada en {path}")

    def _transmit(self) -> None:
        signal = self._mod_signal
        if signal is None:
            return
        self._stop()
        self._player = Player(signal, int(round(self._fs_tx)))
        self._player.start()
        self.tx_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_tx.set(f"Transmitiendo {self._tech} por el altavoz…")
        self.app.set_status(
            f"Transmitiendo {self._tech} a {self._fs_tx:.0f} Hz por el altavoz…")
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
            self.tx_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            return
        if p.is_playing:
            try:
                self._poll_id = self.after(150, self._poll_player)
            except tk.TclError:
                self._poll_id = None
            return
        self.tx_btn.configure(state="normal")
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
        self.stop_btn.configure(state="disabled")
        if self._mod_signal is not None:
            self.tx_btn.configure(state="normal")

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        if self._mod_signal is None or self._base_signal is None:
            self.message("Elija la técnica y pulse «Generar modulada».")
            return
        signal = self._mod_signal
        base = self._base_signal
        fs = self._fs_tx

        if self._is_digital and self._baud > 0:
            symbol = fs / self._baud
            span = max(int(round(0.02 * fs)), int(round(8 * symbol)))
        else:
            span = int(round(0.05 * fs))
        n = max(2, min(len(signal), span))
        t = np.arange(n) / fs * 1e3

        ax1, ax2 = self.axes(2, height_ratios=[1.0, 1.2])
        ax1.plot(t, base[:n], color="#1f77b4", lw=1.0, label=self._base_label)
        ax1.set_xlim(0.0, t[-1])
        ax1.set_xlabel("Tiempo [ms]")
        ax1.set_ylabel("Amplitud")
        ax1.set_title("Señal de banda base / bits modulantes")
        ax1.grid(alpha=0.25)
        ax1.legend(fontsize=8, loc="upper right")

        ax2.plot(t, signal[:n], color="#d62728", lw=0.8,
                 label=f"señal {self._tech} transmitida")
        ax2.set_xlim(0.0, t[-1])
        ax2.set_xlabel("Tiempo [ms]")
        ax2.set_ylabel("Amplitud")
        ax2.set_title(f"Señal modulada {self._tech} · fs = {fs:.0f} Hz")
        ax2.grid(alpha=0.25)
        ax2.legend(fontsize=8, loc="upper right")
        self.draw()

        self.info_var.set(self._mod_info)

    def shutdown(self) -> None:
        self._stop()
