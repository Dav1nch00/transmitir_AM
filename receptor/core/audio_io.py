"""Captura de micrófono, lectura/escritura de WAV y reproducción.

Todo el audio que circula por el programa es mono, float64 y normalizado a
±1.0 (escala completa).  La conversión a entero solo se hace al guardar o al
entregar muestras a PyAudio.
"""
from __future__ import annotations

import os
import struct
import threading
import time
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.io import wavfile

PA_CONTINUE = 0


class AudioError(Exception):
    """Error de audio que se puede mostrar al usuario."""


@dataclass
class Audio:
    x: np.ndarray
    fs: int
    source: str = "archivo"
    path: str = ""
    channels: int = 1
    bits: int = 16
    is_float: bool = False

    @property
    def duration(self) -> float:
        return len(self.x) / float(self.fs)

    @property
    def name(self) -> str:
        return os.path.basename(self.path) if self.path else self.source


def _read_fmt_chunk(path: str) -> tuple[int, int, int, int]:
    """Devuelve (formato, canales, fs, bits) del chunk 'fmt ' del WAV."""
    with open(path, "rb") as f:
        head = f.read(12)
        if len(head) < 12 or head[:4] not in (b"RIFF", b"RF64") or head[8:12] != b"WAVE":
            raise AudioError("El archivo no es un WAV válido (cabecera RIFF/WAVE).")
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            cid, size = struct.unpack("<4sI", hdr)
            if cid == b"fmt ":
                raw = f.read(size)
                if len(raw) < 16:
                    break
                tag, ch, fs, _, _, bits = struct.unpack("<HHIIHH", raw[:16])
                return tag, ch, fs, bits
            f.seek(size + (size & 1), 1)
    raise AudioError("El WAV no contiene el chunk 'fmt '.")


def _to_float(data: np.ndarray, bits: int) -> np.ndarray:
    if data.dtype == np.uint8:
        return (data.astype(np.float64) - 128.0) / 128.0
    if data.dtype.kind == "i":
        if bits == 24:
            scale = 2.0 ** 31
        elif bits == 32:
            scale = 2.0 ** 31
        elif bits == 16:
            scale = 2.0 ** 15
        else:
            scale = float(np.iinfo(data.dtype).max + 1)
        return data.astype(np.float64) / scale
    if data.dtype.kind == "f":
        return data.astype(np.float64)
    raise AudioError(f"Tipo de muestra no soportado: {data.dtype}")


def read_wav(path: str) -> Audio:
    try:
        _, _, _, bits = _read_fmt_chunk(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fs, data = wavfile.read(path)
    except AudioError:
        raise
    except Exception as exc:
        raise AudioError(f"No se pudo leer el WAV: {exc}") from exc

    x = _to_float(data, int(bits))
    channels = 1 if x.ndim == 1 else int(x.shape[1])
    if x.ndim == 2:
        x = x.mean(axis=1)
    if len(x) < 256:
        raise AudioError("El audio es demasiado corto (mínimo 256 muestras).")
    return Audio(
        x=x.astype(np.float64),
        fs=int(fs),
        source="archivo",
        path=path,
        channels=channels,
        bits=int(bits),
        is_float=data.dtype.kind == "f",
    )


def save_wav(path: str, x: np.ndarray, fs: int, bits: int = 16) -> None:
    if bits not in (8, 16):
        raise AudioError("Solo se exportan WAV PCM de 8 o 16 bits.")
    folder = os.path.dirname(os.path.abspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)
    y = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    if bits == 16:
        data = np.round(y * 32767.0).astype(np.int16)
    else:
        data = np.clip(np.round(y * 127.0) + 128.0, 0, 255).astype(np.uint8)
    wavfile.write(path, int(fs), data)


def list_input_devices() -> list[dict]:
    try:
        import pyaudio
    except ImportError as exc:
        raise AudioError("PyAudio no está instalado. Instale PyAudio para grabar.") from exc

    p = pyaudio.PyAudio()
    try:
        devices = []
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            if int(d.get("maxInputChannels", 0)) > 0:
                devices.append({
                    "index": i,
                    "name": str(d.get("name", f"Entrada {i}")),
                    "channels": int(d["maxInputChannels"]),
                    "fs": int(float(d.get("defaultSampleRate", 44100))),
                })
        if not devices:
            raise AudioError("No se encontraron micrófonos disponibles.")
        return devices
    finally:
        p.terminate()


def default_input_index() -> int | None:
    try:
        import pyaudio
    except ImportError:
        return None
    p = pyaudio.PyAudio()
    try:
        return int(p.get_default_input_device_info()["index"])
    except Exception:
        return None
    finally:
        p.terminate()


class MicRecorder:
    def __init__(self, fs: int = 44100, device: int | None = None,
                 channels: int = 1, blocksize: int = 1024):
        self.fs = int(fs)
        self.device = device
        self.channels = int(channels)
        self.blocksize = int(blocksize)
        self._chunks: list[np.ndarray] = []
        self._frames = 0
        self._level = 0.0
        self._lock = threading.Lock()
        self._stream = None
        self._pa = None
        self._active = False
        self.started_at = 0.0

    @property
    def elapsed(self) -> float:
        return self._frames / float(self.fs)

    @property
    def level(self) -> float:
        return self._level

    @property
    def is_recording(self) -> bool:
        return self._active

    def start(self) -> None:
        if self._active:
            return
        try:
            import pyaudio
        except ImportError as exc:
            raise AudioError("PyAudio no está instalado: no se puede grabar.") from exc

        self._pa = pyaudio.PyAudio()
        try:
            if self.device is None:
                info = self._pa.get_default_input_device_info()
            else:
                info = self._pa.get_device_info_by_index(int(self.device))
            index = int(info["index"])
            max_ch = max(1, int(info.get("maxInputChannels", 1)))
            self.channels = 1 if max_ch == 1 else min(self.channels, max_ch)
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.fs,
                input=True,
                input_device_index=index,
                frames_per_buffer=self.blocksize,
                stream_callback=self._callback,
            )
            self._stream.start_stream()
            self._active = True
            self.started_at = time.monotonic()
        except Exception as exc:
            self._close()
            raise AudioError(
                f"No se pudo abrir el micrófono a {self.fs} Hz.\n{exc}\n"
                "Pruebe con 44100 Hz o elija otro dispositivo."
            ) from exc

    def _callback(self, in_data, frame_count, time_info, status):
        if not self._active:
            return (None, PA_CONTINUE)
        raw = np.frombuffer(in_data, dtype=np.int16).astype(np.float64) / 32768.0
        if self.channels > 1:
            raw = raw.reshape(-1, self.channels).mean(axis=1)
        with self._lock:
            self._chunks.append(raw)
            self._frames += len(raw)
            if len(raw):
                self._level = float(np.max(np.abs(raw)))
        return (None, PA_CONTINUE)

    def stop(self) -> Audio:
        if not self._active and not self._chunks:
            raise AudioError("No hay ninguna grabación en curso.")
        self._active = False
        self._close()
        with self._lock:
            chunks = list(self._chunks)
        if not chunks:
            raise AudioError("La grabación quedó vacía. Revise el micrófono elegido.")
        x = np.concatenate(chunks).astype(np.float64)
        return Audio(x=x, fs=self.fs, source="micrófono", path="",
                     channels=1, bits=16, is_float=False)

    def cancel(self) -> None:
        self._active = False
        self._close()
        with self._lock:
            self._chunks = []
            self._frames = 0

    def _close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop_stream()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None


class Player:
    def __init__(self, x: np.ndarray, fs: int, device: int | None = None):
        self.fs = int(fs)
        self.device = device
        self.x = np.asarray(x, dtype=np.float64)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: str | None = None

    @property
    def is_playing(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def error(self) -> str | None:
        return self._error

    def start(self) -> None:
        if self.is_playing:
            return
        self._stop.clear()
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        try:
            import pyaudio
        except ImportError:
            self._error = "PyAudio no está instalado."
            return
        p = pyaudio.PyAudio()
        stream = None
        try:
            data = (np.clip(self.x, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.fs,
                            output=True, frames_per_buffer=1024)
            step = 2048
            for i in range(0, len(data), step):
                if self._stop.is_set():
                    break
                stream.write(data[i:i + step])
        except Exception as exc:
            self._error = f"No se pudo reproducir a {self.fs} Hz: {exc}"
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            p.terminate()


def wait_player(player: Player, timeout: float = 30.0) -> None:
    end = time.monotonic() + timeout
    while player.is_playing and time.monotonic() < end:
        time.sleep(0.02)