"""Modulación y demodulación AM (emisor + receptor).

Agrupa la lógica de ``referencia/Emisor.py`` y ``referencia/Receptor.py``:

Emisión (DSB-FC)
    s(t) = (1 + m·x(t))·cos(2π·fc·t), normalizada para el altavoz.

Recepción
    1. Filtro pasa-bandas alrededor de la portadora (fc ± B).
    2. Envolvente: rectificación + filtro pasa-bajas, o transformada de
       Hilbert.
    3. Eliminación del offset DC y normalización a pico 1.

Todos los filtros se implementan con Butterworth en formato SOS y
``sosfiltfilt`` (filtrado cero-fase, estable en órdenes altos).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

from .dsp import resample_to

# Ancho de banda de demodulación alrededor de la portadora (Hz).
# Coincide con la referencia (Receptor.py): fc ± 2000 Hz.
DEMOD_BANDWIDTH = 2000.0

# fs de transmisión por altavoz.
FS_TX = 44100.0


def _normalize_peak(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0.0:
        x = x / peak
    return x


def _edge_margin(n: int, fs: float) -> int:
    """Margen que descarta transitorios de filtrado en los bordes (20 ms)."""
    return min(int(round(0.02 * float(fs))), max(1, n // 10))


def _finish_demod(envelop: np.ndarray, fs: float) -> tuple[np.ndarray, float]:
    """Quita el DC (media central), normaliza y recorta los bordes."""
    n = len(envelop)
    m = _edge_margin(n, fs)
    center = envelop[m:n - m] if n > 2 * m else envelop
    demod = envelop - float(np.mean(center))
    peak = float(np.max(np.abs(center - np.mean(center)))) if center.size else 0.0
    if peak > 0.0:
        demod = demod / peak
    return demod[m:n - m], float(fs)


def prepare_baseband(signal, fs_in: float, fs_out: float = FS_TX) -> tuple[np.ndarray, float]:
    """Normaliza a pico 1 y remuestrea la banda base a ``fs_out``.

    Usa remuestreo polifásico antialiasing (``core.dsp.resample_to``).
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    if x.size == 0:
        return x, float(fs_in)
    if float(fs_in) != float(fs_out):
        try:
            x, fs_out = resample_to(x, fs_in, fs_out, antialias=True)
        except (ValueError, ZeroDivisionError):
            pass
    return _normalize_peak(x), float(fs_out)


def modulate_am(audio_signal, sample_rate: float, carrier_freq: float = 17000.0,
                index_mod: float = 0.7) -> np.ndarray:
    """Modula una señal de banda base en AM (DSB-FC).

    ``s(t) = (1 + m·x(t))·cos(2π·fc·t)`` y se normaliza al final para
    evitar distorsión en el altavoz.
    """
    x = np.asarray(audio_signal, dtype=np.float64).ravel()
    if x.size == 0:
        return x
    t = np.arange(x.size, dtype=np.float64) / float(sample_rate)
    carrier = np.cos(2.0 * np.pi * float(carrier_freq) * t)
    am_signal = (1.0 + float(index_mod) * x) * carrier
    return _normalize_peak(am_signal)


def bandpass_filter(data, lowcut: float, highcut: float, fs: float,
                    order: int = 4) -> np.ndarray:
    """Filtro pasa-bandas Butterworth cero-fase (SOS + sosfiltfilt)."""
    nyq = 0.5 * float(fs)
    if not 0.0 < lowcut < highcut < nyq:
        raise ValueError(
            f"Corte inválido: necesita 0 < {lowcut:g} < {highcut:g} < fs/2 = {nyq:g} Hz.")
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, np.asarray(data, dtype=np.float64))


def lowpass_filter(data, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """Filtro pasa-bajas Butterworth cero-fase (SOS + sosfiltfilt)."""
    nyq = 0.5 * float(fs)
    if not 0.0 < cutoff < nyq:
        raise ValueError(f"Corte inválido: necesita 0 < {cutoff:g} < fs/2 = {nyq:g} Hz.")
    sos = butter(order, cutoff / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, np.asarray(data, dtype=np.float64))


def demodulate_am(received_signal, sample_rate: float, carrier_freq: float = 17000.0,
                  cutoff_freq: float = 4000.0,
                  bandwidth: float = DEMOD_BANDWIDTH) -> tuple[np.ndarray, float]:
    """Demodula una señal AM grabada del aire y devuelve el audio recuperado.

    Pasos: filtro pasa-bandas alrededor de la portadora, rectificación y
    filtro pasa-bajas para obtener la envolvente; se elimina el offset DC y
    se normaliza.
    """
    x = np.asarray(received_signal, dtype=np.float64).ravel()
    if x.size == 0:
        return x, float(sample_rate)
    filtered = bandpass_filter(x, carrier_freq - bandwidth, carrier_freq + bandwidth,
                               float(sample_rate))
    rectified = np.abs(filtered)
    envelop = lowpass_filter(rectified, cutoff_freq, float(sample_rate))
    return _finish_demod(envelop, sample_rate)


def demodulate_am_hilbert(received_signal, sample_rate: float,
                          carrier_freq: float = 17000.0,
                          cutoff_freq: float = 4000.0,
                          bandwidth: float = DEMOD_BANDWIDTH) -> tuple[np.ndarray, float]:
    """Variante con envolvente por la transformada de Hilbert."""
    x = np.asarray(received_signal, dtype=np.float64).ravel()
    if x.size == 0:
        return x, float(sample_rate)
    filtered = bandpass_filter(x, carrier_freq - bandwidth, carrier_freq + bandwidth,
                               float(sample_rate))
    analytic = hilbert(filtered)
    envelop = np.abs(analytic)
    return _finish_demod(envelop, sample_rate)
