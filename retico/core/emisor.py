"""Lógica de emisión por modulación AM.

Se apoya en el código de apoyo ``Emisor.py``: graba/señal base, genera la
portadora de alta frecuencia y modula en AM para transmitir por el altavoz.
"""
from __future__ import annotations

import numpy as np


def modulate_am(audio_signal, sample_rate: float, carrier_freq: float = 17000.0,
                index_mod: float = 0.7) -> np.ndarray:
    """Modula una señal de banda base en AM (DSB-FC).

    ``s(t) = (1 + m * x(t)) * cos(2*pi*fc*t)`` y se normaliza al final para
    evitar distorsión en el altavoz.
    """
    x = np.asarray(audio_signal, dtype=np.float64).ravel()
    if x.size == 0:
        return x
    t = np.arange(x.size, dtype=np.float64) / float(sample_rate)
    carrier = np.cos(2.0 * np.pi * float(carrier_freq) * t)
    am_signal = (1.0 + float(index_mod) * x) * carrier
    peak = float(np.max(np.abs(am_signal))) if am_signal.size else 0.0
    if peak > 0.0:
        am_signal = am_signal / peak
    return am_signal.astype(np.float64)


def prepare_baseband(signal, fs_in: float, fs_out: float = 44100.0) -> tuple[np.ndarray, float]:
    """Normaliza y remuestrea una señal de banda base a ``fs_out``."""
    x = np.asarray(signal, dtype=np.float64).ravel()
    if x.size == 0:
        return x, float(fs_in)
    if float(fs_in) != float(fs_out):
        try:
            from scipy.signal import resample

            n_out = max(2, int(round(x.size * float(fs_out) / float(fs_in))))
            x = resample(x, n_out)
        except Exception:  # noqa: BLE001
            pass
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0.0:
        x = x / peak
    return np.asarray(x, dtype=np.float64), float(fs_out)