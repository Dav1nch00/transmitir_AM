"""Demodulación AM del receptor (lógica de codigo_apoyo/Receptor.py)."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data)


def lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, data)


def demodulate_am(received_signal, sample_rate: float, carrier_freq: float = 17000.0,
                  cutoff_freq: float = 4000.0, duration: float | None = None):
    """Demodula una señal AM grabada del aire y devuelve el audio recuperado.

    Pasos: filtro pasa-banda alrededor de la portadora, rectificación y
    filtro pasa-bajas para obtener la envolvente; se elimina el offset DC y
    se normaliza.
    """
    x = np.asarray(received_signal, dtype=np.float64).ravel()
    if x.size == 0:
        return x, float(sample_rate)
    filtered = bandpass_filter(x, carrier_freq - 1000.0, carrier_freq + 1000.0,
                               float(sample_rate))
    rectified = np.abs(filtered)
    envelop = lowpass_filter(rectified, cutoff_freq, float(sample_rate))
    demod = envelop - np.mean(envelop)
    peak = float(np.max(np.abs(demod))) if demod.size else 0.0
    if peak > 0.0:
        demod = demod / peak
    return np.asarray(demod, dtype=np.float64), float(sample_rate)


def demodulate_am_hilbert(received_signal, sample_rate: float,
                          carrier_freq: float = 17000.0, cutoff_freq: float = 4000.0):
    """Variante con envolvente de Hilbert (enviar_recibir.py)."""
    x = np.asarray(received_signal, dtype=np.float64).ravel()
    if x.size == 0:
        return x, float(sample_rate)
    filtered = bandpass_filter(x, carrier_freq - 1000.0, carrier_freq + 1000.0,
                               float(sample_rate))
    analytic = hilbert(filtered)
    envelop = np.abs(analytic)
    demod = envelop - np.mean(envelop)
    peak = float(np.max(np.abs(demod))) if demod.size else 0.0
    if peak > 0.0:
        demod = demod / peak
    return np.asarray(demod, dtype=np.float64), float(sample_rate)