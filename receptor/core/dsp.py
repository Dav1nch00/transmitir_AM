"""Muestreo, FFT, aliasing y reconstrucción.

Convenios:
    x[n] = x(n·Ts),  Ts = 1/fs
    |X(f)| = 2·|FFT{x[n]·w[n]}| / Σw[n]     (espectro unilateral de amplitud)
    Δf = fs / N
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import signal

HEADROOM = 0.95
WINDOWS = {
    "Hann": "hann",
    "Hamming": "hamming",
    "Blackman": "blackman",
    "Rectangular": "boxcar",
}


def normalize(x: np.ndarray, headroom: float = HEADROOM) -> tuple[np.ndarray, float]:
    y = np.asarray(x, dtype=np.float64)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak <= 0:
        return y.copy(), 1.0
    gain = headroom / peak
    return y * gain, gain


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0


def peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(x))) if len(x) else 0.0


def window_key(name: str) -> str:
    return WINDOWS.get(name, name)


def spectrum(x: np.ndarray, fs: float, window: str = "hann") -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 8:
        return np.zeros(0), np.zeros(0)
    w = signal.get_window(window_key(window), n, fftbins=True)
    xw = (x - np.mean(x)) * w
    mag = np.abs(np.fft.rfft(xw)) * (2.0 / np.sum(w))
    mag[0] *= 0.5
    if n % 2 == 0:
        mag[-1] *= 0.5
    f = np.fft.rfftfreq(n, 1.0 / float(fs))
    return f, mag


def db(mag: np.ndarray, ref: float = 1.0, floor_db: float = -140.0) -> np.ndarray:
    mag = np.asarray(mag, dtype=np.float64)
    with np.errstate(divide="ignore"):
        y = 20.0 * np.log10(np.maximum(mag, 1e-15) / max(ref, 1e-15))
    return np.maximum(y, floor_db)


def resolution(fs: float, n: int) -> float:
    return float(fs) / max(1, n)


def peak_frequency(f: np.ndarray, mag: np.ndarray, fmin: float = 20.0) -> float:
    m = f >= fmin
    if not np.any(m):
        return 0.0
    return float(f[m][np.argmax(mag[m])])


def cumulative_frequency(f: np.ndarray, power: np.ndarray, fraction: float = 0.99,
                         fmin: float = 20.0) -> float:
    m = f >= fmin
    ff, pp = f[m], power[m]
    if ff.size == 0 or pp.sum() <= 0:
        return 0.0
    c = np.cumsum(pp)
    idx = int(np.searchsorted(c, fraction * c[-1]))
    return float(ff[min(idx, len(ff) - 1)])


def bandwidth_99(f: np.ndarray, mag: np.ndarray, fraction: float = 0.99) -> float:
    return cumulative_frequency(f, mag ** 2, fraction)


def spectral_centroid(f: np.ndarray, mag: np.ndarray, fmin: float = 20.0,
                      fmax: float = 20000.0) -> float:
    m = (f >= fmin) & (f <= fmax)
    p = (mag[m] ** 2)
    total = p.sum()
    return float((f[m] * p).sum() / total) if total > 0 else 0.0


def energy_above(f: np.ndarray, mag: np.ndarray, f_cut: float, fmin: float = 1.0) -> float:
    m = f >= fmin
    p = mag[m] ** 2
    total = float(p.sum())
    if total <= 0:
        return 0.0
    return float(p[f[m] > f_cut].sum() / total)


@dataclass
class NyquistInfo:
    fs: float
    f_max: float
    fraction: float = 0.99

    @property
    def fs_min(self) -> float:
        return 2.0 * self.f_max

    @property
    def ok(self) -> bool:
        return self.fs >= self.fs_min

    def verdict(self) -> str:
        if self.ok:
            return f"CUMPLE: fs = {self.fs:.0f} Hz ≥ 2·f_max = {self.fs_min:.0f} Hz"
        return f"NO CUMPLE: fs = {self.fs:.0f} Hz < 2·f_max = {self.fs_min:.0f} Hz"


def nyquist_info(x: np.ndarray, fs: float, window: str = "hann",
                 fraction: float = 0.99) -> NyquistInfo:
    f, mag = spectrum(x, fs, window)
    return NyquistInfo(fs=float(fs), f_max=bandwidth_99(f, mag, fraction),
                       fraction=fraction)


def resample_to(x: np.ndarray, fs_in: float, fs_out: float,
                antialias: bool = True) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64)
    fs_in = float(fs_in)
    fs_out = float(fs_out)
    if fs_in <= 0 or fs_out <= 0:
        raise ValueError("Las frecuencias de muestreo deben ser positivas.")
    if abs(fs_in - fs_out) < 1e-9:
        return x.copy(), fs_out

    g = math.gcd(int(round(fs_in)), int(round(fs_out)))
    up = int(round(fs_out)) // g
    down = int(round(fs_in)) // g

    if antialias:
        y = signal.resample_poly(x, up, down, window=("kaiser", 5.0))
        return y.astype(np.float64), fs_out

    n_out = max(1, int(round(len(x) * fs_out / fs_in)))
    idx = np.round(np.arange(n_out) * fs_in / fs_out).astype(np.int64)
    idx = np.clip(idx, 0, len(x) - 1)
    return x[idx].astype(np.float64), fs_out


def alias_frequency(f: float, fs: float) -> float:
    if fs <= 0:
        return float(f)
    k = round(f / fs)
    return float(abs(f - k * fs))


def zero_order_hold(x: np.ndarray, fs: float, t_out: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return np.zeros_like(t_out)
    idx = np.floor(np.asarray(t_out) * fs).astype(np.int64)
    return x[np.clip(idx, 0, len(x) - 1)]


def sinc_interp(x: np.ndarray, fs: float, t_out: np.ndarray,
                half_width: int = 32) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    t_out = np.asarray(t_out, dtype=np.float64)
    n = len(x)
    if n == 0:
        return np.zeros_like(t_out)

    y = np.zeros_like(t_out)
    taps = 2 * half_width + 1
    offsets = np.arange(taps)
    chunk = 2048
    for start in range(0, len(t_out), chunk):
        tt = t_out[start:start + chunk]
        center = tt * fs
        k0 = np.floor(center).astype(np.int64) - half_width
        ks = k0[:, None] + offsets[None, :]
        valid = (ks >= 0) & (ks < n)
        arg = center[:, None] - ks
        vals = x[np.clip(ks, 0, n - 1)] * np.sinc(arg)
        y[start:start + len(tt)] = np.where(valid, vals, 0.0).sum(axis=1)
    return y


def pick_window_start(x: np.ndarray, fs: float, win_s: float) -> int:
    n = int(win_s * fs)
    if n <= 0 or len(x) <= n:
        return 0
    c = np.concatenate(([0.0], np.cumsum(np.asarray(x, dtype=np.float64) ** 2)))
    hop = max(1, n // 2)
    starts = np.arange(0, len(x) - n + 1, hop)
    return int(starts[np.argmax(c[starts + n] - c[starts])])


def spectrogram(x: np.ndarray, fs: float, window: str = "hann",
                nperseg: int = 1024, noverlap: int = 768) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    nperseg = min(max(64, int(nperseg)), len(x))
    noverlap = min(noverlap, nperseg - 1)
    f, t, sxx = signal.spectrogram(x, fs=fs, window=window_key(window),
                                   nperseg=nperseg, noverlap=noverlap,
                                   mode="magnitude", scaling="spectrum")
    sxx_db = 20.0 * np.log10(np.maximum(sxx, 1e-12))
    return f, t, sxx_db


def display_slice(t: np.ndarray, x: np.ndarray, max_points: int = 6000):
    n = len(x)
    if n <= max_points:
        return t, x
    step = int(math.ceil(n / max_points))
    return t[::step], x[::step]