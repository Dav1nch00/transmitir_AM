"""Comparación entre el audio original y el audio recuperado.

La comparación nunca superpone las dos señales en una misma traza: entrega
segmentos alineados y las métricas para que la interfaz las dibuje en ejes
separados.  También construye una secuencia A → silencio → B para escuchar los
dos audios uno después del otro, sin mezclarlos.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import signal

from .dsp import resample_to

_EPS = 1e-15


@dataclass(frozen=True)
class ComparisonResult:
    """Resultado de comparar dos audios ya alineados y escalados."""

    fs: float
    reference: np.ndarray
    test: np.ndarray
    error: np.ndarray
    t: np.ndarray
    delay_samples: int
    delay_s: float
    gain: float
    corr: float
    snr_db: float
    rmse: float
    max_error: float
    rms_ref: float
    rms_test: float
    overlap_ref: float
    overlap_test: float
    resampled: bool
    fs_ref_original: float
    fs_test_original: float
    n_ref: int
    n_test: int

    @property
    def n(self) -> int:
        """Número de muestras comparadas."""
        return int(self.t.size)

    @property
    def duration(self) -> float:
        """Duración del tramo comparado en segundos."""
        return self.n / float(self.fs)

    @property
    def correlation_percent(self) -> float:
        return 100.0 * max(0.0, min(1.0, self.corr))

    @property
    def fidelity(self) -> str:
        """Clasificación cualitativa de la correlación."""
        if not math.isfinite(self.corr) or self.rms_ref <= _EPS or self.rms_test <= _EPS:
            return "sin correlación"
        if self.corr >= 0.99:
            return "muy alta"
        if self.corr >= 0.95:
            return "alta"
        if self.corr >= 0.80:
            return "aceptable"
        return "baja"


def _as_signal(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError(f"El audio {name} está vacío.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"El audio {name} contiene valores no finitos.")
    return arr


def compare_audio(reference, fs_ref: float, test, fs_test: float, *,
                  align: bool = True, match_gain: bool = True,
                  max_delay_s: float = 0.5) -> ComparisonResult:
    """Compara dos audios y devuelve sus trazas alineadas y las métricas.

    El audio de prueba se remuestrea a la frecuencia del audio de referencia.
    Con ``align=True`` se busca el desfase que maximiza la correlación dentro
    de ±``max_delay_s`` segundos.  Con ``match_gain=True`` se aplica a la señal
    de prueba la ganancia de mínimos cuadrados que mejor la ajusta a la de
    referencia (una ganancia negativa indica polaridad invertida).

    Las señales devueltas están recortadas al tramo común: no se superponen ni
    se rellenan con ceros, de modo que el error se calcula solo donde ambas
    tienen datos.
    """
    a = _as_signal(reference, "original")
    b = _as_signal(test, "recuperado")
    fs_ref = float(fs_ref)
    fs_test = float(fs_test)
    if not math.isfinite(fs_ref) or fs_ref <= 0:
        raise ValueError("La fs del audio original debe ser positiva.")
    if not math.isfinite(fs_test) or fs_test <= 0:
        raise ValueError("La fs del audio recuperado debe ser positiva.")
    if not math.isfinite(float(max_delay_s)) or float(max_delay_s) < 0:
        raise ValueError("La búsqueda de desfase debe ser ≥ 0 s.")

    n_ref, n_test = int(a.size), int(b.size)
    fs_a, fs_b = fs_ref, fs_test

    # Unificar la frecuencia de muestreo en la del audio original.
    resampled = abs(fs_b - fs_a) > 1e-9
    if resampled:
        b, _ = resample_to(b, fs_b, fs_a)

    fs = fs_a
    lag = 0
    if align and a.size > 1 and b.size > 1:
        a0 = a - float(np.mean(a))
        b0 = b - float(np.mean(b))
        sa = float(np.std(a0))
        sb = float(np.std(b0))
        if sa > _EPS and sb > _EPS:
            xc = signal.fftconvolve(a0 / sa, (b0 / sb)[::-1], mode="full")
            lags = np.arange(-b.size + 1, a.size)
            max_lag = min(int(round(float(max_delay_s) * fs)),
                          int(max(a.size, b.size)))
            mask = np.abs(lags) <= max_lag
            if np.any(mask):
                lag = int(lags[np.flatnonzero(mask)[
                    int(np.argmax(np.abs(xc[mask])))]])

    # Tramo común: test[0] corresponde a reference[lag].
    if lag >= 0:
        n = min(a.size - lag, b.size)
        seg_a = a[lag:lag + n]
        seg_b = b[:n]
    else:
        n = min(a.size, b.size + lag)
        seg_a = a[:n]
        seg_b = b[-lag:-lag + n]
    n = int(min(seg_a.size, seg_b.size))
    if n == 0:
        raise ValueError(
            "Las señales no se solapan con el desfase encontrado; "
            "amplíe la búsqueda o revise los audios.")
    seg_a = seg_a[:n]
    seg_b = seg_b[:n]

    # Ganancia de mínimos cuadrados (puede ser negativa si hay inversión).
    gain = 1.0
    if match_gain:
        denom = float(np.dot(seg_b, seg_b))
        if denom > _EPS:
            gain = float(np.dot(seg_a, seg_b) / denom)
            if not math.isfinite(gain):
                gain = 1.0
    seg_b_scaled = seg_b * gain

    error = seg_b_scaled - seg_a
    p_ref = float(np.mean(seg_a ** 2))
    p_err = float(np.mean(error ** 2))
    if p_err <= _EPS:
        snr = math.inf if p_ref > _EPS else math.inf
    elif p_ref <= _EPS:
        snr = -math.inf
    else:
        snr = 10.0 * math.log10(p_ref / p_err)

    sa_c = float(np.std(seg_a))
    sb_c = float(np.std(seg_b_scaled))
    if sa_c <= _EPS or sb_c <= _EPS:
        corr = 0.0
    else:
        corr = float(np.corrcoef(seg_a, seg_b_scaled)[0, 1])
        if not math.isfinite(corr):
            corr = 0.0

    rms_ref = float(np.sqrt(np.mean(seg_a ** 2)))
    rms_test = float(np.sqrt(np.mean(seg_b ** 2)))
    rmse = float(np.sqrt(p_err))
    max_error = float(np.max(np.abs(error))) if n else 0.0
    t = np.arange(n, dtype=np.float64) / fs

    return ComparisonResult(
        fs=fs,
        reference=seg_a,
        test=seg_b_scaled,
        error=error,
        t=t,
        delay_samples=int(-lag),
        delay_s=float(-lag) / fs,
        gain=float(gain),
        corr=float(corr),
        snr_db=float(snr),
        rmse=rmse,
        max_error=max_error,
        rms_ref=rms_ref,
        rms_test=rms_test,
        overlap_ref=float(n / max(1, n_ref)),
        overlap_test=float(n / max(1, int(b.size))),
        resampled=bool(resampled),
        fs_ref_original=float(fs_ref),
        fs_test_original=float(fs_test),
        n_ref=n_ref,
        n_test=n_test,
    )


def sequential_signal(a, fs_a: float, b, fs_b: float,
                      gap_s: float = 0.25) -> tuple[np.ndarray, int, float]:
    """Construye A → silencio → B, sin superponer los audios.

    Devuelve la señal concatenada, su frecuencia de muestreo y el instante en
    segundos donde empieza B.  Se usa para escuchar y exportar la comparación
    de forma secuencial.
    """
    xa = _as_signal(a, "original")
    xb = _as_signal(b, "recuperado")
    if float(fs_a) <= 0 or float(fs_b) <= 0:
        raise ValueError("Las frecuencias de muestreo deben ser positivas.")
    if float(gap_s) < 0:
        raise ValueError("El silencio entre audios no puede ser negativo.")
    fs = float(max(fs_a, fs_b))
    ya, _ = resample_to(xa, fs_a, fs)
    yb, _ = resample_to(xb, fs_b, fs)
    gap = np.zeros(max(0, int(round(float(gap_s) * fs))), dtype=np.float64)
    out = np.concatenate([ya, gap, yb])
    start_b = (ya.size + gap.size) / fs
    return out, int(round(fs)), float(start_b)
