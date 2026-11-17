"""PCM (Pulse Code Modulation): palabras binarias, trama serie y línea.

La digitalización de la pestaña 2 obtiene las muestras x[n] y sus códigos
PCM.  Este módulo añade lo necesario para estudiar la codificación PCM:

- flujo serie de bits (MSB primero),
- agrupación de bits en palabras de ``N`` bits,
- códigos de línea NRZ unipolar, NRZ bipolar, RZ unipolar y Manchester,
- métricas de la trama (bits por muestra, tasa de bits, duración de palabra).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .modulation import bits_from_codes

LINE_CODES = ("NRZ unipolar", "NRZ bipolar", "RZ unipolar", "Manchester")


class PCMError(ValueError):
    """Parámetros PCM inválidos."""


@dataclass(frozen=True)
class PCMMetrics:
    """Resumen de la trama PCM."""

    fs: float
    bits: int
    bitrate: float

    @property
    def levels(self) -> int:
        return 2 ** int(self.bits)

    def __post_init__(self) -> None:
        if float(self.fs) <= 0.0:
            raise PCMError("La frecuencia de muestreo debe ser > 0 Hz.")
        if not 1 <= int(self.bits) <= 24:
            raise PCMError("Los bits por muestra deben estar entre 1 y 24.")

    @property
    def bit_period(self) -> float:
        """Duración de un bit, en segundos."""
        return 1.0 / self.bitrate

    @property
    def word_period(self) -> float:
        """Duración de una palabra PCM completa, en segundos."""
        return float(self.bits) / self.bitrate

    @property
    def bytes_per_second(self) -> float:
        return self.bitrate / 8.0


def pcm_bit_stream(codes_u: np.ndarray, bits: int,
                   max_samples: int | None = None) -> np.ndarray:
    """Códigos PCM → flujo de bits MSB primero."""
    codes = np.asarray(codes_u, dtype=np.int64).ravel()
    if max_samples is not None:
        codes = codes[:max(0, int(max_samples))]
    return bits_from_codes(codes, int(bits))


def codes_from_bit_stream(bits: np.ndarray, bits_per_word: int,
                          max_words: int | None = None) -> np.ndarray:
    """Flujo de bits → códigos PCM (descarta el resto incompleto)."""
    stream = np.asarray(bits, dtype=np.uint8).ravel()
    nbits = int(bits_per_word)
    if not 1 <= nbits <= 24:
        raise PCMError("Los bits por muestra deben estar entre 1 y 24.")
    n = stream.size // nbits
    if max_words is not None:
        n = min(n, max(0, int(max_words)))
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    groups = stream[:n * nbits].reshape(n, nbits)
    weights = (1 << np.arange(nbits - 1, -1, -1)).astype(np.int64)
    return (groups.astype(np.int64) * weights).sum(axis=1)


def pcm_metrics(fs: float, bits: int) -> PCMMetrics:
    """Métricas de una trama PCM sin compresión."""
    b = int(bits)
    return PCMMetrics(fs=float(fs), bits=b,
                      bitrate=float(fs) * float(b))


def word_bits(code_u: int, bits: int) -> np.ndarray:
    """Palabra PCM de una muestra como vector de bits (MSB primero)."""
    b = int(bits)
    if not 1 <= b <= 24:
        raise PCMError("Los bits por muestra deben estar entre 1 y 24.")
    value = int(code_u)
    if value < 0:
        value = 0
    value &= (1 << b) - 1
    return np.array([(value >> shift) & 1 for shift in range(b - 1, -1, -1)],
                    dtype=np.uint8)


def line_waveform(bits, samples_per_bit: int = 24,
                  mode: str = "NRZ bipolar") -> tuple[np.ndarray, np.ndarray]:
    """Forma de onda de la trama serie para un código de línea.

    Devuelve ``(t, y)`` con ``t`` en unidades de bit y ``y`` normalizada.
    """
    stream = np.asarray(bits, dtype=np.uint8).ravel()
    if mode not in LINE_CODES:
        raise PCMError(f"Código de línea no soportado: {mode}")
    if stream.size == 0:
        return np.zeros(0), np.zeros(0)
    spb = max(2, int(samples_per_bit))
    half = spb // 2
    y = np.empty(stream.size * spb, dtype=np.float64)
    for i, bit in enumerate(stream.astype(np.int64)):
        start = i * spb
        if mode == "NRZ unipolar":
            y[start:start + spb] = float(bit)
        elif mode == "NRZ bipolar":
            y[start:start + spb] = 2.0 * float(bit) - 1.0
        elif mode == "RZ unipolar":
            y[start:start + half] = float(bit)
            y[start + half:start + spb] = 0.0
        else:  # Manchester
            first = 1.0 if bit else -1.0
            y[start:start + half] = first
            y[start + half:start + spb] = -first
    t = np.arange(y.size, dtype=np.float64) / spb
    return t, y


def frame_words(codes_u: np.ndarray, bits: int,
                count: int = 8) -> list[tuple[int, str]]:
    """Primeras ``count`` palabras PCM como ``(código, binario)``."""
    nbits = int(bits)
    codes = np.asarray(codes_u, dtype=np.int64).ravel()[:max(0, int(count))]
    return [(int(code), format(int(code), f"0{nbits}b")) for code in codes]
