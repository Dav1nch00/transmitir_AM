"""Constelaciones teóricas de PCM/PAM y de las técnicas digitales.

La constelación representa cada símbolo como un punto en el plano I/Q
(ASK, PSK, QAM y PCM/PAM) o como un tono en un eje de frecuencia (FSK,
MFSK y GFSK).  El mapeo reproduce exactamente el que usa
:func:`core.modulation.modulate_digital`, de modo que las figuras sirven
para interpretar las señales que genera la pestaña de transmisión:

- **ASK**: amplitud unipolar ``0 … 1`` sobre el eje I.
- **PSK**: fase ``2π·k/M`` sobre la circunferencia unitaria.
- **QAM**: rejilla cuadrada con niveles centrados y normalizados.
- **PCM (PAM)**: niveles PAM centrados ``−(M−1)/2 … +(M−1)/2``.
- **FSK/MFSK/GFSK**: ``M`` tonos equiespaciados en ``−1 … +1``.

Las etiquetas de cada símbolo son los bits que lo forman, con el MSB
primero, tal como los agrupa el modulador.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TECHNIQUES = ("PCM (PAM)", "ASK", "PSK", "QAM", "FSK", "MFSK", "GFSK")
IQ_TECHNIQUES = ("PCM (PAM)", "ASK", "PSK", "QAM")
FSK_TECHNIQUES = ("FSK", "MFSK", "GFSK")

#: Bits/símbolo admitidos por cada técnica (QAM exige M cuadrado perfecto).
ALLOWED_BITS = {
    "PCM (PAM)": (1, 2, 3, 4, 5, 6, 7, 8),
    "ASK": (1, 2, 3, 4, 5, 6),
    "PSK": (1, 2, 3, 4, 5, 6),
    "QAM": (2, 4, 6),
    "FSK": (1, 2, 3, 4, 5, 6),
    "MFSK": (1, 2, 3, 4, 5, 6),
    "GFSK": (1, 2, 3, 4, 5, 6),
}


class ConstellationError(ValueError):
    """Parámetros de constelación inválidos."""


def _canonical(technique: str) -> str:
    text = str(technique).strip().upper()
    aliases = {
        "PCM": "PCM (PAM)",
        "PAM": "PCM (PAM)",
        "PCM (PAM)": "PCM (PAM)",
    }
    text = aliases.get(text, text)
    if text not in TECHNIQUES:
        raise ConstellationError(f"Técnica no soportada: {technique}")
    return text


@dataclass(frozen=True)
class Constellation:
    """Puntos de una constelación y sus métricas principales."""

    technique: str
    bits_per_symbol: int
    points: np.ndarray
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in ("iq", "fsk"):
            raise ConstellationError(f"Tipo de constelación inválido: {self.kind}")
        if not 1 <= int(self.bits_per_symbol) <= 8:
            raise ConstellationError("Los bits por símbolo deben estar entre 1 y 8.")
        if self.points.size != (1 << int(self.bits_per_symbol)):
            raise ConstellationError(
                "El número de puntos debe ser 2^(bits por símbolo).")

    # -- datos -------------------------------------------------------------
    @property
    def is_iq(self) -> bool:
        return self.kind == "iq"

    @property
    def m(self) -> int:
        """Número de símbolos, M = 2^bits."""
        return 1 << int(self.bits_per_symbol)

    @property
    def labels(self) -> tuple[str, ...]:
        """Etiquetas binarias (MSB primero) en orden de símbolo."""
        width = int(self.bits_per_symbol)
        return tuple(format(i, f"0{width}b") for i in range(self.m))

    @property
    def i_values(self) -> np.ndarray:
        return np.real(self.points)

    @property
    def q_values(self) -> np.ndarray:
        return np.imag(self.points)

    @property
    def frequencies(self) -> np.ndarray:
        """Frecuencias normalizadas (solo FSK); en I/Q devuelve vacío."""
        if self.kind != "fsk":
            return np.zeros(0, dtype=np.float64)
        return np.real(self.points)

    # -- métricas ----------------------------------------------------------
    @property
    def mean_energy(self) -> float:
        """Energía media por símbolo ``E[|s|²]`` (frecuencia en FSK)."""
        mag = np.abs(self.points) ** 2
        return float(np.mean(mag)) if mag.size else 0.0

    @property
    def peak_power(self) -> float:
        mag = np.abs(self.points) ** 2
        return float(np.max(mag)) if mag.size else 0.0

    @property
    def papr_db(self) -> float:
        """Relación potencia de pico a potencia media, en dB."""
        mean = self.mean_energy
        if mean <= 0.0:
            return float("inf")
        return float(10.0 * np.log10(self.peak_power / mean))

    @property
    def min_distance(self) -> float:
        """Distancia euclídea mínima entre dos símbolos distintos."""
        if self.m < 2:
            return 0.0
        diff = np.abs(self.points[:, None] - self.points[None, :])
        np.fill_diagonal(diff, np.inf)
        return float(np.min(diff))

    @property
    def dmin_normalized(self) -> float:
        """Distancia mínima referida a ``sqrt(E_media)``."""
        energy = self.mean_energy
        if energy <= 0.0:
            return 0.0
        return self.min_distance / float(np.sqrt(energy))

    # -- tablas ------------------------------------------------------------
    def rows(self) -> list[tuple[str, ...]]:
        """Filas (símbolo, bits, I/f, Q, amplitud, fase) para la tabla."""
        out: list[tuple[str, ...]] = []
        for k, label in enumerate(self.labels):
            i = float(np.real(self.points[k]))
            q = float(np.imag(self.points[k]))
            amp = float(np.hypot(i, q))
            phase = float(np.degrees(np.arctan2(q, i)))
            if self.kind == "fsk":
                out.append((str(k), label, f"{i:+.4f}", "—", "—", "—"))
            else:
                out.append((str(k), label, f"{i:+.4f}", f"{q:+.4f}",
                            f"{amp:.4f}", f"{phase:+.1f}"))
        return out


@dataclass(frozen=True)
class PCMWord:
    """Una palabra PCM y su posición en la constelación PAM.

    ``level`` es el nivel normalizado del punto (de −1 a +1), mientras que
    ``signed_code`` y ``volts`` describen el mismo código en el ADC bipolar
    de la pestaña 2: ``q = k − M/2`` y ``v = q·2·Vref/M``.
    """

    word: str
    code: int
    bits: int
    level: float
    signed_code: int
    vref: float = 1.0

    @property
    def amplitude(self) -> float:
        return abs(self.level)

    @property
    def volts(self) -> float:
        """Tensión que asignaría el DAC al código, en voltios."""
        return float(self.signed_code) * 2.0 * float(self.vref) / (1 << self.bits)


def pcm_word(word: str | int, bits: int, vref: float = 1.0) -> PCMWord:
    """Relaciona una palabra binaria con su punto de la constelación PCM/PAM.

    Parameters
    ----------
    word:
        Texto binario de exactamente ``bits`` caracteres (MSB primero) o un
        entero, que se recorta a ``bits``.
    bits:
        Longitud de la palabra; ``M = 2^bits``.
    vref:
        Semirrango del ADC, para calcular la tensión equivalente.
    """
    b = int(bits)
    if not 1 <= b <= 8:
        raise ConstellationError("Los bits de la palabra deben estar entre 1 y 8.")
    if float(vref) <= 0.0:
        raise ConstellationError("Vref debe ser mayor que 0 V.")
    if isinstance(word, str):
        text = word.strip().replace(" ", "")
        if len(text) != b or any(ch not in "01" for ch in text):
            raise ConstellationError(
                f"La palabra debe tener exactamente {b} bits (solo 0 y 1).")
        code = int(text, 2)
    else:
        code = int(word) & ((1 << b) - 1)
        text = format(code, f"0{b}b")
    m = 1 << b
    half = (m - 1) / 2.0
    level = (code - half) / half
    return PCMWord(word=text, code=code, bits=b, level=level,
                   signed_code=code - m // 2, vref=float(vref))


def constellation(technique: str, bits_per_symbol: int) -> Constellation:
    """Construye la constelación teórica de una técnica digital.

    Parameters
    ----------
    technique:
        ``PCM (PAM)``, ``ASK``, ``PSK``, ``QAM``, ``FSK``, ``MFSK`` o ``GFSK``.
    bits_per_symbol:
        Bits que transporta cada símbolo; ``M = 2^bits``.
    """
    name = _canonical(technique)
    bps = int(bits_per_symbol)
    allowed = ALLOWED_BITS[name]
    if bps not in allowed:
        if name == "QAM":
            raise ConstellationError(
                "QAM necesita 2, 4 o 6 bits/símbolo (4, 16 o 64-QAM).")
        raise ConstellationError(
            f"{name} admite {', '.join(str(b) for b in allowed)} bits/símbolo.")
    m = 1 << bps

    if name == "ASK":
        levels = np.arange(m, dtype=np.float64) / max(1, m - 1)
        points = levels.astype(np.complex128)
        kind = "iq"
    elif name == "PSK":
        angle = 2.0 * np.pi * np.arange(m, dtype=np.float64) / float(m)
        points = np.exp(1j * angle)
        kind = "iq"
    elif name == "QAM":
        side = int(round(np.sqrt(m)))
        levels = np.arange(side, dtype=np.float64) - (side - 1) / 2.0
        levels = levels / np.max(np.abs(levels))
        i_levels = levels[np.arange(m) // side]
        q_levels = levels[np.arange(m) % side]
        points = i_levels + 1j * q_levels
        kind = "iq"
    elif name == "PCM (PAM)":
        levels = np.arange(m, dtype=np.float64) - (m - 1) / 2.0
        levels = levels / np.max(np.abs(levels))
        points = levels.astype(np.complex128)
        kind = "iq"
    else:  # FSK, MFSK y GFSK comparten los mismos tonos nominales.
        if m == 1:
            freq = np.zeros(1, dtype=np.float64)
        else:
            freq = np.linspace(-1.0, 1.0, m, dtype=np.float64)
        points = freq.astype(np.complex128)
        kind = "fsk"

    return Constellation(technique=name, bits_per_symbol=bps,
                         points=points, kind=kind)
