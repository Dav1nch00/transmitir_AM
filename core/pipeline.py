"""Cadena completa: audio capturado → muestreo → cuantización."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio_io import Audio
from .dsp import HEADROOM, normalize, resample_to
from .quantization import AdcConfig, QuantResult, quantize


class ParamsError(ValueError):
    """Parámetros globales inválidos."""


@dataclass(frozen=True)
class Params:
    fs_target: int = 16000
    bits: int = 8
    vref: float = 1.0
    window: str = "Hann"
    antialias: bool = True
    normalize: bool = True

    def validate(self) -> "Params":
        if not 1000 <= int(self.fs_target) <= 192000:
            raise ParamsError("La fs objetivo debe estar entre 1000 y 192000 Hz.")
        if not 2 <= int(self.bits) <= 24:
            raise ParamsError("Los bits del ADC deben estar entre 2 y 24.")
        if float(self.vref) <= 0:
            raise ParamsError("Vref debe ser mayor que 0 V.")
        return self

    def key(self) -> tuple:
        return (int(self.fs_target), int(self.bits), float(self.vref),
                self.window, bool(self.antialias), bool(self.normalize))


@dataclass
class Digitized:
    audio: Audio
    params: Params
    x_orig: np.ndarray      # audio completo, normalizado a HEADROOM
    x_n: np.ndarray         # muestras en la fs objetivo
    t_n: np.ndarray         # tiempos de x_n [s]
    fs_eff: float
    quant: QuantResult
    gain: float             # ganancia aplicada al normalizar

    @property
    def duration(self) -> float:
        return len(self.x_n) / self.fs_eff

    @property
    def n_samples(self) -> int:
        return len(self.x_n)

    def window_original(self, seconds: float) -> tuple[np.ndarray, np.ndarray]:
        n = min(len(self.x_orig), max(2, int(round(seconds * self.audio.fs))))
        t = np.arange(n) / float(self.audio.fs)
        return t, self.x_orig[:n]

    def window_sampled(self, seconds: float) -> tuple[np.ndarray, np.ndarray]:
        n = min(len(self.x_n), max(2, int(round(seconds * self.fs_eff))))
        return self.t_n[:n], self.x_n[:n]


def digitize(audio: Audio, p: Params) -> Digitized:
    """Normaliza, muestrea a la fs objetivo y cuantiza."""
    p.validate()
    x = np.asarray(audio.x, dtype=np.float64)
    gain = 1.0
    if p.normalize:
        x_norm, gain = normalize(x, HEADROOM)
    else:
        x_norm = x.copy()

    x_n, fs_eff = resample_to(x_norm, audio.fs, p.fs_target, p.antialias)
    qr = quantize(x_n, AdcConfig(int(p.bits), float(p.vref)), fs=fs_eff)
    t_n = np.arange(len(x_n), dtype=np.float64) / fs_eff
    return Digitized(audio=audio, params=p, x_orig=x_norm, x_n=x_n, t_n=t_n,
                     fs_eff=float(fs_eff), quant=qr, gain=gain)


def fs_key(fs: int) -> str:
    return f"{fs / 1000:g} kHz"
