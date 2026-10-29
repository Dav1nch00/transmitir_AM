"""Cuantización uniforme, ADC bipolar y DAC ideal."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class QuantizationError(Exception):
    pass


@dataclass(frozen=True)
class AdcConfig:
    bits: int = 8
    vref: float = 1.0

    def __post_init__(self) -> None:
        if not 2 <= int(self.bits) <= 24:
            raise QuantizationError("Los bits del ADC deben estar entre 2 y 24.")
        if float(self.vref) <= 0:
            raise QuantizationError("Vref debe ser mayor que 0 V.")

    @property
    def levels(self) -> int:
        return 2 ** int(self.bits)

    @property
    def lsb(self) -> float:
        return 2.0 * float(self.vref) / self.levels

    @property
    def scale_adc(self) -> float:
        return 1.0 / self.lsb

    @property
    def scale_dac(self) -> float:
        return self.lsb

    @property
    def vpp(self) -> float:
        return 2.0 * float(self.vref)

    @property
    def dynamic_range_db(self) -> float:
        return 6.02 * int(self.bits)

    def __str__(self) -> str:
        return f"{self.bits} bits, ±{self.vref:g} V"


@dataclass
class QuantResult:
    cfg: AdcConfig
    fs: float
    volts: np.ndarray
    codes: np.ndarray
    codes_u: np.ndarray
    volts_q: np.ndarray
    error: np.ndarray
    sqnr_db: float
    sqnr_expected_db: float
    sqnr_sine_db: float
    clipped: int

    @property
    def error_max(self) -> float:
        return float(np.max(np.abs(self.error))) if len(self.error) else 0.0

    @property
    def error_rms(self) -> float:
        return float(np.sqrt(np.mean(self.error ** 2))) if len(self.error) else 0.0

    @property
    def bitrate(self) -> float:
        return float(self.cfg.bits) * float(self.fs)

    @property
    def enob(self) -> float:
        if not np.isfinite(self.sqnr_db):
            return float(self.cfg.bits)
        return max(0.0, (self.sqnr_db - 1.76) / 6.02)


def _sqnr(p_sig: float, p_err: float) -> float:
    if p_sig <= 0:
        return float("inf")
    if p_err <= 0:
        return float("inf")
    return float(10.0 * np.log10(p_sig / p_err))


def quantize(x_norm: np.ndarray, cfg: AdcConfig, fs: float = 1.0) -> QuantResult:
    x = np.asarray(x_norm, dtype=np.float64)
    v = x * float(cfg.vref)
    half = cfg.levels // 2
    raw = np.round(v / cfg.lsb)
    q = np.clip(raw, -half, half - 1).astype(np.int64)
    vq = q.astype(np.float64) * cfg.lsb
    err = vq - v
    p_sig = float(np.mean(v ** 2)) if len(v) else 0.0
    return QuantResult(
        cfg=cfg,
        fs=float(fs),
        volts=v,
        codes=q,
        codes_u=q + half,
        volts_q=vq,
        error=err,
        sqnr_db=_sqnr(p_sig, float(np.mean(err ** 2)) if len(err) else 0.0),
        sqnr_expected_db=_sqnr(p_sig, cfg.lsb ** 2 / 12.0),
        sqnr_sine_db=6.02 * cfg.bits + 1.76,
        clipped=int(np.count_nonzero(raw != q)),
    )


def dac_output(codes_u: np.ndarray, cfg: AdcConfig) -> np.ndarray:
    return (np.asarray(codes_u, dtype=np.int64) - cfg.levels // 2) * cfg.lsb


def to_binary(code_u: int, bits: int) -> str:
    return format(int(code_u), f"0{int(bits)}b")


def level_voltages(cfg: AdcConfig) -> np.ndarray:
    half = cfg.levels // 2
    return (np.arange(cfg.levels, dtype=np.float64) - half) * cfg.lsb


def level_table(cfg: AdcConfig, codes_u: np.ndarray | None = None,
                max_rows: int = 64) -> tuple[list[tuple[str, ...]], str]:
    if codes_u is None:
        idx = np.arange(cfg.levels)
        note = f"Se muestran los {cfg.levels} niveles (2^{cfg.bits})."
    else:
        idx = np.unique(np.asarray(codes_u, dtype=np.int64))
        note = (f"Se muestran los {len(idx)} códigos usados por la señal "
                f"de {cfg.levels} posibles (2^{cfg.bits}).")
    if len(idx) > max_rows:
        half = max_rows // 2
        idx = np.concatenate((idx[:half], idx[-half:]))
        note += f" (se recortan los intermedios: {max_rows} filas)."
    volts = (idx.astype(np.float64) - cfg.levels // 2) * cfg.lsb
    rows = [(str(int(i)), f"{v:+.6f}", to_binary(int(i), cfg.bits))
            for i, v in zip(idx, volts)]
    return rows, note


def pcm_stream(codes_u: np.ndarray, cfg: AdcConfig, count: int = 16) -> str:
    n = min(int(count), len(codes_u))
    return " ".join(to_binary(int(c), cfg.bits) for c in codes_u[:n])


def sample_table(t_n: np.ndarray, x_n: np.ndarray, qr: QuantResult,
                 start: int = 0, rows: int = 40) -> list[tuple[str, ...]]:
    end = min(len(x_n), int(start) + int(rows))
    out = []
    for i in range(max(0, int(start)), end):
        out.append((
            str(i),
            f"{t_n[i] * 1e3:.3f}",
            f"{x_n[i]:+.6f}",
            f"{qr.volts[i]:+.6f}",
            f"{int(qr.codes[i]):+d}",
            str(int(qr.codes_u[i])),
            to_binary(int(qr.codes_u[i]), qr.cfg.bits),
            f"{qr.volts_q[i]:+.6f}",
            f"{qr.error[i] * 1e3:+.4f}",
        ))
    return out


def bits_comparison(x_norm: np.ndarray, vref: float, bits_list: list[int]) -> list[dict]:
    out = []
    for b in bits_list:
        qr = quantize(x_norm, AdcConfig(b, vref))
        out.append({
            "bits": b,
            "levels": qr.cfg.levels,
            "lsb": qr.cfg.lsb,
            "sqnr_db": qr.sqnr_db,
            "sqnr_theo_db": qr.sqnr_sine_db,
            "enob": qr.enob,
        })
    return out