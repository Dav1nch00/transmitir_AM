"""Modulación/demodulación FM y digital (ASK, FSK, PSK, QAM, GFSK, MFSK).

La modulación AM se mantiene en :mod:`core.am` porque ya se usa en la cadena
de transmisión/recepción acústica. Este módulo añade el resto de técnicas, las
utilidades para convertir las muestras PCM en un flujo de bits y los
demoduladores correspondientes (FM y las seis técnicas digitales).

Convenios
---------
- Todas las señales de salida son mono, float64 y se normalizan para que el
  pico no supere 1.0 (lo que necesita el altavoz y el WAV de 16 bits).
- Las técnicas digitales reciben un vector de bits (0/1). Cada grupo de
  ``bits_per_symbol`` bits forma un símbolo.
- La fase de FSK/MFSK/GFSK se obtiene integrando la frecuencia instantánea,
  así que no hay saltos de fase en los cambios de símbolo.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert

from .am import _finish_demod, bandpass_filter, lowpass_filter

ANALOG_TECHNIQUES = ("AM", "FM")
DIGITAL_TECHNIQUES = ("ASK", "FSK", "PSK", "QAM", "GFSK", "MFSK")


class ModulationError(ValueError):
    """Parámetros de modulación inválidos."""


@dataclass
class ModulationResult:
    signal: np.ndarray
    fs: int
    technique: str
    base: np.ndarray
    base_label: str
    info: str
    duration: float
    bits_used: int = 0


def _normalize_peak(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0.0:
        x = x / peak
    return x


def bits_from_codes(codes_u: np.ndarray, bits: int,
                    max_bits: int | None = None) -> np.ndarray:
    """Convierte códigos PCM en un flujo de bits MSB primero.

    ``codes_u`` son los códigos del ADC en offset binario (0 … 2^bits−1).
    Si ``max_bits`` se indica, solo se devuelven los primeros ``max_bits``
    bits; así se puede limitar la duración de la señal modulada sin construir
    todo el flujo de un audio largo.
    """
    codes = np.asarray(codes_u, dtype=np.uint32).ravel()
    nbits = int(bits)
    if not 1 <= nbits <= 24:
        raise ModulationError("El número de bits del PCM debe estar entre 1 y 24.")
    if codes.size == 0:
        return np.zeros(0, dtype=np.uint8)

    limit = None if max_bits is None else max(0, int(max_bits))
    if limit is not None:
        if limit == 0:
            return np.zeros(0, dtype=np.uint8)
        n_codes = (limit + nbits - 1) // nbits
        codes = codes[:n_codes]

    shifts = np.arange(nbits - 1, -1, -1, dtype=np.uint32)
    out = ((codes[:, None] >> shifts[None, :]) & 1).astype(np.uint8).reshape(-1)
    return out if limit is None else out[:limit]


def _pad_bits(bits: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    bps = int(bits_per_symbol)
    if bps < 1:
        raise ModulationError("Los bits por símbolo deben ser ≥ 1.")
    pad = (-len(bits)) % bps
    if pad:
        bits = np.concatenate((bits, np.zeros(pad, dtype=np.uint8)))
    return bits


def _symbols_from_bits(bits: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    bps = int(bits_per_symbol)
    groups = bits.reshape(-1, bps)
    weights = (1 << np.arange(bps - 1, -1, -1)).astype(np.int64)
    return (groups.astype(np.int64) * weights).sum(axis=1)


def _tech_name(technique: str, m: int, bt: float) -> str:
    if technique == "ASK":
        return f"{m}-ASK"
    if technique == "FSK":
        return f"{m}-FSK"
    if technique == "PSK":
        return f"{m}-PSK"
    if technique == "QAM":
        return f"{m}-QAM"
    if technique == "GFSK":
        return f"GFSK (BT = {bt:g})"
    if technique == "MFSK":
        return f"{m}-FSK (MFSK)"
    return technique


def modulate_fm(baseband, fs: float, carrier_freq: float,
                deviation: float) -> np.ndarray:
    """FM de banda ancha: f_i(t) = fc + Δf·x(t), con x normalizada a ±1."""
    x = np.asarray(baseband, dtype=np.float64).ravel()
    if x.size == 0:
        return x
    # Quitar el DC evita que la portadora se desplace de forma permanente.
    x = _normalize_peak(x - float(np.mean(x)))
    fs = float(fs)
    fc = float(carrier_freq)
    dev = float(deviation)
    if not 0.0 < fc < fs / 2.0:
        raise ModulationError(f"La portadora debe estar entre 0 y {fs / 2:.0f} Hz.")
    if dev <= 0.0:
        raise ModulationError("La desviación de FM debe ser > 0 Hz.")
    if fc - dev <= 0.0 or fc + dev >= fs / 2.0:
        raise ModulationError(
            f"Con fc = {fc:.0f} Hz y Δf = {dev:.0f} Hz, las frecuencias "
            f"instantáneas se salen del rango 0 … {fs / 2:.0f} Hz.")
    t = np.arange(x.size, dtype=np.float64) / fs
    phase = (2.0 * np.pi * fc * t
             + 2.0 * np.pi * dev * np.cumsum(x) / fs)
    return _normalize_peak(np.cos(phase))


def modulate_digital(technique: str, bits, fs: float, carrier_freq: float,
                     symbol_rate: float, bits_per_symbol: int,
                     deviation: float = 3000.0, bt: float = 0.5) -> ModulationResult:
    """Modula un flujo de bits con ASK, FSK, PSK, QAM, GFSK o MFSK.

    ``bits_per_symbol`` define M = 2^bps. Para QAM se admiten 2, 4 y 6
    bits/símbolo (4-QAM, 16-QAM y 64-QAM). El mapeo es binario (sin Gray),
    suficiente para la demostración y para la demodulación posterior.
    """
    technique = str(technique).upper()
    if technique not in DIGITAL_TECHNIQUES:
        raise ModulationError(f"Técnica digital no soportada: {technique}")
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    if bits.size == 0:
        raise ModulationError("No hay bits para modular.")
    fs = float(fs)
    fc = float(carrier_freq)
    baud = float(symbol_rate)
    bps = int(bits_per_symbol)
    if not 0.0 < fc < fs / 2.0:
        raise ModulationError(f"La portadora debe estar entre 0 y {fs / 2:.0f} Hz.")
    if baud <= 0.0:
        raise ModulationError("La tasa de símbolos debe ser > 0 baudios.")
    sps = fs / baud
    if sps < 4.0:
        raise ModulationError(
            f"La tasa de símbolos es demasiado alta: {baud:.0f} baudios deja "
            f"{sps:.1f} muestras por símbolo. Use como máximo {fs / 4:.0f} baudios.")
    if bps < 1:
        raise ModulationError("Los bits por símbolo deben ser ≥ 1.")

    m = 1 << bps
    if technique == "QAM":
        side = int(round(np.sqrt(m)))
        if side * side != m:
            raise ModulationError("QAM necesita 2, 4 o 6 bits/símbolo (4, 16 o 64-QAM).")
    if technique in ("FSK", "GFSK", "MFSK"):
        dev = float(deviation)
        if dev <= 0.0:
            raise ModulationError("La desviación de frecuencia debe ser > 0 Hz.")
        if fc - dev <= 0.0 or fc + dev >= fs / 2.0:
            raise ModulationError(
                f"Con fc = {fc:.0f} Hz y Δf = {dev:.0f} Hz, las frecuencias "
                f"se salen del rango 0 … {fs / 2:.0f} Hz.")
    else:
        dev = float(deviation)
    if technique == "GFSK" and float(bt) <= 0.0:
        raise ModulationError("El producto BT de GFSK debe ser > 0.")

    padded = _pad_bits(bits, bps)
    symbols = _symbols_from_bits(padded, bps)
    n_symbols = len(symbols)
    sps = fs / baud
    n = max(1, int(round(n_symbols * sps)))
    t = np.arange(n, dtype=np.float64) / fs
    idx = np.floor(np.arange(n, dtype=np.float64) / sps).astype(np.int64)
    idx = np.clip(idx, 0, n_symbols - 1)
    bit_wave = np.repeat(padded, max(1, int(round(sps)))).astype(np.float64)
    if bit_wave.size < n:
        bit_wave = np.pad(bit_wave, (0, n - bit_wave.size),
                          mode="edge")
    bit_wave = bit_wave[:n]

    if technique == "ASK":
        levels = (np.arange(m, dtype=np.float64) / max(1, m - 1))
        amp = levels[symbols][idx]
        signal = amp * np.cos(2.0 * np.pi * fc * t)
    elif technique == "PSK":
        phase = 2.0 * np.pi * symbols / float(m)
        signal = np.cos(2.0 * np.pi * fc * t + phase[idx])
    elif technique == "QAM":
        levels = np.arange(side, dtype=np.float64) - (side - 1) / 2.0
        levels = levels / np.max(np.abs(levels))
        i_levels = levels[symbols // side]
        q_levels = levels[symbols % side]
        signal = (i_levels[idx] * np.cos(2.0 * np.pi * fc * t)
                  - q_levels[idx] * np.sin(2.0 * np.pi * fc * t))
    else:
        spacing = 2.0 * dev / max(1, m - 1)
        f_dev = -dev + symbols.astype(np.float64) * spacing
        f_dev_samples = f_dev[idx]
        if technique == "GFSK":
            bandwidth = float(bt) * baud
            sigma_t = np.sqrt(np.log(2.0)) / (2.0 * np.pi * bandwidth)
            sigma = max(1.0, sigma_t * fs)
            f_inst = fc + gaussian_filter1d(f_dev_samples, sigma=sigma,
                                            mode="nearest")
        else:
            f_inst = fc + f_dev_samples
        phase = 2.0 * np.pi * np.cumsum(f_inst) / fs
        signal = np.cos(phase)

    signal = _normalize_peak(signal)
    info = (f"{_tech_name(technique, m, bt)} · fc = {fc:.0f} Hz · "
            f"baudios = {baud:.0f} · {bps} bit(s)/símbolo · "
            f"{bits.size} bits · {n_symbols} símbolos · "
            f"duración = {n / fs:.2f} s")
    return ModulationResult(
        signal=signal,
        fs=int(round(fs)),
        technique=technique,
        base=bit_wave,
        base_label="bits modulantes (NRZ)",
        info=info,
        duration=n / fs,
        bits_used=int(bits.size),
    )


# ==========================================================================
# Demodulación
# ==========================================================================
@dataclass
class DemodResult:
    """Resultado de una demodulación digital."""

    signal: np.ndarray
    fs: int
    technique: str
    info: str
    bits: np.ndarray | None = None
    symbols: np.ndarray | None = None
    error: float = 0.0
    timing_offset: float = 0.0
    n_symbols: int = 0


def _safe_bandpass(data: np.ndarray, low: float, high: float,
                   fs: float) -> np.ndarray:
    x = np.asarray(data, dtype=np.float64)
    if x.size < 128 or low >= high:
        return x
    try:
        return bandpass_filter(x, low, high, fs)
    except ValueError:
        return x


def _safe_lowpass(data: np.ndarray, cutoff: float, fs: float) -> np.ndarray:
    x = np.asarray(data, dtype=np.float64)
    if x.size < 128 or not 0.0 < cutoff < fs / 2.0:
        return x
    try:
        return lowpass_filter(x, cutoff, fs)
    except ValueError:
        return x


def _find_burst(x: np.ndarray, sps: float) -> tuple[int, int]:
    """Encuentra el intervalo activo de una captura del micrófono.

    Se usa un suavizado de un símbolo y un umbral relativo al ruido de fondo.
    En archivos generados por la propia aplicación no hace falta: la señal
    empieza en la muestra 0.
    """
    n = len(x)
    if n < 8:
        return 0, n
    win = max(1, int(round(sps)))
    kernel = np.ones(win, dtype=np.float64) / win
    env = np.convolve(np.abs(x), kernel, mode="same")
    noise = float(np.percentile(env, 1.0))
    peak = float(np.percentile(env, 99.5))
    if peak <= 1e-12 or noise > 0.25 * peak:
        # No hay un tramo claramente más silencioso: la ráfaga ocupa toda la
        # captura o la señal empieza en la muestra 0.
        return 0, n
    threshold = max(noise * 5.0, peak * 0.10)
    idx = np.flatnonzero(env > threshold)
    if idx.size < 2:
        return 0, n
    margin = max(1, int(round(sps)))
    start = max(0, int(idx[0]) - margin)
    end = min(n, int(idx[-1]) + margin + 1)
    if end - start < max(8, int(round(2.0 * sps))):
        return 0, n
    return start, end


def _closest_to_zero(candidates: np.ndarray, errors: np.ndarray,
                     tol: float = 1.01) -> float:
    """Elige el candidato de menor error y, en empate, el más cercano a 0."""
    if candidates.size == 0:
        return 0.0
    best = float(np.min(errors))
    selected = candidates[errors <= best * tol + 1e-12]
    if selected.size == 0:
        return 0.0
    # Para fases, la distancia debe medirse de forma circular.
    wrapped = np.abs((selected + np.pi) % (2.0 * np.pi) - np.pi)
    return float(selected[int(np.argmin(wrapped))])


def _best_psk_phase_offset(phi: np.ndarray, m: int,
                           steps: int = 36) -> float:
    """Corrige un desfase constante de la constelación PSK."""
    offsets = np.linspace(0.0, 2.0 * np.pi, steps, endpoint=False)
    errors = []
    for off in offsets:
        residual = np.angle(np.exp(1j * (phi - off) * m)) / float(m)
        errors.append(float(np.mean(np.abs(residual))))
    return _closest_to_zero(offsets, np.asarray(errors))


def _best_qam_phase_offset(i_sig: np.ndarray, q_sig: np.ndarray, side: int,
                           steps: int = 24) -> float:
    """Corrige un desfase constante de la constelación QAM."""
    offsets = np.linspace(0.0, 2.0 * np.pi, steps, endpoint=False)
    levels = np.arange(side, dtype=np.float64) - (side - 1) / 2.0
    levels = levels / np.max(np.abs(levels))
    errors = []
    for off in offsets:
        cos_o, sin_o = np.cos(off), np.sin(off)
        ir = i_sig * cos_o + q_sig * sin_o
        qr = -i_sig * sin_o + q_sig * cos_o
        scale = float(np.percentile(np.maximum(np.abs(ir), np.abs(qr)), 95.0))
        if scale <= 1e-12:
            scale = 1.0
        ir = ir / scale
        qr = qr / scale
        ii = np.argmin(np.abs(ir[:, None] - levels[None, :]), axis=1)
        qi = np.argmin(np.abs(qr[:, None] - levels[None, :]), axis=1)
        err = np.mean((ir - levels[ii]) ** 2 + (qr - levels[qi]) ** 2)
        errors.append(float(err))
    return _closest_to_zero(offsets, np.asarray(errors))


def demodulate_fm(received_signal, sample_rate: float,
                  carrier_freq: float = 10000.0, deviation: float = 3000.0,
                  cutoff_freq: float = 4000.0,
                  bandwidth: float | None = None) -> tuple[np.ndarray, float]:
    """Demodula FM con un discriminador de frecuencia (fase instantánea).

    1. Filtro pasa-bandas alrededor de la portadora.
    2. Señal analítica de Hilbert y fase instantánea.
    3. Frecuencia instantánea = derivada de la fase.
    4. Se resta la portadora, se divide por la desviación y se filtra pasa-bajas.
    """
    x = np.asarray(received_signal, dtype=np.float64).ravel()
    if x.size == 0:
        return x, float(sample_rate)
    fs = float(sample_rate)
    fc = float(carrier_freq)
    dev = float(deviation)
    cutoff = float(cutoff_freq)
    if not 0.0 < fc < fs / 2.0:
        raise ModulationError(f"La portadora debe estar entre 0 y {fs / 2:.0f} Hz.")
    if dev <= 0.0:
        raise ModulationError("La desviación de FM debe ser > 0 Hz.")
    if cutoff <= 0.0 or cutoff >= fs / 2.0:
        raise ModulationError("El corte de audio debe estar entre 0 y fs/2.")

    if bandwidth is None:
        bandwidth = dev + cutoff
    low = max(1.0, fc - float(bandwidth))
    high = min(fs / 2.0 - 1.0, fc + float(bandwidth))
    filtered = _safe_bandpass(x, low, high, fs)

    analytic = hilbert(filtered)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) * fs / (2.0 * np.pi)
    audio = (inst_freq - fc) / dev
    audio = _safe_lowpass(audio, cutoff, fs)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return _finish_demod(audio, fs)


def _symbol_slices(start: float, offset: float, sps: float, n_sym: int,
                   n: int) -> tuple[np.ndarray, np.ndarray]:
    k = np.arange(n_sym, dtype=np.float64)
    starts = np.floor(start + offset + k * sps).astype(np.int64)
    ends = np.floor(start + offset + (k + 1.0) * sps).astype(np.int64)
    starts = np.clip(starts, 0, max(0, n - 1))
    ends = np.clip(ends, 0, n)
    valid = ends > starts
    return starts[valid], ends[valid]


def _bits_from_symbols(symbols: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    bps = int(bits_per_symbol)
    shifts = np.arange(bps - 1, -1, -1, dtype=np.int64)
    values = ((symbols[:, None].astype(np.int64) >> shifts[None, :]) & 1)
    return values.astype(np.uint8).reshape(-1)


def _codes_from_bits(bits: np.ndarray, pcm_bits: int) -> np.ndarray:
    n = len(bits) // int(pcm_bits)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    groups = bits[:n * int(pcm_bits)].reshape(n, int(pcm_bits))
    weights = (1 << np.arange(int(pcm_bits) - 1, -1, -1)).astype(np.int64)
    return (groups.astype(np.int64) * weights).sum(axis=1)


def _align_pcm_phase(bits: np.ndarray, pcm_bits: int) -> np.ndarray:
    """Elige el desplazamiento de bits que produce el audio PCM más suave.

    Al buscar el inicio de una ráfaga capturada por micrófono, el primer bit
    puede caer en cualquier posición dentro de una muestra PCM.  Como el audio
    es una señal continua, la agrupación correcta de ``pcm_bits`` bits es la
    que minimiza la variación entre muestras consecutivas.
    """
    nbits = int(pcm_bits)
    if nbits <= 1 or bits.size < nbits + 1:
        return bits
    best_shift = 0
    best_score = float("inf")
    for shift in range(nbits):
        codes = _codes_from_bits(bits[shift:], nbits)
        if codes.size < 2:
            continue
        values = codes.astype(np.float64)
        delta = np.diff(values - values.mean())
        score = float(np.mean(delta * delta))
        if score < best_score:
            best_score = score
            best_shift = shift
    return bits[best_shift:] if best_shift else bits


def _pcm_from_psk(symbols: np.ndarray, bps: int, m: int,
                  pcm_bits: int) -> np.ndarray:
    """Resuelve la ambigüedad de fase de PSK y devuelve el mejor flujo PCM.

    Una rotación constante de la constelación PSK suma un entero a todos los
    símbolos y produce una permutación de los bits.  Se prueban las M
    rotaciones posibles y se elige la que da el audio más suave.
    """
    best_bits = None
    best_score = float("inf")
    for rotation in range(int(m)):
        rotated = (symbols + rotation) % int(m)
        bits = _bits_from_symbols(rotated, bps)
        aligned = _align_pcm_phase(bits, pcm_bits)
        codes = _codes_from_bits(aligned, pcm_bits)
        if codes.size < 2:
            continue
        values = codes.astype(np.float64)
        delta = np.diff(values - values.mean())
        score = float(np.mean(delta * delta))
        if score < best_score:
            best_score = score
            best_bits = aligned
    return best_bits if best_bits is not None else bits


# Presupuesto aproximado de memoria para las matrices de trabajo de la
# demodulación digital (segmento, acumuladas I/Q y banco de correladores FSK).
# Así una captura muy larga no agota la memoria, pero se procesan varios
# segundos de audio en lugar de recortar la salida a una fracción de segundo.
_DEMOD_MEMORY_BUDGET = 320 * 1024 * 1024  # bytes
_DEMOD_MAX_SPAN = 6_000_000  # muestras (≈136 s a 44100 Hz)


def _max_span_samples(technique: str, m: int, sps: float) -> int:
    """Máximo de muestras a procesar según lo que consume cada técnica."""
    banks = 2 * int(m) if technique in ("FSK", "GFSK", "MFSK") else 0
    bytes_per_sample = 8.0 * (2 + banks + 4)  # acumuladas + segmento/temporales
    span = int(_DEMOD_MEMORY_BUDGET / bytes_per_sample)
    return max(int(64 * sps), min(span, _DEMOD_MAX_SPAN))


def demodulate_digital(technique: str, received_signal, fs: float,
                       carrier_freq: float, symbol_rate: float,
                       bits_per_symbol: int, pcm_bits: int,
                       audio_fs: float = 16000.0,
                       deviation: float = 3000.0, bt: float = 0.5,
                       find_start: bool = True,
                       max_symbols: int | None = None) -> DemodResult:
    """Demodula ASK, FSK, PSK, QAM, GFSK o MFSK y reconstruye el PCM.

    La recepción digital hace:
    1. Filtro pasa-bandas alrededor de la portadora.
    2. Búsqueda del inicio de la ráfaga (si ``find_start`` es True).
    3. Búsqueda del instante de muestreo dentro del símbolo.
    4. Decisión de símbolos (envolvente, fase, constelación o frecuencia).
    5. Símbolos → bits → códigos PCM → audio.

    ``max_symbols`` limita explícitamente cuántos símbolos se procesan.  Si es
    ``None`` se procesa toda la señal, acotada solo por el presupuesto de
    memoria de la técnica.
    """
    technique = str(technique).upper()
    if technique not in DIGITAL_TECHNIQUES:
        raise ModulationError(f"Técnica digital no soportada: {technique}")
    x = np.asarray(received_signal, dtype=np.float64).ravel()
    if x.size == 0:
        raise ModulationError("No hay señal para demodular.")
    fs = float(fs)
    fc = float(carrier_freq)
    baud = float(symbol_rate)
    bps = int(bits_per_symbol)
    nbits_pcm = int(pcm_bits)
    fs_out = float(audio_fs)
    dev = float(deviation)

    if not 0.0 < fc < fs / 2.0:
        raise ModulationError(f"La portadora debe estar entre 0 y {fs / 2:.0f} Hz.")
    if baud <= 0.0:
        raise ModulationError("La tasa de símbolos debe ser > 0 baudios.")
    sps = fs / baud
    if sps < 4.0:
        raise ModulationError(
            f"La tasa de símbolos es demasiado alta: {baud:.0f} baudios deja "
            f"{sps:.1f} muestras por símbolo.")
    if not 1 <= bps <= 6:
        raise ModulationError("Los bits por símbolo deben estar entre 1 y 6.")
    if not 1 <= nbits_pcm <= 24:
        raise ModulationError("Los bits PCM deben estar entre 1 y 24.")
    if fs_out <= 0.0:
        raise ModulationError("La fs del audio debe ser > 0 Hz.")
    m = 1 << bps
    side = 0
    if technique == "QAM":
        side = int(round(np.sqrt(m)))
        if side * side != m:
            raise ModulationError("QAM necesita 2, 4 o 6 bits/símbolo.")
    if technique in ("FSK", "GFSK", "MFSK") and dev <= 0.0:
        raise ModulationError("La desviación de frecuencia debe ser > 0 Hz.")

    # -- filtro de radiofrecuencia -------------------------------------
    if technique in ("FSK", "GFSK", "MFSK"):
        bw = dev + max(2.0 * baud, 2000.0)
    else:
        bw = max(2.0 * baud, 2000.0)
    low = max(1.0, fc - bw)
    high = min(fs / 2.0 - 1.0, fc + bw)
    filtered = _safe_bandpass(x, low, high, fs)

    # -- ráfaga y bases de demodulación --------------------------------
    if find_start:
        start, end = _find_burst(filtered, sps)
    else:
        start, end = 0, len(filtered)
    # No se procesan más muestras de las que permite el presupuesto de
    # memoria (o ``max_symbols`` si se indicó); así una captura larga no
    # dispara el uso de memoria, pero la salida no se recorta de más.
    if max_symbols is None:
        max_span = _max_span_samples(technique, m, sps)
    else:
        max_span = int((int(max_symbols) + 2) * sps) + 1
    total_span = end - start
    end = min(end, start + max(2, max_span))
    if end - start < 2:
        raise ModulationError("La señal es demasiado corta para demodular.")
    truncated = (end - start) < total_span

    # Se trabaja solo sobre el segmento activo (coordenadas desde 0).
    segment = filtered[start:end]
    n = len(segment)
    t = np.arange(n, dtype=np.float64) / fs
    cos_w = np.cos(2.0 * np.pi * fc * t)
    sin_w = np.sin(2.0 * np.pi * fc * t)
    cum_i = np.concatenate(([0.0], np.cumsum(segment * cos_w)))
    cum_q = np.concatenate(([0.0], np.cumsum(-segment * sin_w)))

    # Para FSK/GFSK/MFSK se usa un banco de correladores no coherentes en las
    # M frecuencias posibles (detector óptimo para M-FSK). Las sumas
    # acumuladas permiten obtener la energía de cada símbolo con dos restas.
    fsk_c = fsk_s = None
    if technique in ("FSK", "GFSK", "MFSK"):
        spacing = 2.0 * dev / max(1, m - 1)
        freqs = fc - dev + np.arange(m, dtype=np.float64) * spacing
        fsk_c = np.empty((m, n + 1), dtype=np.float64)
        fsk_s = np.empty((m, n + 1), dtype=np.float64)
        for i, f in enumerate(freqs):
            w = 2.0 * np.pi * f * t
            fsk_c[i] = np.concatenate(([0.0], np.cumsum(segment * np.cos(w))))
            fsk_s[i] = np.concatenate(([0.0], np.cumsum(segment * np.sin(w))))

    def decode_at(offset: float, correct: bool):
        available = n - offset
        n_sym = int(np.floor(available / sps + 0.5))
        if n_sym < 1:
            return None
        if max_symbols is not None:
            n_sym = min(n_sym, int(max_symbols))
        starts, ends = _symbol_slices(0.0, offset, sps, n_sym, n)
        if starts.size < 1:
            return None
        lengths = (ends - starts).astype(np.float64)
        i_sig = (cum_i[ends] - cum_i[starts]) / lengths
        q_sig = (cum_q[ends] - cum_q[starts]) / lengths

        if technique == "ASK":
            env = np.hypot(i_sig, q_sig)
            scale = float(np.percentile(env, 99.0)) if env.size else 0.0
            if scale <= 1e-12:
                scale = 1.0
            env = np.clip(env / scale, 0.0, 1.0)
            levels = np.arange(m, dtype=np.float64) / max(1, m - 1)
            idx = np.argmin(np.abs(env[:, None] - levels[None, :]), axis=1)
            err = float(np.mean((env - levels[idx]) ** 2))
            return idx.astype(np.int64), err

        if technique == "PSK":
            phi = np.arctan2(q_sig, i_sig)
            off = _best_psk_phase_offset(phi, m) if correct else 0.0
            phi = phi - off
            idx = np.rint(phi * m / (2.0 * np.pi)).astype(np.int64) % m
            resid = np.angle(np.exp(1j * phi * m)) / float(m)
            err = float(np.mean(np.abs(resid)))
            return idx.astype(np.int64), err

        if technique == "QAM":
            off = _best_qam_phase_offset(i_sig, q_sig, side) if correct else 0.0
            cos_o, sin_o = np.cos(off), np.sin(off)
            ir = i_sig * cos_o + q_sig * sin_o
            qr = -i_sig * sin_o + q_sig * cos_o
            scale = float(np.percentile(np.maximum(np.abs(ir), np.abs(qr)), 95.0))
            if scale <= 1e-12:
                scale = 1.0
            ir = ir / scale
            qr = qr / scale
            levels = np.arange(side, dtype=np.float64) - (side - 1) / 2.0
            levels = levels / np.max(np.abs(levels))
            ii = np.argmin(np.abs(ir[:, None] - levels[None, :]), axis=1)
            qi = np.argmin(np.abs(qr[:, None] - levels[None, :]), axis=1)
            idx = ii * side + qi
            err = float(np.mean((ir - levels[ii]) ** 2 + (qr - levels[qi]) ** 2))
            return idx.astype(np.int64), err

        # FSK, MFSK y GFSK: banco de correladores no coherentes. El símbolo
        # elegido es el de mayor energía y el error mide qué fracción de la
        # energía cae en la frecuencia ganadora (1 = decisión perfecta). Este
        # criterio también sirve para elegir el instante de muestreo: una
        # ventana desalineada reparte la energía entre dos frecuencias.
        c_delta = fsk_c[:, ends] - fsk_c[:, starts]
        s_delta = fsk_s[:, ends] - fsk_s[:, starts]
        energies = c_delta ** 2 + s_delta ** 2
        idx = np.argmax(energies, axis=0).astype(np.int64)
        best = energies[idx, np.arange(idx.size)]
        total = np.maximum(np.sum(energies, axis=0), 1e-30)
        err = float(np.mean(1.0 - best / total))
        return idx, err

    # -- búsqueda del instante de muestreo ----------------------------
    n_offsets = min(32, max(8, int(round(sps))))
    offsets = np.linspace(0.0, sps, n_offsets, endpoint=False)
    results = []
    for off in offsets:
        decoded = decode_at(float(off), correct=bool(find_start))
        if decoded is None:
            continue
        symbols, err = decoded
        results.append((float(off), symbols, float(err)))
    if not results:
        raise ModulationError("No se pudieron extraer símbolos de la señal.")

    # Se prefiere el desfase que recupera más símbolos (evita perder el
    # primero por un transitorio) y, entre esos, el de menor error y más
    # cercano a 0 (la señal empieza en la muestra 0).
    best_err = min(r[2] for r in results)
    tolerance = best_err + max(0.02, 0.15 * best_err)
    near = [r for r in results if r[2] <= tolerance]
    max_count = max(int(r[1].size) for r in near)
    finalists = [r for r in near if int(r[1].size) == max_count]
    best_off, symbols, error = min(finalists,
                                   key=lambda r: (r[2], abs(r[0])))

    bits = _bits_from_symbols(symbols, bps)
    # En recepción por micrófono el inicio de la ráfaga no tiene por qué caer
    # en un límite de muestra PCM; se busca la agrupación de bits más suave.
    if find_start and technique == "PSK":
        pcm_stream = _pcm_from_psk(symbols, bps, m, nbits_pcm)
    elif find_start:
        pcm_stream = _align_pcm_phase(bits, nbits_pcm)
    else:
        pcm_stream = bits
    codes = _codes_from_bits(pcm_stream, nbits_pcm)
    if codes.size == 0:
        raise ModulationError(
            "No se completó ninguna muestra PCM: aumente la duración de la "
            "recepción o revise los parámetros.")
    half = 1 << (nbits_pcm - 1)
    pcm = (codes.astype(np.float64) - half) / half
    pcm = np.clip(pcm, -1.0, 1.0)

    name = _tech_name(technique, m, bt)
    info = (f"{name} · fc = {fc:.0f} Hz · baudios = {baud:.0f} · "
            f"{bps} bit(s)/símbolo · {symbols.size} símbolos · "
            f"{bits.size} bits · {codes.size} muestras PCM de {nbits_pcm} bits · "
            f"fs audio = {fs_out:.0f} Hz · duración = {codes.size / fs_out:.2f} s · "
            f"offset = {best_off:.2f} muestras · error = {error:.3f}")
    if truncated:
        info += (f" · procesados los primeros {n / fs:.1f} s de la señal "
                 "recibida por límite de memoria")
    if technique == "GFSK" and float(bt) < 0.5:
        info += (" · aviso: con BT < 0.5 la interferencia entre símbolos "
                 "puede degradar la decisión")
    if technique in ("FSK", "GFSK", "MFSK"):
        spacing = 2.0 * dev / max(1, m - 1)
        if spacing < baud:
            info += (" · aviso: la separación entre frecuencias "
                     f"({spacing:.0f} Hz) es menor que los baudios; "
                     "aumente Δf o reduzca la tasa de símbolos")
    return DemodResult(
        signal=pcm,
        fs=int(round(fs_out)),
        technique=technique,
        info=info,
        bits=bits,
        symbols=symbols,
        error=float(error),
        timing_offset=float(best_off),
        n_symbols=int(symbols.size),
    )

