"""Compresión de audio sin pérdida: WAV (PCM) ↔ FLAC.

FLAC guarda exactamente las muestras PCM de entrada, por lo que después de
comprimir y descomprimir el hash de las muestras es idéntico. Se conservan la
frecuencia de muestreo, los canales y el subtipo PCM cuando FLAC lo admite
(PCM_16 y PCM_24). PCM_32 se reduce a PCM_24 desplazando 8 bits (solo se
descartan los 8 bits menos significativos, que no caben en FLAC).

Los WAV con subtipos PCM_U8 o float se rechazan con un mensaje claro: desde la
pestaña 1 de RETICO se pueden exportar como PCM de 16 bits.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .compression_common import size_change

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - depende del entorno
    sf = None  # type: ignore[assignment]


class CompressionError(Exception):
    """Error de compresión que se puede mostrar al usuario."""


#: dtype con el que libsndfile entrega cada subtipo PCM de entrada.
PCM_DTYPES = {"PCM_16": "int16", "PCM_24": "int32", "PCM_32": "int32"}
#: subtipo con el que se escribe el FLAC para cada subtipo de entrada.
FLAC_SUBTYPES = {"PCM_16": "PCM_16", "PCM_24": "PCM_24", "PCM_32": "PCM_24"}


@dataclass
class AudioCompressionResult:
    """Resultado de comprimir, descomprimir o verificar un audio."""

    source: Path
    compressed: Path | None = None
    restored: Path | None = None
    original_size: int = 0
    compressed_size: int = 0
    restored_size: int = 0
    fs: int = 0
    channels: int = 0
    frames: int = 0
    subtype: str = ""        # subtipo del WAV original
    flac_subtype: str = ""   # subtipo usado dentro del FLAC
    pcm_sha256: str = ""     # hash de las muestras escritas en el FLAC
    restored_sha256: str = ""
    exact: bool = False
    max_abs_error: int = 0

    @property
    def ratio(self) -> float:
        """Cuántas veces más pequeño quedó el archivo."""
        if self.compressed_size <= 0:
            return 0.0
        return self.original_size / self.compressed_size

    @property
    def saving_percent(self) -> float:
        if self.original_size <= 0:
            return 0.0
        return 100.0 * (1.0 - self.compressed_size / self.original_size)

    def summary(self) -> str:
        lines = [
            f"Original : {self.source.name} · {self.original_size:,} bytes",
            f"FLAC     : {self.compressed.name if self.compressed else '-'} · "
            f"{self.compressed_size:,} bytes",
            f"Ratio    : {size_change(self.original_size, self.compressed_size)}",
            f"Audio    : {self.fs} Hz · {self.channels} canal(es) · "
            f"{self.frames:,} muestras · {self.subtype} → {self.flac_subtype}",
        ]
        if self.restored is not None:
            lines += [
                f"Reconstr.: {self.restored.name} · {self.restored_size:,} bytes",
                f"Igualdad : {'EXACTA (bit a bit)' if self.exact else 'DIFERENCIAS'} · "
                f"error máx. {self.max_abs_error}",
            ]
        lines.append(f"SHA-256 muestras PCM: {self.pcm_sha256}")
        return "\n".join(lines)


def _require_soundfile() -> None:
    if sf is None:  # pragma: no cover
        raise CompressionError(
            "Falta soundfile. Instale las dependencias con: pip install soundfile")


def _hash_pcm(data: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()


def _read_audio(path: str | Path) -> tuple[np.ndarray, int, str, int]:
    """Devuelve (muestras int, fs, subtipo WAV, canales)."""
    _require_soundfile()
    p = Path(path)
    if not p.is_file():
        raise CompressionError(f"No existe el archivo: {p}")
    try:
        info = sf.info(str(p))
    except Exception as exc:  # noqa: BLE001
        raise CompressionError(f"No se pudo leer {p.name}: {exc}") from exc
    subtype = info.subtype
    if subtype not in PCM_DTYPES:
        raise CompressionError(
            f"El WAV usa el subtipo '{subtype}', que no se puede comprimir en FLAC. "
            "Exporte el audio a PCM de 16 bits desde la pestaña 1 de RETICO.")
    try:
        data, fs = sf.read(str(p), dtype=PCM_DTYPES[subtype], always_2d=False)
    except Exception as exc:  # noqa: BLE001
        raise CompressionError(f"No se pudieron leer las muestras de {p.name}: {exc}") from exc
    return np.asarray(data), int(fs), subtype, int(info.channels)


def _to_flac_pcm(data: np.ndarray, subtype: str) -> np.ndarray:
    """Adapta las muestras al subtipo que FLAC puede almacenar."""
    if subtype == "PCM_32":
        # Descartamos los 8 bits menos significativos: PCM_24 usa 24 bits
        # justificados a la izquierda dentro de un int32.
        return (data >> 8).astype(np.int32)
    return data


def compress_audio_wav(src_wav: str | Path, dst_flac: str | Path) -> AudioCompressionResult:
    """Comprime un WAV PCM a FLAC sin pérdida y devuelve las métricas."""
    _require_soundfile()
    src, dst = Path(src_wav), Path(dst_flac)
    data, fs, subtype, channels = _read_audio(src)
    flac_pcm = _to_flac_pcm(data, subtype)
    flac_subtype = FLAC_SUBTYPES[subtype]
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        sf.write(str(dst), flac_pcm, fs, subtype=flac_subtype, format="FLAC")
    except Exception as exc:  # noqa: BLE001
        raise CompressionError(f"No se pudo escribir el FLAC: {exc}") from exc
    return AudioCompressionResult(
        source=src, compressed=dst,
        original_size=src.stat().st_size, compressed_size=dst.stat().st_size,
        fs=fs, channels=channels, frames=int(flac_pcm.shape[0]),
        subtype=subtype, flac_subtype=flac_subtype,
        pcm_sha256=_hash_pcm(flac_pcm),
    )


def decompress_audio_flac(
    src_flac: str | Path,
    dst_wav: str | Path,
    *,
    expected_sha256: str = "",
    restored_subtype: str = "",
) -> AudioCompressionResult:
    """Descomprime un FLAC a WAV PCM y comprueba que las muestras coinciden."""
    _require_soundfile()
    src, dst = Path(src_flac), Path(dst_wav)
    if not src.is_file():
        raise CompressionError(f"No existe el FLAC: {src}")
    try:
        info = sf.info(str(src))
    except Exception as exc:  # noqa: BLE001
        raise CompressionError(f"No se pudo leer el FLAC {src.name}: {exc}") from exc
    subtype = restored_subtype or info.subtype
    if subtype not in PCM_DTYPES:
        subtype = "PCM_16"
    dtype = PCM_DTYPES[subtype]
    try:
        data, fs = sf.read(str(src), dtype=dtype, always_2d=False)
    except Exception as exc:  # noqa: BLE001
        raise CompressionError(f"No se pudo descomprimir {src.name}: {exc}") from exc
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        sf.write(str(dst), data, fs, subtype=subtype, format="WAV")
    except Exception as exc:  # noqa: BLE001
        raise CompressionError(f"No se pudo escribir el WAV: {exc}") from exc
    digest = _hash_pcm(data)
    exact = bool(expected_sha256) and digest == expected_sha256
    return AudioCompressionResult(
        source=src, restored=dst,
        compressed_size=src.stat().st_size, restored_size=dst.stat().st_size,
        fs=fs, channels=int(info.channels), frames=int(np.asarray(data).shape[0]),
        subtype=subtype, flac_subtype=info.subtype,
        pcm_sha256=expected_sha256, restored_sha256=digest, exact=exact,
    )


def verify_audio_roundtrip(src_wav: str | Path, restored_wav: str | Path) -> AudioCompressionResult:
    """Compara dos WAV muestra a muestra (para la pestaña de verificación)."""
    a, fs_a, sub_a, ch_a = _read_audio(src_wav)
    b, fs_b, sub_b, ch_b = _read_audio(restored_wav)
    if a.shape != b.shape:
        raise CompressionError(
            f"Los audios tienen distinta longitud: {a.shape} vs {b.shape}.")
    if fs_a != fs_b or ch_a != ch_b:
        raise CompressionError("Los audios tienen distinta fs o número de canales.")
    a32, b32 = a.astype(np.int64), b.astype(np.int64)
    if sub_a == "PCM_32":
        a32 >>= 8
    if sub_b == "PCM_32":
        b32 >>= 8
    diff = np.abs(a32 - b32)
    return AudioCompressionResult(
        source=Path(src_wav), restored=Path(restored_wav),
        restored_size=Path(restored_wav).stat().st_size,
        fs=fs_a, channels=ch_a, frames=int(a.shape[0]),
        subtype=sub_a, flac_subtype=sub_b,
        pcm_sha256=_hash_pcm(a.astype(np.int32)),
        restored_sha256=_hash_pcm(b.astype(np.int32)),
        exact=bool(np.array_equal(a32, b32)),
        max_abs_error=int(diff.max()) if diff.size else 0,
    )


def audio_roundtrip(src_wav: str | Path, workdir: str | Path) -> AudioCompressionResult:
    """Comprime + descomprime + verifica; útil en pruebas y en la GUI."""
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    src = Path(src_wav)
    flac = work / f"{src.stem}_comprimido.flac"
    wav = work / f"{src.stem}_reconstruido.wav"
    compressed = compress_audio_wav(src, flac)
    result = decompress_audio_flac(
        flac, wav, expected_sha256=compressed.pcm_sha256,
        restored_subtype=compressed.subtype,
    )
    result.original_size = compressed.original_size
    result.compressed = flac
    return result
