"""Pruebas del Desafío 4: compresión sin pérdida de audio (WAV ↔ FLAC)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from core.compression_audio import (CompressionError, audio_roundtrip,
                                    compress_audio_wav, decompress_audio_flac)
from core.compression_common import size_change


def make_wav(path: Path, subtype: str = "PCM_16") -> Path:
    t = np.arange(8000) / 8000.0
    x = 0.7 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
    dtype = "int16" if subtype == "PCM_16" else "int32"
    pcm = np.round(x * (2 ** 14 if subtype == "PCM_16" else 2 ** 30)).astype(dtype)
    sf.write(str(path), pcm, 8000, subtype=subtype, format="WAV")
    return path


class TestSizeChange(unittest.TestCase):
    def test_smaller_file_uses_positive_percentage(self) -> None:
        text = size_change(1_000_000, 400_000)
        self.assertIn("2.50×", text)
        self.assertIn("más pequeño", text)

    def test_bigger_file_is_described_as_bigger(self) -> None:
        text = size_change(1_000_000, 2_000_000)
        self.assertIn("0.50×", text)
        self.assertIn("más grande", text)
        self.assertNotIn("-", text)

    def test_unknown_sizes_do_not_divide_by_zero(self) -> None:
        self.assertEqual(size_change(0, 1234), "n/d (tamaños no disponibles)")


class TestAudioCompression(unittest.TestCase):
    def test_roundtrip_16_bits_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wav = make_wav(Path(tmp) / "tono.wav")
            result = audio_roundtrip(wav, Path(tmp) / "salida")
            self.assertTrue(result.exact)
            self.assertEqual(result.pcm_sha256, result.restored_sha256)
            self.assertEqual(result.fs, 8000)
            self.assertEqual(result.max_abs_error, 0)
            self.assertGreater(result.ratio, 1.0)
            self.assertIn("EXACTA", result.summary())

    def test_roundtrip_24_bits_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wav = make_wav(Path(tmp) / "tono24.wav", "PCM_24")
            result = audio_roundtrip(wav, Path(tmp) / "salida")
            self.assertTrue(result.exact)
            self.assertEqual(result.flac_subtype, "PCM_24")
            self.assertEqual(result.subtype, "PCM_24")

    def test_unsupported_subtype_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "float.wav"
            sf.write(str(wav), np.zeros(1000, dtype=np.float64), 8000,
                     subtype="FLOAT", format="WAV")
            with self.assertRaises(CompressionError) as ctx:
                compress_audio_wav(wav, Path(tmp) / "x.flac")
            self.assertIn("16 bits", str(ctx.exception))

    def test_corrupt_flac_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "roto.flac"
            bad.write_bytes(b"esto no es un flac")
            with self.assertRaises(CompressionError):
                decompress_audio_flac(bad, Path(tmp) / "out.wav")


if __name__ == "__main__":
    unittest.main()
