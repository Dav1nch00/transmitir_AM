"""Pruebas del núcleo DSP y de audio, sin abrir la interfaz gráfica."""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from core.audio_io import Audio, read_wav, save_wav
from core.dsp import (alias_frequency, bandwidth_99, energy_above, resample_to,
                      sinc_interp, spectrum)
from core.pipeline import Params, digitize
from core.quantization import AdcConfig, QuantizationError, quantize, to_binary


def sine(freq: float, fs: float, seconds: float, amp: float = 0.9) -> np.ndarray:
    t = np.arange(int(round(fs * seconds))) / float(fs)
    return amp * np.sin(2.0 * np.pi * freq * t)


class TestQuantization(unittest.TestCase):
    def test_sqnr_sine_16_bits(self) -> None:
        x = sine(997.0, 8000.0, 1.0, amp=0.999)
        qr = quantize(x, AdcConfig(16, 1.0), fs=8000.0)
        theoretical = 6.02 * 16 + 1.76
        self.assertAlmostEqual(qr.sqnr_sine_db, theoretical, places=6)
        self.assertGreater(qr.sqnr_db, theoretical - 1.5)
        self.assertLess(qr.sqnr_db, theoretical + 1.5)

    def test_error_is_bounded_by_half_lsb(self) -> None:
        x = sine(440.0, 8000.0, 0.5, amp=0.4)
        for bits in (2, 4, 8, 12):
            qr = quantize(x, AdcConfig(bits, 1.0))
            self.assertLessEqual(qr.error_max, qr.cfg.lsb / 2.0 + 1e-12)
            self.assertEqual(qr.clipped, 0)

    def test_clipping_is_detected(self) -> None:
        x = np.array([-1.5, -1.0, 0.0, 1.0, 1.5])
        qr = quantize(x, AdcConfig(4, 1.0))
        self.assertGreaterEqual(qr.clipped, 2)
        self.assertTrue(np.all(qr.codes_u >= 0))
        self.assertTrue(np.all(qr.codes_u < 16))

    def test_binary_code(self) -> None:
        self.assertEqual(to_binary(5, 4), "0101")
        self.assertEqual(to_binary(0, 8), "00000000")

    def test_invalid_config(self) -> None:
        with self.assertRaises(QuantizationError):
            AdcConfig(1, 1.0)
        with self.assertRaises(QuantizationError):
            AdcConfig(8, 0.0)


class TestSpectrum(unittest.TestCase):
    def test_peak_frequency(self) -> None:
        fs = 8000.0
        x = sine(1000.0, fs, 1.0)
        f, mag = spectrum(x, fs, "Hann")
        peak = f[np.argmax(mag)]
        self.assertAlmostEqual(peak, 1000.0, delta=2.0)

    def test_bandwidth_detects_second_tone(self) -> None:
        fs = 8000.0
        t = np.arange(int(fs)) / fs
        x = 0.5 * np.sin(2 * np.pi * 500 * t) + 0.4 * np.sin(2 * np.pi * 2500 * t)
        f, mag = spectrum(x, fs, "Hann")
        self.assertGreater(bandwidth_99(f, mag), 2400.0)
        self.assertLess(bandwidth_99(f, mag), 2700.0)

    def test_energy_above_nyquist(self) -> None:
        fs = 8000.0
        x = sine(3000.0, fs, 1.0)
        f, mag = spectrum(x, fs, "Hann")
        self.assertGreater(energy_above(f, mag, 2000.0), 0.8)
        self.assertLess(energy_above(f, mag, 4000.0), 0.01)


class TestSamplingAndAliasing(unittest.TestCase):
    def test_alias_frequency(self) -> None:
        self.assertAlmostEqual(alias_frequency(7000.0, 8000.0), 1000.0, places=6)
        self.assertAlmostEqual(alias_frequency(3000.0, 8000.0), 3000.0, places=6)
        self.assertAlmostEqual(alias_frequency(5000.0, 8000.0), 3000.0, places=6)

    def test_antialias_filter_reduces_folded_tone(self) -> None:
        fs = 44100.0
        x = sine(7000.0, fs, 1.0, amp=0.8)
        y_no, fs_no = resample_to(x, fs, 8000.0, antialias=False)
        y_aa, fs_aa = resample_to(x, fs, 8000.0, antialias=True)
        f_no, m_no = spectrum(y_no, fs_no, "Hann")
        f_aa, m_aa = spectrum(y_aa, fs_aa, "Hann")
        band_no = (f_no > 950) & (f_no < 1050)
        band_aa = (f_aa > 950) & (f_aa < 1050)
        amp_no = float(np.max(m_no[band_no]))
        amp_aa = float(np.max(m_aa[band_aa]))
        self.assertGreater(amp_no, 0.5)
        self.assertLess(amp_aa, amp_no / 20.0)

    def test_resample_length(self) -> None:
        x = np.zeros(44100)
        y, fs = resample_to(x, 44100, 8000, antialias=True)
        self.assertAlmostEqual(fs, 8000.0)
        self.assertAlmostEqual(len(y), 8000, delta=2)

    def test_sinc_interpolation(self) -> None:
        fs = 1000.0
        f0 = 50.0
        x = np.sin(2 * np.pi * f0 * np.arange(500) / fs)
        t_out = np.linspace(0.05, 0.20, 1500)
        y = sinc_interp(x, fs, t_out, half_width=32)
        ideal = np.sin(2 * np.pi * f0 * t_out)
        self.assertLess(float(np.max(np.abs(y - ideal))), 0.01)


class TestAudioAndPipeline(unittest.TestCase):
    def test_wav_roundtrip(self) -> None:
        fs = 8000
        x = 0.5 * np.sin(2 * np.pi * 440 * np.arange(fs) / fs)
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            save_wav(path, x, fs, bits=16)
            audio = read_wav(path)
            self.assertEqual(audio.fs, fs)
            self.assertEqual(len(audio.x), len(x))
            self.assertLess(float(np.max(np.abs(audio.x - x))), 1e-4)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def test_digitize_pipeline(self) -> None:
        fs = 44100
        audio = Audio(x=sine(1000.0, fs, 1.0), fs=fs, source="prueba")
        d = digitize(audio, Params(fs_target=16000, bits=8, antialias=True,
                                   normalize=True))
        self.assertAlmostEqual(d.fs_eff, 16000.0, places=6)
        self.assertAlmostEqual(len(d.x_n), 16000, delta=2)
        self.assertLessEqual(float(np.max(np.abs(d.x_n))), 0.96)
        self.assertEqual(d.quant.clipped, 0)
        self.assertEqual(len(d.t_n), len(d.x_n))
        self.assertGreater(d.quant.sqnr_db, 40.0)

    def test_pipeline_without_normalization_can_clip(self) -> None:
        fs = 8000
        audio = Audio(x=4.0 * sine(500.0, fs, 0.2, amp=1.0), fs=fs, source="prueba")
        d = digitize(audio, Params(fs_target=8000, bits=4, antialias=False,
                                   normalize=False))
        self.assertGreater(d.quant.clipped, 0)


if __name__ == "__main__":
    unittest.main()
