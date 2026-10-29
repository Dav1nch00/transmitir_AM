"""Pruebas de modulación y demodulación AM, sin abrir la interfaz gráfica."""
from __future__ import annotations

import unittest

import numpy as np

from core.am import (DEMOD_BANDWIDTH, bandpass_filter, demodulate_am,
                     demodulate_am_hilbert, lowpass_filter, modulate_am,
                     prepare_baseband)
from core.dsp import spectrum


def sine(freq: float, fs: float, seconds: float, amp: float = 0.9) -> np.ndarray:
    t = np.arange(int(round(fs * seconds))) / float(fs)
    return amp * np.sin(2.0 * np.pi * freq * t)


def peak_of(f: np.ndarray, mag: np.ndarray, fmin: float = 100.0) -> float:
    m = f >= fmin
    return float(f[m][np.argmax(mag[m])])


class TestModulation(unittest.TestCase):
    def test_modulated_carrier_frequency(self) -> None:
        fs = 44100.0
        x = sine(1000.0, fs, 0.5, amp=0.5)
        am = modulate_am(x, fs, carrier_freq=17000.0, index_mod=0.7)
        f, mag = spectrum(am, fs, "Hann")
        self.assertAlmostEqual(peak_of(f, mag), 17000.0, delta=5.0)
        self.assertLessEqual(float(np.max(np.abs(am))), 1.0 + 1e-12)

    def test_modulation_sidebands(self) -> None:
        fs = 44100.0
        x = sine(1000.0, fs, 0.5, amp=0.5)
        am = modulate_am(x, fs, carrier_freq=17000.0, index_mod=0.7)
        f, mag = spectrum(am, fs, "Hann")
        # Banda lateral superior: fc + 1000 Hz.
        self.assertAlmostEqual(peak_of(f, mag, fmin=17000.0 + 500.0), 18000.0, delta=5.0)

    def test_modulate_empty_signal(self) -> None:
        am = modulate_am(np.zeros(0), 44100.0)
        self.assertEqual(len(am), 0)

    def test_prepare_baseband_resamples(self) -> None:
        fs = 16000.0
        x = sine(1000.0, fs, 1.0, amp=0.8)
        y, fs_out = prepare_baseband(x, fs, 44100.0)
        self.assertAlmostEqual(fs_out, 44100.0)
        self.assertAlmostEqual(len(y), 44100, delta=4)
        # Se normaliza a pico 1.
        self.assertAlmostEqual(float(np.max(np.abs(y))), 1.0, places=6)

    def test_prepare_baseband_same_fs(self) -> None:
        x = sine(500.0, 44100.0, 0.2)
        y, fs_out = prepare_baseband(x, 44100.0, 44100.0)
        self.assertAlmostEqual(fs_out, 44100.0)
        self.assertEqual(len(y), len(x))


class TestDemodulation(unittest.TestCase):
    def roundtrip(self, demod_fn) -> None:
        fs = 44100.0
        fmsg = 1000.0
        x = sine(fmsg, fs, 1.0, amp=0.5)
        am = modulate_am(x, fs, carrier_freq=17000.0, index_mod=0.7)
        demod, fs_out = demod_fn(am, fs, carrier_freq=17000.0, cutoff_freq=4000.0)
        self.assertAlmostEqual(fs_out, fs)
        f, mag = spectrum(demod, fs_out, "Hann")
        self.assertAlmostEqual(peak_of(f, mag, fmin=100.0), fmsg, delta=10.0)
        # El error frente a la señal original es pequeño tras alinear la escala.
        y = demod[:len(x)] * 0.5
        err = float(np.max(np.abs(y - x[:len(y)])))
        self.assertLess(err, 0.2)

    def test_envelope_demod_roundtrip(self) -> None:
        self.roundtrip(demodulate_am)

    def test_hilbert_demod_roundtrip(self) -> None:
        self.roundtrip(demodulate_am_hilbert)

    def test_demodulate_empty_signal(self) -> None:
        y, fs = demodulate_am(np.zeros(0), 44100.0)
        self.assertEqual(len(y), 0)
        self.assertAlmostEqual(fs, 44100.0)

    def test_demodulation_bandwidth_covers_voice(self) -> None:
        # El ancho de banda debe cubrir las bandas laterales de la voz (≥ 4 kHz).
        self.assertGreaterEqual(DEMOD_BANDWIDTH, 4000.0)

    def test_amplitude_of_carrier_alone_is_removed(self) -> None:
        # Solo portadora: la envolvente debe ser plana (rizo < 5 % del DC).
        fs = 44100.0
        am = modulate_am(np.zeros(int(fs)), fs, carrier_freq=17000.0, index_mod=0.7)
        filtered = bandpass_filter(am, 15000.0, 19000.0, fs)
        env = lowpass_filter(np.abs(filtered), 4000.0, fs)
        center = env[2000:-2000]
        self.assertLess(float(center.std() / center.mean()), 0.05)

    def test_hilbert_demod_of_silence_is_quiet(self) -> None:
        # Con Hilbert, la demodulación de solo portadora debe ser mucho más
        # silenciosa que la de un tono real.
        fs = 44100.0
        t = np.arange(int(fs)) / fs
        silence = modulate_am(np.zeros(int(fs)), fs, carrier_freq=17000.0, index_mod=0.7)
        tone = modulate_am(0.5 * np.sin(2 * np.pi * 1000 * t), fs,
                           carrier_freq=17000.0, index_mod=0.7)
        d_sil, _ = demodulate_am_hilbert(silence, fs, carrier_freq=17000.0)
        d_tone, _ = demodulate_am_hilbert(tone, fs, carrier_freq=17000.0)
        r_sil = float(np.sqrt(np.mean(d_sil ** 2)))
        r_tone = float(np.sqrt(np.mean(d_tone ** 2)))
        self.assertLess(r_sil, 0.5 * r_tone)


class TestFilters(unittest.TestCase):
    def test_bandpass_isolates_carrier(self) -> None:
        fs = 44100.0
        t = np.arange(int(fs)) / fs
        x = 0.5 * np.sin(2 * np.pi * 1000 * t) + 0.8 * np.sin(2 * np.pi * 17000 * t)
        y = bandpass_filter(x, 15000.0, 19000.0, fs, order=4)
        f, mag = spectrum(y, fs, "Hann")
        low = float(np.max(mag[(f > 500) & (f < 2000)]))
        high = float(np.max(mag[(f > 16500) & (f < 17500)]))
        self.assertGreater(high, 20.0 * low)

    def test_lowpass_removes_carrier(self) -> None:
        fs = 44100.0
        t = np.arange(int(fs)) / fs
        x = np.sin(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 17000 * t)
        y = lowpass_filter(x, 4000.0, fs, order=4)
        f, mag = spectrum(y, fs, "Hann")
        low = float(np.max(mag[(f > 500) & (f < 2000)]))
        high = float(np.max(mag[(f > 16000) & (f < 18000)]))
        self.assertGreater(low, 20.0 * high)

    def test_invalid_cuts_raise(self) -> None:
        with self.assertRaises(ValueError):
            bandpass_filter(np.zeros(100), 19000.0, 24000.0, 44100.0)
        with self.assertRaises(ValueError):
            bandpass_filter(np.zeros(100), 2000.0, 1000.0, 44100.0)
        with self.assertRaises(ValueError):
            lowpass_filter(np.zeros(100), 30000.0, 44100.0)


if __name__ == "__main__":
    unittest.main()
