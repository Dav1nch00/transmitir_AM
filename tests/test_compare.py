"""Pruebas de la comparación original vs. recuperado."""
from __future__ import annotations

import math
import unittest

import numpy as np

from core.compare import compare_audio, sequential_signal

FS = 16000.0


def tone(seconds: float = 1.0, fs: float = FS, freq: float = 440.0):
    t = np.arange(int(round(seconds * fs)), dtype=np.float64) / fs
    return 0.6 * np.sin(2.0 * np.pi * freq * t) + 0.2 * np.sin(2.0 * np.pi * 2 * freq * t)


class TestComparison(unittest.TestCase):
    def test_identical_audio_is_perfect(self) -> None:
        x = tone()
        r = compare_audio(x, FS, x.copy(), FS)
        self.assertEqual(r.n, len(x))
        self.assertEqual(r.delay_samples, 0)
        self.assertAlmostEqual(r.gain, 1.0)
        self.assertAlmostEqual(r.corr, 1.0, places=6)
        self.assertTrue(math.isinf(r.snr_db) and r.snr_db > 0)
        self.assertAlmostEqual(r.rmse, 0.0)
        self.assertEqual(r.fidelity, "muy alta")

    def test_late_audio_is_detected_with_positive_delay(self) -> None:
        x = tone()
        delayed = np.concatenate([np.zeros(100), x])[: len(x)]
        r = compare_audio(x, FS, delayed, FS, max_delay_s=0.5)
        self.assertEqual(r.delay_samples, 100)
        self.assertAlmostEqual(r.delay_s, 100 / FS)
        self.assertAlmostEqual(r.corr, 1.0, places=6)
        self.assertLess(r.overlap_ref, 1.0)
        self.assertLess(r.overlap_test, 1.0)

    def test_early_audio_is_detected_with_negative_delay(self) -> None:
        x = tone()
        early = x[100:]
        r = compare_audio(x, FS, early, FS, max_delay_s=0.5)
        self.assertEqual(r.delay_samples, -100)
        self.assertAlmostEqual(r.corr, 1.0, places=6)
        self.assertGreater(r.overlap_ref, 0.9)

    def test_gain_matching_recovers_scale(self) -> None:
        x = tone()
        r = compare_audio(x, FS, 0.5 * x, FS)
        self.assertAlmostEqual(r.gain, 2.0, places=6)
        self.assertAlmostEqual(r.corr, 1.0, places=6)
        self.assertAlmostEqual(r.rmse, 0.0)
        self.assertAlmostEqual(r.rms_ref, float(np.sqrt(np.mean(x ** 2))))

    def test_polarity_inversion_gives_negative_gain(self) -> None:
        x = tone()
        r = compare_audio(x, FS, -x, FS)
        self.assertAlmostEqual(r.gain, -1.0, places=6)
        self.assertAlmostEqual(r.corr, 1.0, places=6)
        self.assertAlmostEqual(r.rmse, 0.0)

    def test_noise_lowers_correlation_and_snr(self) -> None:
        x = tone()
        rng = np.random.default_rng(0)
        noisy = x + 0.03 * rng.standard_normal(x.size)
        r = compare_audio(x, FS, noisy, FS)
        self.assertLess(r.corr, 1.0)
        self.assertGreater(r.corr, 0.95)
        self.assertTrue(math.isfinite(r.snr_db))
        self.assertGreater(r.rmse, 0.0)
        self.assertGreater(r.max_error, r.rmse)

    def test_different_sample_rates_are_resampled(self) -> None:
        x = tone(fs=FS)
        t = np.arange(x.size) / FS
        y = np.interp(np.arange(2 * x.size) / (2 * FS), t, x)
        r = compare_audio(x, FS, y, 2 * FS)
        self.assertTrue(r.resampled)
        self.assertEqual(r.fs, FS)
        self.assertEqual(r.fs_test_original, 2 * FS)
        self.assertGreater(r.corr, 0.99)

    def test_different_lengths_compare_only_the_overlap(self) -> None:
        x = tone(seconds=1.0)
        y = tone(seconds=0.6)
        r = compare_audio(x, FS, y, FS)
        self.assertEqual(r.n, len(y))
        self.assertAlmostEqual(r.overlap_test, 1.0)
        self.assertAlmostEqual(r.overlap_ref, 0.6)
        self.assertAlmostEqual(r.corr, 1.0, places=6)

    def test_alignment_can_be_disabled(self) -> None:
        x = tone()
        delayed = np.concatenate([np.zeros(100), x])[: len(x)]
        r = compare_audio(x, FS, delayed, FS, align=False)
        self.assertEqual(r.delay_samples, 0)
        self.assertLess(r.corr, 0.99)

    def test_silence_is_handled(self) -> None:
        a = np.zeros(1600)
        b = np.zeros(1600)
        r = compare_audio(a, FS, b, FS)
        self.assertEqual(r.corr, 0.0)
        self.assertEqual(r.rmse, 0.0)
        self.assertTrue(math.isinf(r.snr_db))
        self.assertEqual(r.fidelity, "sin correlación")

    def test_empty_or_invalid_input_raises(self) -> None:
        x = tone()
        with self.assertRaises(ValueError):
            compare_audio([], FS, x, FS)
        with self.assertRaises(ValueError):
            compare_audio(x, 0.0, x, FS)
        with self.assertRaises(ValueError):
            compare_audio(x, FS, x, -1.0)
        with self.assertRaises(ValueError):
            compare_audio(x, FS, x, FS, max_delay_s=-0.5)
        with self.assertRaises(ValueError):
            compare_audio(np.array([np.nan, 0.0]), FS, x, FS)


class TestSequentialSignal(unittest.TestCase):
    def test_order_is_a_then_silence_then_b(self) -> None:
        a = np.ones(int(0.10 * FS))
        b = -np.ones(int(0.20 * FS))
        out, fs, start_b = sequential_signal(a, FS, b, FS, gap_s=0.05)
        self.assertEqual(fs, int(FS))
        self.assertEqual(len(out), len(a) + int(0.05 * FS) + len(b))
        self.assertAlmostEqual(start_b, (len(a) + int(0.05 * FS)) / FS)
        self.assertTrue(np.all(out[: len(a)] == 1.0))
        self.assertTrue(np.all(out[len(a): len(a) + int(0.05 * FS)] == 0.0))
        self.assertTrue(np.all(out[-len(b):] == -1.0))

    def test_resamples_both_to_the_higher_rate(self) -> None:
        a = tone(seconds=0.2, fs=8000.0)
        b = tone(seconds=0.2, fs=16000.0)
        out, fs, start_b = sequential_signal(a, 8000.0, b, 16000.0, gap_s=0.1)
        self.assertEqual(fs, 16000)
        self.assertAlmostEqual(start_b, 0.2 + 0.1)
        self.assertEqual(len(out), int(0.2 * 16000) + int(0.1 * 16000)
                         + int(0.2 * 16000))

    def test_empty_or_negative_gap_raises(self) -> None:
        with self.assertRaises(ValueError):
            sequential_signal([], FS, tone(), FS)
        with self.assertRaises(ValueError):
            sequential_signal(tone(), FS, tone(), FS, gap_s=-1.0)


if __name__ == "__main__":
    unittest.main()
