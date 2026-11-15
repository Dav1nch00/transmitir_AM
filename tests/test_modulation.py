"""Pruebas de FM y de las modulaciones digitales, sin abrir la interfaz."""
from __future__ import annotations

import unittest

import numpy as np

from core.modulation import (ModulationError, bits_from_codes, modulate_digital,
                             modulate_fm)


def alternating_bits(n: int) -> np.ndarray:
    return (np.arange(n) % 2).astype(np.uint8)


class TestBits(unittest.TestCase):
    def test_codes_to_bits_msb_first(self) -> None:
        codes = np.array([0, 1, 2, 3], dtype=np.uint32)
        bits = bits_from_codes(codes, 2)
        self.assertEqual(bits.tolist(), [0, 0, 0, 1, 1, 0, 1, 1])

    def test_max_bits_truncates(self) -> None:
        codes = np.array([0b1010, 0b0101], dtype=np.uint32)
        bits = bits_from_codes(codes, 4, max_bits=5)
        self.assertEqual(bits.tolist(), [1, 0, 1, 0, 0])

    def test_zero_bits(self) -> None:
        self.assertEqual(bits_from_codes(np.zeros(0), 8).size, 0)
        codes = np.array([255], dtype=np.uint32)
        self.assertEqual(bits_from_codes(codes, 8, max_bits=0).size, 0)


class TestFM(unittest.TestCase):
    def test_fm_length_and_peak(self) -> None:
        fs = 44100.0
        t = np.arange(int(fs * 0.25)) / fs
        x = 0.7 * np.sin(2.0 * np.pi * 300.0 * t)
        y = modulate_fm(x, fs, carrier_freq=10000.0, deviation=2000.0)
        self.assertEqual(len(y), len(x))
        self.assertLessEqual(float(np.max(np.abs(y))), 1.0 + 1e-12)
        self.assertTrue(np.all(np.isfinite(y)))

    def test_invalid_deviation_raises(self) -> None:
        x = np.ones(1000)
        with self.assertRaises(ModulationError):
            modulate_fm(x, 44100.0, carrier_freq=1000.0, deviation=5000.0)
        with self.assertRaises(ModulationError):
            modulate_fm(x, 44100.0, carrier_freq=10000.0, deviation=0.0)


class TestDigital(unittest.TestCase):
    def _modulate(self, technique: str, bps: int = 2, **kwargs):
        bits = alternating_bits(64)
        return modulate_digital(technique, bits, fs=44100.0,
                                carrier_freq=10000.0, symbol_rate=2000.0,
                                bits_per_symbol=bps, **kwargs)

    def test_all_techniques_generate_valid_signals(self) -> None:
        for technique in ("ASK", "FSK", "PSK", "QAM", "GFSK", "MFSK"):
            with self.subTest(technique=technique):
                result = self._modulate(technique, deviation=1500.0, bt=0.3)
                self.assertGreater(result.signal.size, 0)
                self.assertTrue(np.all(np.isfinite(result.signal)))
                self.assertLessEqual(float(np.max(np.abs(result.signal))),
                                     1.0 + 1e-12)
                self.assertEqual(result.technique, technique)
                self.assertEqual(result.bits_used, 64)
                self.assertEqual(len(result.base), len(result.signal))

    def test_symbol_count_and_duration(self) -> None:
        result = self._modulate("PSK", bps=2)
        # 64 bits = 32 símbolos a 2000 baudios → 16 ms.
        self.assertAlmostEqual(result.duration, 32 / 2000.0, places=3)

    def test_qam_rejects_non_square_constellation(self) -> None:
        with self.assertRaises(ModulationError):
            self._modulate("QAM", bps=3)

    def test_too_fast_symbol_rate_raises(self) -> None:
        with self.assertRaises(ModulationError):
            modulate_digital("ASK", alternating_bits(20), fs=44100.0,
                             carrier_freq=10000.0, symbol_rate=20000.0,
                             bits_per_symbol=1)

    def test_bad_frequency_range_raises(self) -> None:
        with self.assertRaises(ModulationError):
            modulate_digital("FSK", alternating_bits(20), fs=44100.0,
                             carrier_freq=10000.0, symbol_rate=2000.0,
                             bits_per_symbol=1, deviation=12000.0)

    def test_empty_bits_raise(self) -> None:
        with self.assertRaises(ModulationError):
            modulate_digital("ASK", np.zeros(0), fs=44100.0,
                             carrier_freq=10000.0, symbol_rate=2000.0,
                             bits_per_symbol=1)


if __name__ == "__main__":
    unittest.main()
