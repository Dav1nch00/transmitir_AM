"""Pruebas de FM y de las demodulaciones digitales, sin abrir la interfaz."""
from __future__ import annotations

import unittest

import numpy as np

from core.modulation import (DIGITAL_TECHNIQUES, ModulationError, bits_from_codes,
                             demodulate_digital, demodulate_fm, modulate_digital,
                             modulate_fm)

FS = 44100.0
CARRIER = 10000.0
BAUD = 2000.0

BPS_BY_TECHNIQUE = {
    "ASK": 2,
    "FSK": 2,
    "PSK": 2,
    "QAM": 4,
    "GFSK": 2,
    "MFSK": 3,
}


def random_bits(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2, n).astype(np.uint8)


def modulation_kwargs(technique: str) -> dict:
    return {
        "deviation": 3000.0,
        "bt": 0.5 if technique == "GFSK" else 0.3,
    }


class TestFMDemodulation(unittest.TestCase):
    def test_fm_round_trip_recovers_the_tone(self) -> None:
        fs = FS
        t = np.arange(int(fs * 0.35)) / fs
        x = 0.6 * np.sin(2.0 * np.pi * 440.0 * t) + \
            0.3 * np.sin(2.0 * np.pi * 880.0 * t)
        y = modulate_fm(x, fs, carrier_freq=CARRIER, deviation=3000.0)
        z, fs_out = demodulate_fm(y, fs, carrier_freq=CARRIER,
                                  deviation=3000.0, cutoff_freq=4000.0)
        self.assertEqual(fs_out, fs)
        margin = int(round(0.02 * fs))
        xx = x[margin:len(x) - margin]
        zz = z[:len(xx)]
        n = min(len(xx), len(zz))
        corr = float(np.corrcoef(xx[:n], zz[:n])[0, 1])
        self.assertGreater(corr, 0.90)

    def test_fm_invalid_frequency_raises(self) -> None:
        with self.assertRaises(ModulationError):
            demodulate_fm(np.ones(1000), FS, carrier_freq=30000.0,
                          deviation=3000.0)
        with self.assertRaises(ModulationError):
            demodulate_fm(np.ones(1000), FS, carrier_freq=CARRIER,
                          deviation=0.0)


class TestDigitalDemodulation(unittest.TestCase):
    def _round_trip(self, technique: str, bits: np.ndarray, find_start: bool,
                    bps: int | None = None):
        bps = BPS_BY_TECHNIQUE[technique] if bps is None else bps
        kwargs = modulation_kwargs(technique)
        mod = modulate_digital(technique, bits, fs=FS, carrier_freq=CARRIER,
                               symbol_rate=BAUD, bits_per_symbol=bps, **kwargs)
        demod = demodulate_digital(
            technique, mod.signal, FS, CARRIER, BAUD, bps, 8,
            audio_fs=16000.0, find_start=find_start, **kwargs)
        return demod

    def test_all_techniques_round_trip_without_burst_search(self) -> None:
        bits = random_bits(120, seed=11)
        for technique in DIGITAL_TECHNIQUES:
            with self.subTest(technique=technique):
                demod = self._round_trip(technique, bits, find_start=False)
                self.assertEqual(demod.bits.size, bits.size)
                np.testing.assert_array_equal(demod.bits, bits)
                self.assertLess(demod.error, 0.6)

    def test_constant_envelope_techniques_find_the_burst(self) -> None:
        bits = random_bits(120, seed=5)
        for technique in ("FSK", "PSK", "MFSK"):
            with self.subTest(technique=technique):
                demod = self._round_trip(technique, bits, find_start=True)
                np.testing.assert_array_equal(demod.bits, bits)

    def test_pcm_codes_are_reconstructed_exactly(self) -> None:
        codes = np.array([0, 1, 2, 127, 128, 129, 200, 254, 255,
                          64, 192, 32, 96, 160, 224, 17], dtype=np.uint32)
        bits = bits_from_codes(codes, 8)
        for technique in DIGITAL_TECHNIQUES:
            with self.subTest(technique=technique):
                demod = self._round_trip(technique, bits, find_start=False)
                half = 128.0
                expected = (codes.astype(np.float64) - half) / half
                n = min(expected.size, demod.signal.size)
                self.assertEqual(n, codes.size)
                np.testing.assert_allclose(demod.signal[:n], expected[:n],
                                           atol=1e-12)

    def test_audio_rate_is_preserved(self) -> None:
        bits = random_bits(64, seed=2)
        demod = self._round_trip("PSK", bits, find_start=False)
        self.assertEqual(demod.fs, 16000)

    def test_empty_signal_raises(self) -> None:
        with self.assertRaises(ModulationError):
            demodulate_digital("PSK", np.zeros(0), FS, CARRIER, BAUD, 2, 8)

    def test_too_fast_symbol_rate_raises(self) -> None:
        with self.assertRaises(ModulationError):
            demodulate_digital("PSK", np.ones(1000), FS, CARRIER,
                               symbol_rate=20000.0, bits_per_symbol=2,
                               pcm_bits=8)

    def test_not_enough_bits_for_a_pcm_sample_raises(self) -> None:
        bits = np.array([0, 1], dtype=np.uint8)
        mod = modulate_digital("PSK", bits, fs=FS, carrier_freq=CARRIER,
                               symbol_rate=BAUD, bits_per_symbol=2)
        with self.assertRaises(ModulationError):
            demodulate_digital("PSK", mod.signal, FS, CARRIER, BAUD, 2, 16,
                               find_start=False)


if __name__ == "__main__":
    unittest.main()
