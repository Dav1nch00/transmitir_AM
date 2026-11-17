"""Pruebas de las constelaciones teóricas de PCM/PAM y técnicas digitales."""
from __future__ import annotations

import unittest

import numpy as np

from core.constellation import (ALLOWED_BITS, TECHNIQUES, ConstellationError,
                                constellation, pcm_word)


class TestConstruction(unittest.TestCase):
    def test_all_techniques_build(self) -> None:
        for technique in TECHNIQUES:
            for bps in ALLOWED_BITS[technique]:
                with self.subTest(technique=technique, bits=bps):
                    c = constellation(technique, bps)
                    self.assertEqual(c.m, 1 << bps)
                    self.assertEqual(len(c.labels), c.m)
                    self.assertEqual(c.points.size, c.m)
                    self.assertTrue(np.all(np.isfinite(c.points)))
                    self.assertEqual(len(c.rows()), c.m)

    def test_aliases(self) -> None:
        self.assertEqual(constellation("pcm", 3).technique, "PCM (PAM)")
        self.assertEqual(constellation("PAM", 3).technique, "PCM (PAM)")

    def test_labels_msb_first(self) -> None:
        c = constellation("PSK", 3)
        self.assertEqual(c.labels[:3], ("000", "001", "010"))
        self.assertEqual(c.labels[-1], "111")

    def test_unknown_technique_raises(self) -> None:
        with self.assertRaises(ConstellationError):
            constellation("OFDM", 2)

    def test_qam_rejects_non_square_bits(self) -> None:
        with self.assertRaises(ConstellationError):
            constellation("QAM", 3)
        with self.assertRaises(ConstellationError):
            constellation("QAM", 5)


class TestMappings(unittest.TestCase):
    def test_ask_is_unipolar(self) -> None:
        c = constellation("ASK", 2)
        self.assertAlmostEqual(float(np.min(c.i_values)), 0.0)
        self.assertAlmostEqual(float(np.max(c.i_values)), 1.0)
        self.assertTrue(np.allclose(c.q_values, 0.0))
        self.assertAlmostEqual(c.min_distance, 1.0 / 3.0)

    def test_psk_uses_unit_circle(self) -> None:
        for bps in (1, 2, 3, 4):
            c = constellation("PSK", bps)
            self.assertTrue(np.allclose(np.abs(c.points), 1.0))
            self.assertAlmostEqual(c.mean_energy, 1.0)
        c = constellation("PSK", 2)
        self.assertAlmostEqual(c.min_distance, np.sqrt(2.0), places=9)
        self.assertAlmostEqual(c.dmin_normalized, np.sqrt(2.0), places=9)
        c = constellation("PSK", 4)
        self.assertAlmostEqual(c.min_distance, 2.0 * np.sin(np.pi / 16.0),
                               places=9)

    def test_pam_is_centered_and_symmetric(self) -> None:
        c = constellation("PCM (PAM)", 3)
        self.assertEqual(c.i_values.size, 8)
        self.assertAlmostEqual(float(c.i_values[0]), -1.0)
        self.assertAlmostEqual(float(c.i_values[-1]), 1.0)
        self.assertTrue(np.allclose(c.i_values, -c.i_values[::-1]))
        self.assertTrue(np.allclose(c.q_values, 0.0))

    def test_pam_4_levels(self) -> None:
        c = constellation("PCM (PAM)", 2)
        self.assertTrue(np.allclose(c.i_values, [-1.0, -1.0 / 3.0,
                                                 1.0 / 3.0, 1.0]))

    def test_qam_16_grid(self) -> None:
        c = constellation("QAM", 4)
        self.assertEqual(c.m, 16)
        expected = np.array([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0])
        self.assertTrue(np.allclose(np.unique(np.round(c.i_values, 9)),
                                    expected))
        self.assertTrue(np.allclose(np.unique(np.round(c.q_values, 9)),
                                    expected))
        self.assertAlmostEqual(c.min_distance, 2.0 / 3.0, places=9)
        self.assertAlmostEqual(c.mean_energy, 10.0 / 9.0, places=9)

    def test_qam_4_has_maximum_papr(self) -> None:
        c = constellation("QAM", 2)
        self.assertAlmostEqual(c.peak_power, 2.0)
        self.assertAlmostEqual(c.mean_energy, 2.0)
        self.assertAlmostEqual(c.papr_db, 0.0, places=9)

    def test_fsk_uses_frequency_axis(self) -> None:
        for technique in ("FSK", "MFSK", "GFSK"):
            c = constellation(technique, 3)
            self.assertFalse(c.is_iq)
            self.assertEqual(c.kind, "fsk")
            self.assertAlmostEqual(float(c.frequencies[0]), -1.0)
            self.assertAlmostEqual(float(c.frequencies[-1]), 1.0)
            self.assertTrue(np.all(np.diff(c.frequencies) > 0.0))
            self.assertAlmostEqual(c.min_distance, 2.0 / 7.0, places=9)
            self.assertTrue(np.allclose(c.q_values, 0.0))

    def test_rows_include_frequency_for_fsk(self) -> None:
        c = constellation("FSK", 2)
        row = c.rows()[1]
        self.assertEqual(row[1], "01")
        self.assertAlmostEqual(float(row[2]), -1.0 / 3.0, places=4)
        self.assertEqual(row[3], "—")

    def test_rows_include_iq_for_psk(self) -> None:
        c = constellation("PSK", 2)
        row = c.rows()[1]
        self.assertEqual(row[1], "01")
        self.assertAlmostEqual(float(row[2]), 0.0, places=6)
        self.assertAlmostEqual(float(row[3]), 1.0, places=6)


class TestPCMWord(unittest.TestCase):
    def test_five_bit_word_maps_to_point_and_adc(self) -> None:
        w = pcm_word("10110", 5)
        self.assertEqual(w.word, "10110")
        self.assertEqual(w.code, 22)
        self.assertEqual(w.signed_code, 6)
        self.assertAlmostEqual(w.level, (22 - 15.5) / 15.5)
        self.assertAlmostEqual(w.volts, 6.0 * 2.0 / 32.0)
        c = constellation("PCM (PAM)", 5)
        self.assertAlmostEqual(w.level, float(c.i_values[22].real))

    def test_boundaries(self) -> None:
        low = pcm_word("00000", 5)
        high = pcm_word("11111", 5)
        self.assertAlmostEqual(low.level, -1.0)
        self.assertAlmostEqual(high.level, 1.0)
        self.assertEqual(low.signed_code, -16)
        self.assertEqual(high.signed_code, 15)
        self.assertAlmostEqual(low.volts, -1.0)
        self.assertAlmostEqual(high.volts, 0.9375)

    def test_integer_word_is_masked(self) -> None:
        w = pcm_word(22, 5)
        self.assertEqual(w.word, "10110")
        self.assertEqual(w.code, 22)
        self.assertEqual(pcm_word(0b110110, 5).word, "10110")

    def test_vref_scales_voltage(self) -> None:
        w = pcm_word("10110", 5, vref=2.0)
        self.assertAlmostEqual(w.volts, 0.75)

    def test_invalid_words_raise(self) -> None:
        for word in ("1011", "101100", "10210", ""):
            with self.subTest(word=word):
                with self.assertRaises(ConstellationError):
                    pcm_word(word, 5)
        with self.assertRaises(ConstellationError):
            pcm_word("10110", 9)
        with self.assertRaises(ConstellationError):
            pcm_word("10110", 5, vref=0.0)


if __name__ == "__main__":
    unittest.main()
