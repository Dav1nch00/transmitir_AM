"""Pruebas del módulo PCM: palabras, trama serie y códigos de línea."""
from __future__ import annotations

import unittest

import numpy as np

from core.pcm import (LINE_CODES, PCMError, codes_from_bit_stream, frame_words,
                      line_waveform, pcm_bit_stream, pcm_metrics, word_bits)


class TestMetrics(unittest.TestCase):
    def test_bitrate_and_periods(self) -> None:
        m = pcm_metrics(8000.0, 8)
        self.assertEqual(m.bitrate, 64000.0)
        self.assertAlmostEqual(m.bit_period, 1.0 / 64000.0)
        self.assertAlmostEqual(m.word_period, 8.0 / 64000.0)
        self.assertAlmostEqual(m.bytes_per_second, 8000.0)
        self.assertEqual(m.levels, 256)

    def test_invalid_parameters(self) -> None:
        with self.assertRaises(PCMError):
            pcm_metrics(0.0, 8)
        with self.assertRaises(PCMError):
            pcm_metrics(8000.0, 0)
        with self.assertRaises(PCMError):
            pcm_metrics(8000.0, 25)


class TestBits(unittest.TestCase):
    def test_word_bits_msb_first(self) -> None:
        self.assertEqual(word_bits(0b1011, 4).tolist(), [1, 0, 1, 1])
        self.assertEqual(word_bits(0b00000010, 8).tolist(),
                         [0, 0, 0, 0, 0, 0, 1, 0])

    def test_word_bits_masks_extra_bits(self) -> None:
        self.assertEqual(word_bits(0b11011, 4).tolist(), [1, 0, 1, 1])

    def test_stream_roundtrip(self) -> None:
        codes = np.array([0, 1, 127, 128, 255], dtype=np.int64)
        stream = pcm_bit_stream(codes, 8)
        self.assertEqual(stream.size, 40)
        recovered = codes_from_bit_stream(stream, 8)
        self.assertEqual(recovered.tolist(), codes.tolist())

    def test_stream_truncates_incomplete_word(self) -> None:
        stream = np.array([1, 0, 1, 0, 0], dtype=np.uint8)
        self.assertEqual(codes_from_bit_stream(stream, 4).tolist(), [0b1010])

    def test_frame_words(self) -> None:
        words = frame_words(np.array([2, 15]), 4, count=8)
        self.assertEqual(words, [(2, "0010"), (15, "1111")])


class TestLineCodes(unittest.TestCase):
    def test_all_codes_have_expected_length(self) -> None:
        bits = np.array([1, 0, 1], dtype=np.uint8)
        for mode in LINE_CODES:
            with self.subTest(mode=mode):
                t, y = line_waveform(bits, samples_per_bit=4, mode=mode)
                self.assertEqual(t.size, 12)
                self.assertEqual(y.size, 12)
                self.assertTrue(np.all(np.isfinite(y)))

    def test_nrz_unipolar_levels(self) -> None:
        _t, y = line_waveform([1, 0], samples_per_bit=4, mode="NRZ unipolar")
        self.assertEqual(y.tolist(), [1, 1, 1, 1, 0, 0, 0, 0])

    def test_nrz_bipolar_levels(self) -> None:
        _t, y = line_waveform([1, 0], samples_per_bit=4, mode="NRZ bipolar")
        self.assertEqual(y.tolist(), [1, 1, 1, 1, -1, -1, -1, -1])

    def test_rz_unipolar_returns_to_zero(self) -> None:
        _t, y = line_waveform([1, 0], samples_per_bit=4, mode="RZ unipolar")
        self.assertEqual(y.tolist(), [1, 1, 0, 0, 0, 0, 0, 0])

    def test_manchester_transition_in_the_middle(self) -> None:
        _t, y = line_waveform([1, 0], samples_per_bit=4, mode="Manchester")
        self.assertEqual(y.tolist(), [1, 1, -1, -1, -1, -1, 1, 1])

    def test_empty_frame(self) -> None:
        t, y = line_waveform([], samples_per_bit=4, mode="NRZ bipolar")
        self.assertEqual(t.size, 0)
        self.assertEqual(y.size, 0)

    def test_unknown_code_raises(self) -> None:
        with self.assertRaises(PCMError):
            line_waveform([1, 0], 4, mode="AMI")


if __name__ == "__main__":
    unittest.main()
