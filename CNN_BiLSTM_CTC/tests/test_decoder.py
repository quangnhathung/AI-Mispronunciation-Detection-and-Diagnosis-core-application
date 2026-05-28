from __future__ import annotations

import unittest

import torch

from src.decoders.greedy import GreedyDecoder
from src.decoders.beam_search import BeamSearchDecoder


class TestGreedyDecoder(unittest.TestCase):
    def setUp(self):
        self.decoder = GreedyDecoder(blank_id=0)

    def test_decode_simple(self):
        B, T, V = 1, 10, 5
        log_probs = torch.randn(B, T, V)
        result = self.decoder.decode(log_probs)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), B)
        self.assertIsInstance(result[0], list)

    def test_collapse_blanks(self):
        log_probs = torch.full((1, 8, 4), -10.0)
        log_probs[0, 0, 0] = 0.0
        log_probs[0, 1, 1] = 0.0
        log_probs[0, 2, 1] = 0.0
        log_probs[0, 3, 0] = 0.0
        log_probs[0, 4, 2] = 0.0
        log_probs[0, 5, 0] = 0.0
        log_probs[0, 6, 3] = 0.0
        log_probs[0, 7, 0] = 0.0
        result = self.decoder.decode(log_probs)
        self.assertEqual(result[0], [1, 2, 3])

    def test_decode_with_confidence(self):
        B, T, V = 2, 5, 4
        log_probs = torch.full((B, T, V), -10.0)
        log_probs[0, 0, 1] = 0.0
        log_probs[1, 0, 0] = 0.0
        log_probs[1, 1, 2] = 0.0
        log_probs[1, 2, 0] = 0.0
        results = self.decoder.decode_with_confidence(log_probs)
        self.assertEqual(len(results), B)


class TestBeamSearchDecoder(unittest.TestCase):
    def setUp(self):
        self.decoder = BeamSearchDecoder(blank_id=0, beam_width=5)

    def test_decode(self):
        B, T, V = 1, 8, 5
        log_probs = torch.randn(B, T, V)
        result = self.decoder.decode(log_probs)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), B)

    def test_decode_with_confidence(self):
        B, T, V = 1, 5, 4
        log_probs = torch.randn(B, T, V)
        results = self.decoder.decode_with_confidence(log_probs)
        self.assertEqual(len(results), B)


if __name__ == "__main__":
    unittest.main()
