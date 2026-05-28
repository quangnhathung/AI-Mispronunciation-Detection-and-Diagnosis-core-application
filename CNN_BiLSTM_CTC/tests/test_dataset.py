from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import torch

from src.datasets.tokenizer import PhonemeTokenizer
from src.datasets.collator import Collator


class TestCollator(unittest.TestCase):
    def setUp(self):
        self.tokenizer = PhonemeTokenizer(include_stress=True)
        self.collator = Collator(pad_token_id=self.tokenizer.blank_id)

    def test_collate_basic(self):
        batch = [
            {"audio": torch.randn(1, 5), "phonemes": torch.tensor([1, 2, 3]),
             "audio_length": 5, "feature_length": 5, "phoneme_length": 3},
            {"audio": torch.randn(1, 3), "phonemes": torch.tensor([4, 5]),
             "audio_length": 3, "feature_length": 3, "phoneme_length": 2},
        ]
        collated = self.collator(batch)
        self.assertIn("audio", collated)
        self.assertIn("phonemes", collated)
        self.assertIn("audio_lengths", collated)
        self.assertIn("phoneme_lengths", collated)
        self.assertEqual(collated["phonemes"].shape[0], 2)
        self.assertEqual(collated["phonemes"].shape[1], 3)

    def test_collate_empty(self):
        batch = []
        with self.assertRaises(Exception):
            self.collator(batch)

    def test_collate_padding(self):
        batch = [
            {"audio": torch.randn(1, 5), "phonemes": torch.tensor([1, 2, 3]),
             "audio_length": 5, "feature_length": 5, "phoneme_length": 3},
            {"audio": torch.randn(1, 3), "phonemes": torch.tensor([4]),
             "audio_length": 3, "feature_length": 3, "phoneme_length": 1},
        ]
        collated = self.collator(batch)
        self.assertEqual(collated["phonemes"][1, 1].item(), self.tokenizer.blank_id)


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = PhonemeTokenizer(include_stress=True)

    def test_round_trip(self):
        phonemes = ["HH", "AH0", "L", "OW1"]
        ids = self.tokenizer.encode(phonemes)
        decoded = self.tokenizer.decode(ids)
        self.assertEqual(phonemes, decoded)

    def test_batch_encode(self):
        batch = [["AA", "B"], ["K", "S"]]
        encoded = self.tokenizer.encode_batch(batch)
        self.assertEqual(len(encoded), 2)


if __name__ == "__main__":
    unittest.main()
