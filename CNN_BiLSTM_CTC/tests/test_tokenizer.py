from __future__ import annotations

import unittest

import torch

from src.datasets.tokenizer import PhonemeTokenizer


class TestPhonemeTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = PhonemeTokenizer(include_stress=True)

    def test_vocab_size(self):
        self.assertGreater(len(self.tokenizer), 40)

    def test_blank_id(self):
        self.assertEqual(self.tokenizer.blank_id, 0)

    def test_unk_id(self):
        self.assertEqual(self.tokenizer.unk_id, 1)

    def test_encode_decode(self):
        phonemes = ["B", "K", "S", "HH", "L"]
        ids = self.tokenizer.encode(phonemes)
        decoded = self.tokenizer.decode(ids)
        self.assertEqual(phonemes, decoded)

    def test_encode_stressed(self):
        phonemes = ["AA1", "IY0", "ER2"]
        ids = self.tokenizer.encode(phonemes)
        decoded = self.tokenizer.decode(ids)
        self.assertEqual(phonemes, decoded)

    def test_encode_unknown(self):
        phonemes = ["ZZZ"]
        ids = self.tokenizer.encode(phonemes)
        decoded = self.tokenizer.decode(ids)
        self.assertEqual(decoded, ["<unk>"])

    def test_round_trip_tensor(self):
        phonemes = ["HH", "AH0", "L", "OW1"]
        ids = self.tokenizer.encode(phonemes)
        self.assertIsInstance(ids, torch.Tensor)
        self.assertEqual(ids.dtype, torch.long)
        decoded = self.tokenizer.decode(ids)
        self.assertEqual(phonemes, decoded)

    def test_decode_to_string(self):
        phonemes = ["HH", "AH0", "L", "OW1"]
        ids = self.tokenizer.encode(phonemes)
        result = self.tokenizer.decode_to_string(ids)
        self.assertEqual(result, "HH AH0 L OW1")

    def test_blank_id_value(self):
        ids = self.tokenizer.encode(["<blank>"])
        self.assertEqual(ids[0].item(), 0)

    def test_vocab_no_stress(self):
        tokenizer = PhonemeTokenizer(include_stress=False)
        self.assertIn("AA", tokenizer.vocab)
        self.assertNotIn("AA1", tokenizer.vocab)


if __name__ == "__main__":
    unittest.main()
