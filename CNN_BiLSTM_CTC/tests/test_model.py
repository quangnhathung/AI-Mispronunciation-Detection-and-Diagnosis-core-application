from __future__ import annotations

import unittest

import torch

from src.models.cnn_bilstm_ctc import CNNBiLSTMCTC
from src.losses.ctc_loss import CTCLossWrapper


class TestCNNBiLSTMCTC(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 42
        self.input_dim = 80
        self.model = CNNBiLSTMCTC(
            input_dim=self.input_dim,
            vocab_size=self.vocab_size,
            cnn_channels=[16, 32],
            cnn_kernel_sizes=[3, 3],
            cnn_strides=[2, 2],
            cnn_dropout=0.1,
            rnn_hidden_size=64,
            rnn_num_layers=2,
            rnn_dropout=0.1,
            rnn_bidirectional=True,
        )

    def test_forward_shape(self):
        B, F, T = 2, 80, 100
        x = torch.randn(B, F, T)
        output = self.model(x)
        expected_T = self.model.get_feat_lengths(torch.tensor([T]))[0].item()
        self.assertEqual(output.shape, (B, expected_T, self.vocab_size))

    def test_forward_with_lengths(self):
        B, F, T = 2, 80, 100
        x = torch.randn(B, F, T)
        lengths = torch.tensor([100, 80])
        output = self.model(x, lengths)
        expected_T = self.model.get_feat_lengths(torch.tensor([T]))[0].item()
        self.assertEqual(output.shape, (B, expected_T, self.vocab_size))

    def test_output_log_softmax(self):
        B, F, T = 1, 80, 50
        x = torch.randn(B, F, T)
        output = self.model(x)
        output_T = output.size(1)
        self.assertTrue(torch.allclose(output.exp().sum(dim=-1), torch.ones(B, output_T), atol=1e-6))

    def test_feat_lengths(self):
        audio_lengths = torch.tensor([1600, 800])
        feat_lengths = self.model.get_feat_lengths(audio_lengths)
        self.assertTrue(torch.all(feat_lengths > 0))

    def test_ctc_loss_compatible(self):
        B, F, T = 2, 80, 100
        x = torch.randn(B, F, T)
        log_probs = self.model(x)
        input_lengths = self.model.get_feat_lengths(torch.tensor([100, 80]))
        targets = torch.randint(1, self.vocab_size, (2, 20))
        target_lengths = torch.tensor([15, 10])

        loss_fn = CTCLossWrapper(blank_id=0)
        loss = loss_fn(log_probs, targets, input_lengths, target_lengths)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

    def test_count_params(self):
        params = self.model.count_params()
        self.assertIn("total", params)
        self.assertIn("trainable", params)
        self.assertGreater(params["total"], 0)


if __name__ == "__main__":
    unittest.main()
