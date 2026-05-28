from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.callbacks.early_stopping import EarlyStopping
from src.callbacks.model_checkpoint import ModelCheckpoint
from src.losses.ctc_loss import CTCLossWrapper
from src.metrics.per import PERMetric
from src.mdd.detector import MispronunciationDetector
from src.models.cnn_bilstm_ctc import CNNBiLSTMCTC


class TestEarlyStopping(unittest.TestCase):
    def test_early_stopping_min(self):
        es = EarlyStopping(patience=3, mode="min")
        self.assertFalse(es(10.0))
        self.assertFalse(es(9.0))
        self.assertFalse(es(8.0))
        self.assertFalse(es(8.0))
        self.assertFalse(es(8.0))
        self.assertTrue(es(8.0))

    def test_early_stopping_improving(self):
        es = EarlyStopping(patience=3, mode="min")
        self.assertFalse(es(10.0))
        self.assertFalse(es(5.0))
        self.assertFalse(es(3.0))
        self.assertFalse(es(2.0))
        self.assertFalse(es(1.0))
        self.assertFalse(es(1.0))
        self.assertFalse(es(1.0))
        self.assertFalse(es(0.5))
        self.assertFalse(es(0.5))
        self.assertFalse(es(0.5))


class TestModelCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ckpt = ModelCheckpoint(
            checkpoint_dir=self.tmpdir,
            monitor="val_loss",
            mode="min",
            save_top_k=2,
        )

    def test_save_checkpoint(self):
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        path = self.ckpt(model, optimizer, scheduler, 1, {"val_loss": 1.0}, None)
        self.assertIsNotNone(path)
        self.assertTrue(Path(str(path)).exists())
        self.assertTrue(Path(self.tmpdir, "best.pt").exists())


class TestPERMetric(unittest.TestCase):
    def setUp(self):
        self.metric = PERMetric(blank_id=0)

    def test_per_perfect(self):
        self.metric.update([[1, 2, 3]], [[1, 2, 3]])
        self.assertEqual(self.metric.compute(), 0.0)

    def test_per_worst(self):
        self.metric.update([[]], [[1, 2, 3]])
        self.assertGreater(self.metric.compute(), 0.0)


class TestMispronunciationDetector(unittest.TestCase):
    def setUp(self):
        from src.datasets.tokenizer import PhonemeTokenizer
        self.tokenizer = PhonemeTokenizer(include_stress=True)
        self.detector = MispronunciationDetector(self.tokenizer)

    def test_detect_correct(self):
        phonemes = ["HH", "AH0", "L", "OW1"]
        ids = self.tokenizer.encode(phonemes).tolist()
        feedback = self.detector.detect(ids, ids)
        self.assertEqual(feedback.per, 0.0)
        self.assertEqual(len(feedback.substitutions), 0)

    def test_detect_substitution(self):
        gt = ["HH", "AH0", "L", "OW1"]
        pred = ["HH", "AE1", "L", "OW1"]
        gt_ids = self.tokenizer.encode(gt).tolist()
        pred_ids = self.tokenizer.encode(pred).tolist()
        feedback = self.detector.detect(pred_ids, gt_ids)
        self.assertGreater(feedback.per, 0.0)
        self.assertEqual(len(feedback.substitutions), 1)

    def test_detect_deletion(self):
        gt = ["HH", "AH0", "L", "OW1"]
        pred = ["HH", "L", "OW1"]
        gt_ids = self.tokenizer.encode(gt).tolist()
        pred_ids = self.tokenizer.encode(pred).tolist()
        feedback = self.detector.detect(pred_ids, gt_ids)
        self.assertEqual(len(feedback.deletions), 1)

    def test_detect_insertion(self):
        gt = ["HH", "L", "OW1"]
        pred = ["HH", "AH0", "L", "OW1"]
        gt_ids = self.tokenizer.encode(gt).tolist()
        pred_ids = self.tokenizer.encode(pred).tolist()
        feedback = self.detector.detect(pred_ids, gt_ids)
        self.assertEqual(len(feedback.insertions), 1)


if __name__ == "__main__":
    unittest.main()
