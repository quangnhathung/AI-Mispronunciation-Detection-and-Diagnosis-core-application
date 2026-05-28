from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from loguru import logger
from torch.utils.data import DataLoader

from src.datasets.collator import Collator
from src.datasets.l2arctic_dataset import L2ArcticDataset
from src.datasets.tokenizer import PhonemeTokenizer
from src.evaluators.evaluator import Evaluator
from src.features.mel_spec import MelFeatureExtractor
from src.models.cnn_bilstm_ctc import CNNBiLSTMCTC
from src.utils.config import Config
from src.utils.helpers import load_checkpoint
from src.visualization.plots import ConfusionPlotter


class EvalPipeline:
    def __init__(self, config: Config, checkpoint_path: str):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(
            getattr(config.inference, "device", "cuda")
            if torch.cuda.is_available()
            else "cpu"
        )
        logger.info(f"Using device: {self.device}")

    def run(self) -> Dict[str, Any]:
        tokenizer = PhonemeTokenizer(include_stress=True)

        self.config.model.vocab_size = tokenizer.vocab_size

        model = CNNBiLSTMCTC(
            input_dim=self.config.model.input_dim,
            vocab_size=tokenizer.vocab_size,
            cnn_channels=self.config.model.cnn_channels,
            cnn_kernel_sizes=self.config.model.cnn_kernel_sizes,
            cnn_strides=self.config.model.cnn_strides,
            cnn_dropout=self.config.model.cnn_dropout,
            rnn_hidden_size=self.config.model.rnn_hidden_size,
            rnn_num_layers=self.config.model.rnn_num_layers,
            rnn_dropout=self.config.model.rnn_dropout,
            rnn_bidirectional=self.config.model.rnn_bidirectional,
        ).to(self.device)

        load_checkpoint(self.checkpoint_path, model, device=self.device)
        logger.info(f"Loaded checkpoint from {self.checkpoint_path}")

        feature_fn = MelFeatureExtractor(
            sample_rate=self.config.data.sample_rate,
            n_fft=self.config.data.n_fft,
            win_length=self.config.data.win_length,
            hop_length=self.config.data.hop_length,
            n_mels=self.config.data.n_mels,
        )

        manifest_dir = Path(self.config.data.manifest_dir)
        test_manifest = manifest_dir / "test.jsonl"

        if not test_manifest.exists():
            test_manifest = manifest_dir / "val.jsonl"

        test_dataset = L2ArcticDataset(
            manifest_path=str(test_manifest),
            tokenizer=tokenizer,
            sample_rate=self.config.data.sample_rate,
            max_audio_length=self.config.data.max_audio_length,
            min_audio_length=self.config.data.min_audio_length,
            feature_fn=feature_fn,
            augmentation_fn=None,
            cache_dir=self.config.data.cache_dir,
            use_cache=False,
        )

        collator = Collator(pad_token_id=tokenizer.blank_id)
        test_loader = DataLoader(
            test_dataset,
            batch_size=getattr(self.config.inference, "batch_size", 32),
            shuffle=False,
            num_workers=getattr(self.config.inference, "num_workers", 2),
            collate_fn=collator,
        )

        evaluator = Evaluator(
            model=model,
            test_loader=test_loader,
            tokenizer=tokenizer,
            config=self.config,
            device=self.device,
        )

        results = evaluator.evaluate()

        logger.info(f"Evaluation Results:")
        logger.info(f"  PER: {results['per']:.4f}")
        logger.info(f"  CER: {results['cer']:.4f}")
        logger.info(f"  Confusion Accuracy: {results['confusion_accuracy']:.4f}")
        logger.info(f"  Samples: {results['num_samples']}")

        output_dir = Path("./outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        evaluator.export_predictions(str(output_dir / "predictions.csv"), results)
        evaluator.export_confusion_report(str(output_dir / "confusion_report.csv"), results)

        plotter = ConfusionPlotter()
        id_to_phoneme = {i: p for i, p in enumerate(tokenizer.vocab)}
        plot_path = plotter.plot(
            matrix=results.get("confusion_matrix", [[]]),
            id_to_phoneme=id_to_phoneme,
            top_k=30,
        )
        if plot_path:
            logger.info(f"Confusion matrix plot saved to {plot_path}")

        return results
