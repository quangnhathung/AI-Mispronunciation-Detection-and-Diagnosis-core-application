from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from src.pipelines.eval_pipeline import EvalPipeline
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CNN-BiLSTM-CTC model on L2-ARCTIC"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/config.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override data directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda/cpu)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for evaluation results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    config = Config.from_yaml(str(config_path))

    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.device:
        config.inference.device = args.device

    setup_logger(
        log_dir=config.logging.log_dir,
        level=config.logging.log_level,
    )

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    logger.info(f"Config: {config_path}")
    logger.info(f"Checkpoint: {checkpoint_path}")

    pipeline = EvalPipeline(config, str(checkpoint_path))
    results = pipeline.run()

    logger.info(f"Evaluation Results:")
    logger.info(f"  PER: {results['per']:.4f} ({results['per'] * 100:.2f}%)")
    logger.info(f"  CER: {results['cer']:.4f} ({results['cer'] * 100:.2f}%)")
    logger.info(f"  Confusion Accuracy: {results['confusion_accuracy']:.4f}")
    logger.info(f"  F1 Macro: {results['f1_macro']:.4f} ({results['f1_macro']*100:.2f}%)")
    logger.info(f"  F1 Micro: {results['f1_micro']:.4f} ({results['f1_micro']*100:.2f}%)")
    logger.info(f"  Samples Evaluated: {results['num_samples']}")

    print(f"\n{'='*50}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"  PER (Phoneme Error Rate):  {results['per']:.4f} ({results['per']*100:.2f}%)")
    print(f"  CER (Character Error Rate): {results['cer']:.4f} ({results['cer']*100:.2f}%)")
    print(f"  Confusion Accuracy:         {results['confusion_accuracy']:.4f}")
    print(f"  F1 Macro:                  {results['f1_macro']:.4f} ({results['f1_macro']*100:.2f}%)")
    print(f"  F1 Micro:                  {results['f1_micro']:.4f} ({results['f1_micro']*100:.2f}%)")
    print(f"  Total Samples:              {results['num_samples']}")
    print(f"{'='*50}")
    print(f"  Predictions:  ./outputs/predictions.csv")
    print(f"  Confusions:   ./outputs/confusion_report.csv")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
