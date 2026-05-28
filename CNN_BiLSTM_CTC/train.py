from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from src.pipelines.train_pipeline import TrainPipeline
from src.utils.config import Config, get_config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CNN-BiLSTM-CTC model for L2-ARCTIC MDD"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/config.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override data directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda/cpu)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
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
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.optimizer.lr = args.lr
    if args.device:
        config.training.device = args.device
    if args.resume:
        config.training.resume_from = args.resume

    set_seed(getattr(args, "seed", 42))
    setup_logger(
        log_dir=config.logging.log_dir,
        level=config.logging.log_level,
    )

    logger.info(f"Config: {config_path}")
    logger.info(f"Training config: epochs={config.training.epochs}, "
                f"batch_size={config.training.batch_size}, "
                f"lr={config.optimizer.lr}")

    pipeline = TrainPipeline(config)
    results = pipeline.run()

    logger.info("Training completed successfully")
    final_metrics = results["history"]
    logger.info(
        f"Final train loss: {final_metrics['train_loss'][-1]:.4f} | "
        f"Final val loss: {final_metrics['val_loss'][-1]:.4f} | "
        f"Final val PER: {final_metrics['val_per'][-1]:.4f}"
    )


if __name__ == "__main__":
    main()
