from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from loguru import logger

from src.pipelines.infer_pipeline import InferPipeline
from src.utils.config import Config
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with trained CNN-BiLSTM-CTC model"
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
        "--audio",
        type=str,
        default=None,
        help="Path to audio file for inference",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default=None,
        help="Directory with audio files for batch inference",
    )
    parser.add_argument(
        "--target_phonemes",
        type=str,
        default=None,
        nargs="+",
        help="Ground truth phoneme sequence for MDD evaluation",
    )
    parser.add_argument(
        "--microphone",
        action="store_true",
        help="Use microphone for real-time inference",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Recording duration in seconds (microphone mode)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda/cpu)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    config = Config.from_yaml(str(config_path))

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

    pipeline = InferPipeline(config, str(checkpoint_path))

    results = None

    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            sys.exit(1)

        if args.target_phonemes:
            result = pipeline.infer_with_ground_truth(
                audio_path=str(audio_path),
                target_phonemes=args.target_phonemes,
                utterance_id=audio_path.stem,
                speaker="unknown",
            )
            results = result
        else:
            result = pipeline.infer_file(str(audio_path))
            print(f"\nPredicted Phonemes: {result['phoneme_string']}")
            print(f"Confidence: {result['confidence']:.4f}")
            results = result

    elif args.audio_dir:
        results = pipeline.batch_infer(args.audio_dir)
        print(f"\nBatch Inference Results ({len(results)} files):")
        for r in results:
            print(f"  {r.get('file', 'unknown')}: {r.get('phoneme_string', 'N/A')}")

    elif args.microphone:
        result = pipeline.infer_microphone(duration=args.duration)
        results = result

    else:
        print("No input specified. Use --audio, --audio_dir, or --microphone")
        parser.print_help()
        sys.exit(1)

    if args.output and results:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
