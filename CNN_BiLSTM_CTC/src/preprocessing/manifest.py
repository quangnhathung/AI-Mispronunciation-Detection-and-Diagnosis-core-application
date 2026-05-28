from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

from src.datasets.l2arctic_parser import L2ArcticParser


class ManifestBuilder:
    def __init__(
        self,
        data_dir: str,
        manifest_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.manifest_dir = Path(manifest_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.parser = L2ArcticParser(str(self.data_dir))

        self.manifest_dir.mkdir(parents=True, exist_ok=True)

    def build_all(self) -> Dict[str, str]:
        all_manifest = self.manifest_dir / "all.jsonl"
        logger.info("Building full manifest...")
        all_utterances = self.parser.build_manifest(str(all_manifest))
        logger.info(f"Total utterances: {len(all_utterances)}")

        random.seed(self.seed)
        random.shuffle(all_utterances)

        total = len(all_utterances)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)

        train_utts = all_utterances[:train_end]
        val_utts = all_utterances[train_end:val_end]
        test_utts = all_utterances[val_end:]

        splits = {
            "train": train_utts,
            "val": val_utts,
            "test": test_utts,
        }

        saved_paths = {}
        for split_name, utts in splits.items():
            split_path = self.manifest_dir / f"{split_name}.jsonl"
            with open(split_path, "w", encoding="utf-8") as f:
                for utt in utts:
                    serializable = {}
                    for k, v in utt.items():
                        if k == "phoneme_timings":
                            serializable[k] = [
                                {"phoneme": p[0], "start": p[1], "end": p[2]}
                                for p in v
                            ]
                        elif isinstance(v, (str, int, float, list, dict)):
                            serializable[k] = v
                    f.write(json.dumps(serializable) + "\n")
            saved_paths[split_name] = str(split_path)
            logger.info(f"{split_name}: {len(utts)} utterances -> {split_path}")

        return saved_paths

    def build_speaker_split(
        self,
        train_speakers: Optional[List[str]] = None,
        val_speakers: Optional[List[str]] = None,
        test_speakers: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        if train_speakers is None:
            all_speakers = self.parser.get_speakers()
            random.seed(self.seed)
            random.shuffle(all_speakers)
            total = len(all_speakers)
            n_train = max(1, int(total * self.train_ratio))
            n_val = max(1, int(total * self.val_ratio))
            train_speakers = all_speakers[:n_train]
            val_speakers = all_speakers[n_train:n_train + n_val]
            test_speakers = all_speakers[n_train + n_val:]

        split_map = {"train": {}, "val": {}, "test": {}}
        for spk in train_speakers:
            split_map["train"][spk] = self.parser.load_all_utterances(spk)
        for spk in val_speakers:
            split_map["val"][spk] = self.parser.load_all_utterances(spk)
        for spk in test_speakers:
            split_map["test"][spk] = self.parser.load_all_utterances(spk)

        saved_paths = {}
        for split_name in ["train", "val", "test"]:
            split_path = self.manifest_dir / f"{split_name}.jsonl"
            utt_count = 0
            with open(split_path, "w", encoding="utf-8") as f:
                for spk, utts in split_map[split_name].items():
                    for utt in utts:
                        serializable = {}
                        for k, v in utt.items():
                            if k == "phoneme_timings":
                                serializable[k] = [
                                    {"phoneme": p[0], "start": p[1], "end": p[2]}
                                    for p in v
                                ]
                            elif isinstance(v, (str, int, float, list, dict)):
                                serializable[k] = v
                        f.write(json.dumps(serializable) + "\n")
                        utt_count += 1
            saved_paths[split_name] = str(split_path)
            logger.info(
                f"{split_name}: {len(split_map[split_name])} speakers, "
                f"{utt_count} utterances -> {split_path}"
            )

        return saved_paths
