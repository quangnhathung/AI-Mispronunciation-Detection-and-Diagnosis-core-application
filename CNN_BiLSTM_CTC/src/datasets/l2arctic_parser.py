from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from loguru import logger


_SPEAKER_META_FIELDS = ["accent", "gender", "native_language"]


def _parse_phoneme_line(content: str) -> List[Tuple[str, float, float]]:
    """Parse TIMIT-style .phn file (3 or 4 columns: [index] start end phoneme)."""
    items = []
    pattern = re.compile(
        r"^\s*(?:\d+\s+)?(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\S+)\s*$"
    )
    for single_line in content.strip().split("\n"):
        m = pattern.match(single_line)
        if m:
            start = float(m.group(1))
            end = float(m.group(2))
            phoneme = m.group(3).strip()
            items.append((phoneme, start, end))
    return items


def _parse_textgrid_phones(textgrid_path: Path) -> List[Tuple[str, float, float]]:
    """Parse Praat TextGrid file and extract the 'phones' interval tier."""
    items = []
    try:
        with open(textgrid_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return []

    phones_tier_match = re.search(
        r'item\s+\[\d+\]:\s*\n'
        r'\s+class\s*=\s*"IntervalTier"\s*\n'
        r'\s+name\s*=\s*"phones"\s*\n'
        r'.*?intervals:\s*size\s*=\s*(\d+)',
        content, re.DOTALL
    )
    if not phones_tier_match:
        return []

    # Only scan intervals WITHIN the phones tier block (from the matched header
    # to the start of the next item or end of file).
    phones_start = phones_tier_match.start()
    next_item = re.search(r'\nitem\s+\[\d+\]:', content[phones_start + 1:])
    if next_item:
        phones_section = content[phones_start:phones_start + 1 + next_item.start()]
    else:
        phones_section = content[phones_start:]

    interval_pattern = re.compile(
        r'intervals\s+\[\d+\]:\s*\n'
        r'\s+xmin\s*=\s*([\d.eE+-]+)\s*\n'
        r'\s+xmax\s*=\s*([\d.eE+-]+)\s*\n'
        r'\s+text\s*=\s*"([^"]*)"'
    )

    for match in interval_pattern.finditer(phones_section):
        start = float(match.group(1))
        end = float(match.group(2))
        phoneme = match.group(3).strip()
        if phoneme:
            items.append((phoneme.upper(), start, end))

    return items


class L2ArcticParser:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"L2-ARCTIC data directory not found: {data_dir}")

    def get_speakers(self) -> List[str]:
        speakers = []
        for item in self.data_dir.iterdir():
            if item.is_dir():
                speakers.append(item.name)
        return sorted(speakers)

    def parse_speaker_metadata(self, speaker: str) -> Dict[str, str]:
        meta: Dict[str, str] = {"speaker_id": speaker}
        speaker_dir = self.data_dir / speaker
        if not speaker_dir.exists():
            return meta

        meta_path = speaker_dir / f"{speaker}.txt"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if ":" in line:
                        k, v = line.split(":", 1)
                        k = k.strip().lower().replace(" ", "_")
                        v = v.strip()
                        if k in _SPEAKER_META_FIELDS:
                            meta[k] = v
        return meta

    def get_utterance_ids(self, speaker: str) -> List[str]:
        wav_dir = self.data_dir / speaker / "wav"
        if not wav_dir.exists():
            return []
        ids = sorted([f.stem for f in wav_dir.glob("*.wav")])
        return ids

    def load_wav_path(self, speaker: str, utterance_id: str) -> str:
        wav_path = self.data_dir / speaker / "wav" / f"{utterance_id}.wav"
        if not wav_path.exists():
            raise FileNotFoundError(f"WAV file not found: {wav_path}")
        return str(wav_path)

    def load_transcript(self, speaker: str, utterance_id: str) -> str:
        transcript_dir = self.data_dir / speaker / "transcript"
        txt_path = transcript_dir / f"{utterance_id}.txt"
        if txt_path.exists():
            with open(txt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        return ""

    def load_phoneme_annotation(self, speaker: str, utterance_id: str) -> List[Tuple[str, float, float]]:
        textgrid_dirs = [
            self.data_dir / speaker / "textgrid",
            self.data_dir / speaker / "annotation",
        ]
        for tg_dir in textgrid_dirs:
            if not tg_dir.exists():
                continue
            tg_path = tg_dir / f"{utterance_id}.TextGrid"
            if tg_path.exists():
                result = _parse_textgrid_phones(tg_path)
                if result:
                    return result
            phn_path = tg_dir / f"{utterance_id}.phn"
            if phn_path.exists():
                with open(phn_path, "r", encoding="utf-8") as f:
                    result = _parse_phoneme_line(f.read())
                    if result:
                        return result
            phn_path = tg_dir / f"{utterance_id}.phones"
            if phn_path.exists():
                with open(phn_path, "r", encoding="utf-8") as f:
                    result = _parse_phoneme_line(f.read())
                    if result:
                        return result
        return []

    def load_all_utterances(self, speaker: str) -> List[Dict]:
        utterances = []
        for utt_id in self.get_utterance_ids(speaker):
            try:
                wav_path = self.load_wav_path(speaker, utt_id)
                transcript = self.load_transcript(speaker, utt_id)
                phonemes = self.load_phoneme_annotation(speaker, utt_id)
                phoneme_seq = [p[0] for p in phonemes]
                utterances.append({
                    "speaker": speaker,
                    "utterance_id": utt_id,
                    "wav_path": wav_path,
                    "transcript": transcript,
                    "phonemes": phoneme_seq,
                    "phoneme_timings": phonemes,
                    "duration": phonemes[-1][2] if phonemes else 0.0,
                })
            except Exception as e:
                logger.warning(f"Failed to load {speaker}/{utt_id}: {e}")
                continue
        return utterances

    def build_manifest(self, output_path: str) -> List[Dict]:
        all_utterances = []
        for speaker in self.get_speakers():
            meta = self.parse_speaker_metadata(speaker)
            utterances = self.load_all_utterances(speaker)
            for utt in utterances:
                utt.update(meta)
            all_utterances.extend(utterances)
            logger.info(f"Speaker {speaker}: {len(utterances)} utterances")

        with open(output_path, "w", encoding="utf-8") as f:
            for utt in all_utterances:
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

        logger.info(f"Manifest saved to {output_path}: {len(all_utterances)} utterances")
        return all_utterances

    def __repr__(self) -> str:
        return f"L2ArcticParser(data_dir={self.data_dir})"
