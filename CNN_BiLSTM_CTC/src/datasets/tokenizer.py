from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Union

import torch

ARPAbet_PHONEMES: List[str] = [
    "AA", "AE", "AH", "AO", "AW", "AY",
    "B", "CH", "D", "DH", "EH", "ER", "EY",
    "F", "G", "HH", "IH", "IY", "JH",
    "K", "L", "M", "N", "NG", "OW", "OY",
    "P", "R", "S", "SH", "T", "TH", "UH", "UW",
    "V", "W", "Y", "Z", "ZH",
]

STRESSED_VOWEL_MAP: Dict[str, List[str]] = {
    "AA": ["AA0", "AA1", "AA2"],
    "AE": ["AE0", "AE1", "AE2"],
    "AH": ["AH0", "AH1", "AH2"],
    "AO": ["AO0", "AO1", "AO2"],
    "AW": ["AW0", "AW1", "AW2"],
    "AY": ["AY0", "AY1", "AY2"],
    "EH": ["EH0", "EH1", "EH2"],
    "ER": ["ER0", "ER1", "ER2"],
    "EY": ["EY0", "EY1", "EY2"],
    "IH": ["IH0", "IH1", "IH2"],
    "IY": ["IY0", "IY1", "IY2"],
    "OW": ["OW0", "OW1", "OW2"],
    "OY": ["OY0", "OY1", "OY2"],
    "UH": ["UH0", "UH1", "UH2"],
    "UW": ["UW0", "UW1", "UW2"],
}

# SIL and SP are L2Arctic annotation markers for silence/short-pause.
# They must be in the vocab so the model learns them instead of <unk>.
SPECIAL_TOKENS: List[str] = ["<blank>", "<unk>", "<sos/eos>"]
SILENCE_TOKENS: List[str] = ["SIL", "SP"]


class PhonemeTokenizer:
    def __init__(
        self,
        include_stress: bool = True,
        custom_phonemes: Optional[List[str]] = None,
        blank_token: str = "<blank>",
        unk_token: str = "<unk>",
    ):
        self.blank_token = blank_token
        self.unk_token = unk_token
        self.include_stress = include_stress

        self.vocab: List[str] = []
        self._id_to_phoneme: Dict[int, str] = {}
        self._phoneme_to_id: Dict[str, int] = {}

        self._build_vocab(custom_phonemes)

    def _build_vocab(self, custom_phonemes: Optional[List[str]]) -> None:
        self.vocab.append(self.blank_token)
        self.vocab.append(self.unk_token)

        all_phonemes: List[str] = []
        if custom_phonemes:
            all_phonemes = custom_phonemes
        elif self.include_stress:
            for vowel in STRESSED_VOWEL_MAP:
                all_phonemes.extend(STRESSED_VOWEL_MAP[vowel])
            for cons in ARPAbet_PHONEMES:
                if cons not in STRESSED_VOWEL_MAP:
                    all_phonemes.append(cons)
        else:
            all_phonemes = list(ARPAbet_PHONEMES)

        # Add silence markers used in L2Arctic annotations
        for st in SILENCE_TOKENS:
            if st not in all_phonemes:
                all_phonemes.append(st)

        self.vocab.extend(sorted(set(all_phonemes)))

        self._id_to_phoneme = {i: p for i, p in enumerate(self.vocab)}
        self._phoneme_to_id = {p: i for i, p in enumerate(self.vocab)}

    @property
    def blank_id(self) -> int:
        return self._phoneme_to_id[self.blank_token]

    @property
    def unk_id(self) -> int:
        return self._phoneme_to_id[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, phonemes: Sequence[str]) -> torch.LongTensor:
        ids = []
        for p in phonemes:
            p = p.strip()
            if p.startswith("<") and p.endswith(">"):
                key = p
            else:
                key = p.upper()
            if key in self._phoneme_to_id:
                ids.append(self._phoneme_to_id[key])
            else:
                ids.append(self.unk_id)
        return torch.tensor(ids, dtype=torch.long)

    def encode_batch(self, batch: Sequence[Sequence[str]]) -> List[torch.LongTensor]:
        return [self.encode(seq) for seq in batch]

    def decode(self, ids: Union[torch.LongTensor, List[int]]) -> List[str]:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return [self._id_to_phoneme.get(i, self.unk_token) for i in ids]

    def decode_to_string(self, ids: Union[torch.LongTensor, List[int]], separator: str = " ") -> str:
        return separator.join(self.decode(ids))

    def save_vocab(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for idx, phoneme in self.vocab:
                f.write(f"{idx}\t{phoneme}\n")

    @classmethod
    def load_vocab(cls, path: str) -> PhonemeTokenizer:
        custom_phonemes = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    custom_phonemes.append(parts[1])
        return cls(custom_phonemes=custom_phonemes)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"PhonemeTokenizer(vocab_size={self.vocab_size}, include_stress={self.include_stress})"
