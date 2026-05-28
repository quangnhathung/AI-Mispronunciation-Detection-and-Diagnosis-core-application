import os
import re
import torch
import torchaudio

import soundfile as sf
import tgt
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2FeatureExtractor
from .dictionary import get_phoneme_id

class L2ArcticDataset(Dataset):
    def __init__(self, root_dir, speaker_list, target_sr=16000, feature_extractor_name="facebook/wav2vec2-base-960h"):
        self.root_dir = root_dir
        self.target_sr = target_sr
        self.data_items = self._build_file_list(speaker_list)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(feature_extractor_name)

    def _build_file_list(self, speaker_list):
        items = []
        for spk in speaker_list:
            wav_dir = os.path.join(self.root_dir, spk, 'wav')
            tg_dir = os.path.join(self.root_dir, spk, 'annotation')
            
            if not os.path.exists(wav_dir):
                continue
            
            if not os.path.exists(tg_dir):
                alt = os.path.join(self.root_dir, spk, 'textgrid')
                if os.path.exists(alt):
                    tg_dir = alt

            for wav_file in os.listdir(wav_dir):
                if not wav_file.lower().endswith('.wav'):
                    continue
                base_name = wav_file[:-4]
                
                for ext in ('.TextGrid', '.textgrid', '.TEXTGRID'):
                    tg_path = os.path.join(tg_dir, base_name + ext)
                    if os.path.exists(tg_path):
                        items.append({
                            'wav_path': os.path.join(wav_dir, wav_file),
                            'tg_path': tg_path,
                            'speaker': spk  
                        })
                        break
        return items

    def _find_phones_tier(self, tg):
        """Tìm đúng tier chứa âm vị, loại bỏ triệt để tier words và IPA."""
        candidates = ["phones", "phone", "phones_tier", "phones_tier_1"]
        for name in candidates:
            try:
                return tg.get_tier_by_name(name)
            except Exception:
                continue
                
        # Fallback an toàn
        for tier in tg.tiers:
            # BỎ QUA CÁC TIER KHÔNG PHẢI ÂM VỊ
            if tier.name.lower() in ["words", "ipa", "comments"]:
                continue
            non_empty = sum(1 for it in tier if (it.text and it.text.strip()))
            if non_empty > 3:
                return tier
        raise ValueError("No suitable phones tier found in TextGrid")

    def _clean_label(self, raw_label: str):
        """Làm sạch khoảng trắng thừa nhưng GIỮ LẠI dấu phẩy."""
        if raw_label is None:
            return ""
        # Xóa toàn bộ khoảng trắng (vd: "Z,S,s " -> "Z,S,s")
        return re.sub(r'\s+', '', raw_label).strip()

    def _parse_textgrid(self, tg_path):
        """Phân tích TextGrid và trả về Tensor."""
        # ===== THÊM KHỐI TRY-EXCEPT ĐỂ CHỐNG SẬP =====
        try:
            tg = tgt.io.read_textgrid(tg_path, include_empty_intervals=True)
            phones_tier = self._find_phones_tier(tg)
        except Exception as e:
            # Nếu file hỏng, in cảnh báo và trả về fallback (SIL)
            print(f"\n[CẢNH BÁO] File TextGrid bị lỗi/hỏng cấu trúc: {tg_path} -> Bỏ qua.")
            canonical_ids = [get_phoneme_id("SIL")]
            labels = [1.0]
            intervals = [[0.0, 0.0]]
            return torch.tensor(canonical_ids, dtype=torch.long), \
                   torch.tensor(labels, dtype=torch.float), \
                   torch.tensor(intervals, dtype=torch.float)
        # =============================================

        canonical_ids = []
        labels = []
        intervals = []

        for interval in phones_tier:
            raw = self._clean_label(interval.text)
            start_t = float(interval.start_time)
            end_t = float(interval.end_time)

            # Bỏ qua các khoảng rỗng hoàn toàn
            if raw == "":
                continue

            # Xử lý nhãn lỗi (vd: "sil,K,a" hoặc "d,t,s")
            if "," in raw:
                parts = raw.split(",")
                canonical_ph = parts[0].strip().upper()
                if canonical_ph == "":
                    canonical_ph = "SIL"
                is_correct = 0.0 # Bị đánh dấu sai
            else:
                canonical_ph = raw.upper()
                is_correct = 1.0 # Đọc đúng

            # Chuẩn hóa token khoảng lặng
            if canonical_ph in ("SIL", "S", "SP", "PAU", "PAUSE"):
                canonical_ph = "SIL" if canonical_ph in ("SIL", "S") else "SP"

            # Ánh xạ ra ID
            pid = get_phoneme_id(canonical_ph)
            canonical_ids.append(pid)
            labels.append(is_correct)
            intervals.append([start_t, end_t])

        # Fallback nếu câu rỗng (nhưng file không hỏng)
        if len(canonical_ids) == 0:
            canonical_ids = [get_phoneme_id("SIL")]
            labels = [1.0]
            intervals = [[0.0, 0.0]]

        return torch.tensor(canonical_ids, dtype=torch.long), \
               torch.tensor(labels, dtype=torch.float), \
               torch.tensor(intervals, dtype=torch.float)

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        item = self.data_items[idx]
        
        # Load và chuẩn hóa Audio bằng soundfile
        # Chỉ định dtype='float32' để tương thích tốt nhất với PyTorch
        waveform_np, sr = sf.read(item['wav_path'], dtype='float32')
        
        # Chuyển đổi từ NumPy Array sang PyTorch Tensor
        waveform = torch.from_numpy(waveform_np)
        
        # Xử lý chiều ma trận: soundfile trả về (T,) hoặc (T, C). PyTorch yêu cầu (C, T)
        if waveform.dim() == 1:
            # Nếu là mono (Time,), thêm chiều channel để thành (1, Time)
            waveform = waveform.unsqueeze(0)
        else:
            # Nếu là stereo (Time, Channels), xoay ma trận thành (Channels, Time)
            waveform = waveform.t()
        
        # Đưa về Mono bằng cách lấy trung bình cộng các kênh (nếu có > 1 kênh)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample tần số lấy mẫu về 16kHz (chuẩn của Wav2Vec 2.0)
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.target_sr)

        # Squeeze mảng (1, Time) về mảng 1D (Time,) để đưa vào bộ
        audio_tensor = waveform.squeeze(0)
        audio_len = audio_tensor.shape[0]

        # Chuẩn hóa âm học với Wav2Vec2 (Zero-mean, unit-variance)
        processed = self.processor(audio_tensor, sampling_rate=self.target_sr, return_tensors="pt")
        input_values = processed.input_values.squeeze(0)      

        # 2. Phân tích nhãn TextGrid
        canonical_ids, target_scores, phoneme_intervals = self._parse_textgrid(item['tg_path'])

        return {
            "input_values": input_values,                 # Tensor
            "audio_len": audio_len,                       # Độ dài mẫu
            "canonical_ids": canonical_ids,               # Tensor: (N,)
            "target_scores": target_scores,               # Tensor: (N,)
            "phoneme_intervals": phoneme_intervals,       # Tensor: (N, 2)
            "speaker_id": item['speaker'],
            "utt_id": os.path.splitext(os.path.basename(item['wav_path']))[0]
        }


class MDDCollate:
    def __init__(self, pad_phoneme_id=0):
        self.pad_phoneme_id = pad_phoneme_id

    def __call__(self, batch):
        # Audio
        input_list = [b["input_values"] for b in batch]           
        lengths = torch.tensor([iv.shape[0] for iv in input_list], dtype=torch.long)
        padded_inputs = pad_sequence(input_list, batch_first=True, padding_value=0.0)  

        max_len = padded_inputs.shape[1]
        attention_mask = (torch.arange(max_len)[None, :] < lengths[:, None]).long()

        # Phonemes & Targets
        ph_list = [b["canonical_ids"] for b in batch]
        score_list = [b["target_scores"] for b in batch]
        intervals_list = [b["phoneme_intervals"] for b in batch]

        ph_padded = pad_sequence(ph_list, batch_first=True, padding_value=self.pad_phoneme_id)   
        scores_padded = pad_sequence(score_list, batch_first=True, padding_value=-100.0)    
        
        # Intervals Padding
        interval_padded = []
        max_ph = ph_padded.shape[1]
        for intr in intervals_list:
            if intr.shape[0] < max_ph:
                pad = torch.zeros((max_ph - intr.shape[0], 2), dtype=torch.float)
                interval_padded.append(torch.cat([intr, pad], dim=0))
            else:
                interval_padded.append(intr[:max_ph, :])
        interval_padded = torch.stack(interval_padded, dim=0)  

        speakers = [b["speaker_id"] for b in batch]
        utt_ids = [b["utt_id"] for b in batch]

        return {
            "input_values": padded_inputs,          # (B, T_max)
            "attention_mask": attention_mask,       # (B, T_max)
            "audio_lengths": lengths,               # (B,)
            "canonical_ids": ph_padded,             # (B, N_max)
            "target_scores": scores_padded,         # (B, N_max)
            "phoneme_intervals": interval_padded,   # (B, N_max, 2)
            "speakers": speakers,
            "utt_ids": utt_ids
        }