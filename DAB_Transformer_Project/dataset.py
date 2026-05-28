"""

dataset.py — Load dữ liệu & batch



  [PRE] preprocess_resample.py (offline 16kHz)

  [A]   Index

  [B]   __getitem__ — load + cắt (train dùng MAX_SAMPLES_TRAIN ngắn hơn → ít OOM)

  [C]   collate_fn

  [D]   BucketBatchSampler — tránh batch toàn file quá dài

  [E]   make_dataloader

"""

import random

import torch

import torchaudio

import os

from torch.utils.data import Dataset, Sampler

from config import Config





class L2ArcticDataset(Dataset):

    def __init__(self, folder_path, text_process, max_samples=None):

        self.samples = []

        self.lengths = []

        self.text_process = text_process

        self.max_samples = max_samples if max_samples is not None else Config.MAX_SAMPLES



        if not os.path.exists(folder_path):

            print(f"[ERROR] Thu muc khong ton tai: {folder_path}")

            print("   Chạy trước: python preprocess_resample.py")

            return



        speakers = [

            d for d in os.listdir(folder_path)

            if os.path.isdir(os.path.join(folder_path, d))

        ]

        for spk in speakers:

            wav_dir = os.path.join(folder_path, spk, "wav")

            txt_dir = os.path.join(folder_path, spk, "transcript")

            if os.path.exists(wav_dir):

                for f in os.listdir(wav_dir):

                    if f.endswith(".wav"):

                        wav_path = os.path.join(wav_dir, f)

                        txt_path = os.path.join(txt_dir, f.replace(".wav", ".txt"))

                        if os.path.exists(txt_path):

                            self.samples.append({"audio": wav_path, "text": txt_path})

                            self.lengths.append(self._probe_length(wav_path))



        print(f"[OK] Da nap {len(self.samples)} tep tu: {folder_path}")



    def _probe_length(self, wav_path):

        info = torchaudio.info(wav_path)

        if info.sample_rate != Config.TARGET_SAMPLE_RATE:

            raise ValueError(

                f"File chưa 16 kHz: {wav_path}\nChạy: python preprocess_resample.py"

            )

        return min(info.num_frames, self.max_samples)



    def __len__(self):

        return len(self.samples)



    def __getitem__(self, idx):

        s = self.samples[idx]

        wav, sr = torchaudio.load(s["audio"])

        if sr != Config.TARGET_SAMPLE_RATE:

            raise ValueError(f"Wav chưa 16 kHz: {s['audio']}")

        wav = wav[0]



        if wav.shape[0] > self.max_samples:

            wav = wav[: self.max_samples]



        with open(s["text"], "r", encoding="utf-8") as f:

            transcript = f.read().strip()



        label = torch.tensor(self.text_process.text_to_int(transcript))

        return wav, label





class BucketBatchSampler(Sampler):

    """

    [D] Gom mẫu cùng độ dài; tách batch nếu max độ dài > BUCKET_MAX_WAV_LEN (phòng OOM 4GB).

    """



    def __init__(self, lengths, batch_size, shuffle=True, max_wav_len=None):

        self.lengths = lengths

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.max_wav_len = max_wav_len or Config.BUCKET_MAX_WAV_LEN
        self._cached_batches = None  # cache tránh build 2 lần

    def _build_batches(self):
        # Trả cache nếu đã build (__len__ gọi trước __iter__ → chỉ sort 1 lần/epoch)
        if self._cached_batches is not None:
            return self._cached_batches

        order = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])

        batches = []

        current = []



        for idx in order:

            current.append(idx)

            if len(current) < self.batch_size:

                continue

            max_len = max(self.lengths[i] for i in current)

            if max_len > self.max_wav_len and len(current) > 1:

                batches.append(current[:-1])

                current = [idx]

            else:

                batches.append(current)

                current = []



        if current:

            batches.append(current)

        self._cached_batches = batches
        return batches



    def __iter__(self):

        batches = list(self._build_batches())  # list() để shuffle không ảnh hưởng cache
        self._cached_batches = None  # reset sau mỗi epoch để shuffle lại đúng

        if self.shuffle:

            random.shuffle(batches)

        yield from batches



    def __len__(self):

        return len(self._build_batches())





def collate_fn(batch):

    waveforms, labels = zip(*batch)

    wav_lens = torch.tensor([w.shape[0] for w in waveforms])

    label_lens = torch.tensor([len(l) for l in labels])

    waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)

    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return waveforms, labels, wav_lens, label_lens





def make_dataloader(dataset, shuffle=True):

    loader_kw = {

        "collate_fn": collate_fn,

        "num_workers": Config.NUM_WORKERS,

        "pin_memory": Config.DEVICE.type == "cuda",

    }

    if Config.NUM_WORKERS > 0:

        loader_kw["persistent_workers"] = True

        loader_kw["prefetch_factor"] = Config.PREFETCH_FACTOR



    if Config.BUCKET_BATCHING and len(dataset.lengths) > 0:

        sampler = BucketBatchSampler(dataset.lengths, Config.BATCH_SIZE, shuffle=shuffle)

        return torch.utils.data.DataLoader(dataset, batch_sampler=sampler, **loader_kw)



    return torch.utils.data.DataLoader(

        dataset, batch_size=Config.BATCH_SIZE, shuffle=shuffle, **loader_kw

    )