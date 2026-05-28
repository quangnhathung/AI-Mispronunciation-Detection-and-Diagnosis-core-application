# 🎤 CNN-BiLSTM-CTC — Phát Hiện Lỗi Phát Âm Tiếng Anh (MDD)

> **Mispronunciation Detection System** — Nhận diện lỗi phát âm ở cấp độ âm vị (phoneme) cho người học tiếng Anh (L2) sử dụng mô hình **CNN + BiLSTM + CTC** trên bộ dữ liệu **L2-ARCTIC**.

---

## 📋 Tổng Quan

Hệ thống này giải quyết bài toán **Mispronunciation Detection (MDD)**: đầu vào là file audio giọng nói tiếng Anh của người học (non-native), hệ thống sẽ:

1. **Nhận dạng chuỗi âm vị (phoneme)** từ audio
2. **So khớp** với chuỗi âm vị chuẩn (canonical phonemes)
3. **Phát hiện và phân loại lỗi** ở 3 dạng: Substitution, Deletion, Insertion
4. **Đo lường độ chính xác** qua các metrics PER, F1, CER

Ứng dụng thực tế: hỗ trợ giáo viên và người học trong việc **luyện phát âm tiếng Anh**, xác định âm nào cần cải thiện.

---

## ✨ Tính Năng

| Tính năng | Mô tả |
|-----------|-------|
| **⏱️ CTC-based** | Không cần alignment frame-level, giải mã toàn bộ câu |
| **🛡️ Mixed Precision (AMP)** | Huấn luyện nhanh hơn 2-3x trên GPU |
| **🔍 SpecAugment** | Tăng cường dữ liệu (frequency + time masking) |
| **📈 Early Stopping** | Dừng tự động khi F1 không cải thiện (patience=10) |
| **💾 Checkpoint Resume** | Tiếp tục huấn luyện từ checkpoint bất kỳ |
| **📊 TensorBoard + CSV** | Log metrics real-time, theo dõi loss/F1/PER |
| **🔬 MDD Feedback** | Báo cáo chi tiết từng lỗi (substitution/deletion/insertion) |
| **🎤 Microphone Inference** | Nhận dạng real-time qua mic |
| **📦 Docker** | Container sẵn sàng train/eval |
| **📁 ONNX Export** | Xuất mô hình sang ONNX cho production |
| **🧪 Test Suite** | 35+ unit tests |

---

## 🏗️ Kiến Trúc Mô Hình

### Pipeline xử lý

```
Audio (16kHz WAV)
    │
    ▼
┌─────────────────────────────┐
│  Mel Spectrogram (80 bands) │  ← 25ms window, 10ms hop
│  Normalize (mean-std)       │  → shape: (1, 80, T)
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  CNN Encoder (3× Conv2D)    │  ← BatchNorm + GELU + Dropout2D
│  [64 → 128 → 256 channels]  │  → 8× time downsampling
│  stride=2 per block         │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  BiLSTM Encoder (4 layers)  │  ← 256 hidden dim, bidirectional
│  PackedSequence support     │  → output dim: 512
│  LayerNorm                  │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  CTC Projection Head        │  ← Linear → LogSoftmax
│  vocab_size=42 (phonemes)   │   → shape: (B, T', 42)
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Decoder (Greedy/Beam)      │  ← collapse blanks + merge repeats
│  → chuỗi phoneme IDs        │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  MDD Detector               │  ← Levenshtein alignment
│  → Phân loại lỗi + PER     │  → Substitution / Deletion / Insertion
└─────────────────────────────┘
```

### Chi tiết các khối

| Component | Tham số | Ghi chú |
|-----------|---------|---------|
| **CNN Frontend** | 3× Conv2D (kernel=3, stride=2) | Giảm time dimension 8× |
| **BatchNorm2D** | Theo sau mỗi Conv2D | Chuẩn hóa feature map |
| **GELU Activation** | phi(x) = x * Φ(x) | Mượt hơn ReLU |
| **Dropout2D** | rate=0.2 | Chống overfitting |
| **BiLSTM** | 4 layers, hidden=256, bidirectional | 512 output dim |
| **LayerNorm** | Sau BiLSTM, trên 512 dim | Ổn định training |
| **CTC Head** | Linear(512, 42) + LogSoftmax | Đầu ra log-probability |

### Thông số mô hình

- **Input**: Mel spectrogram (80 bands) với SpecAugment
- **Time stride tổng**: 8× (128 frames → 16 features)
- **Vocab size**: 42 (40 phonemes + blank + unk)
- **Loss**: CTC (Connectionist Temporal Classification) — không cần alignment frame
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler**: Cosine Annealing (T_max=100, eta_min=1e-6) + Warmup 10%

---

## 📊 Dataset: L2-ARCTIC

**L2-ARCTIC** là bộ dữ liệu giọng nói tiếng Anh của 24 người học (L2) đến từ 6 quốc gia:

| Ngôn ngữ mẹ đẻ | Số người | Ký hiệu |
|----------------|----------|---------|
| Tiếng Ả Rập | 4 | ABA, ASM, BMA, HMS |
| Tiếng Trung | 4 | BWC, LXC, NCC, YDCK |
| Tiếng Hindi | 4 | EBVS, FCJH, HKK, RNSK |
| Tiếng Hàn | 4 | HJK, HKL, HYW, YKWS |
| Tiếng Tây Ban Nha | 4 | NJS, RPEO, SDFA, YNWS |
| Tiếng Việt | 4 | HQTV, MNTV, NTT, PNT |

- **Tổng số câu**: ~27,000 câu (1.5h mỗi người)
- **Sampling rate**: 44.1kHz → downsampled to 16kHz
- **Annotation**: TextGrid (Praat) — phoneme boundaries + canonical phonemes
- **Phoneme set**: ARPABET (39 phonemes + stress markers)

### Cấu trúc thư mục

```
data/raw/L2_ARCTIC/
├── ABA/                 # Speaker ID
│   ├── wav/            # File .wav (44.1kHz → 16kHz)
│   │   ├── ABA_1.wav
│   │   └── ...
│   ├── textgrid/       # Praat .TextGrid annotation
│   │   ├── ABA_1.TextGrid
│   │   └── ...
│   └── transcript/     # Transcript plain text
│       └── ...
├── BWC/
├── EBVS/
├── FCJH/
├── ...                 # 24 speakers total
└── README.md
```

### Train / Val / Test Split

| Split | Tỷ lệ | Số câu (ước lượng) |
|-------|-------|-------------------|
| Train | 80% | ~21,600 |
| Validation | 10% | ~2,700 |
| Test | 10% | ~2,700 |

Phương pháp split: **speaker-independent** — tách riêng người nói, đảm bảo không rò rỉ dữ liệu.

---

## 🔧 Cài Đặt

### Yêu cầu

- Python **3.9 – 3.13**
- PyTorch **2.0+** (CPU hoặc CUDA)
- GPU **NVIDIA** với CUDA 12+ (khuyến nghị, không bắt buộc)
- ~8GB VRAM cho batch_size=16

### Cài đặt thủ công

```bash
# 1. Clone repository
git clone <repo-url>
cd CNN_BiLSTM_CTC

# 2. (Khuyến nghị) Tạo virtual environment
python -m venv venv
.\venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/macOS

# 3. Cài dependencies
pip install -r requirements.txt
```

### Docker

```bash
# Build image
docker build -t mdd-cnn-bilstm .

# Train trên GPU
docker run --gpus all \
  -v D:/data/L2_ARCTIC:/app/data/raw/L2_ARCTIC \
  mdd-cnn-bilstm

# Evaluate
docker run --gpus all \
  -v D:/data/L2_ARCTIC:/app/data/raw/L2_ARCTIC \
  mdd-cnn-bilstm python evaluate.py --checkpoint checkpoints/best.pt
```

---

## 📖 Hướng Dẫn Sử Dụng

### 1. Chuẩn bị dữ liệu

```bash
# Tải L2-ARCTIC về và giải nén vào data/raw/L2_ARCTIC/
# Sau đó build manifest (train/val/test split + cache)
python -c "
from src.preprocessing.manifest import ManifestBuilder
builder = ManifestBuilder('data/raw/L2_ARCTIC', 'data/manifests')
builder.build_all()
"
```

### 2. Cấu hình

Tất cả tham số trong `configs/config.yaml`:

```yaml
model:
  input_dim: 80
  cnn_channels: [64, 128, 256]
  cnn_kernel_sizes: [3, 3, 3]
  cnn_strides: [2, 2, 2]
  rnn_hidden_size: 256
  rnn_num_layers: 4
  vocab_size: 42

training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.001
  mixed_precision: true
  monitor_metric: val_f1_macro
  early_stopping_patience: 10
  gradient_clip: 5.0

data:
  sample_rate: 16000
  n_mels: 80
  spec_augment: true
```

Override nhanh qua CLI:

```bash
python train.py --epochs 50 --batch_size 32 --lr 0.0005 --device cuda
```

### 3. Huấn luyện (Training)

```bash
# Train từ đầu
python train.py --config configs/config.yaml

# Train với override
python train.py --epochs 200 --batch_size 32 --lr 0.0001

# Resume từ checkpoint
python train.py --resume checkpoints/last.pt
```

**Tham số CLI:**

| Flag | Mô tả | Default |
|------|-------|---------|
| `--config` | Đường dẫn config | `configs/config.yaml` |
| `--data_dir` | Dataset path override | — |
| `--epochs` | Số epoch | 100 |
| `--batch_size` | Batch size | 16 |
| `--lr` | Learning rate | 0.001 |
| `--device` | `cuda` hoặc `cpu` | cuda |
| `--resume` | Resume từ checkpoint | — |
| `--seed` | Random seed | 42 |

**Đầu ra (outputs):**

```
checkpoints/
├── best.pt              # Checkpoint có F1 cao nhất
├── last.pt              # Checkpoint epoch cuối
└── epoch_XXX_*.pt       # Top-K checkpoints
logs/
├── tensorboard/         # Logs cho TensorBoard
├── training_log.csv     # CSV metrics (loss, PER, F1)
└── training_*.log       # Log file chi tiết
outputs/plots/
└── training_metrics.png # Biểu đồ loss / PER / F1 / LR
```

**Theo dõi training:**

```bash
tensorboard --logdir logs/tensorboard --port 6006
# Mở http://localhost:6006 trong browser
```

### 4. Đánh giá (Evaluation)

```bash
# Đánh giá trên test set
python evaluate.py --checkpoint checkpoints/best.pt

# Đánh giá với override
python evaluate.py --checkpoint checkpoints/last.pt --device cuda
```

**Đầu ra:**

```
outputs/
├── predictions.csv        # Dự đoán theo từng câu
├── confusion_report.csv   # Top substitution pairs
└── plots/
    ├── confusion_matrix.png   # Heatmap ma trận nhầm lẫn (top-30 phoneme)
    └── training_metrics.png   # Đồ thị huấn luyện
```

**Metrics báo cáo:**

```
Evaluation Results:
  PER:                 0.4235 (42.35%)
  CER:                 0.3150 (31.50%)
  Confusion Accuracy:  0.5812
  F1 Macro:            0.5578 (55.78%)
  F1 Micro:            0.5976 (59.76%)
  Samples Evaluated:   2690
```

### 5. Suy luận (Inference)

```bash
# File đơn
python inference.py --checkpoint checkpoints/best.pt --audio sample.wav

# File đơn + ground truth (MDD)
python inference.py --checkpoint checkpoints/best.pt \
    --audio sample.wav \
    --target_phonemes HH AH0 L OW1 W ER1 L D

# Batch nhiều file
python inference.py --checkpoint checkpoints/best.pt \
    --audio_dir data/test_audio/ \
    --output results.json

# Microphone real-time
python inference.py --checkpoint checkpoints/best.pt \
    --microphone --duration 5.0

# Xuất kết quả JSON
python inference.py --checkpoint checkpoints/best.pt \
    --audio sample.wav --output result.json
```

---

## 📈 Metrics & Giải Thích

### Core Metrics

| Metric | Công thức | Ý nghĩa | Khoảng |
|--------|-----------|---------|--------|
| **PER** | `(S + D + I) / N` | Phoneme Error Rate — tỷ lệ lỗi âm vị | 0% → ∞ |
| **CER** | Character Error Rate | Tỷ lệ lỗi ký tự | 0% → ∞ |
| **F1 Macro** | mean(F1 per class) | F1 trung bình mỗi phoneme (công bằng) | 0–1 |
| **F1 Micro** | global TP / (TP + FP/2 + FN/2) | F1 tổng thể (thiên về phoneme phổ biến) | 0–1 |
| **Accuracy** | 1 - PER | Độ chính xác cấp phoneme | 0–1 |

### MDD Error Types

| Loại lỗi | Mô tả | Ví dụ | Hiển thị |
|-----------|-------|-------|----------|
| ✅ **Correct** | Phát âm đúng | `HH` → `HH` | 🟢 Xanh |
| 🔴 **Substitution** | Sai âm vị | `AE1` → `AH0` | 🔴 Đỏ |
| 🟡 **Deletion** | Thiếu âm vị | `L` → ∅ | 🟡 Vàng |
| 🔵 **Insertion** | Thừa âm vị | ∅ → `AH0` | 🔵 Xanh dương |

### Confusion Matrix

Ma trận nhầm lẫn (confusion matrix) là công cụ trực quan mạnh để phân tích lỗi:

- **Hàng (truth)**: phoneme đúng
- **Cột (predicted)**: phoneme dự đoán
- **Đường chéo**: số lần dự đoán đúng
- **Off-diagonal**: các cặp dễ nhầm lẫn (ví dụ: `IY`↔`IH`, `S`↔`Z`)

---

## 🗂️ Cấu Trúc Dự Án

```
CNN_BiLSTM_CTC/
│
├── configs/
│   └── config.yaml              # ⚙️ File cấu hình chính
│
├── data/
│   ├── raw/L2_ARCTIC/           # 📁 Dataset gốc (user tự tải)
│   ├── processed/               # 📁 Feature tensor đã xử lý
│   ├── cache/                   # 📁 Cache dataset (tăng tốc)
│   └── manifests/               # 📁 Train/val/test split (.jsonl)
│
├── src/
│   ├── datasets/
│   │   ├── l2arctic_parser.py   # ️ Parser TextGrid + .phn
│   │   ├── l2arctic_dataset.py  # 📦 Dataset PyTorch
│   │   ├── tokenizer.py         # 🔤 Mã hóa phoneme ↔ ID
│   │   └── collator.py          # 📐 Padding + batch
│   │
│   ├── features/
│   │   ├── mel_spec.py          # 🎵 Mel spectrogram extraction
│   │   └── spec_augment.py      # 🎭 SpecAugment (freq+time mask)
│   │
│   ├── models/
│   │   ├── base.py              # 🏗️ Base model class
│   │   └── cnn_bilstm_ctc.py    # 🧠 CNN + BiLSTM + CTC
│   │
│   ├── losses/
│   │   └── ctc_loss.py          # 📉 CTC loss wrapper
│   │
│   ├── decoders/
│   │   ├── greedy.py            # ⚡ Greedy decoder (nhanh)
│   │   └── beam_search.py       # 🔍 Beam search decoder (chính xác hơn)
│   │
│   ├── metrics/
│   │   ├── per.py               # 📊 Phoneme Error Rate
│   │   ├── cer.py               # 📊 Character Error Rate
│   │   ├── confusion.py         # 📊 Confusion matrix
│   │   └── f1.py                # 📊 F1 score (macro/micro)
│   │
│   ├── mdd/
│   │   └── detector.py          # 🔬 Mispronunciation detector
│   │
│   ├── callbacks/
│   │   ├── early_stopping.py    # ⏹️ Early stopping (F1 monitor)
│   │   └── model_checkpoint.py  # 💾 Checkpoint saver
│   │
│   ├── trainers/
│   │   └── trainer.py           # 🏋️ Training loop (AMP + gradient clip)
│   │
│   ├── evaluators/
│   │   └── evaluator.py         # ✅ Full evaluation pipeline
│   │
│   ├── inference/
│   │   └── predictor.py         # 🎤 Inference engine
│   │
│   ├── pipelines/
│   │   ├── train_pipeline.py    # 🔄 Training pipeline
│   │   ├── eval_pipeline.py     # 🔄 Evaluation pipeline
│   │   └── infer_pipeline.py    # 🔄 Inference pipeline
│   │
│   ├── preprocessing/
│   │   └── manifest.py          # 📋 Manifest builder
│   │
│   ├── visualization/
│   │   └── plots.py             # 📈 Training plots + confusion heatmap
│   │
│   └── utils/
│       ├── config.py            # ⚙️ Config loader (YAML)
│       ├── helpers.py           # 🛠️ Utility functions
│       ├── logger.py            # 📝 Logger setup (loguru)
│       └── seed.py              # 🌱 Random seed
│
├── tests/
│   ├── test_dataset.py          # Dataset + tokenizer + collator tests
│   ├── test_decoder.py          # Greedy + beam search tests
│   ├── test_model.py            # Model forward + shape tests
│   ├── test_tokenizer.py        # Vocab + encode/decode tests
│   └── test_trainer.py          # Early stopping + checkpoint + MDD tests
│
├── checkpoints/                 # 📁 Model checkpoints
├── logs/                        # 📁 Training logs
├── outputs/                     # 📁 Evaluation results + plots
├── scripts/                     # 📁 Utility scripts
│
├── train.py                     # 🚀 Entry point: training
├── evaluate.py                  # 🚀 Entry point: evaluation
├── inference.py                 # 🚀 Entry point: inference
├── requirements.txt             # 📦 Python dependencies
├── Dockerfile                   # 📦 Docker build
├── .gitignore                   # 🙈 Git ignore
└── README.md                    # 📘 Bạn đang đọc đây!
```

---

## 📦 Dependencies

```
torch>=2.0.0              # Deep learning framework
torchaudio>=2.0.0          # Audio I/O & augmentation
torchmetrics>=1.0.0        # Metric computation
librosa>=0.10.0            # Audio analysis
jiwer>=3.0.0               # Word error rate
pandas>=2.0.0              # Data processing
numpy>=1.24.0              # Numerical computing
scikit-learn>=1.3.0        # Evaluation utilities
tensorboard>=2.14.0        # Training visualization
pyyaml>=6.0                # Config parsing
tqdm>=4.65.0               # Progress bars
loguru>=0.7.0              # Logging
matplotlib>=3.7.0          # Plotting
seaborn>=0.12.0            # Statistical plots
sounddevice>=0.4.6         # Microphone recording
onnx>=1.14.0               # ONNX export
onnxruntime>=1.15.0        # ONNX inference
```

---

## 🐳 Docker

### Build

```bash
docker build -t mdd-cnn-bilstm:latest .
```

Dockerfile sử dụng base image `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`, cài sẵn sox + libsndfile cho audio processing.

### Run

```bash
# Training
docker run --gpus all --rm \
  -v D:/data/L2_ARCTIC:/app/data/raw/L2_ARCTIC \
  -v D:/projects/checkpoints:/app/checkpoints \
  mdd-cnn-bilstm

# Evaluation
docker run --gpus all --rm \
  -v D:/data/L2_ARCTIC:/app/data/raw/L2_ARCTIC \
  -v D:/projects/checkpoints:/app/checkpoints \
  mdd-cnn-bilstm python evaluate.py --checkpoint checkpoints/best.pt

# Interactive
docker run --gpus all -it --rm \
  -v D:/data/L2_ARCTIC:/app/data/raw/L2_ARCTIC \
  mdd-cnn-bilstm /bin/bash
```

---

## 🔬 Kết Quả Tham Khảo

| Metric | Baseline | Sau tuning | Mục tiêu |
|--------|----------|------------|----------|
| PER (phoneme error rate) | ~57.8% | < 50% | < 35% |
| F1 Macro | ~42.5% | > 50% | > 60% |
| Phoneme Accuracy | ~42.5% | > 50% | > 65% |

> **Lưu ý**: Kết quả phụ thuộc nhiều vào điều kiện dữ liệu, thời lượng train, hyperparameter. Các số trên là giá trị tham khảo từ lần chạy baseline.

---

## 🧪 Testing

```bash
# Chạy tất cả tests (35 tests)
python -m pytest tests/ -v

# Chạy test theo module
python -m pytest tests/test_model.py -v
python -m pytest tests/test_tokenizer.py -v
python -m pytest tests/test_trainer.py -v

# Test với coverage
python -m pytest tests/ --cov=src --cov-report=term
```

---

## 🚀 Cải Tiến Tương Lai

### Ngắn hạn
- [ ] **Attention-based Decoder**: Thay thế CTC bằng RNN-T hoặc Transformer
- [ ] **Speaker Adaptation**: Fine-tune riêng theo từng người nói
- [ ] **Data Augmentation**: Thêm noise, speed perturbation, pitch shift
- [ ] **Ensemble**: Kết hợp nhiều checkpoint để tăng độ chính xác

### Trung hạn
- [ ] **End-to-end MDD**: Loại bỏ bước canonical phoneme alignment
- [ ] **Multi-task Learning**: Học đồng thời phoneme + word + fluency score
- [ ] **Self-supervised Pretraining**: Dùng Wav2Vec2 / HuBERT làm feature extractor
- [ ] **Web API**: Flask/FastAPI endpoint cho inference

### Dài hạn
- [ ] **Triển khai mobile**: ONNX → CoreML / TFLite / NCNN
- [ ] **Hệ thống đề xuất luyện tập**: Gợi ý âm cần cải thiện dựa trên confusion matrix
- [ ] **Hỗ trợ đa ngôn ngữ**: Mở rộng sang các ngôn ngữ khác (Nhật, Hàn, Việt)

---

## 🛠️ Troubleshooting

| Vấn đề | Nguyên nhân | Giải pháp |
|--------|-------------|-----------|
| `CUDA out of memory` | Batch size quá lớn | Giảm `batch_size` hoặc tăng `gradient_accumulation` |
| `NaN loss` | Learning rate quá cao | Giảm `lr`, tăng `gradient_clip` |
| `PER = 0` từ epoch 1 | Không load được phoneme annotation | Kiểm tra `data/raw/L2_ARCTIC/{speaker}/textgrid/` |
| `No module named 'pytest'` | Thiếu dev dependencies | `pip install pytest` |
| `list object has no attribute 'shape'` | Lỗi type trong plotting | Cập nhật `ConfusionPlotter.plot()` |
| Training quá lâu | Không có GPU / `num_workers` thấp | Tăng `num_workers`, bật `mixed_precision` |
| Manifest cũ chứa `phonemes=[]` | Build từ khi bug Parser | Xóa `data/manifests/` và build lại |

---

## 📚 Tài Liệu Tham Khảo

1. **L2-ARCTIC Dataset** — Zhao et al., INTERSPEECH 2018
2. **CTC (Connectionist Temporal Classification)** — Graves et al., ICML 2006
3. **SpecAugment** — Park et al., INTERSPEECH 2019
4. **AdamW Optimizer** — Loshchilov & Hutter, ICLR 2019
5. **Cosine Annealing** — Loshchilov & Hutter, ICLR 2017

### Citation

```bibtex
@software{mdd-cnn-bilstm-ctc-2024,
  title  = {CNN-BiLSTM-CTC Mispronunciation Detection for L2-ARCTIC},
  author = {MDD Team},
  year   = {2024},
  url    = {https://github.com/your-repo/CNN_BiLSTM_CTC}
}

@inproceedings{zhao2018l2arctic,
  title     = {L2-ARCTIC: A non-native English speech corpus},
  author    = {Zhao, Gang and Sonsaat, Sinem and Silpachai, Alif and
               Lu, Ivana and Levis, John and Chukharev-Hudilainen, Evgeny and
               Gutierrez-Osuna, Ricardo},
  booktitle = {Proceedings of INTERSPEECH},
  year      = {2018}
}
```

---

## 👥 Tác Giả & Liên Hệ

- **Phát triển**: Đội ngũ MDD Team
- **Email**: [liên hệ qua GitHub Issues](https://github.com/your-repo/CNN_BiLSTM_CTC/issues)
- **Báo cáo lỗi / Đóng góp**: Mở Issue hoặc Pull Request tại repository

---

<div align="center">
  <sub>
    Built with ❤️ and PyTorch
  </sub>
</div>
