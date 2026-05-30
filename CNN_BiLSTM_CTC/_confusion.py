import sys; sys.path.insert(0, 'CNN_BiLSTM_CTC')
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils.config import Config
from src.utils.helpers import load_checkpoint
from src.datasets.tokenizer import PhonemeTokenizer
from src.datasets.l2arctic_dataset import L2ArcticDataset
from src.datasets.collator import Collator
from src.features.mel_spec import MelFeatureExtractor
from src.models.cnn_bilstm_ctc import CNNBiLSTMCTC
from src.decoders.greedy import GreedyDecoder
from src.mdd.detector import MispronunciationDetector
from torch.utils.data import DataLoader

config = Config.from_yaml('CNN_BiLSTM_CTC/configs/config.yaml')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

tokenizer = PhonemeTokenizer(include_stress=True)
vocab_size = tokenizer.vocab_size
print('Vocab size:', vocab_size)

model = CNNBiLSTMCTC(
    input_dim=config.model.input_dim,
    vocab_size=vocab_size,
    cnn_channels=config.model.cnn_channels,
    cnn_kernel_sizes=config.model.cnn_kernel_sizes,
    cnn_strides=config.model.cnn_strides,
    cnn_dropout=config.model.cnn_dropout,
    rnn_hidden_size=config.model.rnn_hidden_size,
    rnn_num_layers=config.model.rnn_num_layers,
    rnn_dropout=config.model.rnn_dropout,
    rnn_bidirectional=config.model.rnn_bidirectional,
)

ckpt = load_checkpoint('CNN_BiLSTM_CTC/checkpoints/best.pt', model, device=device)
print('Loaded epoch:', ckpt.get('epoch', 'unknown'))
model = model.to(device)
model.eval()

feat = MelFeatureExtractor(
    sample_rate=config.data.sample_rate,
    n_fft=config.data.n_fft,
    win_length=config.data.win_length,
    hop_length=config.data.hop_length,
    n_mels=config.data.n_mels,
)

test_manifest = Path('CNN_BiLSTM_CTC/data/manifests/test.jsonl')
if not test_manifest.exists():
    test_manifest = Path('CNN_BiLSTM_CTC/data/manifests/val.jsonl')

ds = L2ArcticDataset(
    manifest_path=str(test_manifest),
    tokenizer=tokenizer,
    sample_rate=config.data.sample_rate,
    feature_fn=feat,
)
print('Test samples:', len(ds))

collator = Collator(pad_token_id=tokenizer.blank_id)
dl = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collator, num_workers=0)

decoder = GreedyDecoder(blank_id=tokenizer.blank_id)
detector = MispronunciationDetector(tokenizer)

n = vocab_size
conf_matrix = np.zeros((n, n), dtype=np.int64)

with torch.no_grad():
    for batch in tqdm(dl, desc='Evaluating'):
        audio = batch['audio'].to(device)
        phonemes = batch['phonemes'].to(device)
        audio_lengths = batch['audio_lengths'].to(device)
        phoneme_lengths = batch['phoneme_lengths']

        logits = model(audio, audio_lengths)
        feat_lengths = model.get_feat_lengths(audio_lengths)
        preds = decoder.decode(logits, feat_lengths)

        for pred, targ, plen in zip(preds, phonemes, phoneme_lengths):
            target = targ[:plen].tolist()
            aligned_pred, aligned_target = detector._align_ids(pred, target)
            for p, t in zip(aligned_pred, aligned_target):
                if p >= 0 and t >= 0 and p < n and t < n:
                    conf_matrix[t, p] += 1

id_to_ph = tokenizer._id_to_phoneme
skip_ids = {tokenizer.blank_id, tokenizer.unk_id}
sos_eos = tokenizer._phoneme_to_id.get('<sos/eos>', -1)
if sos_eos >= 0:
    skip_ids.add(sos_eos)

phonemes_to_show = []
for i in range(n):
    if i in skip_ids:
        continue
    total = conf_matrix[i, :].sum()
    if total == 0:
        continue
    tp = conf_matrix[i, i]
    fp = conf_matrix[:, i].sum() - tp
    fn = total - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    phonemes_to_show.append((i, id_to_ph[i], int(total), int(tp), prec, rec, f1, fp, fn))

phonemes_to_show.sort(key=lambda x: -x[2])

header = "{:>6} {:>7} {:>6} {:>7} {:>7} {:>7}".format("Phoneme", "Count", "TP", "Prec", "Recall", "F1")
print(header)
print('-' * 45)
total_err = 0
total_ph = 0
for pid, ph, cnt, tp, prec, rec, f1, fp, fn in phonemes_to_show:
    err = cnt - tp
    total_err += err
    total_ph += cnt
    print(f'{ph:>6} {cnt:>7} {tp:>6} {prec:.4f} {rec:.4f} {f1:.4f}')
per = total_err / total_ph if total_ph > 0 else 0
print('-' * 45)
print(f'Total PER: {per:.4f} ({total_err}/{total_ph})')

# === Confusion Matrix Plot ===
top_k = min(30, len(phonemes_to_show))
top_phones = phonemes_to_show[:top_k]
labels = [p[1] for p in top_phones]
indices = [p[0] for p in top_phones]

cm = conf_matrix[np.ix_(indices, indices)]
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True).clip(min=1)

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=7, rotation=90)
ax.set_yticklabels(labels, fontsize=7)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Target', fontsize=12)
ax.set_title(f'Ma tr\u1eadn nh\u1ea7m l\u1eabn \u2014 CNN-BiLSTM-CTC (Top {top_k} phonemes, PER={per:.4f})', fontsize=14)

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
for i in range(len(labels)):
    for j in range(len(labels)):
        val = cm_norm[i, j]
        text_color = 'white' if val > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=5, color=text_color)

plt.tight_layout()
out_path = 'CNN_BiLSTM_CTC/plots/confusion_matrix.png'
fig.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved: {out_path}')

# === Per-Phoneme F1 Bar Chart ===
fig2, ax2 = plt.subplots(figsize=(12, 6))
ph_names = [p[1] for p in phonemes_to_show]
f1_scores = [p[6] for p in phonemes_to_show]
colors = ['green' if f >= 0.7 else 'orange' if f >= 0.4 else 'red' for f in f1_scores]
bars = ax2.bar(range(len(ph_names)), f1_scores, color=colors)
ax2.set_xticks(range(len(ph_names)))
ax2.set_xticklabels(ph_names, fontsize=6, rotation=90)
ax2.set_ylabel('F1 Score', fontsize=12)
ax2.set_title(f'F1 Score theo t\u1eebng \u00e2m v\u1ecb \u2014 CNN-BiLSTM-CTC ({len(phonemes_to_show)} phonemes)', fontsize=14)
ax2.set_ylim(0, 1.05)
ax2.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='F1=0.7')
ax2.legend()
plt.tight_layout()
f1_path = 'CNN_BiLSTM_CTC/plots/per_phoneme_f1.png'
fig2.savefig(f1_path, dpi=200, bbox_inches='tight')
plt.close(fig2)
print(f'Saved: {f1_path}')
