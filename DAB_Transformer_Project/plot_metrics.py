"""

plot_metrics.py — Giai đoạn báo cáo sau train



Đọc loss / Val WER / Val CER từ các checkpoint → vẽ biểu đồ.

Không load dữ liệu âm thanh, không chạy mô hình.

"""

import torch
import matplotlib.pyplot as plt
import glob, os, re

import numpy as np
import jiwer

from config import Config
from model import DAB_Transformer
from dataset import L2ArcticDataset, make_dataloader
from utils import text_process, decode_logits



def draw_full_report():

    """Đọc checkpoints/model_e*.pt và lưu full_report_metrics.png."""

    checkpoints = glob.glob(os.path.join(Config.SAVE_DIR, "model_e*.pt"))

    checkpoints.sort(key=lambda f: int(re.search(r'e(\d+)', f).group(1)))



    epochs, losses, wers, cers = [], [], [], []



    for cp in checkpoints:

        data = torch.load(cp, map_location='cpu')

        if isinstance(data, dict) and 'loss' in data:

            epochs.append(data['epoch'])

            losses.append(data['loss'])

            wers.append(data.get('wer', 1.0))

            cers.append(data.get('cer', 1.0))



    if not epochs:

        print("❌ Chưa có dữ liệu chỉ số trong checkpoint!")

        return



    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))



    ax1.plot(epochs, losses, 'r-o', label='Training Loss')

    ax1.set_title('Sơ đồ độ hội tụ mô hình (Loss Curve)')

    ax1.set_ylabel('Loss Value')

    ax1.grid(True)

    ax1.legend()



    ax2.plot(epochs, [w*100 for w in wers], 'b-s', label='Val WER (Lỗi từ)')

    ax2.plot(epochs, [c*100 for c in cers], 'g-d', label='Val CER (Lỗi ký tự)')

    ax2.set_title('Sơ đồ tỷ lệ lỗi trên tập Val (Error Rate)')

    ax2.set_xlabel('Epoch')

    ax2.set_ylabel('Percentage (%)')

    ax2.grid(True)

    ax2.legend()



    plt.tight_layout()

    plt.savefig('full_report_metrics.png')

    print("✅ Đã lưu bộ sơ đồ tại: full_report_metrics.png")

    plt.show()



# ── Confusion Matrix ────────────────────────────────────────────────────────

def compute_confusion_matrix(model, data_loader, device, max_batches=None):
    num_classes = len(text_process.char_map)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    n_deletions = 0
    n_insertions = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, (wavs, labels, w_lens, l_lens) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            logits, _ = model(wavs.to(device))
            preds = decode_logits(logits, w_lens, text_process)

            for idx, pred in enumerate(preds):
                target = text_process.labels_to_text(labels[idx], l_lens[idx].item())
                if not target:
                    continue

                target_clean = " ".join(target.split())
                pred_clean = " ".join(pred.split())

                if not pred_clean:
                    n_deletions += len(target_clean.split())
                    continue

                alignment = jiwer.process_words(target_clean, pred_clean).alignments[0]
                t_words = target_clean.split()
                p_words = pred_clean.split()

                for chunk in alignment:
                    if chunk.type == 'equal':
                        for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                            idx_t = text_process.char_map[t_words[i]]
                            cm[idx_t, idx_t] += 1
                    elif chunk.type == 'substitute':
                        for i, j in zip(range(chunk.ref_start_idx, chunk.ref_end_idx),
                                        range(chunk.hyp_start_idx, chunk.hyp_end_idx)):
                            idx_t = text_process.char_map[t_words[i]]
                            idx_p = text_process.char_map[p_words[j]]
                            cm[idx_t, idx_p] += 1
                    elif chunk.type == 'delete':
                        for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                            n_deletions += 1
                    elif chunk.type == 'insert':
                        for j in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                            n_insertions += 1

    return cm, n_deletions, n_insertions


def plot_confusion_matrix(cm, save_path="confusion_matrix.png"):
    # Exclude blank (index 0) — only show real phonemes + space
    phoneme_ids = list(range(1, len(text_process.char_map)))
    active = [i for i in phoneme_ids if cm[i].sum() > 0]
    if len(active) <= 1:
        print("Not enough data for confusion matrix")
        return

    cm_sub = cm[np.ix_(active, active)]
    labels = [text_process.index_map[i] for i in active]

    row_totals = cm_sub.sum(axis=1, keepdims=True)
    row_totals = np.where(row_totals == 0, 1, row_totals)
    cm_pct = cm_sub.astype(float) / row_totals * 100

    fig, ax = plt.subplots(figsize=(max(10, len(active) * 0.4),
                                    max(8, len(active) * 0.35)))

    im = ax.imshow(cm_pct, cmap='Blues', aspect='auto', vmin=0, vmax=100)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Percentage (%)', fontsize=11)

    ax.set_xticks(range(len(active)))
    ax.set_yticks(range(len(active)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Predicted Phoneme', fontsize=11)
    ax.set_ylabel('True Phoneme', fontsize=11)
    ax.set_title('Phoneme Confusion Matrix', fontsize=13)

    for i in range(len(active)):
        pct = cm_pct[i, i]
        if cm_sub[i, i] > 0:
            c = 'white' if pct > 50 else 'black'
            ax.text(i, i, f"{cm_sub[i, i]:.0f}", ha='center', va='center',
                    fontsize=5, fontweight='bold', color=c)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix: {save_path}")


def draw_confusion_matrix(epoch_num=None, max_batches=None,
                          save_path="confusion_matrix.png"):
    if epoch_num is None:
        epoch_num = input("Epoch number: ").strip()

    ckpt_path = f"{Config.SAVE_DIR}/model_e{epoch_num}.pt"
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    model = DAB_Transformer(
        len(text_process.char_map), Config.D_MODEL, Config.NHEAD,
        Config.NUM_LAYERS
    ).to(Config.DEVICE)

    ckpt = torch.load(ckpt_path, map_location=Config.DEVICE, weights_only=True)
    state = (ckpt["model_state_dict"]
             if isinstance(ckpt, dict) and "model_state_dict" in ckpt
             else ckpt)
    model.load_state_dict(state)
    model.eval()

    test_loader = make_dataloader(
        L2ArcticDataset(Config.TEST_PATH, text_process), shuffle=False
    )

    print(f"Computing confusion matrix — epoch {epoch_num} ...")
    cm, n_del, n_ins = compute_confusion_matrix(
        model, test_loader, Config.DEVICE, max_batches
    )

    total = cm[1:].sum() + n_del + n_ins
    correct = int(np.trace(cm[1:, 1:]))
    subs = int(cm[1:, 1:].sum()) - correct
    print(f"  Total phonemes : {total}")
    print(f"  Correct        : {correct} ({correct/total*100:.2f}%)")
    print(f"  Substitutions  : {subs} ({subs/total*100:.2f}%)")
    print(f"  Deletions      : {n_del} ({n_del/total*100:.2f}%)")
    print(f"  Insertions     : {n_ins} ({n_ins/total*100:.2f}%)")

    plot_confusion_matrix(cm, save_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?",
                        choices=["report", "confusion"], default="report")
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--save", type=str, default="confusion_matrix.png")
    parser.add_argument("--show-samples", type=int, default=3)
    args = parser.parse_args()

    if args.mode == "confusion":
        draw_confusion_matrix(args.epoch, args.max_batches, args.save)
    else:
        draw_full_report()


