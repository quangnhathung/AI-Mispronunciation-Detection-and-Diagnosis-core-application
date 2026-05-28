import os
import csv
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import L2ArcticDataset, MDDCollate
from src.model.mmd_model_v2 import MDDModelV2
from src.data.dictionary import ID_TO_PHONEME

def evaluate_model():
    print("=== BẮT ĐẦU KIỂM THỬ MÔ HÌNH TRÊN TẬP TEST ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    root_dir = "./data/raw"
    test_speakers = ["suitcase_corpus"] 
    
    print(f"[*] Đang load Test Dataset với các speakers: {test_speakers}...")
    test_dataset = L2ArcticDataset(root_dir=root_dir, speaker_list=test_speakers)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=2,
        shuffle=False,
        collate_fn=MDDCollate(pad_phoneme_id=0),
        num_workers=2
    )


    print("[*] Đang load trọng số best_mdd_model_v4.pt...")
    vocab_size = 46 
    model = MDDModelV2(vocab_size=vocab_size).to(device)
    
    model_path = "./checkpoints/best_mdd_model_v4.pt"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    model.eval()

    all_records = []

    progress_bar = tqdm(test_loader, desc="Đang chạy kiểm thử")
    
    with torch.no_grad():
        for batch in progress_bar:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            canonical_ids = batch["canonical_ids"].to(device)
            target_scores = batch["target_scores"].to(device)
            
            logits, _ = model(
                input_values=input_values, 
                attention_mask=attention_mask, 
                canonical_ids=canonical_ids
            )
            
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            for b in range(len(batch["utt_ids"])):
                utt_id = batch["utt_ids"][b]
                speaker = batch["speakers"][b]
                for n in range(canonical_ids.shape[1]):
                    if target_scores[b, n] == -100.0:
                        continue
                    phoneme_id = canonical_ids[b, n].item()
                    phoneme = ID_TO_PHONEME.get(phoneme_id, "UNK")
                    target = target_scores[b, n].item()
                    pred = preds[b, n].item()
                    score = probs[b, n].item()
                    all_records.append({
                        "utterance": utt_id,
                        "speaker": speaker,
                        "phoneme_id": phoneme_id,
                        "phoneme": phoneme,
                        "target": target,
                        "prediction": pred,
                        "probability": f"{score:.6f}"
                    })

    total_phonemes = len(all_records)
    tp = sum(1 for r in all_records if r["target"] == 0.0 and r["prediction"] == 0.0)
    fp = sum(1 for r in all_records if r["target"] == 1.0 and r["prediction"] == 0.0)
    fn = sum(1 for r in all_records if r["target"] == 0.0 and r["prediction"] == 1.0)
    tn = sum(1 for r in all_records if r["target"] == 1.0 and r["prediction"] == 1.0)

    accuracy = (tp + tn) / (total_phonemes + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)

    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "evaluation_predictions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["utterance", "speaker", "phoneme_id", "phoneme", "target", "prediction", "probability"])
        writer.writeheader()
        writer.writerows(all_records)
    print(f"[*] Đã lưu chi tiết dự đoán tại: {csv_path}")

    print("\n")
    print("TEST SET EVALUATION")
    print(f"Tổng số âm vị đã kiểm tra : {total_phonemes}")
    print(f"Tổng số lỗi thực tế có    : {tp + fn}")
    print(f"Số lỗi mô hình bắt trúng  : {tp}")
    print("-" * 50)
    print(f"Độ chính xác tổng (Accuracy) : {accuracy:.4f}")
    print(f"Độ chuẩn xác báo lỗi (Precision) : {precision:.4f}")
    print(f"Độ bao phủ lỗi (Recall)      : {recall:.4f}")
    print(f"Điểm F1-Score (Lớp Lỗi)      : {f1_score:.4f}")

    print("\n" + "=" * 55)
    print("MA TRẬN NHẪM LẪN (Confusion Matrix)")
    print("=" * 55)
    print(f"{'':>14} {'Pred: Đúng':>14} {'Pred: Sai':>14}")
    print(f"{'Actual: Đúng':>14} {tn:>14} {fp:>14}")
    print(f"{'Actual: Sai ':>14} {fn:>14} {tp:>14}")
    print("=" * 55)

    cm = [[tn, fp], [fn, tp]]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    classes = ["Đúng (1.0)", "Sai (0.0)"]

    im = ax.imshow(cm, cmap="Blues")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Số lượng")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    max_val = max(tn, tp, fp, fn)
    for i in range(2):
        for j in range(2):
            val = cm[i][j]
            color = "white" if val > max_val * 0.6 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=15, fontweight="bold", color=color)

    ax.set_xlabel("Dự đoán (Predicted)", fontsize=12)
    ax.set_ylabel("Thực tế (Actual)", fontsize=12)
    ax.set_title("Confusion Matrix – MDD Model V4", fontsize=13, fontweight="bold")

    fig.tight_layout()
    chart_path = os.path.join(log_dir, "confusion_matrix.png")
    fig.savefig(chart_path, dpi=150)
    print(f"[*] Đã lưu biểu đồ confusion matrix tại: {chart_path}")
    plt.close(fig)
