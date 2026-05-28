"""
test_inference.py — Cỗ máy bắt lỗi phát âm (Trả về JSON cho App Flutter)
"""
import os
import argparse
import json
import torch
import jiwer

from config import Config
from model import DAB_Transformer
from dataset import L2ArcticDataset, make_dataloader
from utils import text_process, decode_logits

def build_pronunciation_feedback(target, predicted):
    """
    Sử dụng Jiwer Alignment để gióng hàng và phân tích lỗi đọc.
    Trả về cấu trúc JSON sẵn sàng cho Frontend.
    """
    target_clean = " ".join(target.split())
    pred_clean = " ".join(predicted.split())
    
    # Nếu rỗng thì báo lỗi toàn bộ
    if not pred_clean:
        return [{"status": "error", "message": "Không nhận diện được giọng nói"}]
        
    alignment = jiwer.process_words(target_clean, pred_clean).alignments[0]
    
    t_words = target_clean.split()
    p_words = pred_clean.split()
    
    feedback_list = []
    
    for chunk in alignment:
        error_type = chunk.type
        
        # Đọc chuẩn xác
        if error_type == 'equal':
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                feedback_list.append({
                    "phoneme": t_words[i], 
                    "status": "correct", 
                    "predicted": t_words[i]
                })
        
        # Đọc nhầm âm (Ví dụ: B thành P)
        elif error_type == 'substitute':
            for i, j in zip(range(chunk.ref_start_idx, chunk.ref_end_idx), range(chunk.hyp_start_idx, chunk.hyp_end_idx)):
                feedback_list.append({
                    "phoneme": t_words[i], 
                    "status": "substitution", 
                    "predicted": p_words[j],
                    "message": f"Nhầm /{t_words[i]}/ thành /{p_words[j]}/"
                })
                
        # Nuốt âm (Không phát âm chữ đó)
        elif error_type == 'delete':
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                feedback_list.append({
                    "phoneme": t_words[i], 
                    "status": "deletion", 
                    "predicted": "-",
                    "message": f"Bạn bị nuốt âm /{t_words[i]}/"
                })
                
        # Dư âm (Tự chèn thêm âm lạ)
        elif error_type == 'insert':
            for j in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                feedback_list.append({
                    "phoneme": "-", 
                    "status": "insertion", 
                    "predicted": p_words[j],
                    "message": f"Phát âm thừa âm /{p_words[j]}/"
                })

    return feedback_list


def test_model(epoch_num=None, show_samples=3):
    if epoch_num is None:
        epoch_num = input("➡️ Nhập số Epoch bạn muốn test (ví dụ: 10): ").strip()

    checkpoint_path = f"{Config.SAVE_DIR}/model_e{epoch_num}.pt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Không tìm thấy file checkpoint: {checkpoint_path}")
        return

    # Khởi tạo model với đầu ra 41 Âm vị
    model = DAB_Transformer(
        len(text_process.char_map), Config.D_MODEL, Config.NHEAD, Config.NUM_LAYERS
    ).to(Config.DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE, weights_only=True)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()

    test_loader = make_dataloader(L2ArcticDataset(Config.TEST_PATH, text_process), shuffle=False)
    
    print(f"\n🎧 [PRONUNCIATION ASSESSMENT] - Chấm điểm Epoch {epoch_num}...\n")

    if show_samples > 0:
        with torch.no_grad():
            for i, (wavs, labels, w_lens, l_lens) in enumerate(test_loader):
                wavs = wavs.to(Config.DEVICE)
                logits, _ = model(wavs)
                
                pred = decode_logits(logits, w_lens, text_process)[0]
                target = text_process.labels_to_text(labels[0], l_lens[0].item())
                
                # Chạy AI bóc tách lỗi
                feedback_json = build_pronunciation_feedback(target, pred)
                
                print(f"================ MẪU {i+1} ================")
                # Biểu đồ lỗi ASCII trực quan của Jiwer
                if pred.strip():
                    print(jiwer.visualize_alignment(jiwer.process_words(" ".join(target.split()), " ".join(pred.split()))))
                
                print("\n[📦 Dữ liệu JSON Backend trả về App Flutter]:")
                print(json.dumps(feedback_json, indent=2, ensure_ascii=False))
                print("\n")
                
                if i + 1 >= show_samples:
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("epoch", nargs="?", help="Số epoch checkpoint")
    parser.add_argument("--show-samples", type=int, default=3)
    args = parser.parse_args()
    test_model(epoch_num=args.epoch, show_samples=args.show_samples)