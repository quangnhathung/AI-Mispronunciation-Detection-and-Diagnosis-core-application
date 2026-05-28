import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import từ các module của dự án
from src.data.dataset import L2ArcticDataset, MDDCollate
from src.model.mdd_model import MDDModel

def train_model_v2(epoch=25):
    print("=== KHỞI ĐỘNG QUÁ TRÌNH HUẤN LUYỆN MDD (V2 - TẬP TRUNG BẮT LỖI) ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Đang sử dụng thiết bị: {device}")
    if torch.cuda.is_available():
        print(f"[*] Tên GPU: {torch.cuda.get_device_name(0)}")


    # 1. CHUẨN BỊ DỮ LIỆU
    root_dir = "./data/raw"
    # Lấy thêm người nói để tăng tính đa dạng (ví dụ: Ấn Độ, Hàn Quốc, Ả Rập)
    train_speakers = ["ABA", "ASI", "BWC","EBVS","ERMS",
                      "HUK,", "HKK", "HQTV", "LXC", "MBMPS",
                    "NCC", "NJS","PNV","RRBI","SKA","SVBI","THV"] 
    
    print("[*] Đang load Dataset...")
    train_dataset = L2ArcticDataset(root_dir=root_dir, speaker_list=train_speakers)
    
    batch_size = 2 
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=MDDCollate(pad_phoneme_id=0),
        num_workers=2
    )

    # 2. KHỞI TẠO MÔ HÌNH
    print("[*] Đang khởi tạo mô hình...")
    vocab_size = 46 
    model = MDDModel(vocab_size=vocab_size).to(device)

    # 3. TỐI ƯU HÓA PHÂN TẦNG
    optimizer = torch.optim.AdamW([
        {'params': model.wav2vec2.parameters(), 'lr': 1e-5},
        {'params': model.phoneme_embedding.parameters(), 'lr': 1e-4},
        {'params': model.cross_attention.parameters(), 'lr': 1e-4},
        {'params': model.scoring_head.parameters(), 'lr': 1e-4}
    ], weight_decay=0.01)

    # Hàm Loss thô (Reduction='none' để tự nhân trọng số)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    # Trọng số phạt (Phạt gấp 4 lần nếu đoán sai các âm vị lỗi)
    ERROR_WEIGHT = 4.0 
    CORRECT_WEIGHT = 1.0

    # 4. VÒNG LẶP HUẤN LUYỆN
    num_epochs = epoch
    best_f1_error = 0.0 # Chuyển sang lưu model dựa trên F1-Score thay vì Loss
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n[*] Bắt đầu huấn luyện trong {num_epochs} Epochs...\n")
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        
        # Biến đếm để tính F1-Score cho lớp LỖI (Target = 0.0)
        total_true_errors = 0   # Mô hình báo Lỗi, Thực tế là Lỗi
        total_false_errors = 0  # Mô hình báo Lỗi, Thực tế là Đúng (Báo động giả)
        total_actual_errors = 0 # Tổng số Lỗi thực tế có trong dữ liệu
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            canonical_ids = batch["canonical_ids"].to(device)
            target_scores = batch["target_scores"].to(device)
            
            optimizer.zero_grad()
            
            logits, _ = model(
                input_values=input_values, 
                attention_mask=attention_mask, 
                canonical_ids=canonical_ids
            )
            
            # === BƯỚC 1: TÍNH WEIGHTED MASKED LOSS ===
            raw_loss = criterion(logits, target_scores)
            valid_mask = (target_scores != -100.0).float()
            
            # Phân bổ trọng số: target == 0.0 thì nhận ERROR_WEIGHT, ngược lại nhận CORRECT_WEIGHT
            weight_matrix = torch.where(target_scores == 0.0, 
                                        torch.tensor(ERROR_WEIGHT, device=device), 
                                        torch.tensor(CORRECT_WEIGHT, device=device))
            
            # Tính Loss cuối cùng
            loss = (raw_loss * weight_matrix * valid_mask).sum() / valid_mask.sum()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # === BƯỚC 2: THỐNG KÊ MA TRẬN NHẦM LẪN (Dành cho lớp Lỗi) ===
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float() # Threshold 0.5
                
                # Tìm các vị trí hợp lệ
                valid_indices = (target_scores != -100.0)
                
                # Đếm số lượng
                preds_valid = preds[valid_indices]
                targets_valid = target_scores[valid_indices]
                
                # Lớp LỖI là lớp 0.0
                true_errors = ((preds_valid == 0.0) & (targets_valid == 0.0)).sum().item()
                false_errors = ((preds_valid == 0.0) & (targets_valid == 1.0)).sum().item()
                actual_errors = (targets_valid == 0.0).sum().item()
                
                total_true_errors += true_errors
                total_false_errors += false_errors
                total_actual_errors += actual_errors

            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # === BƯỚC 3: TÍNH TOÁN CHỈ SỐ SAU MỖI EPOCH ===
        avg_loss = total_train_loss / len(train_loader)
        
        # Tránh chia cho 0
        precision = total_true_errors / (total_true_errors + total_false_errors + 1e-8)
        recall = total_true_errors / (total_actual_errors + 1e-8)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        print(f"\n[Epoch {epoch+1} Báo Cáo]")
        print(f" - Train Loss : {avg_loss:.4f}")
        print(f" - Bắt Lỗi (Error Class) -> Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1_score:.4f}")
        
        # Lưu Checkpoint dựa trên F1-Score của việc bắt lỗi
        if f1_score > best_f1_error:
            best_f1_error = f1_score
            save_path = os.path.join(save_dir, "best_mdd_model_v2.pt")
            torch.save(model.state_dict(), save_path)
            print(f"[+] Mô hình BẮT LỖI đỉnh nhất! Đã lưu tại: {save_path}")
