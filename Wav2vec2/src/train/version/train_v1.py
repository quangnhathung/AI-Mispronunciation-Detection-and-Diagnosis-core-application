import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data.dataset import L2ArcticDataset, MDDCollate
from src.model.mdd_model import MDDModel

def train_model(epoch):
    print("=== KHỞI ĐỘNG QUÁ TRÌNH HUẤN LUYỆN MDD ===")
    
    # 1. THIẾT LẬP THIẾT BỊ (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Đang sử dụng thiết bị: {device}")
    if torch.cuda.is_available():
        print(f"[*] Tên GPU: {torch.cuda.get_device_name(0)}")

    # 2. CHUẨN BỊ DỮ LIỆU
    root_dir = "./data/raw"
    # Lấy 2 người nói để train thử. (Thực tế bạn sẽ chia train/val speakers)
    train_speakers = ["ABA", "SKA"] 
    
    print("[*] Đang load Dataset...")
    train_dataset = L2ArcticDataset(root_dir=root_dir, speaker_list=train_speakers)
    
    # Kích thước Batch Size: Nên để nhỏ (2-4) vì Wav2Vec2 tốn rất nhiều VRAM
    batch_size = 2 
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=MDDCollate(pad_phoneme_id=0), # SỬA CHỖ NÀY
        num_workers=2
    )

    # 3. KHỞI TẠO MÔ HÌNH
    print("[*] Đang khởi tạo mô hình...")
    vocab_size = 46 # Tổng số phonemes trong dictionary.py của bạn
    model = MDDModel(vocab_size=vocab_size).to(device)

    # 4. TỐI ƯU HÓA PHÂN TẦNG (Layer-wise Learning Rate)
    # Cơ thể Wav2Vec2 đã được pre-train kỹ nên chỉ cần LR nhỏ (1e-5)
    # Các khối mới thêm (Attention, Scoring) bắt đầu từ con số 0 nên cần LR lớn hơn (1e-4)
    optimizer = torch.optim.AdamW([
        {'params': model.wav2vec2.parameters(), 'lr': 1e-5},
        {'params': model.phoneme_embedding.parameters(), 'lr': 1e-4},
        {'params': model.cross_attention.parameters(), 'lr': 1e-4},
        {'params': model.scoring_head.parameters(), 'lr': 1e-4}
    ], weight_decay=0.01)

    # 5. HÀM LOSS ĐẶC CHẾ (Masked BCE)
    # Để reduction='none' để trả về ma trận Loss thay vì 1 con số
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    # 6. VÒNG LẶP HUẤN LUYỆN
    num_epochs = epoch
    best_loss = float('inf')
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n[*] Bắt đầu huấn luyện trong {num_epochs} Epochs...\n")
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        
        # Thanh tiến trình TQDM cho đẹp và dễ theo dõi
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # Chuyển dữ liệu lên GPU
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            canonical_ids = batch["canonical_ids"].to(device)
            target_scores = batch["target_scores"].to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass: Đưa dữ liệu qua mô hình
            logits, _ = model(
                input_values=input_values, 
                attention_mask=attention_mask, 
                canonical_ids=canonical_ids
            )
            
            # === XỬ LÝ MASKED LOSS ĐỂ BỎ QUA -100.0 ===
            # Tính loss thô trên toàn bộ ma trận (B, N_max)
            raw_loss = criterion(logits, target_scores)
            
            # Tạo mặt nạ: Chỗ nào != -100.0 thì là 1 (giữ lại), ngược lại là 0 (bỏ đi)
            valid_mask = (target_scores != -100.0).float()
            
            # Nhân loss với mask và tính trung bình chỉ trên các phần tử hợp lệ
            loss = (raw_loss * valid_mask).sum() / valid_mask.sum()
            
            # Backward Pass: Tính đạo hàm
            loss.backward()
            
            # Gradient Clipping: Cắt xén gradient để chống "nổ" gradient (đặc biệt quan trọng với Wav2Vec)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Cập nhật trọng số
            optimizer.step()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_train_loss / len(train_loader)
        print(f"\n[Epoch {epoch+1} Kết thúc] - Average Training Loss: {avg_loss:.4f}")
        
        # Lưu lại Checkpoint nếu loss giảm
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(save_dir, "best_mdd_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"[+] Đã lưu checkpoint tốt nhất tại: {save_path}")