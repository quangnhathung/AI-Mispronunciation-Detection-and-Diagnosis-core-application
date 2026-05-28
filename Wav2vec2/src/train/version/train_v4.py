import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import L2ArcticDataset, MDDCollate
from src.model.mmd_model_v2 import MDDModelV2
from src.util.stopping import EarlyStopping
from src.train.draw.draw import plot_training_history 

from torch.utils.tensorboard import SummaryWriter

def train_v4(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    ERROR_WEIGHT, CORRECT_WEIGHT = 4.0, 1.0

    progress_bar = tqdm(dataloader, desc="[Train] Đang học")
    for batch in progress_bar:
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        canonical_ids = batch["canonical_ids"].to(device)
        target_scores = batch["target_scores"].to(device)
        
        optimizer.zero_grad()
        logits, _ = model(input_values, attention_mask, canonical_ids)
        
        raw_loss = criterion(logits, target_scores)
        valid_mask = (target_scores != -100.0).float()
        weight_matrix = torch.where(target_scores == 0.0, 
                                    torch.tensor(ERROR_WEIGHT, device=device), 
                                    torch.tensor(CORRECT_WEIGHT, device=device))
        
        loss = (raw_loss * weight_matrix * valid_mask).sum() / valid_mask.sum()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, device):
    model.eval() 
    
    total_true_errors = 0
    total_false_errors = 0
    total_actual_errors = 0

    progress_bar = tqdm(dataloader, desc="[Valid] Đang thi thử")
    with torch.no_grad():
        for batch in progress_bar:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            canonical_ids = batch["canonical_ids"].to(device)
            target_scores = batch["target_scores"].to(device)
            
            logits, _ = model(input_values, attention_mask, canonical_ids)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            
            valid_indices = (target_scores != -100.0)
            preds_valid = preds[valid_indices]
            targets_valid = target_scores[valid_indices]
            
            # Thống kê bắt lỗi
            total_true_errors += ((preds_valid == 0.0) & (targets_valid == 0.0)).sum().item()
            total_false_errors += ((preds_valid == 0.0) & (targets_valid == 1.0)).sum().item()
            total_actual_errors += (targets_valid == 0.0).sum().item()

    precision = total_true_errors / (total_true_errors + total_false_errors + 1e-8)
    recall = total_true_errors / (total_actual_errors + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return precision, recall, f1_score

def train_model_with_validation():
    print("=== KHỞI ĐỘNG HUẤN LUYỆN (V4 - LOGGING & VISUALIZATION) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- SETUP THƯ MỤC LOG ---
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 1. Khởi tạo CSV Logger
    csv_path = os.path.join(log_dir, "training_history.csv")
    csv_file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Epoch", "Train_Loss", "Val_Precision", "Val_Recall", "Val_F1"])
    
    # 2. Khởi tạo TensorBoard Writer
    tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard_runs"))
    
    # 3. Dictionary lưu cục bộ để vẽ Matplotlib
    history = {'train_loss': [], 'val_precision': [], 'val_recall': [], 'val_f1': []}
    
    # --- DATA & MODEL SETUP ---
    root_dir = "./data/raw"
    train_speakers = ["ABA", "ASI", "BWC", "EBVS", "ERMS", 
                      "HUK", "HKK", "HQTV", "LXC", "MBMPS",
                      "NCC", "NJS", "PNV", "RRBI", "SKA", "SVBI", "THV", "TXHC", "YBAA"] 
    val_speakers = ["TLV", "TNI"]
    
    train_loader = DataLoader(L2ArcticDataset(root_dir, train_speakers), batch_size=2, shuffle=True, collate_fn=MDDCollate(), num_workers=2)
    val_loader = DataLoader(L2ArcticDataset(root_dir, val_speakers), batch_size=2, shuffle=False, collate_fn=MDDCollate(), num_workers=2)

    model = MDDModelV2(vocab_size=46).to(device)
    optimizer = torch.optim.AdamW([
        {'params': model.wav2vec2.parameters(), 'lr': 1e-5},
        {'params': model.phoneme_embedding.parameters(), 'lr': 1e-4},
        {'params': model.phoneme_rnn.parameters(), 'lr': 1e-4},
        {'params': model.cross_attention.parameters(), 'lr': 1e-4},
        {'params': model.scoring_head.parameters(), 'lr': 1e-4}
    ], weight_decay=0.01)
    
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    early_stopping = EarlyStopping(patience=5, save_dir="./checkpoints", filename="best_mdd_model_v4.pt")

    # --- VÒNG LẶP HUẤN LUYỆN ---
    num_epochs = 50 
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        train_loss = train_v4(model, train_loader, optimizer, criterion, device)
        val_precision, val_recall, val_f1 = validate_epoch(model, val_loader, device)
        
        print(f"[Báo Cáo] Train Loss: {train_loss:.4f} | Val P: {val_precision:.4f} | Val R: {val_recall:.4f} | Val F1: {val_f1:.4f}")
        
        # Ghi log vào CSV
        csv_writer.writerow([epoch+1, train_loss, val_precision, val_recall, val_f1])
        csv_file.flush()
        
        # Ghi log lên TensorBoard
        tb_writer.add_scalar("Loss/Train", train_loss, epoch)
        tb_writer.add_scalar("Metrics/Precision", val_precision, epoch)
        tb_writer.add_scalar("Metrics/Recall", val_recall, epoch)
        tb_writer.add_scalar("Metrics/F1_Score", val_f1, epoch)
        
        # Lưu vào history cho Matplotlib
        history['train_loss'].append(train_loss)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        
        early_stopping(val_f1, model)
        if early_stopping.early_stop:
            print("[!] Dừng sớm do mô hình không cải thiện.")
            break

    csv_file.close()
    tb_writer.close()
    
    plot_training_history(history)
