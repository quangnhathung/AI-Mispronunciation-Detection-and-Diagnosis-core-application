import os
import torch

"""
Trọng tài theo dõi quá trình huấn luyện. 
Nếu điểm F1 trên tập Validation không tăng sau 'patience' vòng, tự động ngắt.
"""

class EarlyStopping:
    def __init__(self, patience=5, save_dir="./checkpoints", filename="best_mdd_model_v3.pt"):
        self.patience = patience
        self.counter = 0
        self.best_score = 0.0
        self.early_stop = False
        self.save_path = os.path.join(save_dir, filename)
        os.makedirs(save_dir, exist_ok=True)

    def __call__(self, current_f1_score, model):
        # Nếu điểm số mới cao hơn kỷ lục cũ -> Cập nhật và lưu mô hình
        if current_f1_score > self.best_score:
            print(f"\n[+] Validation F1 TĂNG ({self.best_score:.4f} --> {current_f1_score:.4f}). Đã lưu mô hình tốt nhất!")
            self.best_score = current_f1_score
            self.save_checkpoint(model)
            self.counter = 0 # Reset bộ đếm sự kiên nhẫn
            
        # Nếu điểm số giảm hoặc đi ngang (Bắt đầu học vẹt) -> Tăng bộ đếm
        else:
            self.counter += 1
            print(f"\n[-] Validation F1 không tăng. Sự kiên nhẫn: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print("\n[!] KÍCH HOẠT EARLY STOPPING! Mô hình bắt đầu Overfitting. Dừng huấn luyện ngay lập tức.")

    def save_checkpoint(self, model):
        """Lưu lại trạng thái tốt nhất của mô hình."""
        torch.save(model.state_dict(), self.save_path)