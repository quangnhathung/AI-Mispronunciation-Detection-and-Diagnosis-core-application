import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter # Thêm TensorBoard

from src.data.dataset import L2ArcticDataset, MDDCollate
from src.model.mmd_model_v2 import MDDModelV2
from src.util.stopping import EarlyStopping

def plot_training_history(history, save_path="./logs/training_chart.png"):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Biểu đồ Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='red', marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Biểu đồ Metrics
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_precision'], label='Val Precision', linestyle='--')
    plt.plot(epochs, history['val_recall'], label='Val Recall', linestyle='-.')
    plt.plot(epochs, history['val_f1'], label='Val F1', color='blue', marker='s')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[*] Đã lưu biểu đồ tổng kết tại: {save_path}")
    plt.close()