"""
GIAI ĐOẠN 1 — Cấu hình tối ưu cho 8GB VRAM
"""
import platform
import torch

class Config:
    # --- Đường dẫn ---
    RAW_TRAIN_PATH = r"D:\HungQuang_WorkSpace\MDD_AI\data\raw_splitted\train"
    RAW_VAL_PATH   = r"D:\HungQuang_WorkSpace\MDD_AI\data\raw_splitted\val"
    RAW_TEST_PATH  = r"D:\HungQuang_WorkSpace\MDD_AI\data\raw_splitted\test"

    TRAIN_PATH = r"D:\HungQuang_WorkSpace\MDD_AI\data\processed_16k\train"
    VAL_PATH   = r"D:\HungQuang_WorkSpace\MDD_AI\data\processed_16k\val"
    TEST_PATH  = r"D:\HungQuang_WorkSpace\MDD_AI\data\processed_16k\test"

    TARGET_SAMPLE_RATE = 16000
    SAVE_DIR = "./checkpoints_phoneme_8vram" 

    # --- Độ dài (Tăng lên 10s) ---
    MAX_SAMPLES       = 160000   
    MAX_SAMPLES_TRAIN = 160000   

    # --- Bộ não (Giữ nguyên hoặc tăng D_MODEL nếu muốn sâu hơn) ---
    D_MODEL    = 256   
    NHEAD      = 8     
    NUM_LAYERS = 4    
    DROPOUT    = 0.15  

    # --- CẤU HÌNH BATCH CHO 8GB ---
    BATCH_SIZE          = 4     
    ACCUMULATION_STEPS  = 8     # 8 * 4 = 32 (Effective Batch)
    LR                  = 3e-4   
    WEIGHT_DECAY        = 1e-2   
    NUM_EPOCHS          = 50     
    DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    USE_GRADIENT_CHECKPOINT = True  

    # --- LR Schedule ---
    USE_ONE_CYCLE_LR    = True
    WARMUP_PCT          = 0.10   
    DIV_FACTOR          = 25     
    FINAL_DIV_FACTOR    = 1e4    

    # --- SpecAugment ---
    USE_SPEC_AUGMENT    = True
    SPEC_FREQ_MASK_MAX  = 32   
    SPEC_TIME_MASK_MAX  = 50
    SPEC_NUM_FREQ_MASKS = 2
    SPEC_NUM_TIME_MASKS = 2

    # --- Decode ---
    VAL_DECODER  = "greedy"   
    TEST_DECODER = "greedy"
    BEAM_SIZE    = 1            

    EARLY_STOPPING_PATIENCE = 8   
    EARLY_STOPPING_MIN_DELTA = 0.001

    VAL_EVAL_MAX_BATCHES = 50 
    NUM_WORKERS          = 0  # Tăng lên 4 để máy 8GB load dữ liệu nhanh hơn
    PREFETCH_FACTOR      = 2       
    BUCKET_BATCHING      = True
    BUCKET_MAX_WAV_LEN   = MAX_SAMPLES_TRAIN