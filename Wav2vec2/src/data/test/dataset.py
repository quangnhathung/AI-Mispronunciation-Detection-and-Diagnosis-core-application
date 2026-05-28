import torch
from torch.utils.data import DataLoader
from  ..dataset import L2ArcticDataset,collate_fn_factory
import os
import time

def run_pipeline_test():
    print("=== BẮT ĐẦU KIỂM THỬ DATA PIPELINE ===")
    
    # 1. Cấu hình đường dẫn (Hãy đổi thành đường dẫn thực tế trên máy bạn)
    root=os.getcwd();
    root_data_dir = root+"/data/raw" 
    print(f"Root path: {root_data_dir}")
    time.sleep(1)
    # Test thử với 2 speaker theo cấu trúc thư mục của L2-ARCTIC
    test_speakers = ["ABA"] 
    
    print(f"1. Đang khởi tạo L2ArcticDataset cho speakers: {test_speakers}...")
    try:
        dataset = L2ArcticDataset(
            root_dir=root_data_dir, 
            speaker_list=test_speakers, 
            target_sr=16000
        )
    except Exception as e:
        print(f"[LỖI CRITICAL] Không thể khởi tạo Dataset. Chi tiết: {e}")
        return

    print(f"-> Tổng số mẫu dữ liệu quét được: {len(dataset)}")
    if len(dataset) == 0:
        print("[CẢNH BÁO] Không tìm thấy dữ liệu! Hãy kiểm tra lại cấu trúc thư mục hoặc đường dẫn root_dir.")
        return

    # 2. Khởi tạo DataLoader
    batch_size = 4
    print(f"\n2. Đang khởi tạo DataLoader với Batch Size = {batch_size}...")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True, # Lấy ngẫu nhiên để kiểm tra padding của các câu dài/ngắn khác nhau
        collate_fn=collate_fn_factory(pad_phoneme_id=0),
        num_workers=0 # Để 0 khi test để dễ dàng in lỗi (stack trace) nếu có
    )

    # 3. Kéo thử 1 Batch và mổ xẻ các Tensor
    print("\n3. Đang kéo (fetch) thử 1 batch dữ liệu...")
    for batch_idx, batch in enumerate(dataloader):
        print("\n" + "="*40)
        print(f"THÔNG TIN BATCH THỨ {batch_idx + 1}")
        print("="*40)
        
        # --- Khối Âm thanh (Audio) ---
        print("\n[KHỐI ÂM THANH - Dành cho Wav2Vec 2.0]")
        print(f" - input_values shape: {batch['input_values'].shape} -> (Batch, T_max)")
        print(f" - attention_mask shape: {batch['attention_mask'].shape} -> (Batch, T_max)")
        print(f" - audio_lengths thực tế: {batch['audio_lengths'].tolist()}")
        
        # --- Khối Âm vị & Nhãn (Phonemes & Targets) ---
        print("\n[KHỐI ÂM VỊ & NHÃN - Dành cho Cross-Attention & Scoring]")
        print(f" - canonical_ids shape: {batch['canonical_ids'].shape} -> (Batch, N_max)")
        print(f" - target_scores shape: {batch['target_scores'].shape} -> (Batch, N_max)")
        print(f" - phoneme_intervals shape: {batch['phoneme_intervals'].shape} -> (Batch, N_max, 2)")
        
        # --- Khối Metadata ---
        print("\n[METADATA]")
        print(f" - Speakers: {batch['speakers']}")
        print(f" - Utterance IDs: {batch['utt_ids']}")
        
        # --- Đào sâu: Kiểm tra logic Padding ---
        print("\n[ĐÀO SÂU: KIỂM TRA PADDING]")
        sample_idx = 0 # Lấy mẫu đầu tiên trong batch để soi
        real_ph_len = (batch['canonical_ids'][sample_idx] != 0).sum().item() # Đếm số âm vị thật (khác PAD_ID = 0)
        
        print(f" - Mẫu 0 có {real_ph_len} âm vị thực tế. Các vị trí sau đó phải bị pad.")
        print(f" - Target Scores của Mẫu 0 (Rút gọn):")
        # In ra 10 phần tử cuối để xem hàm collate_fn có điền giá trị -100.0 chuẩn không
        print(f"   {batch['target_scores'][sample_idx][-10:].tolist()} (Kỳ vọng thấy các giá trị -100.0 ở cuối)")
        
        # VERIFY LOGIC (RẤT QUAN TRỌNG)
        print("\n[VERIFY LOGIC - CROSS CHECK]")

        # ========== (1) Padding alignment ==========
        print("\n(1) Check padding alignment (canonical_ids vs target_scores)")
        pad_id = 0

        for b in range(batch['canonical_ids'].shape[0]):
            ph = batch['canonical_ids'][b]
            scores = batch['target_scores'][b]

            cond1 = (ph == pad_id)
            cond2 = (scores == -100)

            mismatch = (cond1 != cond2).sum().item()

            if mismatch == 0:
                print(f" - Sample {b}: OK")
            else:
                print(f" - Sample {b}:  Mismatch {mismatch} vị trí")

        # ========== (2) Interval padding ==========
        print("\n(2) Check interval padding")

        for b in range(batch['phoneme_intervals'].shape[0]):
            intervals = batch['phoneme_intervals'][b]
            scores = batch['target_scores'][b]

            pad_mask = (scores == -100)

            if pad_mask.sum() == 0:
                print(f" - Sample {b}: No padding")
                continue

            pad_intervals = intervals[pad_mask]

            if torch.all(pad_intervals == 0):
                print(f" - Sample {b}: OK")
            else:
                print(f" - Sample {b}: ❌ Interval padding sai")

        # ========== (3) Attention mask ==========
        print("\n(3) Check attention mask")

        mask_sum = batch['attention_mask'].sum(dim=1)
        lengths = batch['audio_lengths']

        for b in range(len(lengths)):
            if mask_sum[b].item() == lengths[b].item():
                print(f" - Sample {b}: OK")
            else:
                print(f" - Sample {b}: ❌ Sai mask")
        
        break # Chỉ cần test 1 batch đầu tiên là đủ để verify pipeline

