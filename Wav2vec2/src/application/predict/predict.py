import os
import torch
import soundfile as sf
import torchaudio
from g2p_en import G2p
from transformers import Wav2Vec2FeatureExtractor

# Import từ các file của dự án (Giữ nguyên)
from src.model.mmd_model_v2 import MDDModelV2
from src.data.dictionary import get_phoneme_id, ARPABET_PHONEMES

ID_TO_PHONEME = {i: ph for i, ph in enumerate(ARPABET_PHONEMES)}

# THÊM: Từ điển map ARPABET sang chuẩn IPA
ARPABET_TO_IPA = {
    'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ',
    'AY': 'aɪ', 'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð',
    'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'F': 'f', 'G': 'g',
    'HH': 'h', 'IH': 'ɪ', 'IY': 'i', 'JH': 'dʒ', 'K': 'k',
    'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'OW': 'oʊ',
    'OY': 'ɔɪ', 'P': 'p', 'R': 'ɹ', 'S': 's', 'SH': 'ʃ',
    'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v',
    'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ'
}

class SilenceDetected(Exception):
    pass

SILENCE_THRESHOLD = 0.02

class MDDPredictor:
    def __init__(self, model_path, vocab_size=46, device='cuda'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy file trọng số mô hình tại: {model_path}")
            
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"[*] Đang khởi chạy mô hình trên: {self.device}")
        
        self.g2p = G2p()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        self.target_sr = 16000
        
        self.model = MDDModelV2(vocab_size=vocab_size).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

    def text_to_phonemes(self, text):
        raw_phonemes = self.g2p(text)
        canonical_ids = []
        canonical_tokens = []
        
        for ph in raw_phonemes:
            if ph.isalpha() or ph[-1].isdigit(): 
                pid = get_phoneme_id(ph)
                canonical_ids.append(pid)
                canonical_tokens.append(ID_TO_PHONEME[pid])
                
        return torch.tensor([canonical_ids], dtype=torch.long), canonical_tokens

    def process_audio(self, wav_path):
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Không tìm thấy file âm thanh: {wav_path}")
            
        waveform_np, sr = sf.read(wav_path, dtype='float32')
        waveform = torch.from_numpy(waveform_np)
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()
            
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        rms = torch.sqrt(torch.mean(waveform ** 2))
        if rms < SILENCE_THRESHOLD:
            raise SilenceDetected(f"Không phát hiện giọng nói (RMS={rms:.6f} < ngưỡng {SILENCE_THRESHOLD})")
            
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.target_sr)

        processed = self.processor(waveform.squeeze(0), sampling_rate=self.target_sr, return_tensors="pt")
        return processed.input_values

    def predict(self, wav_path, expected_text, threshold=0.5):
        """
        Chẩn đoán phát âm. Trả về cấu trúc Dictionary tiêu chuẩn.
        """
        # THÊM: Block try..except để bắt lỗi trực tiếp mà không làm crash luồng app
        try:
            # 1. Chuẩn bị dữ liệu
            input_values = self.process_audio(wav_path).to(self.device)
            canonical_ids, canonical_tokens = self.text_to_phonemes(expected_text)
            canonical_ids = canonical_ids.to(self.device)
            
            attention_mask = torch.ones_like(input_values, dtype=torch.long).to(self.device)

            # 2. Đưa qua mô hình
            with torch.no_grad():
                logits, _ = self.model(
                    input_values=input_values, 
                    attention_mask=attention_mask, 
                    canonical_ids=canonical_ids
                )
                probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()

            # 3. Đóng gói kết quả (Return instead of Print)
            # Kiểm tra an toàn để tránh lỗi index out of range nếu model output lệch chiều dài
            seq_len = min(len(canonical_tokens), len(probabilities))
            
            details = []
            for i in range(seq_len):
                prob = float(probabilities[i]) # Ép kiểu numpy float -> python float
                is_correct = bool(prob >= threshold) # Ép kiểu numpy bool -> python bool
                
                # THÊM: Xử lý map sang IPA (lọc bỏ số nhấn âm như 0, 1, 2 nếu có)
                clean_phoneme = ''.join([c for c in canonical_tokens[i] if not c.isdigit()])
                ipa_symbol = ARPABET_TO_IPA.get(clean_phoneme, canonical_tokens[i])
                
                details.append({
                    "phoneme": canonical_tokens[i],
                    "ipa": ipa_symbol, # THÊM
                    "score": round(prob, 4),
                    "is_correct": is_correct,
                    "status_text": "ĐÚNG" if is_correct else "LỖI"
                })

            # Cấu trúc trả về bao quát toàn bộ thông tin ngữ cảnh
            return {
                "success": True, # THÊM cờ xác nhận thành công
                "file_path": wav_path,
                "text": expected_text,
                "threshold": threshold,
                "overall_accuracy": round(sum(1 for d in details if d["is_correct"]) / max(1, len(details)) * 100, 2),
                "details": details
            }
            
        except Exception as e:
            # THÊM: Trả về cục Dictionary lỗi thay vì văng Exception
            return {
                "success": False,
                "error_message": str(e),
                "file_path": wav_path,
                "text": expected_text,
                "overall_accuracy": 0.0,
                "details": []
            }

# ==========================================
# KHU VỰC KIỂM THỬ (TESTING) VÀ CÁCH SỬ DỤNG
# ==========================================
# def main():
#     MODEL_PATH = "./checkpoints/best_mdd_model_v4.pt"
#     TEST_WAV = "C:/Users/quang/Downloads/voice/human/everyday_1.wav"
#     TEST_TEXT = "Every day is a new chance to grow."
    
#     try:
#         # Khởi tạo predictor
#         predictor = MDDPredictor(model_path=MODEL_PATH)
        
#         # Lấy kết quả trả về
#         result = predictor.predict(wav_path=TEST_WAV, expected_text=TEST_TEXT, threshold=0.5)
        
#         # Có thể in ra định dạng cũ nếu muốn, hoặc trả thẳng JSON cho Frontend
#         print("\n" + "="*50)
#         print(f"KẾT QUẢ CHẨN ĐOÁN PHÁT ÂM")
#         print(f"File: {result['file_path']}")
#         print(f"Text: '{result['text']}'")
#         print(f"Độ chính xác tổng thể: {result['overall_accuracy']}%")
#         print("="*50)
#         print(f"{'ÂM VỊ':<10} | {'ĐIỂM SỐ':<10} | {'ĐÁNH GIÁ'}")
#         print("-" * 50)
        
#         for item in result['details']:
#             status_icon = "✔" if item['is_correct'] else "❌"
#             print(f"{item['phoneme']:<10} | {item['score']:.4f}     | {status_icon} {item['status_text']}")

#     except Exception as e:
#         print(f"[LỖI] Đã xảy ra sự cố trong quá trình dự đoán: {e}")