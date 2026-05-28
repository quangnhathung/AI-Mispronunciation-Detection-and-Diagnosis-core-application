import torch
import soundfile as sf
import torchaudio
from g2p_en import G2p
from transformers import Wav2Vec2FeatureExtractor

# Import từ các file của dự án
from src.model.mmd_model_v2 import MDDModelV2
from src.data.dictionary import get_phoneme_id, ARPABET_PHONEMES

# Tạo từ điển ngược để dịch ID về lại chữ cái âm vị
ID_TO_PHONEME = {i: ph for i, ph in enumerate(ARPABET_PHONEMES)}

class MDDPredictor:
    def __init__(self, model_path, vocab_size=46, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"[*] Đang khởi chạy mô hình trên: {self.device}")
        
        self.g2p = G2p()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        self.target_sr = 16000
        
        # Khởi tạo và Load trọng số
        self.model = MDDModelV2(vocab_size=vocab_size).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval() # Tắt chế độ train, khóa bộ nhớ Dropout

    def text_to_phonemes(self, text):
        """Chuyển đổi Text -> Danh sách ID âm vị chuẩn."""
        raw_phonemes = self.g2p(text)
        canonical_ids = []
        canonical_tokens = []
        
        for ph in raw_phonemes:
            # Lọc bỏ dấu câu (., ?, !)
            if ph.isalpha() or ph[-1].isdigit(): 
                pid = get_phoneme_id(ph)
                canonical_ids.append(pid)
                canonical_tokens.append(ID_TO_PHONEME[pid])
                
        return torch.tensor([canonical_ids], dtype=torch.long), canonical_tokens

    def process_audio(self, wav_path):
        """Đọc và chuẩn hóa file âm thanh thô."""
        waveform_np, sr = sf.read(wav_path, dtype='float32')
        waveform = torch.from_numpy(waveform_np)
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()
            
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.target_sr)

        processed = self.processor(waveform.squeeze(0), sampling_rate=self.target_sr, return_tensors="pt")
        return processed.input_values

    def predict(self, wav_path, expected_text, threshold=0.5):
        """Chẩn đoán phát âm."""
        # 1. Chuẩn bị dữ liệu
        input_values = self.process_audio(wav_path).to(self.device)
        canonical_ids, canonical_tokens = self.text_to_phonemes(expected_text)
        canonical_ids = canonical_ids.to(self.device)
        
        attention_mask = torch.ones_like(input_values, dtype=torch.long).to(self.device)

        # 2. Đưa qua mô hình
        with torch.no_grad(): # Khóa vi phân, tiết kiệm VRAM
            logits, _ = self.model(
                input_values=input_values, 
                attention_mask=attention_mask, 
                canonical_ids=canonical_ids
            )
            # Ép Logits qua Sigmoid để lấy xác suất [0, 1]
            probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        # 3. In kết quả trực quan
        print("\n" + "="*50)
        print(f"KẾT QUẢ CHẨN ĐOÁN PHÁT ÂM")
        print(f"File: {wav_path}")
        print(f"Text: '{expected_text}'")
        print("="*50)
        print(f"{'ÂM VỊ':<10} | {'ĐIỂM SỐ':<10} | {'ĐÁNH GIÁ'}")
        print("-" * 50)
        
        for i, token in enumerate(canonical_tokens):
            prob = probabilities[i]
            is_correct = prob >= threshold
            status = "✔ ĐÚNG" if is_correct else "❌ LỖI"
            print(f"{token:<10} | {prob:.4f}     | {status}")

#kiểm thử
def predict():
    MODEL_PATH = "./checkpoints/best_mdd_model_v4.pt"
    
    # test
    TEST_WAV = "C:/Users/quang/Downloads/voice/human/everyday_1.wav"
    #TEST_WAV = "C:/Users/quang/Downloads/voice/ai/everyday.wav"
    TEST_TEXT = "Every day is a new chance to grow."
    # TEST_WAV = "C:/Users/quang/Downloads/voice/human/believe_1.wav"
    # TEST_WAV = "C:/Users/quang/Downloads/voice/ai/believe.wav"
    #TEST_TEXT = "Believe in yourself and never give up."

    
    predictor = MDDPredictor(model_path=MODEL_PATH)
    predictor.predict(wav_path=TEST_WAV, expected_text=TEST_TEXT)