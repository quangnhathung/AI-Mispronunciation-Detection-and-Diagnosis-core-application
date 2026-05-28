"""
utils.py — Xử lý Âm vị (Phoneme), CTC Greedy Decode và Bóc tách lỗi phát âm
"""
import re
import torch
import jiwer
from g2p_en import G2p

# 39 Âm vị tiếng Anh chuẩn (ARPAbet) + <blank> cho CTC + khoảng trắng
ARPABET_PHONEMES = [
    "<blank>", " ", "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER",
    "EY", "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY",
    "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"
]

class TextProcess:
    def __init__(self):
        self.g2p = G2p()
        self.char_map = {p: i for i, p in enumerate(ARPABET_PHONEMES)}
        self.index_map = {i: p for p, i in self.char_map.items()}

    def text_to_phonemes(self, text):
        """Dịch Text (VD: HELLO) -> Âm vị (VD: HH AH L OW)"""
        text = text.lower().strip()
        # g2p_en tự động dịch cả số (1 -> W AH N) và xóa dấu câu
        raw_phones = self.g2p(text)
        phones = []
        for p in raw_phones:
            # Xóa số trọng âm (vd: 'AH0' -> 'AH') để model dễ hội tụ
            p_clean = re.sub(r'\d+', '', p)
            if p_clean in self.char_map:
                phones.append(p_clean)
            elif p_clean.strip() == '':
                phones.append(" ") 
        return phones

    def text_to_int(self, text):
        phones = self.text_to_phonemes(text)
        return [self.char_map[p] for p in phones]

    def int_to_text(self, indices):
        return " ".join([self.index_map[i] for i in indices if i in self.index_map and i != 0])

    def labels_to_text(self, label_tensor, length):
        return self.int_to_text(label_tensor[:length].tolist())

text_process = TextProcess()

def calculate_input_lengths(w_lens):
    l = ((w_lens + 2 * 5 - 11) // 8) + 1
    l = ((l + 2 * 3 - 7) // 4) + 1
    return l

def greedy_decoder(output, text_process, max_frames=None):
    if max_frames is not None:
        output = output[:max_frames]
    arg_maxes = torch.argmax(output, dim=-1)
    decodes = []
    for i in range(len(arg_maxes)):
        if arg_maxes[i] != 0:
            if i == 0 or arg_maxes[i] != arg_maxes[i - 1]:
                decodes.append(arg_maxes[i].item())
    return text_process.int_to_text(decodes)

def decode_logits(logits, wav_lens, text_process, **kwargs):
    frame_lens = calculate_input_lengths(wav_lens)
    preds = []
    for idx in range(logits.size(0)):
        t = int(frame_lens[idx].item())
        preds.append(greedy_decoder(logits[idx, :t], text_process))
    return preds

def evaluate_metrics(model, data_loader, device, max_batches=None, log_samples=3, **kwargs):
    model.eval()
    total_per, samples = 0.0, 0
    logged = 0

    with torch.no_grad():
        for batch_idx, (wavs, labels, w_lens, l_lens) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            logits, _ = model(wavs.to(device))
            preds = decode_logits(logits, w_lens, text_process)

            for idx, pred in enumerate(preds):
                target = text_process.labels_to_text(labels[idx], l_lens[idx].item())
                if not target:
                    continue
                
                # Dọn dẹp khoảng trắng để Jiwer tính chính xác
                target_clean = " ".join(target.split())
                pred_clean = " ".join(pred.split())
                
                # Vì các âm vị cách nhau bằng dấu cách, jiwer.wer sẽ đóng vai trò tính PER (Phoneme Error Rate)
                total_per += jiwer.wer(target_clean, pred_clean)
                samples += 1

                if logged < log_samples:
                    print(f"  [Mẫu {logged + 1}]")
                    print(f"  Target (Phonemes): {target_clean[:80]}")
                    print(f"  Predict(Phonemes): {pred_clean[:80] if pred_clean else '(rỗng)'}")
                    logged += 1

    if samples == 0:
        return 1.0, 1.0, 0
    
    # Val WER và Val CER trên Terminal giờ cùng hiển thị giá trị PER
    per = total_per / samples
    return per, per, samples