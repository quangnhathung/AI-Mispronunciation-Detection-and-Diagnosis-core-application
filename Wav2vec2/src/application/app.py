import os
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sounddevice as sd
import soundfile as sf
import numpy as np
import torch          # THÊM
import torchaudio     # THÊM

# Import Predictor từ file inference đã refactor
from src.application.predict.predict import MDDPredictor

# Cấu hình đường dẫn cố định
MODEL_PATH = "./MDD_AI/checkpoints/best_mdd_model_v4.pt"
AUDIO_DIR = os.path.join(os.getcwd(), "data", "user_audio")

# Đảm bảo thư mục lưu audio luôn tồn tại
os.makedirs(AUDIO_DIR, exist_ok=True)

def preprocess_raw_audio(input_path, output_path, target_sr=16000):
    """
    Đọc file âm thanh thô, chuyển về mono và resample,
    sau đó xuất ra một file .wav mới để đưa vào mô hình.
    """
    waveform_np, sr = sf.read(input_path, dtype='float32')
    waveform = torch.from_numpy(waveform_np)
    
    # Xử lý chiều ma trận
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.t()
        
    # Đưa về Mono bằng cách lấy trung bình cộng các kênh
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # Resample tần số lấy mẫu về target_sr (16kHz)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)
        
    # Ghi file wav đã xử lý ra disk. sf.write yêu cầu shape (Time, Channels) nên cần .t()
    sf.write(output_path, waveform.t().numpy(), target_sr)
    return output_path
# =====================================================================

class AudioRecorder:
    def __init__(self):
        self.is_recording = False
        self.frames = []
        self.fs = 16000  # Bắt buộc 16kHz cho Wav2Vec2

    def start(self):
        self.is_recording = True
        self.frames = []
        self.thread = threading.Thread(target=self._record)
        self.thread.start()

    def _record(self):
        # Mở luồng ghi âm mono (channels=1)
        with sd.InputStream(samplerate=self.fs, channels=1) as stream:
            while self.is_recording:
                data, _ = stream.read(1024)
                self.frames.append(data)

    def stop(self, filepath):
        self.is_recording = False
        self.thread.join() # Đợi luồng ghi âm kết thúc hoàn toàn
        audio_data = np.concatenate(self.frames, axis=0)
        sf.write(filepath, audio_data, self.fs)
        return filepath


class MDDApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MDD - Hệ thống Chẩn đoán Phát âm (Demo)")
        self.root.geometry("700x600")
        self.root.resizable(False, False)

        self.recorder = AudioRecorder()
        self.predictor = None
        self.is_recording = False

        self._setup_ui()
        self._init_model_async() # Load model ngầm để không đơ lúc mở app

    def _setup_ui(self):
        # Font chữ chuẩn
        font_label = ("Arial", 11)
        font_entry = ("Arial", 11)

        # --- Frame Input ---
        frame_input = tk.LabelFrame(self.root, text="Dữ liệu đầu vào", padx=10, pady=10)
        frame_input.pack(fill="x", padx=15, pady=10)

        # 1. Text Input
        tk.Label(frame_input, text="Văn bản (Expected Text):", font=font_label).grid(row=0, column=0, sticky="w", pady=5)
        self.entry_text = tk.Entry(frame_input, font=font_entry, width=50)
        self.entry_text.grid(row=0, column=1, columnspan=2, padx=5, pady=5)

        # 2. Audio File Input
        tk.Label(frame_input, text="File âm thanh (.wav):", font=font_label).grid(row=1, column=0, sticky="w", pady=5)
        self.entry_audio = tk.Entry(frame_input, font=font_entry, width=40)
        self.entry_audio.grid(row=1, column=1, padx=5, pady=5)
        
        btn_browse = tk.Button(frame_input, text="Duyệt File", command=self.browse_file)
        btn_browse.grid(row=1, column=2, padx=5)

        # --- Frame Controls (Ghi âm & Chạy) ---
        frame_controls = tk.Frame(self.root, pady=10)
        frame_controls.pack(fill="x", padx=15)

        self.btn_record = tk.Button(frame_controls, text="🎤 Bắt đầu Thu âm", bg="#ffcccc", width=20, command=self.toggle_record)
        self.btn_record.pack(side="left", padx=10)

        self.btn_predict = tk.Button(frame_controls, text="🚀 XÁC NHẬN & PHÂN TÍCH", bg="#ccffcc", width=25, font=("Arial", 10, "bold"), command=self.run_prediction)
        self.btn_predict.pack(side="right", padx=10)

        # Trạng thái hệ thống
        self.lbl_status = tk.Label(self.root, text="Đang tải mô hình, vui lòng đợi...", fg="blue")
        self.lbl_status.pack(pady=5)

        # --- Frame Output (Bảng kết quả) ---
        frame_output = tk.LabelFrame(self.root, text="Kết quả Chẩn đoán", padx=10, pady=10)
        frame_output.pack(fill="both", expand=True, padx=15, pady=10)
        
        self.lbl_score = tk.Label(frame_output, text="Độ chính xác tổng thể: --%", font=("Arial", 12, "bold"))
        self.lbl_score.pack(anchor="w", pady=(0, 10))

        # Dùng Treeview làm bảng (Table)
        columns = ("phoneme", "score", "status")
        self.tree = ttk.Treeview(frame_output, columns=columns, show="headings", height=15)
        self.tree.heading("phoneme", text="Âm vị (Phoneme)")
        self.tree.heading("score", text="Điểm số (Confidence)")
        self.tree.heading("status", text="Đánh giá")
        
        self.tree.column("phoneme", anchor="center", width=150)
        self.tree.column("score", anchor="center", width=150)
        self.tree.column("status", anchor="center", width=150)
        
        self.tree.pack(fill="both", expand=True)

    def _init_model_async(self):
        """Khởi tạo mô hình trên một luồng phụ để UI hiển thị ngay lập tức"""
        def load_task():
            try:
                self.predictor = MDDPredictor(model_path=MODEL_PATH)
                # Dùng root.after để cập nhật UI an toàn từ luồng phụ
                self.root.after(0, lambda: self.lbl_status.config(text="Hệ thống Sẵn sàng!", fg="green"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Lỗi Tải Mô Hình", str(e)))
                self.root.after(0, lambda: self.lbl_status.config(text="Lỗi khởi tạo mô hình!", fg="red"))
        
        threading.Thread(target=load_task, daemon=True).start()

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Chọn file âm thanh",
            filetypes=(("WAV files", "*.wav"), ("All files", "*.*"))
        )
        if filename:
            self.entry_audio.delete(0, tk.END)
            self.entry_audio.insert(0, filename)

    def toggle_record(self):
        if not self.is_recording:
            # Bắt đầu ghi
            self.is_recording = True
            self.btn_record.config(text="⏹ Dừng Thu âm", bg="#ff4d4d", fg="white")
            self.lbl_status.config(text="Đang thu âm...", fg="red")
            self.entry_audio.delete(0, tk.END) # Xóa đường dẫn cũ
            self.recorder.start()
        else:
            # Dừng ghi
            self.is_recording = False
            self.btn_record.config(text="🎤 Bắt đầu Thu âm", bg="#ffcccc", fg="black")
            
            # Tạo tên file theo timestamp
            timestamp = int(time.time())
            save_path = os.path.join(AUDIO_DIR, f"record_{timestamp}.wav")
            
            self.recorder.stop(save_path)
            self.entry_audio.insert(0, save_path)
            self.lbl_status.config(text=f"Đã lưu âm thanh tại: {save_path}", fg="green")

    def run_prediction(self):
        # 1. Kiểm tra đầu vào (Validation)
        if self.predictor is None:
            messagebox.showwarning("Cảnh báo", "Mô hình đang tải, vui lòng đợi!")
            return

        text = self.entry_text.get().strip()
        wav_path = self.entry_audio.get().strip()

        if not text:
            messagebox.showerror("Lỗi", "Vui lòng nhập văn bản tiếng Anh!")
            return
        if not wav_path or not os.path.exists(wav_path):
            messagebox.showerror("Lỗi", "File âm thanh không hợp lệ hoặc không tồn tại!")
            return

        # 2. Khóa nút bấm và cập nhật trạng thái
        self.btn_predict.config(state="disabled", text="Đang xử lý...")
        self.lbl_status.config(text="Mô hình đang phân tích...", fg="orange")
        
        # Xóa bảng cũ
        for item in self.tree.get_children():
            self.tree.delete(item)

        # 3. Chạy inference trên luồng riêng biệt
        def predict_task():
            try:
                # SỬA/THÊM: Tiền xử lý file trước khi gửi cho model
                filename_only = os.path.basename(wav_path)
                name, ext = os.path.splitext(filename_only)
                # Đặt tên file mới kèm hậu tố "_processed" lưu vào thư mục audio
                processed_wav_path = os.path.join(AUDIO_DIR, f"{name}_processed{ext}")
                
                # Chạy hàm tiền xử lý và lưu file
                preprocess_raw_audio(wav_path, processed_wav_path)

                # Gọi hàm predict từ inference.py NHƯNG TRUYỀN VÀO FILE ĐÃ XỬ LÝ
                result = self.predictor.predict(wav_path=processed_wav_path, expected_text=text, threshold=0.5)
                
                # Đẩy kết quả lên UI thông qua root.after
                self.root.after(0, self._update_result_ui, result)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Lỗi xử lý", str(e)))
                self.root.after(0, lambda: self.lbl_status.config(text="Lỗi phân tích!", fg="red"))
            finally:
                # Mở khóa nút
                self.root.after(0, lambda: self.btn_predict.config(state="normal", text="🚀 XÁC NHẬN & PHÂN TÍCH"))

        threading.Thread(target=predict_task, daemon=True).start()

    def _update_result_ui(self, result):
        """Hàm nhận Dictionary từ mô hình và cập nhật lên bảng Treeview"""
        if not result.get("success", True):
            self.lbl_status.config(text=result.get("error_message", "Lỗi không xác định"), fg="red")
            self.lbl_score.config(text="Độ chính xác tổng thể: --%")
            return
        
        self.lbl_status.config(text="Phân tích hoàn tất!", fg="green")
        
        # Cập nhật overall accuracy từ Dictionary
        if "overall_accuracy" in result:
            self.lbl_score.config(text=f"Độ chính xác tổng thể: {result['overall_accuracy']}%")

        # Đổ dữ liệu vào Treeview
        if "details" in result:
            for item in result['details']:
                status_icon = "✔ ĐÚNG" if item['is_correct'] else "❌ LỖI"
                self.tree.insert("", "end", values=(
                    item['phoneme'], 
                    f"{item['score']:.4f}", 
                    status_icon
                ))

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = MDDApp(root)
#     root.mainloop()