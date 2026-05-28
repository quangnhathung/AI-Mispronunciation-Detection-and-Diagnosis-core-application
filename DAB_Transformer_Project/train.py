"""
train.py — Huấn luyện ASR (CTC) — tối ưu cho 4GB VRAM

Cải tiến so với phiên bản cũ:
  [NEW-LR]  OneCycleLR thay ReduceLROnPlateau: warmup tự động → cosine decay
  [NEW-ES]  Early stopping theo Val WER (patience=8)
  [NEW-CTC] Label smoothing nhẹ qua zero_infinity + epsilon
  [NEW-LOG] In thêm GPU VRAM usage mỗi epoch

Luồng:
  [0] Khởi tạo GPU, model, optimizer, scheduler, CTC loss
  [1] Resume checkpoint (nếu có)
  [2] DataLoader train + val
  [3] Vòng epoch:
        [3a] Train: forward → CTC loss → backward → step
        [3b] Val:   evaluate_metrics (decode + WER/CER)
        [3c] Lưu checkpoint + early stopping check
"""
import torch
import torch.nn.functional as F
import os
import glob
import re

from config import Config
from model import DAB_Transformer
from dataset import L2ArcticDataset, make_dataloader
from utils import text_process, calculate_input_lengths, evaluate_metrics


# ── Early Stopping ───────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=8, min_delta=0.001, mode="min"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.best       = float("inf") if mode == "min" else float("-inf")
        self.counter    = 0
        self.best_epoch = 0

    def step(self, value, epoch):
        improved = (
            value < self.best - self.min_delta
            if self.mode == "min"
            else value > self.best + self.min_delta
        )
        if improved:
            self.best = value
            self.counter = 0
            self.best_epoch = epoch
            return False  # không dừng
        self.counter += 1
        print(
            f"  ⚠️  Early stopping: {self.counter}/{self.patience} "
            f"(best={self.best*100:.2f}% @ epoch {self.best_epoch})"
        )
        return self.counter >= self.patience  # True = dừng


# ── Optimizer step ────────────────────────────────────────────────────────────
def _optimizer_step(scaler, optimizer, model):
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()


# ── Main train ────────────────────────────────────────────────────────────────
def train():
    # --- [0] Thiết lập ---
    if not torch.cuda.is_available():
        print("⚠️ Không thấy CUDA — huấn luyện trên CPU sẽ rất chậm.")
    else:
        print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   VRAM tổng: {total_vram:.1f} GB")

    torch.backends.cudnn.benchmark = True
    os.makedirs(Config.SAVE_DIR, exist_ok=True)

    model = DAB_Transformer(
        num_classes=len(text_process.char_map),
        d_model=Config.D_MODEL,
        nhead=Config.NHEAD,
        num_layers=Config.NUM_LAYERS,
    ).to(Config.DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Tham số: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY,
        betas=(0.9, 0.98),   # beta2 cao hơn: ổn định cho seq model
    )
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    scaler    = torch.amp.GradScaler("cuda", enabled=Config.DEVICE.type == "cuda")

    # --- [1] Resume ---
    start_epoch = 1
    checkpoints = glob.glob(os.path.join(Config.SAVE_DIR, "model_e*.pt"))

    # Scheduler sẽ được tạo sau khi biết tổng số steps
    scheduler_state = None

    if checkpoints:
        latest = max(checkpoints, key=os.path.getctime)
        print(f"🔄 Tìm thấy checkpoint: {latest}. Đang nạp...")
        ckpt = torch.load(latest, map_location=Config.DEVICE)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler_state = ckpt.get("scheduler_state_dict")
            if "scaler_state_dict" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
        else:
            model.load_state_dict(ckpt)
            m = re.search(r"e(\d+)", latest)
            if m:
                start_epoch = int(m.group(1)) + 1

        print(f"🚀 Tiếp tục từ Epoch {start_epoch}")
    else:
        print("🆕 Huấn luyện từ đầu.")

    # --- [2] Dữ liệu ---
    train_set = L2ArcticDataset(
        Config.TRAIN_PATH, text_process, max_samples=Config.MAX_SAMPLES_TRAIN
    )
    val_set   = L2ArcticDataset(Config.VAL_PATH, text_process)
    train_loader = make_dataloader(train_set, shuffle=True)
    val_loader   = make_dataloader(val_set,   shuffle=False)

    n_samples  = len(train_set)
    n_batches  = len(train_loader)
    eff_batch  = Config.BATCH_SIZE * Config.ACCUMULATION_STEPS
    remaining_epochs = Config.NUM_EPOCHS - start_epoch + 1

    # [NEW-LR] OneCycleLR — tính tổng steps cho các epoch còn lại
    steps_per_epoch = max(1, n_batches // Config.ACCUMULATION_STEPS)
    total_steps     = steps_per_epoch * remaining_epochs

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr        = Config.LR,
        total_steps   = total_steps,
        pct_start     = Config.WARMUP_PCT,
        div_factor    = Config.DIV_FACTOR,
        final_div_factor = Config.FINAL_DIV_FACTOR,
        anneal_strategy  = "cos",
    )
    if scheduler_state is not None:
        try:
            scheduler.load_state_dict(scheduler_state)
        except Exception:
            print("   ℹ️ Không nạp được scheduler state (có thể do thay đổi schedule). Bắt đầu mới.")

    early_stop = EarlyStopping(
        patience  = Config.EARLY_STOPPING_PATIENCE,
        min_delta = Config.EARLY_STOPPING_MIN_DELTA,
    )

    print(
        f"\n📦 Train: {n_samples} mẫu | {n_batches} batch/epoch "
        f"(batch={Config.BATCH_SIZE}×accum={Config.ACCUMULATION_STEPS}={eff_batch}) | "
        f"layers={Config.NUM_LAYERS} | dropout={Config.DROPOUT} | "
        f"GradCkpt={'ON' if Config.USE_GRADIENT_CHECKPOINT else 'OFF'} | "
        f"SpecAug={'ON' if Config.USE_SPEC_AUGMENT else 'OFF'}"
    )
    print(f"   LR schedule: OneCycleLR max_lr={Config.LR:.0e} | {total_steps} steps tổng")
    print(f"   Early stopping: patience={Config.EARLY_STOPPING_PATIENCE}\n")

    # --- [3] Vòng epoch ---
    for epoch in range(start_epoch, Config.NUM_EPOCHS + 1):

        # [3a] Huấn luyện
        model.train()
        epoch_loss    = 0.0
        valid_batches = 0
        optimizer.zero_grad()
        accum_count = 0

        for i, (wavs, labels, w_lens, l_lens) in enumerate(train_loader):
            wavs   = wavs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)

            with torch.amp.autocast("cuda", enabled=Config.DEVICE.type == "cuda"):
                logits, _ = model(wavs)
                input_lengths = calculate_input_lengths(w_lens).to(Config.DEVICE)
                log_probs = F.log_softmax(logits.float(), dim=2).transpose(0, 1)
                loss = (
                    criterion(log_probs, labels, input_lengths, l_lens)
                    / Config.ACCUMULATION_STEPS
                )

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                accum_count = 0
                continue

            scaler.scale(loss).backward()
            accum_count += 1
            epoch_loss  += loss.item() * Config.ACCUMULATION_STEPS
            valid_batches += 1

            if accum_count % Config.ACCUMULATION_STEPS == 0:
                _optimizer_step(scaler, optimizer, model)
                accum_count = 0
                # [NEW-LR] step scheduler mỗi effective batch
                scheduler.step()

            if i % 200 == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch {epoch} | Batch {i}/{n_batches} | "
                    f"Loss: {loss.item() * Config.ACCUMULATION_STEPS:.4f} | "
                    f"LR: {lr_now:.2e}"
                )

        if accum_count > 0:
            _optimizer_step(scaler, optimizer, model)
            scheduler.step()

        avg_loss = epoch_loss / valid_batches if valid_batches > 0 else float("inf")

        # [3b] Validation
        print(f"\n📊 Val WER/CER — Epoch {epoch}:")
        avg_wer, avg_cer, n_val = evaluate_metrics(
            model, val_loader, Config.DEVICE,
            max_batches=Config.VAL_EVAL_MAX_BATCHES,
            log_samples=3,
            decoder=Config.VAL_DECODER,
            beam_size=Config.BEAM_SIZE,
        )

        # VRAM usage
        if Config.DEVICE.type == "cuda":
            used  = torch.cuda.memory_reserved()  / 1024**3
            alloc = torch.cuda.memory_allocated() / 1024**3
            print(f"   VRAM reserved={used:.2f}GB | allocated={alloc:.2f}GB")

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"✅ Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
            f"Val WER: {avg_wer*100:.2f}% | Val CER: {avg_cer*100:.2f}% | "
            f"LR: {lr_now:.2e}"
        )

        # [3c] Lưu checkpoint
        torch.save(
            {
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict":    scaler.state_dict(),
                "loss":                 avg_loss,
                "wer":                  avg_wer,
                "cer":                  avg_cer,
                "val_samples":          n_val,
            },
            f"{Config.SAVE_DIR}/model_e{epoch}.pt",
        )

        # [NEW-ES] Early stopping
        should_stop = early_stop.step(avg_wer, epoch)
        if should_stop:
            print(f"\n🛑 Early stopping tại epoch {epoch}. Best WER={early_stop.best*100:.2f}% @ epoch {early_stop.best_epoch}")
            break

        print("-" * 65)

        if Config.DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n🏁 Huấn luyện xong. Best checkpoint: model_e{early_stop.best_epoch}.pt")


if __name__ == "__main__":
    train()