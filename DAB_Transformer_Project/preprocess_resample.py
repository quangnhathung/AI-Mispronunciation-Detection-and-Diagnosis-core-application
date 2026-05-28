"""

[PRE] Tiền xử lý OFFLINE — Resample toàn bộ tập về 16 kHz (chạy 1 lần)



Trước khi train:

    python preprocess_resample.py



Đọc:  Config.RAW_*_PATH  (thường 44.1 kHz)

Ghi:   Config.TRAIN/VAL/TEST_PATH  (Dataset_Splitted_16k, cấu trúc giữ nguyên)



Sau khi xong, dataset.py chỉ load wav 16 kHz — không resample trong __getitem__.

"""

import os

import shutil

import argparse



import torch

import torchaudio



from config import Config



SPLITS = [

    ("train", Config.RAW_TRAIN_PATH, Config.TRAIN_PATH),

    ("val", Config.RAW_VAL_PATH, Config.VAL_PATH),

    ("test", Config.RAW_TEST_PATH, Config.TEST_PATH),

]





def resample_wav(src_path, dst_path, target_sr, overwrite=False):

    if os.path.exists(dst_path) and not overwrite:

        try:

            info = torchaudio.info(dst_path)

            if info.sample_rate == target_sr:

                return "skip"

        except Exception:

            pass



    wav, sr = torchaudio.load(src_path)

    if sr != target_sr:

        wav = torchaudio.functional.resample(wav, sr, target_sr)



    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    torchaudio.save(dst_path, wav, target_sr)

    return "ok"





def copy_transcript(src_txt, dst_txt, overwrite=False):

    if not os.path.exists(src_txt):

        return "no_txt"

    if os.path.exists(dst_txt) and not overwrite:

        return "skip"

    os.makedirs(os.path.dirname(dst_txt), exist_ok=True)

    shutil.copy2(src_txt, dst_txt)

    return "ok"





def process_split(name, raw_root, out_root, target_sr, overwrite=False):

    if not os.path.isdir(raw_root):

        print(f"  Bỏ qua [{name}]: không tìm thấy {raw_root}")

        return 0, 0, 0



    n_ok, n_skip, n_fail = 0, 0, 0

    speakers = [

        d for d in os.listdir(raw_root)

        if os.path.isdir(os.path.join(raw_root, d))

    ]



    for spk in speakers:

        wav_dir = os.path.join(raw_root, spk, "wav")

        txt_dir = os.path.join(raw_root, spk, "transcript")

        if not os.path.isdir(wav_dir):

            continue



        out_wav_dir = os.path.join(out_root, spk, "wav")

        out_txt_dir = os.path.join(out_root, spk, "transcript")



        for fname in os.listdir(wav_dir):

            if not fname.endswith(".wav"):

                continue



            src_wav = os.path.join(wav_dir, fname)

            dst_wav = os.path.join(out_wav_dir, fname)

            src_txt = os.path.join(txt_dir, fname.replace(".wav", ".txt"))

            dst_txt = os.path.join(out_txt_dir, fname.replace(".wav", ".txt"))



            try:

                status = resample_wav(src_wav, dst_wav, target_sr, overwrite)

                if status == "ok":

                    n_ok += 1

                else:

                    n_skip += 1

                copy_transcript(src_txt, dst_txt, overwrite)

            except Exception as e:

                n_fail += 1

                print(f"  Lỗi: {src_wav} — {e}")



    return n_ok, n_skip, n_fail





def main():

    parser = argparse.ArgumentParser(description="Resample L2-Arctic về 16 kHz (offline)")

    parser.add_argument(

        "--overwrite",

        action="store_true",

        help="Ghi đè file wav 16k đã tồn tại",

    )

    args = parser.parse_args()

    sr = Config.TARGET_SAMPLE_RATE



    print(f"Resample -> {sr} Hz")

    print(f"  Raw train: {Config.RAW_TRAIN_PATH}")

    print(f"  Out train: {Config.TRAIN_PATH}\n")



    total_ok = total_skip = total_fail = 0

    for name, raw_path, out_path in SPLITS:

        print(f"--- [{name}] ---")

        ok, skip, fail = process_split(name, raw_path, out_path, sr, args.overwrite)

        total_ok += ok

        total_skip += skip

        total_fail += fail

        print(f"  Xong: {ok} mới/chuyển | {skip} đã có | {fail} lỗi\n")



    print("=" * 50)

    print(f"Tổng: {total_ok} resample | {total_skip} bỏ qua | {total_fail} lỗi")

    print("Tiếp theo: python train.py")





if __name__ == "__main__":

    main()


