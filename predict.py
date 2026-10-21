"""predict.py - command line deepfake video detection (same logic as detector.py).

Usage:
    python predict.py path/to/video.mp4
    python predict.py video.mp4 --threshold 0.7 --sample-every 5 --fake-idx 0 --csv report.csv
"""
import argparse
import csv
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LABEL_BGR = {"Fake": (0, 0, 255), "Real": (0, 255, 0), "Uncertain": (0, 255, 255)}


# -------------------------------
# Deepfake Detector Model
# -------------------------------
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights=None)  # weights come from your checkpoint
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # 2 classes

    def forward(self, x):
        return self.model(x)

    def load_model(self, path, device="cpu"):
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.to(device)
        self.model.eval()

    def predict(self, rgb, device="cpu", use_norm=False):
        """Returns softmax probabilities [p_class0, p_class1] as a numpy array."""
        img = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
        if use_norm:
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=1)[0].cpu().numpy()


# -------------------------------
# Helpers
# -------------------------------
def get_cascade():
    # OpenCV 5 moved CascadeClassifier to contrib -> may be missing; fall back to full frame
    try:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        return None if cascade.empty() else cascade
    except Exception:
        return None


def detect_face(bgr, cascade, margin=0.25):
    if cascade is None:
        return None
    h, w = bgr.shape[:2]
    scale = 640.0 / max(h, w) if max(h, w) > 640 else 1.0
    small = cv2.resize(bgr, None, fx=scale, fy=scale) if scale < 1 else bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    x, y, fw, fh = [int(v / scale) for v in (x, y, fw, fh)]
    mx, my = int(fw * margin), int(fh * margin)
    return max(x - mx, 0), max(y - my, 0), min(x + fw + mx, w), min(y + fh + my, h)


def label_for(p_fake, thr):
    if p_fake >= thr:
        return "Fake"
    if p_fake <= 1 - thr:
        return "Real"
    return "Uncertain"


def verdict_for(avg, thr):
    if avg >= thr:
        return "Likely FAKE"
    if avg <= 1 - thr:
        return "Likely REAL"
    return "Inconclusive"


# -------------------------------
# Video Prediction Function
# -------------------------------
def predict_video(video_path, model, device="cpu", threshold=0.7, sample_every=5,
                  fake_idx=0, use_norm=False, use_face=True, output_path=None,
                  csv_path=None, max_seconds=0):
    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open the video.")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_seconds:
        total_frames = min(total_frames, int(fps * max_seconds)) if total_frames > 0 else int(fps * max_seconds)

    cascade = get_cascade() if use_face else None
    if use_face and cascade is None:
        print("⚠️ Face detector unavailable (OpenCV 5?). Using the full frame. "
              "Install opencv-python-headless<5 to enable face crop.")

    output_path = output_path or f"annotated_{int(time.time())}.mp4"
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    print(f"Processing: {video_path} | frames: {total_frames} | FPS: {fps:.1f} | "
          f"analysing every {sample_every}th frame")

    times, p_list, face_list = [], [], []
    last_label, last_p, last_box = None, None, None
    i = 0
    while total_frames <= 0 or i < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if i % sample_every == 0:
            last_box = detect_face(frame, cascade) if cascade is not None else None
            if last_box:
                x0, y0, x1, y1 = last_box
                crop = frame[y0:y1, x0:x1]
            else:
                crop = frame
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            last_p = float(model.predict(rgb, device, use_norm)[fake_idx])
            last_label = label_for(last_p, threshold)
            times.append(i / fps)
            p_list.append(last_p)
            face_list.append(last_box is not None)
            if len(p_list) % 20 == 0:
                print(f"  analysed {len(p_list)} frames (frame {i + 1}/{total_frames or '?'})")

        # annotate every frame with the most recent prediction (smooth output video)
        if last_label is not None:
            color = LABEL_BGR[last_label]
            fscale = max(width / 900.0, 0.6)
            if last_box:
                cv2.rectangle(frame, (last_box[0], last_box[1]), (last_box[2], last_box[3]),
                              color, max(2, int(2 * fscale)))
            cv2.putText(frame, f"{last_label}: P(fake)={last_p*100:.1f}%", (10, int(35 * fscale)),
                        cv2.FONT_HERSHEY_SIMPLEX, fscale, color, max(2, int(2 * fscale)))
        writer.write(frame)
        i += 1

    cap.release()
    writer.release()

    if not p_list:
        print("No frames processed.")
        return None

    labels = [label_for(p, threshold) for p in p_list]
    fake_n, real_n, unc_n = labels.count("Fake"), labels.count("Real"), labels.count("Uncertain")
    n = len(p_list)
    avg = float(np.mean(p_list))

    print("\n✅ Analysis Complete!")
    print(f"Frames analysed : {n}")
    print(f"Fake frames     : {fake_n} ({fake_n / n * 100:.1f}%)")
    print(f"Real frames     : {real_n} ({real_n / n * 100:.1f}%)")
    print(f"Uncertain frames: {unc_n} ({unc_n / n * 100:.1f}%)")
    print(f"Average P(fake) : {avg * 100:.1f}%  (peak {max(p_list) * 100:.1f}%)")
    print(f"Verdict         : {verdict_for(avg, threshold)}  (threshold {threshold})")
    print(f"Annotated video : {output_path}")

    if csv_path:
        with open(csv_path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["time_s", "p_fake", "label", "face_found"])
            for t, p, lab, fd in zip(times, p_list, labels, face_list):
                wr.writerow([round(t, 2), round(p, 4), lab, fd])
        print(f"CSV report      : {csv_path}")

    return {"avg_p_fake": avg, "fake": fake_n, "real": real_n, "uncertain": unc_n,
            "verdict": verdict_for(avg, threshold)}


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Deepfake video detection (CLI)")
    ap.add_argument("video", nargs="?", help="path to the video file")
    ap.add_argument("--model", default="checkpoints/model_best.pth")
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="P(fake) >= t -> Fake, P(fake) <= 1-t -> Real, else Uncertain")
    ap.add_argument("--sample-every", type=int, default=5, help="analyse every Nth frame")
    ap.add_argument("--fake-idx", type=int, default=0, choices=[0, 1],
                    help="class index of 'fake' (ImageFolder alphabetical: fake=0, real=1)")
    ap.add_argument("--imagenet-norm", action="store_true",
                    help="use ImageNet mean/std normalization (only if used in training)")
    ap.add_argument("--no-face", action="store_true", help="skip face detection/crop")
    ap.add_argument("--max-seconds", type=int, default=0, help="limit analysed length (0 = full video)")
    ap.add_argument("--output", default=None, help="annotated output video path")
    ap.add_argument("--csv", default=None, help="save per-frame results to this CSV file")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔹 Using device: {device}")

    model = DeepfakeDetector()
    model.load_model(args.model, device=device)

    video_file = args.video or input("Enter path to video: ").strip()
    predict_video(video_file, model, device=device, threshold=args.threshold,
                  sample_every=args.sample_every, fake_idx=args.fake_idx,
                  use_norm=args.imagenet_norm, use_face=not args.no_face,
                  output_path=args.output, csv_path=args.csv, max_seconds=args.max_seconds)
