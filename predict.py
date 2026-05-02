# predict.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import cv2
import numpy as np
import os
import time

# -------------------------------
# Deepfake Detector Model
# -------------------------------
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights=None)  # your checkpoint model
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # 2 classes: Real / Fake

    def forward(self, x):
        return self.model(x)

    def load_model(self, path, device="cpu"):
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.to(device)
        self.model.eval()

    def predict(self, frame, device="cpu"):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = self.forward(img)
            probs = F.softmax(logits, dim=1)
            conf, class_idx = torch.max(probs, dim=1)
            return class_idx.item(), conf.item()

# -------------------------------
# Video Prediction Function
# -------------------------------
def predict_video(video_path, model, device="cpu", confidence_threshold=0.5):
    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    labels = []
    confidences = []

    print(f"Processing video: {video_path} | Total frames: {total_frames} | FPS: {fps}")

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        class_idx, conf = model.predict(frame, device=device)
        label = "Real" if class_idx == 0 else "Fake"
        display_label = label if conf >= confidence_threshold else "Uncertain"

        # Annotate frame
        cv2.putText(frame, f"{display_label}: {conf*100:.2f}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if label=="Real" else (0,0,255), 2)
        frames.append(frame)
        labels.append(label)
        confidences.append(conf)

        print(f"Frame {i+1}/{total_frames} | Prediction: {display_label} | Confidence: {conf*100:.2f}%")

    cap.release()

    # Summary
    real_count = labels.count("Real")
    fake_count = labels.count("Fake")
    total = len(labels)
    print("\n✅ Analysis Complete!")
    print(f"Real frames: {real_count} ({real_count/total*100:.2f}%)")
    print(f"Fake frames: {fake_count} ({fake_count/total*100:.2f}%)")
    print(f"Average confidence: {np.mean(confidences)*100:.2f}%")

    # Save annotated video
    if len(frames) > 0:
        height, width, _ = frames[0].shape
        output_path = f"annotated_{int(time.time())}.mp4"
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for f in frames:
            out.write(f)
        out.release()
        print(f"Annotated video saved as: {output_path}")
    else:
        print("No frames processed.")

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔹 Using device: {device}")

    model = DeepfakeDetector()
    model.load_model("checkpoints/model_best.pth", device=device)

    video_file = input("Enter path to video: ").strip()
    predict_video(video_file, model, device=device, confidence_threshold=0.5)
