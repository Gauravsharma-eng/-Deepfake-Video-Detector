
# 🛡️ DeepGuard: Deepfake Video Detector 🕵️🎬

An explainable deepfake video detection app built with **PyTorch, OpenCV and Streamlit**. It samples frames from a video, detects and crops the face, classifies each frame as Real or Fake with a ResNet50 model, and shows *where* and *why* the model suspects manipulation.

## 🚀 Live Demo

👉 **https://deepfake-video-detector-rs.streamlit.app/**

*(The app may take a few seconds to wake up if it hasn't been used recently.)*

<!-- Add a screenshot or demo GIF here, e.g.: ![Demo](assets/demo.gif) -->

## 🔹 Features

- **Video input:** Upload MP4 / AVI / MOV files or paste a video URL (YouTube or direct links via `yt-dlp`).
- **Face detection + crop:** The largest face in each sampled frame is detected and cropped, so the model focuses on the face instead of the background.
- **Frame-wise AI analysis:** Real vs Fake prediction using a fine-tuned **ResNet50** backbone.
- **Explainability (Grad-CAM):** Heatmaps highlight the facial regions that pushed the model towards "fake".
- **Threshold-aware results:** Frames are labelled Real / Fake / Uncertain using an adjustable decision threshold. Changing the slider after analysis updates the verdict and charts instantly.
- **Overall verdict card:** Likely Real / Likely Fake / Inconclusive, based on the average fake probability across the video.
- **Timeline chart:** See *when* in the video the fake probability rises.
- **Most suspicious frames:** The top frames with their Grad-CAM overlays.
- **Downloadable report:** Export per-frame results (CSV) and a text summary.
- **Model Performance page:** Architecture, evaluation metrics and known limitations.
- **Dark themed UI** with a custom animated home page.
- **Performance-friendly:** Configurable frame sampling and max analysis length; uses GPU automatically if available.

## 🧠 How It Works

```
Video → Frame sampling → Face detection → Crop 224×224 → ResNet50 → P(fake) → Aggregation → Verdict
                                                              └→ Grad-CAM heatmap
```

1. Every N-th frame is sampled from the video.
2. The face is detected (OpenCV Haar cascade) and cropped with a margin. If no face is found, the full frame is used.
3. ResNet50 outputs the probability that the frame is fake.
4. Frame labels use the decision threshold: `P(fake) ≥ t` → Fake, `P(fake) ≤ 1 − t` → Real, otherwise Uncertain.
5. The video-level verdict is based on the average fake probability.

## 🛠️ Technologies Used

- Python 3.11
- PyTorch & torchvision (ResNet50)
- OpenCV (video processing, face detection)
- Streamlit (web interface)
- Matplotlib, Pandas (visualisation and reports)
- yt-dlp (URL input)
- CSS/HTML (custom UI)

## 📁 File Structure

```
.
├── checkpoints/
│   └── model_best.pth       # Trained model weights (~89.9 MB)
├── .streamlit/
│   └── config.toml          # Dark theme configuration
├── detector.py              # Main Streamlit application
├── requirements.txt         # Python dependencies
├── packages.txt             # System dependencies (libgl1)
└── README.md                # Project documentation
```

## ⚡ How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/Gauravsharma-eng/-Deepfake-Video-Detector.git deepfake-video-detector
   cd deepfake-video-detector
   ```

2. **Install dependencies** (Python 3.11 recommended)
   ```bash
   pip install -r requirements.txt
   ```
   > `opencv-python-headless` is pinned below version 5 because OpenCV 5 moved the Haar `CascadeClassifier` used for face detection into `opencv_contrib`.

3. **Run the app**
   ```bash
   streamlit run detector.py
   ```

## ⚠️ Limitations

- This is a **frame-level** model: it does not use temporal cues such as blinking patterns or lip-sync.
- Accuracy depends on the training data. Unseen manipulation methods or heavy compression can reduce performance.
- Haar-cascade face detection can miss side-profile, very small or occluded faces (the full frame is analysed as a fallback).
- Grad-CAM shows where the model looked, not a guarantee that the region is actually manipulated.
- Results are probability estimates and **not conclusive proof** that a video is real or fake.

## 👤 Author

**Gaurav Sharma**
Final Year B.Tech Student at VITM, Gwalior
