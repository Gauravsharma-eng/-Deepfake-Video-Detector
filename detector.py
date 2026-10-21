import os
import shutil
import tempfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from streamlit_option_menu import option_menu
from torchvision import models

# ======================================================================
# CONFIG  (yahan apni values fill karo)
# ======================================================================
st.set_page_config(
    page_title="DeepGuard | Deepfake Video Detector",
    page_icon="🛡️",
    layout="wide",
)

MODEL_PATH = "checkpoints/model_best.pth"
DEFAULT_FAKE_IDX = 0  # training ka class_to_idx dekh ke set karo (ImageFolder: fake=0, real=1)
DEFAULT_USE_NORM = False  # True sirf tab jab training mein ImageNet mean/std lagi thi

# Model Performance page ke liye apne REAL metrics bharo (None = "not added yet").
PERFORMANCE = {
    "dataset": None,  # e.g. "FaceForensics++ (c23)"
    "train_samples": None,
    "val_samples": None,
    "epochs": None,
    "accuracy": None,  # 0-1 ya % dono chalega, e.g. 0.93
    "precision": None,
    "recall": None,
    "f1": None,
    "auc": None,
    "confusion_matrix": None,  # [[TN, FP], [FN, TP]]  (Real=negative, Fake=positive)
}

ACCENT = "#00ffcc"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ======================================================================
# MODEL (ResNet50 + Grad-CAM)
# ======================================================================
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self._acts = None
        self._grads = None
        self.model.layer4.register_forward_hook(self._save_acts)

    def _save_acts(self, module, inputs, output):
        self._acts = output
        if output.requires_grad:
            output.register_hook(lambda g: setattr(self, "_grads", g))

    def forward(self, x):
        return self.model(x)

    def load_model(self, path, device="cpu"):
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.to(device)
        self.model.eval()

    @staticmethod
    def preprocess(rgb, use_norm, device):
        img = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
        if use_norm:
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)

    def predict(self, rgb, device="cpu", use_norm=False, fake_idx=0, with_cam=False):
        """Returns (softmax probs as numpy [2], Grad-CAM map 7x7 in [0,1] or None)."""
        x = self.preprocess(rgb, use_norm, device)
        if not with_cam:
            with torch.no_grad():
                probs = F.softmax(self.forward(x), dim=1)[0]
            return probs.cpu().numpy(), None

        with torch.enable_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
            self.zero_grad()
            logits[0, fake_idx].backward()
        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._acts).sum(dim=1))[0].detach().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return probs, cam


@st.cache_resource(show_spinner="🔄 Loading AI model...")
def load_detector(device):
    detector = DeepfakeDetector()
    detector.load_model(MODEL_PATH, device=device)
    return detector


@st.cache_resource
def get_face_cascade():
    # OpenCV 5 moved CascadeClassifier to contrib, so it can be missing -> return None (fallback: full frame)
    try:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        return None if cascade.empty() else cascade
    except Exception:
        return None


# ======================================================================
# HELPERS
# ======================================================================
def detect_face(bgr, cascade, margin=0.25):
    """Largest face box (x0, y0, x1, y1) with margin, or None."""
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


def overlay_cam(rgb, cam):
    rgb224 = cv2.resize(rgb, (224, 224))
    heat = cv2.resize(cam, (224, 224))
    heat = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb224, 0.55, heat, 0.45, 0)


def download_video(url):
    """Download from YouTube / direct links using yt-dlp. Returns (file_path, temp_dir)."""
    import yt_dlp

    tmp_dir = tempfile.mkdtemp()
    opts = {
        "format": "best[height<=480][ext=mp4]/best[height<=480]/best",
        "outtmpl": os.path.join(tmp_dir, "video.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    files = os.listdir(tmp_dir)
    if not files:
        raise RuntimeError("Video download failed.")
    return os.path.join(tmp_dir, files[0]), tmp_dir


def label_for(p_fake, thr):
    if p_fake >= thr:
        return "Fake"
    if p_fake <= 1 - thr:
        return "Real"
    return "Uncertain"


LABEL_BGR = {"Fake": (0, 0, 255), "Real": (0, 255, 0), "Uncertain": (0, 255, 255)}


def verdict_for(avg, thr):
    if avg >= thr:
        return "fake", "🔴 Likely FAKE"
    if avg <= 1 - thr:
        return "real", "🟢 Likely REAL"
    return "unc", "🟡 Inconclusive"


def style_ax(ax):
    ax.set_facecolor("none")
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_color("#555555")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")


def new_fig(size):
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_alpha(0)
    style_ax(ax)
    return fig, ax


# ======================================================================
# STYLING
# ======================================================================
st.markdown(
    f"""
<style>
.stApp {{ background-color: #0e1117; color: #f5f5f5; }}
.stButton>button {{ background: linear-gradient(90deg,#ff0040,#ff6a00); color:#fff; border:none;
    border-radius:10px; padding:10px 26px; font-weight:700; }}
.stButton>button:disabled {{ background:#333; color:#777; }}
.hero-title {{ font-size:48px; font-weight:800; text-align:center; color:{ACCENT};
    text-shadow:0 0 25px rgba(0,255,204,.35); margin:10px 0 0 0; }}
.hero-sub {{ font-size:19px; text-align:center; color:#b5b5b5; margin-bottom:18px; }}
.card {{ background:#161b22; border:1px solid #2a313c; border-radius:14px; padding:20px; height:100%; }}
.card h4 {{ color:{ACCENT}; margin:0 0 8px 0; }}
.card p {{ color:#c9d1d9; margin:0; font-size:15px; }}
.verdict {{ border-radius:16px; padding:22px 26px; margin:10px 0 18px 0; border:1px solid; }}
.verdict.fake {{ background:rgba(255,0,60,.10); border-color:#ff3b5c; }}
.verdict.real {{ background:rgba(0,255,120,.08); border-color:#21d07a; }}
.verdict.unc  {{ background:rgba(255,200,0,.08); border-color:#ffc400; }}
.v-title {{ font-size:32px; font-weight:800; }}
.v-sub {{ color:#aaa; font-size:14px; margin-top:4px; }}
.bar {{ background:#2a2f3a; border-radius:8px; height:14px; margin:10px 0 4px 0; overflow:hidden; }}
.bar .fill {{ height:100%; background:linear-gradient(90deg,#21d07a,#ffc400,#ff3b5c); }}
.v-num {{ font-size:15px; color:#ddd; }}
.eyes {{ display:flex; justify-content:center; margin:26px 0 8px 0; }}
.eye {{ width:90px; height:90px; background:radial-gradient(circle,#ff0000 25%,#220000 100%); border-radius:50%;
    margin:0 28px; position:relative; overflow:hidden; box-shadow:0 0 45px #ff0000, inset 0 0 28px #ff0000;
    animation:pulse 3s infinite alternate; }}
@keyframes pulse {{ 0%{{transform:scale(1);}} 100%{{transform:scale(1.1);}} }}
.pupil {{ width:28px; height:28px; background:radial-gradient(circle,#000 30%,#660000 100%); border-radius:50%;
    position:absolute; top:31px; left:31px; animation:move 6s infinite; }}
@keyframes move {{ 0%,100%{{transform:translate(0,0);}} 25%{{transform:translate(10px,-8px);}}
    50%{{transform:translate(-14px,9px);}} 75%{{transform:translate(6px,14px);}} }}
.eyelid {{ position:absolute; inset:0; background:#000; border-radius:50%; opacity:.9; animation:blink 5s infinite; }}
@keyframes blink {{ 0%,90%,100%{{height:0;}} 45%,50%{{height:100%;}} }}
</style>
""",
    unsafe_allow_html=True,
)

selected = option_menu(
    menu_title=None,
    options=["🏠 Home", "🕵️ Detection Tool", "📊 Model Performance", "ℹ️ About"],
    icons=["house", "camera-video", "bar-chart", "info-circle"],
    orientation="horizontal",
    styles={"nav-link-selected": {"background-color": "#ff0040"}},
)


# ======================================================================
# RESULTS RENDERING (reads from session_state, so threshold changes re-render instantly)
# ======================================================================
def render_results(res, thr):
    p = np.array(res["p_fake"])
    t = np.array(res["times"])
    face = np.array(res["face_found"])
    labels = [label_for(v, thr) for v in p]
    fake_n, real_n, unc_n = labels.count("Fake"), labels.count("Real"), labels.count("Uncertain")
    avg = float(p.mean())
    cls, title = verdict_for(avg, thr)

    st.markdown(
        f"""
    <div class="verdict {cls}">
      <div class="v-title">{title}</div>
      <div class="v-sub">Average fake probability across {len(p)} analysed frames</div>
      <div class="bar"><div class="fill" style="width:{avg*100:.1f}%"></div></div>
      <div class="v-num">{avg*100:.1f}% fake probability &nbsp;|&nbsp; {fake_n/len(p)*100:.0f}% of frames flagged fake</div>
    </div>""",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frames analysed", len(p))
    c2.metric("Peak P(fake)", f"{p.max()*100:.1f}%")
    c3.metric("Fake / Real / Uncertain", f"{fake_n} / {real_n} / {unc_n}")
    if res["meta"]["face_crop"]:
        c4.metric("Face detected", f"{face.mean()*100:.0f}% frames")
        if face.mean() < 0.5:
            st.warning("Face was detected in less than half the frames, so results may be unreliable.")
    else:
        c4.metric("Face crop", "Off")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("##### 📈 Fake probability timeline")
        fig, ax = new_fig((7, 3.2))
        ax.plot(t, p, color=ACCENT, linewidth=1.8)
        ax.fill_between(t, p, thr, where=p >= thr, color="#ff3b5c", alpha=0.35)
        ax.axhline(thr, color="#ff3b5c", linestyle="--", linewidth=1, label=f"Fake ≥ {thr:.2f}")
        ax.axhline(1 - thr, color="#21d07a", linestyle="--", linewidth=1, label=f"Real ≤ {1-thr:.2f}")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("P(fake)")
        ax.legend(facecolor="#161b22", labelcolor="white", edgecolor="#333", fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("##### 🥧 Frame distribution")
        data = [(n, c, col) for n, c, col in
                [(real_n, "Real", "#21d07a"), (fake_n, "Fake", "#ff3b5c"), (unc_n, "Uncertain", "#ffc400")] if n > 0]
        fig, ax = new_fig((3.6, 3.2))
        ax.pie([d[0] for d in data], labels=[d[1] for d in data], colors=[d[2] for d in data],
               autopct="%1.1f%%", textprops={"color": "white"})
        st.pyplot(fig)
        plt.close(fig)

    if res["top"]:
        st.markdown("##### 🔍 Most suspicious frames (with Grad-CAM)")
        st.caption("Heatmap shows the regions that pushed the model towards 'fake' (red = strongest evidence).")
        cols = st.columns(len(res["top"]))
        for col, (pf, ts, img, cam_img) in zip(cols, res["top"]):
            with col:
                st.image(img, caption=f"t = {ts:.1f}s | P(fake) = {pf*100:.1f}%", use_container_width=True)
                if cam_img is not None:
                    st.image(cam_img, caption="Grad-CAM", use_container_width=True)

    # ---- Report downloads ----
    df = pd.DataFrame({"time_s": np.round(t, 2), "p_fake": np.round(p, 4),
                       "label": labels, "face_found": face})
    summary = (
        "DEEPFAKE ANALYSIS REPORT\n"
        "========================\n"
        f"Video: {res['meta']['name']}\n"
        f"Verdict: {title}\n"
        f"Average P(fake): {avg*100:.1f}%\n"
        f"Peak P(fake): {p.max()*100:.1f}%\n"
        f"Frames analysed: {len(p)} (every {res['meta']['sample_every']}th frame)\n"
        f"Fake / Real / Uncertain: {fake_n} / {real_n} / {unc_n}\n"
        f"Decision threshold: {thr}\n"
        f"Face crop: {res['meta']['face_crop']} | Grad-CAM: {res['meta']['grad_cam']}\n"
        "Note: automated prediction, not conclusive proof.\n"
    )
    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button("⬇️ Download CSV", df.to_csv(index=False).encode(), "deepfake_report.csv", "text/csv")
    d2.download_button("⬇️ Download summary", summary.encode(), "deepfake_summary.txt", "text/plain")


# ======================================================================
# PAGES
# ======================================================================
if selected == "🏠 Home":
    st.markdown(
        "<div class='eyes'><div class='eye'><div class='pupil'></div><div class='eyelid'></div></div>"
        "<div class='eye'><div class='pupil'></div><div class='eyelid'></div></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='hero-title'>🛡️ DeepGuard</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>AI-powered deepfake video detection with explainable results</div>",
                unsafe_allow_html=True)

    a, b, c = st.columns(3)
    a.markdown("<div class='card'><h4>🎯 Face-focused analysis</h4><p>Detects and crops the face in every "
               "sampled frame so the model looks only at what matters.</p></div>", unsafe_allow_html=True)
    b.markdown("<div class='card'><h4>🔥 Explainable (Grad-CAM)</h4><p>Heatmaps reveal which facial regions "
               "triggered the 'fake' prediction.</p></div>", unsafe_allow_html=True)
    c.markdown("<div class='card'><h4>📈 Timeline & report</h4><p>See exactly where in the video manipulation "
               "is suspected and download the report.</p></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### How it works")
    st.graphviz_chart(
        """digraph G { rankdir=LR; bgcolor="transparent";
        node [shape=box, style="rounded,filled", fillcolor="#161b22", color="#00ffcc", fontcolor="white"];
        edge [color="#888888"];
        Video -> "Frame sampling" -> "Face detection" -> "Crop 224x224" -> ResNet50 -> "P(fake)" -> "Aggregate + Verdict";
        ResNet50 -> "Grad-CAM"; }"""
    )
    st.info("👉 Open the **Detection Tool** tab to analyse a video.")

elif selected == "🕵️ Detection Tool":
    st.markdown(f"<h2 style='text-align:center;color:{ACCENT};'>Upload & Analyze Your Video</h2>",
                unsafe_allow_html=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(f"Running on: {device.upper()}")
    model = load_detector(device)
    cascade = get_face_cascade()
    if cascade is None:
        st.warning("Face detector is unavailable in this environment (OpenCV 5?). "
                   "The full frame will be analysed. Pin `opencv-python-headless<5` in requirements.txt to enable it.")

    in_col, set_col = st.columns(2)
    with in_col:
        st.markdown("##### 🎞️ Video input")
        input_option = st.radio("Input method", ["Upload Video", "Paste Video URL"], horizontal=True,
                                label_visibility="collapsed")
        uploaded_file, video_url = None, ""
        if input_option == "Upload Video":
            uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        else:
            video_url = st.text_input("Video URL (YouTube / direct link)")
    with set_col:
        st.markdown("##### ⚙️ Detection settings")
        threshold = st.slider("Decision threshold", 0.5, 0.95, 0.7, 0.05,
                              help="P(fake) ≥ threshold → Fake; P(fake) ≤ 1 − threshold → Real; otherwise Uncertain. "
                                   "Changing it after analysis updates the results instantly.")
        sample_every = st.slider("Analyze every Nth frame", 1, 15, 5)
        max_seconds = st.slider("Max seconds to analyze", 10, 120, 60)
        use_face = st.checkbox("Face detection + crop", value=True)
        use_cam = st.checkbox("Grad-CAM explainability (slower)", value=True)
        with st.expander("Advanced (match your training setup)"):
            fake_idx = st.selectbox("Fake class index", [0, 1], index=DEFAULT_FAKE_IDX,
                                    help="ImageFolder uses alphabetical order: fake=0, real=1.")
            use_norm = st.checkbox("Use ImageNet normalization", value=DEFAULT_USE_NORM,
                                   help="Turn on only if normalization was used during training.")

    ready = uploaded_file is not None or bool(video_url.strip())
    start = st.button("🚀 Start Analysis", disabled=not ready)

    if start:
        st.session_state.pop("result", None)
        video_path, cleanup_dir = None, None
        try:
            if uploaded_file is not None:
                suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tfile.write(uploaded_file.read())
                tfile.close()
                video_path, video_name = tfile.name, uploaded_file.name
            else:
                with st.spinner("⬇️ Downloading video..."):
                    video_path, cleanup_dir = download_video(video_url.strip())
                video_name = video_url.strip()

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("❌ Could not open the video file.")
            else:
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                limit = int(fps * max_seconds)
                if total <= 0 or total > limit:
                    total = limit

                live_l, live_r = st.columns([3, 2])
                frame_ph = live_l.empty()
                cam_ph = live_r.empty()
                stat_ph = live_r.empty()
                prog = st.progress(0, text="Analyzing video...")

                times, p_list, face_list, top = [], [], [], []

                for i in range(total):
                    if not cap.grab():
                        break
                    if i % sample_every != 0:
                        continue
                    ok, frame = cap.retrieve()
                    if not ok:
                        break

                    box = detect_face(frame, cascade) if use_face else None
                    if box:
                        x0, y0, x1, y1 = box
                        crop = frame[y0:y1, x0:x1]
                    else:
                        crop = frame
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                    probs, cam = model.predict(rgb, device, use_norm, fake_idx, with_cam=use_cam)
                    p_fake = float(probs[fake_idx])
                    label = label_for(p_fake, threshold)
                    color = LABEL_BGR[label]

                    # annotate full frame
                    disp = frame.copy()
                    fscale = max(frame.shape[1] / 900.0, 0.6)
                    if box:
                        cv2.rectangle(disp, (box[0], box[1]), (box[2], box[3]), color, max(2, int(2 * fscale)))
                    cv2.putText(disp, f"{label}: {p_fake*100:.1f}%", (10, int(35 * fscale)),
                                cv2.FONT_HERSHEY_SIMPLEX, fscale, color, max(2, int(2 * fscale)))
                    if disp.shape[1] > 640:
                        s = 640.0 / disp.shape[1]
                        disp = cv2.resize(disp, None, fx=s, fy=s)
                    disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

                    cam_img = overlay_cam(rgb, cam) if cam is not None else None
                    frame_ph.image(disp_rgb, use_container_width=True)
                    if cam_img is not None:
                        cam_ph.image(cam_img, caption="Grad-CAM (model attention)", width=260)
                    else:
                        cam_ph.image(cv2.resize(rgb, (224, 224)), caption="Model input", width=260)
                    stat_ph.markdown(f"**Frame {len(p_list)+1}** · P(fake): **{p_fake*100:.1f}%** · {label}")

                    times.append(i / fps)
                    p_list.append(p_fake)
                    face_list.append(box is not None)

                    top.append((p_fake, i / fps, disp_rgb, cam_img))
                    top.sort(key=lambda r: r[0], reverse=True)
                    top = top[:3]

                    prog.progress(min((i + 1) / total, 1.0), text="Analyzing video...")

                cap.release()
                prog.empty()
                frame_ph.empty()
                cam_ph.empty()
                stat_ph.empty()

                if not p_list:
                    st.error("❌ No frames could be analysed. Please check the video.")
                else:
                    st.session_state["result"] = {
                        "times": times, "p_fake": p_list, "face_found": face_list, "top": top,
                        "meta": {"name": video_name, "sample_every": sample_every,
                                 "face_crop": use_face, "grad_cam": use_cam},
                    }
        except Exception as e:
            st.error(f"❌ Something went wrong: {e}")
        finally:
            if video_path and uploaded_file is not None and os.path.exists(video_path):
                os.remove(video_path)
            if cleanup_dir:
                shutil.rmtree(cleanup_dir, ignore_errors=True)

    if "result" in st.session_state:
        st.markdown("---")
        st.markdown("## 📋 Analysis Result")
        render_results(st.session_state["result"], threshold)
        st.caption("⚠️ Automated prediction from a frame-level model. Treat it as an indicator, not proof.")

elif selected == "📊 Model Performance":
    st.markdown(f"<h2 style='text-align:center;color:{ACCENT};'>Model Performance</h2>", unsafe_allow_html=True)

    st.markdown("##### 🧠 Architecture")
    st.write("ResNet50 backbone (ImageNet-pretrained, fine-tuned) with a 2-class head (Real / Fake), "
             "applied to 224×224 face crops. Video-level verdict = average fake probability over sampled frames.")

    perf = PERFORMANCE
    metric_keys = ["accuracy", "precision", "recall", "f1", "auc"]
    if all(perf[k] is None for k in metric_keys) and perf["confusion_matrix"] is None:
        st.info("Metrics not added yet. Open `detector.py` and fill the `PERFORMANCE` dictionary at the top "
                "with your real evaluation results (accuracy, precision, recall, F1, AUC, confusion matrix).")
    else:
        if perf["dataset"]:
            st.write(f"**Dataset:** {perf['dataset']}"
                     + (f" · Train: {perf['train_samples']}" if perf["train_samples"] else "")
                     + (f" · Val: {perf['val_samples']}" if perf["val_samples"] else "")
                     + (f" · Epochs: {perf['epochs']}" if perf["epochs"] else ""))
        shown = [(k, perf[k]) for k in metric_keys if perf[k] is not None]
        if shown:
            cols = st.columns(len(shown))
            for col, (k, v) in zip(cols, shown):
                val = v * 100 if v <= 1 else v
                col.metric(k.upper() if k in ("auc", "f1") else k.capitalize(), f"{val:.1f}%")
        if perf["confusion_matrix"] is not None:
            cm = np.array(perf["confusion_matrix"])
            fig, ax = new_fig((4, 3.5))
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Real", "Fake"])
            ax.set_yticklabels(["Real", "Fake"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            for (r, c), v in np.ndenumerate(cm):
                ax.text(c, r, str(v), ha="center", va="center",
                        color="white" if v > cm.max() / 2 else "black", fontsize=14)
            st.pyplot(fig)
            plt.close(fig)

    st.markdown("##### ⚠️ Limitations")
    st.write(
        "- Frame-level model: it does not use temporal cues such as blinking or lip-sync.\n"
        "- Accuracy depends on the training data; unseen manipulation methods or heavy compression can reduce it.\n"
        "- Haar-cascade face detection can miss side-profile, small or occluded faces.\n"
        "- Grad-CAM shows where the model looked, not a guarantee that the region is actually manipulated.\n"
        "- Output is a probability estimate and should not be treated as conclusive proof."
    )

elif selected == "ℹ️ About":
    st.markdown("### 🚀 Project Features")
    st.write(
        "- **Face detection + crop** for focused analysis\n"
        "- **Grad-CAM** explainability heatmaps\n"
        "- **Threshold-aware** Real / Fake / Uncertain classification\n"
        "- **Timeline chart** of fake probability across the video\n"
        "- **Downloadable report** (CSV + summary)\n"
        "- **Upload or URL** input (YouTube / direct links via yt-dlp)\n"
        "- **GPU (CUDA)** used automatically when available"
    )
    st.markdown("### 🛠️ Tech stack")
    st.write("Python · PyTorch · torchvision (ResNet50) · OpenCV · Streamlit · Matplotlib · yt-dlp")
    st.info("Developed by Gaurav Sharma & Team")
