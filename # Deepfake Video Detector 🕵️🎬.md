# Deepfake Video Detector 🕵️🎬

A real-time Deepfake Video Detection system built using **PyTorch**, **OpenCV**, and **Streamlit**, featuring frame-wise detection, annotated video output, and interactive UI effects.

---

## 🔹 Features

- Upload any video in **MP4/AVI/MOV** formats.
- **Frame-wise Real vs Fake detection** with confidence scores.
- **Annotated video output** with labels.
- **Live Pie Chart** showing frame-wise Real/Fake/Uncertain distribution.
- **Evil Eyes Horror UI animation** for an engaging interface.
- **Dark Theme Professional UI**.
- Confidence threshold slider for filtering uncertain frames.
- GPU acceleration supported (CUDA if available).

---

## 🛠️ Technologies Used

- **Python 3.10+**
- **Streamlit** for web interface
- **PyTorch** for deep learning model
- **Torchvision** for pretrained ResNet50 backbone
- **OpenCV** for video processing
- **Matplotlib** for live pie charts
- **Streamlit Option Menu** for navigation
- HTML & CSS for custom UI animations

---

## 📁 File Structure


---

## ⚡ How to Run

1. **Clone the repository**:

```bash
git clone https://github.com/Gauravsharma-eng/deepfake-detector.git
cd deepfake-detector

pip install -r requirements.txt

streamlit run app.py



👤 Author
Gaurav Sharma 
