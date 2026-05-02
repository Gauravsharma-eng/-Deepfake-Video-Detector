Deepfake Video Detector 🕵️🎬
A real-time Deepfake Video Detection system built using PyTorch, OpenCV, and Streamlit. This project features frame-wise detection, annotated video output, and a unique horror-themed interactive UI.

🚀 Live Demo
You can try the live version of the Deepfake Video Detector here:
👉 https://deepfake-video-detector-rs.streamlit.app/
(Note: The app might take a few seconds to wake up if it hasn't been used recently.)

🔹 Features
Video Support: Upload any video in MP4/AVI/MOV formats.

AI Analysis: Frame-wise Real vs Fake detection using a ResNet50 backbone.

Visual Insights: Live Pie Chart showing Real/Fake/Uncertain distribution during processing.

Annotated Output: Generates a downloadable video with labels and confidence scores.

Interactive UI: Features "Evil Eyes" horror animations and a sleek Dark Theme.

Customizable: Confidence threshold slider to fine-tune detection sensitivity.

🛠️ Technologies Used
Python 3.11

PyTorch & torchvision (Deep Learning Framework)

Streamlit (Web Interface)

OpenCV (Video Processing)

Matplotlib (Data Visualization)

CSS/HTML (Custom UI Animations)

📁 File Structure
Plaintext
.
├── checkpoints/
│   └── model_best.pth       # Pre-trained AI model (89.9 MB)
├── outputs/                 # Directory for processed videos
├── detector.py              # Main Streamlit application code
├── requirements.txt         # Python dependencies
├── packages.txt             # System-level dependencies (libgl1)
└── README.md                # Project documentation
⚡ How to Run Locally
Clone the repository:

Bash
git clone [https://github.com/Gauravsharma-eng/Deepfake-Video-Detector.git](https://github.com/Gauravsharma-eng/Deepfake-Video-Detector.git)
cd Deepfake-Video-Detector
Install dependencies:

Bash
pip install -r requirements.txt
Run the app:

Bash
streamlit run detector.py
👤 Author
Gaurav Sharma
Final Year B.Tech Student at VITM, Gwalior
