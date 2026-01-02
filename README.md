🚗 License Plate Recognition System

YOLOv11 + GPT-5 Vision powered Streamlit App

A web-based License Plate Detection and Recognition System built using YOLOv11 for license plate detection and OpenAI GPT-5 Vision API for accurate plate text recognition.
The application is implemented using Streamlit and supports image and video processing with detailed analytics.

✨ Key Features
🔍 License Plate Detection

Vehicle license plate detection using YOLOv11

Supports multiple plates in a single image or video frame

Bounding box visualization with confidence scores

🧠 License Plate Recognition (OCR)

Plate text recognition powered by GPT-5 Vision API

Automatic fallback to GPT-4o if GPT-5 is unavailable

Robust handling of:

Blurry images

Low-light conditions

Angled or distorted plates

Various plate formats

📷 Image Processing

Upload images (JPG, PNG, BMP)

Detect and crop license plates

Display bounding boxes and recognized plate numbers

Export results as CSV

Show per-plate latency and average processing time

🎥 Video Processing

Upload videos (MP4, AVI, MOV, MKV)

Flexible detection modes:

Every frame (maximum accuracy)

Every 5 frames

Every 10 frames

Every 20 frames (fastest)

Real-time progress display and preview

Download processed video with bounding boxes

Export detection logs as CSV

Detection timeline visualization

🏗️ System Architecture
Input Image / Video
        │
        ▼
YOLOv11 (License Plate Detection)
        │
        ▼
Plate Cropping
        │
        ▼
GPT-5 Vision API (Text Recognition)
        │
        ▼
Post-processing & Validation
        │
        ▼
Visualization + CSV Export

📁 Project Structure
licenseplate/
├── README.md
├── index4.py              # Main Streamlit application
├── bestyolov11.pt         # YOLOv11 trained model
└── requirements.txt       # Python dependencies

⚙️ Requirements

Python 3.9+

OpenAI API key (GPT-5 / GPT-4o access)

Python Dependencies

All dependencies are listed in requirements.txt.

🚀 Installation & Usage
1️⃣ Clone Repository
git clone https://github.com/nabilamutiara/licenseplate.git
cd licenseplate

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Application
streamlit run index4.py

🔑 OpenAI API Key Setup

Obtain your API key from OpenAI

Enter the key in the sidebar input field inside the application

The system will automatically initialize the detector

⚠️ Note:
GPT-5 requires API access. If unavailable, the system will automatically fallback to GPT-4o.

📊 Output & Results
Image Mode

Bounding boxes with confidence scores

Recognized plate numbers

Detection latency per plate

CSV export of detection results

Video Mode

Processed video with bounding boxes

Detection statistics

Average GPT latency

Detection timeline

CSV export of all detections

⚠️ Performance Notes

Every frame mode

Highest accuracy

Slowest processing

Highest API cost

Every 5–20 frames

Recommended for most use cases

Better balance between speed and accuracy

🎓 Academic Use Case

This project is suitable for:

Intelligent Transportation Systems (ITS)

Smart city surveillance

Parking management systems

Traffic monitoring

Academic research and final-year projects

🧩 Technologies Used

YOLOv11 – License plate detection

OpenAI GPT-5 Vision API – Plate text recognition

Streamlit – Web interface

OpenCV – Image & video processing

NumPy / Pandas – Data processing

PyTorch – Model inference

📌 Disclaimer

This application requires an active OpenAI API key.
Processing videos frame-by-frame may incur high API usage costs.
