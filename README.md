# 🗑️ Smart Urban Waste Management: Garbage Bin Overflow Detection

An end-to-end Computer Vision pipeline designed to detect garbage bins and identify overflow conditions in real-time. This project leverages state-of-the-art object detection (YOLOv9) and instance segmentation to promote smarter urban waste management, optimize collection routes, and reduce environmental hazards.

---

## 🎯 Project Objective
The primary goal of this project was to build a smart retail/urban monitoring solution capable of identifying standard garbage bins and overflowing bins in diverse real-world environments. 

Furthermore, the system required a real-time inference mechanism where a smartphone camera could act as a wireless edge device, streaming frames to a local backend server (PC) for rapid GPU inference.

---

## 🛤️ The Journey: From Concept to Deployment

### 1. What We Started With
When this project began, we faced several significant hurdles regarding data quality and model infrastructure:
*   **Severe Class Imbalance**: The initial raw datasets (`Set1`, `Set2`) had an overwhelming number of empty bins and a severe lack of "overflow" examples.
*   **Inconsistent Annotations**: We encountered conflicting bounding box guidelines across different data sources (e.g., partial boxes vs. full boxes).
*   **Infrastructure Friction**: Moving trained models between cloud compute (RunPod RTX 4090) and local hardware (NVIDIA RTX 3060) caused PyTorch absolute path mapping errors in the checkpoint files.
*   **Basic Tooling**: We started with older, smaller detection models that struggled with the complexities of cluttered urban backgrounds.

### 2. What We Did (Implementation & Solutions)
To overcome these challenges, we executed a massive data refactor and pipeline overhaul:

*   **Data Engineering & Balancing**: We merged 211 high-quality "overflow" images into the base dataset. We converted raw CSV absolute pixel coordinates into normalized YOLO format.
*   **Strict Splitting & Smart Augmentation**: To prevent data leakage, we performed a strict randomized 85/10/5 split *before* any augmentation. We then wrote a custom OpenCV script (`split_and_augment.py`) to apply horizontal flips and brightness adjustments exclusively to the training pool, mathematically recalculating bounding boxes on the fly.
*   **Migrating to Instance Segmentation**: We didn't stop at bounding boxes. We developed `auto_mask.py`, which utilizes **SAM2 (Segment Anything Model 2)** to automatically convert our bounding box annotations into high-quality, normalized polygon masks for YOLO segmentation training.
*   **The YOLOv9 Upgrade**: We migrated our entire training pipeline to **YOLOv9** (specifically `yolov9c.pt` for detection and `yolov9c-seg.pt` for segmentation). We optimized hyperparameters (Box Loss: 7.5, Cls Loss: 0.5, DFL Loss: 1.5) and batch sizes to maximize the GELAN architecture's performance within the 12GB VRAM limit of an RTX 3060.
*   **Live Edge-to-Server Deployment**: We built `web_server.py`, a robust Flask application with a modern, responsive UI. It handles asynchronous frame streaming from any LAN-connected mobile device, processes the frames through the YOLOv9 model, and returns annotated video feeds in real-time.

### 3. What We Learned
This project was a deep dive into the realities of applied Computer Vision. Key takeaways include:
*   **Data Leakage Prevention**: Experiencing firsthand how augmenting before splitting can artificially inflate validation metrics, and implementing the correct pipeline to fix it.
*   **Foundation Models in the Loop**: Learning to leverage powerful foundation models like SAM2 not for end-inference, but as automated data labeling engines to accelerate development.
*   **Hardware-Aware Optimization**: Understanding how to balance batch sizes, image resolutions, and mixed-precision training (`amp=True`) to avoid CUDA Out-Of-Memory errors while maximizing GPU utilization.
*   **State Management in PyTorch**: Building patchers to strip hardcoded cloud paths from `.pt` checkpoint dictionaries to ensure model portability.

### 4. The Final Outcome
*   **A Healthy Dataset**: A robust, finely-tuned dataset of **5,000+ images** boasting a healthy ~1.7:1 ratio of 'bin' to 'overflow' annotations.
*   **High Accuracy Model**: Our final YOLOv9c model achieved a peak **mAP50 of >80%**, demonstrating excellent recall even in cluttered environments.
*   **A Production-Ready App**: A fully functional, network-accessible web application capable of running live inference from mobile cameras with sub-second latency.

---

## 🛠️ Tech Stack
- **Deep Learning**: Python 3.11, PyTorch, Ultralytics YOLOv8/YOLOv9, SAM2
- **Data Engineering**: OpenCV, NumPy, Custom Python Scripts
- **Deployment**: Flask, HTML5/CSS3 (Glassmorphism UI), JavaScript Streams
- **Hardware Profile**: Optimized for NVIDIA RTX 3060 (12GB VRAM)

---

## 📂 Project Structure
```bash
.
├── Main.py               # Unified YOLOv9 training script (Detect/Segment)
├── web_server.py         # Flask server for real-time mobile LAN inference
├── requirements.txt      # Python dependencies
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/pok8thecoder/garbage-bin-detection.git
   cd garbage-bin-detection
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📈 Usage Guide

### 1. Training the Model
The `Main.py` script is fully configured. You can toggle between detection and segmentation tasks directly inside the file.
```bash
python Main.py
```

### 2. Running the Live Inference Server
To stream from your phone's camera to your PC for real-time GPU inference:
1. Start the Flask server (ensure your PC and phone are on the same WiFi network):
   ```bash
   python web_server.py --host 0.0.0.0 --https
   ```
   *(Note: The `--https` flag generates an adhoc SSL certificate, which is required by most modern mobile browsers to grant camera access).*
2. Open the displayed IP address (e.g., `https://192.168.1.X:5000`) on your mobile device.
3. Grant camera permissions and click "Start Camera".
