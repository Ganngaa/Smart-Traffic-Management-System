# Smart-Traffic-Management-System
SMART TRAFFIC MANAGEMENT SYSTEM is a computer vision-based traffic monitoring and control platform built using Flask (backend) and YOLOv8 (AI model). It detects vehicles, violations, emergency vehicles, and traffic conditions in real time using video feeds.

This repository contains:

* A Flask-based web application for traffic monitoring.
* YOLOv8 models for vehicle detection, ambulance detection, and violation detection.
* Video processing modules for real-time traffic analysis.

## 1. Core Capabilities

* Real-time vehicle detection using YOLOv8.
* Traffic signal violation detection.
* Zebra crossing violation detection.
* Ambulance detection with priority handling.
* Traffic density estimation for smart signal control.
* Image/frame storage for violation evidence.

## 2. High-Level Data Flow

1. Video feed is captured from uploaded source.
2. Frames are processed using YOLOv8 models.
3. Vehicles are detected with bounding boxes.
4. System checks for:

   * Signal violations
   * Zebra crossing violations
   * Emergency vehicles
5. Violations are logged.
6. Processed frames are streamed to the web interface.
7. Admin/user monitors results via dashboard.

## 3. Tech Stack

Frontend:

* HTML, CSS, JavaScript
* Flask Templates (Jinja2)

Backend:

* Flask (Python)
* OpenCV (Video Processing)

AI/ML:

* YOLOv8 (Ultralytics)
* Python libraries (NumPy, Torch)

## 4. Repository Structure

* app.py: Main Flask application
* zebra.py: Handles zebra crossing violation detection logic
* combined.py: Integrates multiple detection modules (vehicle, ambulance, and violations)
* templates/: HTML pages for UI
* static/: CSS, JS, uploaded media
* models/: YOLOv8 model files
* uploads/: Input videos/images
* outputs/: Processed results
* utils/: Helper scripts for detection and processing

## 5. Supplementary Files

This project includes additional files for model execution, testing, and system support.

Configuration and environment support:

* `requirements.txt`: Python dependencies

Model and detection support:

* YOLOv8 weight files (`.pt`)
* Custom-trained ambulance detection model

Video processing support:

* Sample input videos/images in `uploads/`
* Output processed frames in `outputs/`

Operational helper scripts:

* Detection scripts integrated inside main Flask app
* Model loading and inference utilities

Supplementary notes:

* Project may include dataset references for training YOLO models
* Pretrained weights are used for faster deployment

## 6. Prerequisites

* Python 3.8+
* Pip (Python package manager)
* OpenCV
* PyTorch
* Ultralytics YOLOv8
* Web browser

## 7. Backend Setup

### 7.1 Create environment and install dependencies

```bash
python -m venv venv
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 7.2 Run the application

```bash
python app.py
```


### 7.3 Access application

* Open browser:

```
http://127.0.0.1:5000
```

## 8. Frontend Setup

* No separate setup required (Flask handles frontend).
* Templates automatically render UI.

## 9. Live Video Processing Contract

Input:

* Uploaded video

Processing includes:

* Frame extraction
* Object detection using YOLOv8
* Bounding box generation
* Violation detection logic

Output:

* Annotated video stream
* Stored violation images
* Detection logs

## 10. ML and Dataset Notes

* YOLOv8 pretrained model is used for vehicle detection.
* Custom dataset used for ambulance and fire engine detection .
* Model trained using:

  * Images of vehicles
  * Traffic scenarios
  * Emergency vehicles

Training tools:

* Ultralytics YOLOv8 framework
* Python-based preprocessing

## 11. Major API Routes

* `/` → Home page
* `/video_feed` → Live video streaming
* `/upload` → Upload video/image
* `/detect` → Run detection
* `/results` → View processed output

## 12. Roles and Responsibilities

* Admin:

  * Monitor traffic system
  * Analyze violations
* System:

  * Detect vehicles and violations
  * Generate alerts
* User:

  * Upload videos/images
  * View results

## 13. Troubleshooting

Application not starting:

* Ensure Python environment is activated
* Check dependencies installed correctly

Model not detecting:

* Verify `.pt` model files exist
* Check correct model path

Video not loading:

* Ensure file format is supported (mp4, avi)

Slow performance:

* Use GPU if available
* Reduce video resolution

---


## 14. Development Command Summary

```bash
# Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run
python app.py
```

