# Computer-Vision-Object-Detection

## Real-time Object Detection with YOLOv5 and OpenCV

This repository provides a practical implementation of real-time object detection using the YOLOv5 model integrated with OpenCV. It demonstrates how to set up a detection pipeline for various applications, including surveillance, autonomous systems, and quality control. The project focuses on efficiency and ease of deployment.

### Features

*   **YOLOv5 Integration:** Utilizes pre-trained YOLOv5 models for accurate and fast object detection.
*   **Real-time Processing:** Optimized for real-time video stream analysis using OpenCV.
*   **Custom Object Detection:** Includes guidelines for training custom YOLOv5 models on new datasets.
*   **Visualization:** Provides clear bounding box and label visualization for detected objects.

### Getting Started

To get this project running on your local machine, follow these steps.

#### Prerequisites

Ensure you have Python 3.8+ installed. Install the necessary libraries:

```bash
pip install -r requirements.txt
```

#### Usage

1.  **Download YOLOv5 weights:**
    ```bash
    # Example: download yolov5s.pt
    wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt
    ```
2.  **Run the object detection script:**
    ```bash
    python detect_objects.py --source 0 # for webcam
    # or
    python detect_objects.py --source video.mp4 # for video file
    ```

### Project Structure

```
. 
├── README.md
├── requirements.txt
├── detect_objects.py
└── yolov5s.pt (downloaded weights)
```

### Badges

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv)
![YOLOv5](https://img.shields.io/badge/YOLOv5-v6.0-red?style=for-the-badge&logo=yolo)

### License

This project is licensed under the MIT License - see the LICENSE file for details.
