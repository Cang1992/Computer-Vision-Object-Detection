import cv2
import torch
import numpy as np
import argparse
import time

# --- Configuration --- #
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 1000

# --- Load YOLOv5 Model --- #
def load_yolov5_model(weights_path=\'yolov5s.pt\'):
    """Loads a pre-trained YOLOv5 model."""
    try:
        model = torch.hub.load(\'ultralytics/yolov5\', \'yolov5s\', pretrained=True)
        # If local weights are preferred, uncomment the line below and ensure yolov5s.pt is in the directory
        # model = torch.hub.load(\'\', \'custom\', path=weights_path, source=\'local\')
        model.conf = CONF_THRESHOLD  # NMS confidence threshold
        model.iou = IOU_THRESHOLD    # NMS IoU threshold
        model.max_det = MAX_DETECTIONS # maximum number of detections per image
        print(f"YOLOv5 model loaded successfully from {weights_path}.")
        return model
    except Exception as e:
        print(f"Error loading YOLOv5 model: {e}")
        print("Please ensure you have an internet connection or \'yolov5s.pt\' is in the current directory.")
        return None

# --- Perform Object Detection --- #
def detect_objects(frame, model):
    """Performs object detection on a single frame."""
    if model is None:
        return frame

    # Convert frame to RGB (YOLOv5 expects RGB)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Inference
    results = model(img_rgb)

    # Process results
    detections = results.xyxy[0].cpu().numpy() # x1, y1, x2, y2, conf, cls

    for *xyxy, conf, cls in detections:
        label = model.names[int(cls)]
        color = (0, 255, 0) # Green bounding box
        x1, y1, x2, y2 = map(int, xyxy)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)

        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    return frame

# --- Main Execution Flow --- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Object Detection with YOLOv5 and OpenCV.")
    parser.add_argument("--source", type=str, default="0",
                        help="Source for video stream (0 for webcam, or path to video file).")
    parser.add_argument("--weights", type=str, default="yolov5s.pt",
                        help="Path to YOLOv5 weights file.")
    args = parser.parse_args()

    model = load_yolov5_model(args.weights)
    if model is None:
        exit()

    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {args.source}.")
        exit()

    print("Starting real-time object detection. Press \'q\' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or error reading frame.")
            break

        start_time = time.time()
        processed_frame = detect_objects(frame, model)
        end_time = time.time()

        fps = 1 / (end_time - start_time)
        cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Object Detection", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord(\'q\'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Object detection finished.")
