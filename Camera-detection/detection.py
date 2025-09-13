"""
detection.py
YOLOv8 detector adapted to work with main.py and only detect inside calibration polygon.
"""

import math
import cvzone
import cv2
from ultralytics import YOLO
import numpy as np
from matplotlib.path import Path

def load_model(weights_path):
    """
    Load YOLO model (best.pt)
    """
    model = YOLO(weights_path)
    print(f"[detection] Loaded YOLO model from {weights_path}")
    return model

def point_in_polygon(x, y, poly):
    """
    Check if a point (x,y) is inside a polygon
    poly = [(x1,y1), (x2,y2), ...]
    """
    path = Path(poly)
    return path.contains_point((x, y))

def detect(model, frame, calib_points=None, conf_thresh=0.3, vehicle_only=True):
    """
    Run detection on a single frame.
    Only return detections inside the calibration polygon.
    Returns list of [x1, y1, x2, y2, conf, cls]
    """
    results = model(frame, stream=True)
    dets = []

    if calib_points is not None:
        poly = np.array(calib_points, dtype=np.int32)

        # Draw the polygon on frame for visualization
        cv2.polylines(frame, [poly], isClosed=True, color=(0,255,0), thickness=2)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if conf < conf_thresh:
                continue

            # Filter only vehicles if needed
            if vehicle_only:
                allowed = [0]  # adapt if your model uses COCO classes
                if cls not in allowed:
                    continue

            # Center of bounding box
            xc = int((x1 + x2) / 2)
            yc = int((y1 + y2) / 2)

            # Only keep if inside calibration polygon
            if calib_points is not None and not point_in_polygon(xc, yc, poly):
                continue

            dets.append([x1, y1, x2, y2, conf, cls])

    return dets
