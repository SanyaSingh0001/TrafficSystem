#!/usr/bin/env python3
"""
Calibration tool for static traffic cameras.
Click 4 reference points on the road, enter their real-world distances, and save calibration.
Automatically grabs the first frame from a video for calibration.
"""

import cv2
import json
import numpy as np
import os

ref_points = []
frame = None

def click_event(event, x, y, flags, param):
    global ref_points, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Calibration", frame)
        print(f"Point {len(ref_points)}: ({x}, {y})")

def calibrate(video_path, out_json="calibration.json", screen_inches=10.0):
    import math
    import tkinter as tk
    global frame, ref_points

    # Capture first frame from video
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read first frame from {video_path}")

    print("Click 4 reference points on the road (clockwise).")
    cv2.imshow("Calibration", frame)
    cv2.setMouseCallback("Calibration", click_event)

    # Wait until 4 points are clicked
    while len(ref_points) < 4:
        cv2.waitKey(100)

    cv2.destroyAllWindows()

    if len(ref_points) != 4:
        raise RuntimeError("Need exactly 4 points for calibration.")

    # Get screen resolution
    root = tk.Tk()
    root.withdraw()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()
    screen_diag_px = math.sqrt(screen_w**2 + screen_h**2)
    ppi = screen_diag_px / screen_inches
    print(f"Screen resolution: {screen_w}x{screen_h}, approx {ppi:.2f} PPI")

    # Ask user for real-world dimensions
    width_m = float(input("Enter real-world width (meters) between left-right points: "))
    height_m = float(input("Enter real-world height (meters) between top-bottom points: "))

    # Destination points in real-world scale
    dst = np.array([
        [0, 0],
        [width_m, 0],
        [width_m, height_m],
        [0, height_m]
    ], dtype=np.float32)

    src = np.array(ref_points, dtype=np.float32)

    # Compute homography
    H, _ = cv2.findHomography(src, dst)

    # Full JSON with all fields
    calib = {
        "ref_points": ref_points,
        "width_m": width_m,
        "height_m": height_m,
        "screen_resolution": [screen_w, screen_h],
        "screen_inches": screen_inches,
        "ppi": ppi,
        "homography": H.tolist()
    }

    import os
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        import json
        json.dump(calib, f, indent=2)

    print(f"[Saved] Calibration data to {out_json}")
    return H


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to video file for calibration")
    ap.add_argument("--out", default="calibration.json", help="Output JSON file")
    args = ap.parse_args()

    calibrate(args.video, args.out)