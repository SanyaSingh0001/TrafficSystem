"""
main.py
Pipeline: always run calibration -> detection -> tracking -> speed estimation
Saves one row per vehicle in CSV with final average speed.
"""

import argparse, time, os, cv2, json
from detection import load_model, detect
from tracking import SimpleTracker
from projection import pixel_to_world
from speed_estimation import SpeedEstimator
from utils import draw_box, save_csv_header, append_csv_row, annotate_and_write_video

import numpy as np
import math
import tkinter as tk

# --- Calibration ---
ref_points = []
raw_frame = None
display_frame = None
screen_w, screen_h = 0, 0

def add_black_border(frame, screen_w, screen_h):
    """Resize frame keeping aspect ratio, pad with black borders if needed.
       Returns (canvas, scale, x_offset, y_offset) so we can map back clicks."""
    h, w = frame.shape[:2]
    scale = min(screen_w / w, screen_h / h)

    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    x_offset = (screen_w - new_w) // 2
    y_offset = (screen_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas, scale, x_offset, y_offset

def click_event(event, x, y, flags, param):
    global ref_points, display_frame, raw_frame, scale, x_offset, y_offset
    if event == cv2.EVENT_LBUTTONDOWN:
        # Map from display coordinates → raw frame coordinates
        raw_x = int((x - x_offset) / scale)
        raw_y = int((y - y_offset) / scale)

        if 0 <= raw_x < raw_frame.shape[1] and 0 <= raw_y < raw_frame.shape[0]:
            ref_points.append((raw_x, raw_y))
            print(f"Point {len(ref_points)}: ({raw_x}, {raw_y})")

            # Draw marker on display frame
            cv2.circle(display_frame, (x, y), 6, (0, 0, 255), -1)
            cv2.imshow("Calibration", display_frame)

def run_calibration(video_path, calib_path, screen_inches=8.0):
    global raw_frame, display_frame, screen_w, screen_h, scale, x_offset, y_offset
    ref_points.clear()

    cap = cv2.VideoCapture(video_path)
    ret, raw_frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read video: {video_path}")

    print("Click 4 reference points on the road (clockwise).")

    # Get monitor resolution
    root = tk.Tk()
    root.withdraw()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()

    # Prepare bordered display frame
    display_frame, scale, x_offset, y_offset = add_black_border(raw_frame, screen_w, screen_h)

    cv2.namedWindow("Calibration", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Calibration", display_frame)
    cv2.setMouseCallback("Calibration", click_event)

    while len(ref_points) < 4:
        cv2.waitKey(100)

    cv2.destroyAllWindows()
    print("4 points selected. Proceeding to real-world dimensions.")


    # Screen diagonal PPI
    screen_diag_px = math.sqrt(screen_w**2 + screen_h**2)
    ppi = screen_diag_px / screen_inches
    print(f"Screen resolution: {screen_w}x{screen_h}, approx {ppi:.2f} PPI")

    # Ask user for real-world dimensions
    width_m = float(input("Enter real-world width (meters) between left-right points: "))
    height_m = float(input("Enter real-world height (meters) between top-bottom points: "))

    dst_pts = np.array([
        [0, 0],
        [width_m, 0],
        [width_m, height_m],
        [0, height_m]
    ], dtype=np.float32)
    src_pts = np.array(ref_points, dtype=np.float32)

    H, _ = cv2.findHomography(src_pts, dst_pts)

    data = {
        "ref_points": ref_points,
        "width_m": width_m,
        "height_m": height_m,
        "screen_resolution": [screen_w, screen_h],
        "screen_inches": screen_inches,
        "ppi": ppi,
        "homography": H.tolist()
    }

    os.makedirs(os.path.dirname(calib_path) or ".", exist_ok=True)
    with open(calib_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[calibration] Saved to {calib_path}")
    return H, ref_points

# --- pipeline ---
def run_pipeline(video_path, calib_path, out_csv, weights, display=False):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = load_model(weights)
    tracker = SimpleTracker(max_lost=20, iou_threshold=0.3)
    estimator = SpeedEstimator(fps=int(fps), history_len=10)

    # Load calibration JSON
    with open(calib_path) as f:
        calib_json = json.load(f)
    H = np.array(calib_json["homography"])
    calib_points = calib_json["ref_points"]

    # Prepare CSV
    csv_f, csv_writer = save_csv_header(out_csv)

    out_video_writer = None
    if display:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video_writer = annotate_and_write_video(
            out_csv.replace('.csv','_annot.mp4'), fourcc, fps, (w,h)
        )

        # Get monitor resolution
        root = tk.Tk()
        root.withdraw()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.destroy()

        cv2.namedWindow("Traffic Analysis", cv2.WINDOW_AUTOSIZE)

    frame_id = 0
    last_positions = {}

    # --- ROI average speed ---
    roi_speeds = []
    last_update = time.time()
    roi_avg_speed = 0.0

    # --- Vehicle counter ---
    counted_ids = set()
    vehicle_count = 0
    count_line_y = int(h * 0.85)  # 85% down the frame
    count_area = (50, count_line_y, w - 50, h)

    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        dets = detect(model, frame, calib_points=calib_points, conf_thresh=0.45, vehicle_only=True)
        tracks_out = tracker.update(dets)

        if display:
            cv2.polylines(frame, [np.array(calib_points, dtype=np.int32)], isClosed=True, color=(0,255,0), thickness=2)

        for t in tracks_out:
            tid = t[0]
            bbox = t[1:5]
            x1, y1, x2, y2 = bbox
            xc = int((x1 + x2) / 2)
            yc = int(y2)

            X, Y = pixel_to_world(xc, yc, H)
            if X is None:
                continue

            # Speed estimation
            speed, acc = estimator.update(tid, X, Y)
            last_positions[tid] = (X, Y, speed, acc)

            # --- ROI average speed collection ---
            if cv2.pointPolygonTest(np.array(calib_points, dtype=np.int32), (xc, yc), False) >= 0:
                roi_speeds.append(speed * 3.6)  # km/h

            # --- Vehicle counter check ---
            if tid not in counted_ids:
                if count_area[0] <= xc <= count_area[2] and count_area[1] <= yc <= count_area[3]:
                    vehicle_count += 1
                    counted_ids.add(tid)
                    print(f"[Counter] Vehicle {tid} crossed line. Total = {vehicle_count}")

            # Draw bbox + speed
            if display:
                draw_box(frame, bbox, tid=tid, label=f"{speed*3.6:.1f} km/h")

        # --- ROI avg speed update ---
        if time.time() - last_update >= 1.0:
            if roi_speeds:
                roi_avg_speed = sum(roi_speeds) / len(roi_speeds)
            else:
                roi_avg_speed = 0.0
            roi_speeds = []
            last_update = time.time()

        # --- Display metrics ---
        if display:
            # Counting area
            cv2.rectangle(frame, (count_area[0], count_area[1]), (count_area[2], count_area[3]), (0,0,255), 2)

            # Vehicle Count
            cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

            # Bulk Avg Speed
            cv2.putText(frame, f"Bulk Avg Speed: {roi_avg_speed:.1f} km/h", (w - 300, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # ✅ Add black borders for correct aspect ratio
            display_frame = add_black_border(frame, screen_w, screen_h)
            cv2.imshow("Traffic Analysis", frame)
            out_video_writer.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # --- Write final average speed per vehicle ---
    for tid, (X, Y, speed, acc) in last_positions.items():
        append_csv_row(csv_writer, frame_id, tid, X, Y, speed, acc)

    csv_f.close()
    cap.release()
    if display:
        out_video_writer.release()
        cv2.destroyAllWindows()

    print(f"✅ Done. Processed {frame_id} frames in {time.time()-start:.1f}s")
    print(f"📂 Results saved: {out_csv}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True, help="Path to video file")
    ap.add_argument('--calib', default="outputs/calibration.json", help="Calibration JSON path")
    ap.add_argument('--out', default='outputs/vehicle_speeds.csv', help="CSV output path")
    ap.add_argument('--weights', default="yolo-weights/best.pt", help="YOLO weights path")
    ap.add_argument('--display', action='store_true', help="Show annotated video")
    args = ap.parse_args()

    # Always run calibration first
    run_calibration(args.video, args.calib)

    # Run full pipeline
    run_pipeline(args.video, args.calib, args.out, weights=args.weights, display=args.display)

# ''' To run:
# python main.py --video "Video_path" --weights "yolo_model_path" --display '''
