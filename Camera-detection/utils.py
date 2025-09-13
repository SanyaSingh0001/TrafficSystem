"""
utils.py
Helper functions: draw boxes, save CSV, annotate frames
"""

import csv, os, cv2

def draw_box(frame, bbox, tid=None, label=None, color=(0,255,0)):
    x1,y1,x2,y2 = map(int, bbox)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    txt = f"ID:{tid}" if tid is not None else ""
    if label: txt += f" {label}"
    cv2.putText(frame, txt, (x1, max(10,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def save_csv_header(path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    f = open(path, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(["frame","id","x_m","y_m","speed_mps","speed_kmph","acc_mps2"])
    return f, writer

def append_csv_row(writer, frame_id, tid, x_m, y_m, speed, acc):
    writer.writerow([frame_id, tid, x_m, y_m, speed, speed*3.6, acc])

def annotate_and_write_video(out_path, fourcc, fps, frame_size):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    writer = cv2.VideoWriter(out_path, fourcc, fps, frame_size)
    return writer
