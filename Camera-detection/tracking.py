"""
tracking.py
Simple centroid-based tracker that assigns persistent IDs to bbox detections.
"""

import numpy as np
from collections import OrderedDict
import time

class Track:
    def __init__(self, tid, bbox, max_history=30):
        self.id = tid
        self.bbox = bbox
        self.hits = 1
        self.time_since_update = 0
        self.history = [bbox]
        self.max_history = max_history
        self.last_seen = time.time()

    def update(self, bbox):
        self.bbox = bbox
        self.hits += 1
        self.time_since_update = 0
        self.history.append(bbox)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        self.last_seen = time.time()

    def predict(self):
        return self.bbox

def iou(b1, b2):
    x11,y11,x12,y12 = b1
    x21,y21,x22,y22 = b2
    xi1 = max(x11, x21); yi1 = max(y11, y21)
    xi2 = min(x12, x22); yi2 = min(y12, y22)
    iw = max(0, xi2 - xi1); ih = max(0, yi2 - yi1)
    inter = iw * ih
    a1 = max(1, (x12-x11)*(y12-y11))
    a2 = max(1, (x22-x21)*(y22-y21))
    union = a1 + a2 - inter
    return inter/union if union > 0 else 0.0

class SimpleTracker:
    def __init__(self, max_lost=30, iou_threshold=0.3):
        self.next_id = 1
        self.tracks = OrderedDict()
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold

    def update(self, detections):
        boxes = [d[:4] for d in detections]
        if len(self.tracks) == 0:
            for b in boxes:
                t = Track(self.next_id, b)
                self.tracks[self.next_id] = t
                self.next_id += 1
            return [(t.id, *t.bbox) for t in self.tracks.values()]

        track_ids = list(self.tracks.keys())
        preds = [self.tracks[tid].predict() for tid in track_ids]
        iou_matrix = np.zeros((len(preds), len(boxes)), dtype=float)
        for i, pb in enumerate(preds):
            for j, b in enumerate(boxes):
                iou_matrix[i, j] = iou(pb, b)

        matched_tracks, matched_boxes = set(), set()
        for _ in range(min(len(preds), len(boxes))):
            i,j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            if iou_matrix[i,j] < self.iou_threshold:
                break
            track_id = track_ids[i]
            self.tracks[track_id].update(boxes[j])
            matched_tracks.add(i); matched_boxes.add(j)
            iou_matrix[i,:] = -1; iou_matrix[:,j] = -1

        for j, b in enumerate(boxes):
            if j not in matched_boxes:
                t = Track(self.next_id, b)
                self.tracks[self.next_id] = t
                self.next_id += 1

        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                tr = self.tracks.get(tid)
                if tr:
                    tr.time_since_update += 1
                    if tr.time_since_update > self.max_lost:
                        del self.tracks[tid]

        return [(t.id, *t.bbox) for t in self.tracks.values()]
