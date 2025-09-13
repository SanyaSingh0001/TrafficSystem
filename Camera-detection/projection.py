"""
projection.py
Load calibration.json and provide pixel_to_world conversion using homography.
"""

import json
import numpy as np

def load_calibration(path="calibration.json"):
    with open(path, "r") as f:
        data = json.load(f)
    H = np.array(data["homography"], dtype=float)
    return H, data

def pixel_to_world(x, y, H):
    pt = np.array([float(x), float(y), 1.0])
    world = H.dot(pt)
    if world[2] == 0:
        return None, None
    world = world / world[2]
    return float(world[0]), float(world[1])
