import time, math
from collections import defaultdict, deque

class SpeedEstimator:
    def __init__(self, fps=30, history_len=10):
        self.fps = fps
        self.dt = 1.0 / fps
        self.history_len = history_len

        # per id: deque of (timestamp, x, y)
        self.positions = defaultdict(lambda: deque(maxlen=history_len))

        # per id: cumulative distance + time for averaging
        self.total_distance = defaultdict(float)
        self.first_time = {}
        self.last_time = {}
        self.avg_speed = {}

    def update(self, track_id, x, y, frame_time=None):
        """
        Update vehicle position (in meters).
        Returns: (avg_speed_mps, avg_acc_mps2)
        The avg_speed is cumulative from first frame the vehicle was seen.
        """
        if frame_time is None:
            frame_time = time.time()

        # Initialize first frame
        if track_id not in self.first_time:
            self.first_time[track_id] = frame_time
            self.last_time[track_id] = frame_time
            self.avg_speed[track_id] = 0.0

        # Append new position
        posdq = self.positions[track_id]
        posdq.append((frame_time, float(x), float(y)))

        if len(posdq) < 2:
            return self.avg_speed[track_id], 0.0

        # Distance between last two positions
        t1, x1, y1 = posdq[-2]
        t2, x2, y2 = posdq[-1]
        dt = t2 - t1 if (t2 - t1) > 0 else self.dt
        dist = math.hypot(x2-x1, y2-y1)

        # Update cumulative distance and time
        self.total_distance[track_id] += dist
        self.last_time[track_id] = t2
        t_total = self.last_time[track_id] - self.first_time[track_id]

        # Average speed
        avg_speed = self.total_distance[track_id] / t_total if t_total > 0 else 0.0

        # Acceleration as change in average speed
        prev_avg = self.avg_speed.get(track_id, 0.0)
        acc = (avg_speed - prev_avg) / dt if dt > 0 else 0.0

        # Update stored average speed
        self.avg_speed[track_id] = avg_speed

        return avg_speed, acc
