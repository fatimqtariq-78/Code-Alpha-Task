"""
tracker.py
----------
A genuine SORT-style ("Simple Online and Realtime Tracking") multi-object
tracker: each tracked object is modeled with a Kalman filter (constant-
velocity motion model), and detections are assigned to tracks each frame
using IOU (Intersection-over-Union) cost solved optimally with the
Hungarian algorithm.

HONEST IMPLEMENTATION NOTE:
The original SORT paper's reference implementation uses the `filterpy`
library for its Kalman filter. To avoid an extra native dependency that
can cause version conflicts on some student machines, the Kalman filter
here is implemented directly with NumPy using the same 7-state constant-
velocity model SORT uses: state = [cx, cy, s, r, vx, vy, vs] where
(cx, cy) is the box center, s is the box area (scale), r is the aspect
ratio (assumed constant), and (vx, vy, vs) are their velocities. This is
a real predict/update Kalman filter combined with the Hungarian algorithm
for assignment - i.e. a proper SORT-style tracker, just without the
filterpy dependency.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


# ------------------------------------------------------------------
# Bounding-box <-> Kalman state-space conversions
# ------------------------------------------------------------------
def bbox_to_measurement(bbox):
    """[x1, y1, x2, y2] -> column vector [[cx], [cy], [s], [r]]."""
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    s = w * h
    r = w / h
    return np.array([[cx], [cy], [s], [r]], dtype=np.float64)


def state_to_bbox(x):
    """State vector (7x1) -> [x1, y1, x2, y2]."""
    cx, cy, s, r = float(x[0, 0]), float(x[1, 0]), float(x[2, 0]), float(x[3, 0])
    s = max(s, 1e-6)
    r = max(r, 1e-6)
    w = np.sqrt(s * r)
    h = s / w if w > 1e-9 else 1e-6
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


def iou(box_a, box_b) -> float:
    """Intersection-over-Union between two [x1, y1, x2, y2] boxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


# ------------------------------------------------------------------
# Single-object Kalman-filter tracker
# ------------------------------------------------------------------
class KalmanBoxTracker:
    """
    Tracks a single object's bounding box over time using a constant-
    velocity Kalman filter. Every instance gets a permanent, unique
    tracking ID.
    """

    _next_id = 1

    def __init__(self, bbox, class_id, class_name, confidence):
        # State: [cx, cy, s, r, vx, vy, vs]^T
        self.x = np.zeros((7, 1), dtype=np.float64)
        self.x[:4] = bbox_to_measurement(bbox)

        # State transition matrix (constant velocity model, dt = 1 frame)
        self.F = np.eye(7)
        for i in range(3):
            self.F[i, i + 4] = 1.0

        # Measurement matrix: we only observe [cx, cy, s, r] directly
        self.H = np.zeros((4, 7))
        for i in range(4):
            self.H[i, i] = 1.0

        # Covariances (standard SORT-style tuning)
        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0     # high initial uncertainty on velocity
        self.Q = np.eye(7)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0       # scale/ratio measurements are noisier

        self.id = KalmanBoxTracker._next_id
        KalmanBoxTracker._next_id += 1

        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence

        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.counted_crossing = False
        self.prev_centroid_y = float(self.x[1, 0])

    # ---- Kalman filter steps ----
    def predict(self):
        """Predicts this track's next state (called once per frame)."""
        if (self.x[6, 0] + self.x[2, 0]) <= 0:
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return state_to_bbox(self.x)

    def update(self, bbox, class_id, class_name, confidence):
        """Corrects the filter using a newly matched detection."""
        self.prev_centroid_y = float(self.x[1, 0])

        z = bbox_to_measurement(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0

    @property
    def bbox(self):
        return state_to_bbox(self.x)

    def centroid(self):
        return np.array([float(self.x[0, 0]), float(self.x[1, 0])])


# ------------------------------------------------------------------
# Multi-object manager: association + track lifecycle
# ------------------------------------------------------------------
def associate_detections_to_trackers(predicted_bboxes, detection_bboxes, iou_threshold):
    """
    Solves the assignment problem between predicted track positions and
    new detections using IOU cost + the Hungarian algorithm.

    Returns:
        matches: list of (track_index, detection_index)
        unmatched_trackers: list of track indices with no match
        unmatched_detections: list of detection indices with no match
    """
    if len(predicted_bboxes) == 0 or len(detection_bboxes) == 0:
        return [], list(range(len(predicted_bboxes))), list(range(len(detection_bboxes)))

    iou_matrix = np.zeros((len(predicted_bboxes), len(detection_bboxes)), dtype=np.float32)
    for i, pred_box in enumerate(predicted_bboxes):
        for j, det_box in enumerate(detection_bboxes):
            iou_matrix[i, j] = iou(pred_box, det_box)

    row_idx, col_idx = linear_sum_assignment(-iou_matrix)

    matches, matched_trk, matched_det = [], set(), set()
    for r, c in zip(row_idx, col_idx):
        if iou_matrix[r, c] >= iou_threshold:
            matches.append((r, c))
            matched_trk.add(r)
            matched_det.add(c)

    unmatched_trackers = [i for i in range(len(predicted_bboxes)) if i not in matched_trk]
    unmatched_detections = [j for j in range(len(detection_bboxes)) if j not in matched_det]
    return matches, unmatched_trackers, unmatched_detections


class SortTracker:
    """
    SORT-style multi-object tracker: predicts every track's next position
    with a Kalman filter, matches detections to tracks via IOU + Hungarian
    assignment, spawns new tracks for unmatched detections, and retires
    tracks that go unseen for too long.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 20, min_hits: int = 1):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []

    def reset(self):
        self.trackers = []
        KalmanBoxTracker._next_id = 1

    def update(self, detections):
        """
        Args:
            detections: list of dicts with keys bbox, class_id, class_name, confidence

        Returns:
            List of KalmanBoxTracker objects currently visible (matched this frame).
        """
        predicted_bboxes = [t.predict() for t in self.trackers]
        detection_bboxes = [d["bbox"] for d in detections]

        matches, _, unmatched_dets = associate_detections_to_trackers(
            predicted_bboxes, detection_bboxes, self.iou_threshold
        )

        for track_idx, det_idx in matches:
            det = detections[det_idx]
            self.trackers[track_idx].update(
                det["bbox"], det["class_id"], det["class_name"], det["confidence"]
            )

        for det_idx in unmatched_dets:
            det = detections[det_idx]
            self.trackers.append(
                KalmanBoxTracker(det["bbox"], det["class_id"], det["class_name"], det["confidence"])
            )

        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        # Only return tracks actually matched this frame - predicted-but-
        # unseen "ghost" boxes are never drawn or counted.
        return [t for t in self.trackers if t.time_since_update == 0]
