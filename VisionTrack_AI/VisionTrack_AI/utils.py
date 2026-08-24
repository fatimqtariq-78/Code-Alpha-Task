"""
utils.py
--------
Shared helper functions and constants used across the VisionTrack AI
pipeline: color assignment, drawing bounding boxes, FPS calculation,
and the list of object classes exposed in the UI filter.
"""

import time
from collections import deque

import cv2
import numpy as np

# ------------------------------------------------------------------
# Theme colors (BGR, since OpenCV draws in BGR not RGB)
# Dark charcoal / warm amber monitoring theme.
# ------------------------------------------------------------------
AMBER = (11, 154, 245)         # primary accent (BGR for a warm amber/orange)
AMBER_DIM = (30, 110, 170)     # muted amber for secondary highlights
OFF_WHITE = (235, 235, 240)    # label text
CHARCOAL = (34, 30, 28)        # label background

# A small, deliberately muted palette (not neon) used to differentiate
# object classes visually while staying inside the dark/amber theme.
_PALETTE = [
    (11, 154, 245),   # amber
    (140, 199, 92),   # muted sage green
    (219, 152, 52),   # soft teal-blue
    (168, 132, 219),  # dusty violet
    (100, 189, 235),  # warm gold
    (170, 170, 170),  # neutral gray
]


def color_for_class(class_id: int):
    """Deterministically assigns a muted color to a class ID."""
    return _PALETTE[class_id % len(_PALETTE)]


# Common COCO classes exposed in the UI's "Object Class Filter".
# (The underlying YOLO model still detects all 80 COCO classes -
# this list only controls what the UI lets the user filter on.)
FILTERABLE_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "dog", "cat", "backpack", "bottle", "chair", "laptop", "cell phone",
]

# Classes grouped for the "People" / "Vehicles" statistic cards.
PEOPLE_CLASSES = {"person"}
VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck"}


class FPSCounter:
    """Rolling-average FPS counter over the last N frames."""

    def __init__(self, window: int = 20):
        self.timestamps = deque(maxlen=window)

    def tick(self) -> float:
        now = time.time()
        self.timestamps.append(now)
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed


def draw_detection(frame, bbox, class_name, track_id, confidence,
                    show_box=True, show_label=True, show_conf=True,
                    show_id=True, color=AMBER):
    """
    Draws a single detection/track on the frame: bounding box + a
    compact label bar (class, tracking ID, confidence).
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]

    if show_box:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if show_label:
        parts = [class_name]
        if show_id and track_id is not None:
            parts.append(f"ID {track_id:02d}")
        if show_conf:
            parts.append(f"{confidence * 100:.0f}%")
        label = "  •  ".join(parts)

        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        label_bg_top = max(0, y1 - text_h - 10)
        cv2.rectangle(
            frame,
            (x1, label_bg_top),
            (x1 + text_w + 12, y1),
            color,
            thickness=-1,
        )
        cv2.putText(
            frame, label, (x1 + 6, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, CHARCOAL, 1, cv2.LINE_AA
        )

    return frame


def draw_crossing_line(frame, line_y_ratio: float, entered: int, exited: int):
    """Draws the optional horizontal line-crossing counter on the frame."""
    h, w = frame.shape[:2]
    y = int(h * line_y_ratio)
    cv2.line(frame, (0, y), (w, y), AMBER, 2, cv2.LINE_AA)
    label = f"Entered: {entered}   Exited: {exited}"
    cv2.putText(
        frame, label, (12, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, AMBER, 2, cv2.LINE_AA
    )
    return frame


def resize_for_processing(frame, max_width: int = 640):
    """
    Resizes a frame down to a maximum width for faster CPU inference,
    preserving aspect ratio. Returns the resized frame and the scale
    factor used (so bounding boxes can be mapped back if needed).
    """
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame, 1.0
    scale = max_width / float(w)
    resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return resized, scale
