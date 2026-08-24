"""
video_processor.py
-------------------
Manages the full video pipeline: opening a webcam or video file, running
each frame through detection + tracking, drawing the results, keeping
real-time statistics (current + unique object counts, optional line
crossing), and handling screenshot capture / processed-video export.

This module deliberately contains NO Streamlit code - it can be tested
and reused independently of the UI layer.
"""

import os
import time
from datetime import datetime

import cv2
import numpy as np

from tracker import SortTracker
from utils import (
    FPSCounter, draw_detection, draw_crossing_line, resize_for_processing,
    color_for_class, PEOPLE_CLASSES, VEHICLE_CLASSES,
)

SCREENSHOT_DIR = os.path.join("outputs", "screenshots")
PROCESSED_VIDEO_DIR = os.path.join("outputs", "processed_videos")


class VideoProcessor:
    """Owns the capture device/file, the tracker, and all running statistics."""

    def __init__(self, detector):
        self.detector = detector
        self.tracker = SortTracker(iou_threshold=0.3, max_age=20, min_hits=1)
        self.cap = None
        self.source_label = None
        self.fps_counter = FPSCounter()

        self.frame_width = None
        self.frame_height = None

        # Statistics
        self.current_counts = {}          # class_name -> count this frame
        self.unique_ids_seen = set()      # every track ID ever confirmed
        self.unique_counts = {}           # class_name -> unique count
        self.last_fps = 0.0

        # Line-crossing (optional)
        self.line_enabled = False
        self.line_y_ratio = 0.5
        self.entered_count = 0
        self.exited_count = 0

        # Recording
        self.video_writer = None
        self.recording_path = None

        self.last_annotated_frame = None
        self.open_error = None

    # ------------------------------------------------------------------
    # Source management
    # ------------------------------------------------------------------
    def open_webcam(self, index: int = 0) -> bool:
        self.release()
        self.open_error = None
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
            if not cap.isOpened():
                self.open_error = (
                    "Unable to access webcam. Please check that another "
                    "application is not using it, and that a camera is connected."
                )
                return False
            self.cap = cap
            self.source_label = f"Webcam {index}"
            self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            return True
        except Exception as exc:
            self.open_error = f"Unexpected error opening webcam: {exc}"
            return False

    def open_video_file(self, path: str) -> bool:
        self.release()
        self.open_error = None
        if not os.path.exists(path):
            self.open_error = "The selected video file could not be found."
            return False
        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                self.open_error = (
                    "This video file could not be opened. It may be corrupted "
                    "or in an unsupported format."
                )
                return False
            self.cap = cap
            self.source_label = os.path.basename(path)
            self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            return True
        except Exception as exc:
            self.open_error = f"Unexpected error opening video file: {exc}"
            return False

    def release(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.stop_recording()

    def reset_statistics(self):
        self.tracker.reset()
        self.current_counts = {}
        self.unique_ids_seen = set()
        self.unique_counts = {}
        self.entered_count = 0
        self.exited_count = 0
        self.fps_counter = FPSCounter()

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------
    def process_next_frame(self, confidence_threshold=0.4, allowed_classes=None,
                            show_boxes=True, show_labels=True, show_conf=True,
                            show_ids=True, tracking_enabled=True):
        """
        Reads and processes the next frame.

        Returns:
            (success: bool, annotated_frame or None, stats: dict)
        """
        if self.cap is None:
            return False, None, self._empty_stats()

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None, self._empty_stats()

        try:
            processed_frame, scale = resize_for_processing(frame, max_width=640)

            detections = self.detector.detect(
                processed_frame,
                confidence_threshold=confidence_threshold,
                allowed_classes=allowed_classes,
            )

            if tracking_enabled:
                visible_tracks = self.tracker.update(detections)
                objects_to_draw = [
                    {
                        "bbox": t.bbox,
                        "class_id": t.class_id,
                        "class_name": t.class_name,
                        "confidence": t.confidence,
                        "track_id": t.id,
                    }
                    for t in visible_tracks
                ]
                self._update_unique_counts(visible_tracks)
                if self.line_enabled:
                    self._update_line_crossing(visible_tracks, processed_frame.shape[0])
            else:
                objects_to_draw = [
                    {
                        "bbox": d["bbox"],
                        "class_id": d["class_id"],
                        "class_name": d["class_name"],
                        "confidence": d["confidence"],
                        "track_id": None,
                    }
                    for d in detections
                ]

            self._update_current_counts(objects_to_draw)

            annotated = processed_frame.copy()
            for obj in objects_to_draw:
                annotated = draw_detection(
                    annotated, obj["bbox"], obj["class_name"], obj["track_id"],
                    obj["confidence"], show_box=show_boxes, show_label=show_labels,
                    show_conf=show_conf, show_id=show_ids,
                    color=color_for_class(obj["class_id"]),
                )

            if self.line_enabled:
                annotated = draw_crossing_line(
                    annotated, self.line_y_ratio, self.entered_count, self.exited_count
                )

            self.last_fps = self.fps_counter.tick()
            self.last_annotated_frame = annotated

            if self.video_writer is not None:
                self.video_writer.write(annotated)

            stats = self._build_stats()
            return True, annotated, stats

        except Exception as exc:
            return False, None, self._empty_stats(error=str(exc))

    def _update_current_counts(self, objects):
        counts = {}
        for obj in objects:
            counts[obj["class_name"]] = counts.get(obj["class_name"], 0) + 1
        self.current_counts = counts

    def _update_unique_counts(self, visible_tracks):
        for t in visible_tracks:
            if t.id not in self.unique_ids_seen:
                self.unique_ids_seen.add(t.id)
                self.unique_counts[t.class_name] = self.unique_counts.get(t.class_name, 0) + 1

    def _update_line_crossing(self, visible_tracks, frame_height):
        line_y = frame_height * self.line_y_ratio
        for t in visible_tracks:
            if t.counted_crossing:
                continue
            prev_y = t.prev_centroid_y
            curr_y = t.centroid()[1]
            if prev_y < line_y <= curr_y:
                self.entered_count += 1
                t.counted_crossing = True
            elif prev_y > line_y >= curr_y:
                self.exited_count += 1
                t.counted_crossing = True

    def _build_stats(self):
        people_current = sum(v for k, v in self.current_counts.items() if k in PEOPLE_CLASSES)
        vehicles_current = sum(v for k, v in self.current_counts.items() if k in VEHICLE_CLASSES)
        return {
            "fps": round(self.last_fps, 1),
            "current_objects": sum(self.current_counts.values()),
            "unique_objects": len(self.unique_ids_seen),
            "people_current": people_current,
            "vehicles_current": vehicles_current,
            "current_counts": dict(self.current_counts),
            "unique_counts": dict(self.unique_counts),
            "entered": self.entered_count,
            "exited": self.exited_count,
            "error": None,
        }

    def _empty_stats(self, error=None):
        return {
            "fps": 0.0, "current_objects": 0, "unique_objects": len(self.unique_ids_seen),
            "people_current": 0, "vehicles_current": 0, "current_counts": {},
            "unique_counts": dict(self.unique_counts), "entered": self.entered_count,
            "exited": self.exited_count, "error": error,
        }

    # ------------------------------------------------------------------
    # Screenshot capture
    # ------------------------------------------------------------------
    def capture_screenshot(self):
        """Saves the last annotated frame to outputs/screenshots/. Returns the path or None."""
        if self.last_annotated_frame is None:
            return None
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path = os.path.join(SCREENSHOT_DIR, filename)
        try:
            cv2.imwrite(path, self.last_annotated_frame)
            return path
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Processed video export
    # ------------------------------------------------------------------
    def start_recording(self):
        if self.cap is None or self.frame_width is None:
            return None
        os.makedirs(PROCESSED_VIDEO_DIR, exist_ok=True)
        filename = f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        path = os.path.join(PROCESSED_VIDEO_DIR, filename)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # We record at the (possibly downscaled) processing resolution.
        width = min(640, self.frame_width)
        height = int(self.frame_height * (width / self.frame_width)) if self.frame_width else 480
        try:
            self.video_writer = cv2.VideoWriter(path, fourcc, 20.0, (width, height))
            self.recording_path = path
            return path
        except Exception:
            self.video_writer = None
            self.recording_path = None
            return None

    def stop_recording(self):
        if self.video_writer is not None:
            try:
                self.video_writer.release()
            except Exception:
                pass
            self.video_writer = None
        path = self.recording_path
        self.recording_path = None
        return path

    def is_recording(self) -> bool:
        return self.video_writer is not None
