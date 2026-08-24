"""
detector.py
-----------
Wraps the Ultralytics YOLO model so it is loaded exactly ONCE and reused
for every frame, rather than being re-initialized repeatedly (which would
be extremely slow).

The model runs entirely on CPU by default (no GPU required) and uses a
lightweight nano model (yolov8n.pt) so it stays usable on a normal
student laptop.
"""

from ultralytics import YOLO


class ObjectDetector:
    """A thin, reusable wrapper around a pretrained YOLOv8 model."""

    def __init__(self, model_name: str = "yolov8n.pt"):
        """
        Loads the YOLO model once. The first time this runs, Ultralytics
        will automatically download the pretrained weights (a few MB for
        the nano model) - this requires an internet connection just once;
        after that, the weights are cached locally.
        """
        self.model_name = model_name
        self.load_error = None
        self.model = None
        self.class_names = {}

        try:
            self.model = YOLO(model_name)
            self.class_names = self.model.names  # {id: class_name}
        except Exception as exc:
            self.load_error = (
                f"Could not load YOLO model '{model_name}'. "
                f"Check your internet connection for the first-time download. ({exc})"
            )

    def is_ready(self) -> bool:
        return self.model is not None and self.load_error is None

    def detect(self, frame, confidence_threshold: float = 0.4, allowed_classes=None):
        """
        Runs YOLO inference on a single BGR frame (as read by OpenCV).

        Args:
            frame: numpy array (H, W, 3), BGR
            confidence_threshold: minimum confidence to keep a detection
            allowed_classes: optional set/list of class names to keep;
                              if None, all detected classes are kept

        Returns:
            List of dicts, each with:
                bbox: [x1, y1, x2, y2]
                confidence: float
                class_id: int
                class_name: str
        """
        if not self.is_ready():
            return []

        detections = []
        try:
            results = self.model.predict(
                source=frame,
                conf=confidence_threshold,
                verbose=False,
            )
        except Exception:
            # A single failed inference on a bad frame shouldn't crash the app.
            return []

        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            try:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.class_names.get(cls_id, str(cls_id))
            except Exception:
                continue

            if allowed_classes and cls_name not in allowed_classes:
                continue

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls_id,
                "class_name": cls_name,
            })

        return detections
