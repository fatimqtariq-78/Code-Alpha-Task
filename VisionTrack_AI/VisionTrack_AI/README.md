# ◎ VisionTrack AI — Real-Time Object Detection &amp; Tracking

> **CodeAlpha Artificial Intelligence Internship — Task 4: Object Detection and Tracking**

A real-time computer vision application that detects and tracks objects from
a webcam or video file using **YOLOv8** and a custom **IOU-based multi-object
tracker**, wrapped in a dark, monitoring-style Streamlit interface.

---

## 1. Project Overview

VisionTrack AI captures live video (webcam or an uploaded file), runs every
frame through a pretrained YOLOv8 model to detect objects, filters detections
by a user-adjustable confidence threshold, and passes them to a lightweight
tracker that assigns each object a **persistent tracking ID** across frames.
The result is displayed live with bounding boxes, labels, confidence scores,
and tracking IDs, alongside real-time statistics, object counting, optional
line-crossing counting, screenshot capture, and processed-video export.

---

## 2. Problem Statement

Manually monitoring video feeds — for counting people, tracking vehicles, or
observing activity in a space — is slow and error-prone. Automated object
detection and tracking makes it possible to identify *what* is in a scene and
*follow it over time*, which is the foundation of real-world applications
like surveillance analytics, retail footfall counting, and traffic
monitoring.

---

## 3. Objectives

- Detect multiple object classes in real time using a genuine pretrained
  deep learning model (not hard-coded or simulated detections).
- Track each object across frames with a persistent ID, distinguishing
  detection (finding objects in a single frame) from tracking (following
  the same object over multiple frames).
- Provide a configurable, professional monitoring-style interface with
  live statistics, object counting, and export capabilities.
- Run entirely on a CPU, with no GPU or paid API required.

---

## 4. Features

- 🎥 Live webcam **and** uploaded video file input
- 🧠 Real YOLOv8 object detection (80 COCO classes, CPU-only)
- 🆔 Persistent tracking IDs via a custom IOU + Hungarian-assignment tracker
- 🎚️ Adjustable confidence threshold (0.20–0.90)
- 🗂️ Object class filter (show only selected classes, or all)
- 📊 Real-time statistics: FPS, current objects, unique objects, people,
  vehicles, active model
- 🔢 Object counting — both **currently detected** and **total unique seen**,
  computed from tracking IDs (not naive per-frame counting)
- 🚧 Optional line-crossing counter (Entered / Exited)
- 📷 One-click frame screenshot capture (`outputs/screenshots/`)
- 🎬 Processed video export with all annotations burned in (`outputs/processed_videos/`)
- ▶️⏸️■↺ Functional Start / Pause-Resume / Stop / Reset controls
- 🎨 Dark charcoal + warm amber monitoring-style UI (no blue/white AI theme)
- 🛡️ Graceful error handling for webcam/file/model failures

---

## 5. How It Works

```
Video Frame
     ↓
YOLOv8 Detection            (detector.py)
     ↓
Confidence Filtering
     ↓
IOU Tracker                 (tracker.py)
     ↓
Persistent Tracking IDs
     ↓
Visualization                (drawing boxes, labels, IDs)
     ↓
Statistics                    (counts, FPS, line crossings)
```

---

## 6. Object Detection

**YOLO (You Only Look Once)** is a single-pass convolutional neural network
that predicts bounding boxes and class probabilities for an entire image in
one forward pass, making it fast enough for real-time video. This project
uses **YOLOv8n** (the "nano" variant from Ultralytics) — the smallest and
fastest model in the YOLOv8 family — specifically so the application runs
smoothly on a normal laptop CPU without needing a GPU. The model is loaded
**once** when the app starts (`detector.py`) and reused for every frame,
rather than being reloaded repeatedly.

## 7. Object Tracking

Detection alone only tells you what's in a *single* frame — it has no memory
of what happened in the previous frame. **Tracking** solves this by linking
detections across frames so the same physical object keeps the same ID over
time, even as it moves.

**Design choice:** this project implements a genuine **SORT-style tracker**
(`tracker.py`) — the same core algorithm as the original "Simple Online and
Realtime Tracking" paper: a **Kalman filter** predicts each object's next
position using a 7-state constant-velocity model (`[cx, cy, scale, aspect
ratio, vx, vy, v_scale]`), and predicted positions are matched to new YOLO
detections each frame using **IOU (Intersection over Union) cost solved
optimally with the Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`).

The reference SORT implementation uses the `filterpy` library for its Kalman
filter; here the Kalman filter's predict/update equations are implemented
directly with NumPy instead, to avoid an extra native dependency that can
cause version conflicts on some student machines. The math is the same
textbook Kalman filter — this is not a simplified stand-in. Each track:
predicts its next position every frame, gets corrected by a matched
detection when one is found, survives brief occlusion by coasting on its
predicted position for up to `max_age` frames, and is retired if it stays
unmatched for too long. New tracks are created for detections that don't
match any existing track.

## 8. Object Counting

- **Currently Detected**: a live count per class, computed directly from the
  objects visible (matched by the tracker) in the current frame.
- **Total Unique Objects Seen**: computed from the *set of tracking IDs* that
  have ever been confirmed during the session — each tracking ID is only
  added to this count the first time it appears, so an object that stays in
  frame for 200 frames is still counted once, not 200 times.

---

## 9. Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3.9+** | Core language |
| **Streamlit** | Web UI / control panel |
| **OpenCV** | Video/webcam capture, drawing, video export |
| **Ultralytics YOLOv8** | Pretrained object detection model |
| **NumPy** | Numerical operations |
| **SciPy** | Hungarian algorithm for optimal detection-to-track assignment |

No paid APIs and no GPU are required.

---

## 10. Project Structure

```
VisionTrack_AI/
│
├── app.py                    # Streamlit UI - control panel, live feed, stats
├── detector.py                 # YOLOv8 wrapper (loads model once)
├── tracker.py                    # Lightweight IOU + Hungarian multi-object tracker
├── video_processor.py              # Capture lifecycle, drawing, counting, export
├── utils.py                          # Colors, class list, FPS counter, drawing helpers
├── requirements.txt
├── README.md
├── .gitignore
│
├── assets/
├── sample_videos/               # Drop test videos here (or upload via the UI)
├── outputs/
│   ├── screenshots/                # Captured frames land here
│   └── processed_videos/             # Exported annotated videos land here
└── screenshots/                        # App screenshots for submission
```

---

## 11. Installation

```bash
# 1. Navigate into the project folder
cd VisionTrack_AI

# 2. (Recommended) create a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

The first time you run detection, Ultralytics will automatically download
the YOLOv8n weights (~6 MB) — this needs an internet connection just once;
after that, the weights are cached locally.

---

## 12. Running the Project

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`. In the sidebar
**Control Panel**, choose **Webcam** or **Video File**, adjust settings if
needed, and press **▶ Start Detection**.

> The live video feed runs as an isolated Streamlit *fragment*
> (`st.fragment(run_every=...)`, requires Streamlit 1.37+, pinned in
> `requirements.txt`) that refreshes on its own timer. This keeps the
> sidebar and every control fully responsive while detection is running,
> instead of the whole page reloading on every single video frame.

---

## 13. Demo Instructions

1. Select **Webcam** as the input source (or upload a short video).
2. Press **▶ Start** — the status chip should change to *TRACKING ACTIVE*.
3. Move within frame — watch the bounding box, class label, tracking ID, and
   confidence percentage update live.
4. Open the **"Detailed counts"** expander to see current vs. total unique
   counts.
5. Try **📷 Capture Frame** and check `outputs/screenshots/`.
6. Try **🔴 Start Recording**, let it run a few seconds, then **⏹ Stop
   Recording**, and check `outputs/processed_videos/`.
7. Press **⏸ Pause** / **▶ Resume** to confirm the feed actually freezes and
   resumes.
8. Press **■ Stop** to end the session cleanly.

---

## 14. Testing Checklist

| # | Test | Expected Result |
|---|---|---|
| 1 | Webcam with one person | A single bounding box with a stable ID appears and follows the person |
| 2 | Webcam with multiple people | Each person gets a separate, stable ID |
| 3 | Video containing cars | Cars are detected and labeled correctly; "Vehicles" stat updates |
| 4 | Multiple object classes at once | Each class is labeled with its own color and name |
| 5 | Objects moving across the frame | Tracking ID persists as the object moves (does not reset each frame) |
| 6 | Low-confidence objects | Raising the confidence slider removes weak detections from view |
| 7 | Unknown/invalid video file | A clear error message appears; the app does not crash |
| 8 | Webcam unavailable (in use elsewhere) | A friendly "unable to access webcam" message appears |
| 9 | Screenshot capture | A `.png` file appears in `outputs/screenshots/` and a confirmation message shows |
| 10 | Processed video export | An `.mp4` file with visible annotations appears in `outputs/processed_videos/` |

---

## 15. Demo Scenarios for the Evaluator

**Scenario 1 — Basic live detection**
Do: Start webcam detection with one person in frame.
Shows: real bounding box + class + confidence updating live.
Say: "This is a live YOLOv8 model running on my CPU, not a pre-recorded demo."

**Scenario 2 — Multi-object tracking with persistent IDs**
Do: Have two people walk into frame and move around/cross paths.
Shows: each person keeps their own ID even after moving.
Say: "Detection alone doesn't know these are the same people frame-to-frame —
the tracker is what assigns and keeps a consistent ID for each object."

**Scenario 3 — Confidence threshold in action**
Do: Lower the threshold to 0.2, then raise it to 0.8 live.
Shows: more/fewer boxes appearing as the threshold changes.
Say: "This threshold controls how confident the model needs to be before we
trust and display a detection."

**Scenario 4 — Object counting (current vs. unique)**
Do: Open the "Detailed counts" expander while people walk in and out of frame.
Shows: "Currently Detected" fluctuates, "Total Unique Objects Seen" only grows.
Say: "Unique counting uses tracking IDs so the same person isn't counted
twice just because they're visible for many frames."

**Scenario 5 — Screenshot and video export**
Do: Capture a frame, then record 5–10 seconds of video.
Shows: files actually appear in `outputs/`.
Say: "These aren't fake buttons — the app is writing real files to disk using
OpenCV's image and video writers."

---

## 16. Limitations

- Detection accuracy depends on lighting, camera quality, and distance from
  the camera.
- Heavy occlusion (objects blocking each other) can cause an ID switch or a
  brief loss of tracking.
- Very fast motion between frames can reduce IOU overlap and occasionally
  create a new ID instead of continuing the old one.
- Running entirely on CPU limits achievable FPS compared to a GPU setup,
  especially with larger YOLO variants.
- The Streamlit live-feed pattern used here (frame-by-frame rerun) is simple
  and reliable but not as optimized as a dedicated video-streaming server.

---

## 17. Future Improvements

- GPU acceleration for higher FPS with larger, more accurate YOLO models
- Upgrade the tracker to a full Kalman-filter-based SORT or DeepSORT with
  appearance embeddings for stronger re-identification after occlusion
- Multi-camera support with a unified dashboard
- Zone-based monitoring (alerts when objects enter/exit defined regions)
- Heatmaps of object movement over time
- Crowd density estimation and analytics dashboards

---

## 18. CodeAlpha Task Mapping

| CodeAlpha Requirement | Implementation |
|---|---|
| Real-time video input (webcam/video file, OpenCV) | `video_processor.py` — `open_webcam()`, `open_video_file()` |
| Pre-trained detection model (YOLO/Faster R-CNN) | `detector.py` — YOLOv8n via Ultralytics |
| Process each frame to detect objects | `video_processor.py` — `process_next_frame()` |
| Draw bounding boxes around detected objects | `utils.py` — `draw_detection()` |
| Object tracking (SORT/Deep SORT or equivalent) | `tracker.py` — custom IOU + Hungarian-assignment tracker |
| Display object labels and tracking IDs in real time | `app.py` — live feed with labels/IDs rendered every frame |

---

## 19. GitHub Setup

```bash
cd VisionTrack_AI
git init
git add .
git commit -m "VisionTrack AI - CodeAlpha Task 4: real-time object detection and tracking"
git branch -M main
git remote add origin <your-repository-url>
git push -u origin main
```

The included `.gitignore` keeps the repository lightweight by excluding
downloaded model weights, generated screenshots/videos, and virtual
environment files, while keeping the folder structure intact via
`.gitkeep` files.

---

## 20. Suggested Screenshots for Submission

- Empty state ("No active feed") showing the control panel layout
- Live detection with a single person, showing box + ID + confidence
- Live detection with multiple object classes at once (e.g. person + laptop + bottle)
- The "Detailed counts" expander open, showing current vs. unique counts
- Line-crossing counter enabled with Entered/Exited values visible
- A saved screenshot file shown in `outputs/screenshots/`
- The confidence threshold slider at a low vs. high value (two screenshots)

---

## 21. Presentation Explanation (for the CodeAlpha Evaluator)

"This project, VisionTrack AI, solves the problem of automatically finding
and following objects in a live video feed. I used YOLOv8 for detection
because it's fast enough to run in real time on a normal CPU while still
being accurate — it looks at the whole frame once and predicts all the
objects and their positions in a single pass, rather than scanning the image
region by region like older methods.

Detection alone only tells you what's in one frame, so I added a tracking
layer that gives every object a persistent ID and follows it across frames.
I implemented my own lightweight tracker based on the same core idea as
SORT — matching predicted object positions to new detections using
Intersection-over-Union and the Hungarian algorithm — but without the
Kalman filter dependency, to keep it reliable across different setups.

The confidence threshold lets the user control how strict the model is
before it trusts a detection enough to show and track it. Object counting
uses the tracking IDs rather than raw per-frame detections, so the same
person isn't counted again every single frame.

The main challenges were keeping the app responsive while continuously
processing video inside Streamlit's rerun-based execution model, and making
sure tracking IDs stayed stable through brief occlusion. Going forward, this
could be extended with GPU acceleration, a full Kalman-filter tracker for
better handling of occlusion, and zone-based analytics."

---

## 22. Author

**Fatima**
Bachelor of Computer Science Student, Capital University of Science and Technology (CUST)
Built for the **CodeAlpha Artificial Intelligence Internship — Task 4**
