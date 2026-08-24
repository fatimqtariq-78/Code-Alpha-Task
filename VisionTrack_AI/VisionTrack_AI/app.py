"""
app.py
------
VisionTrack AI - Real-Time Object Detection & Tracking
CodeAlpha Artificial Intelligence Internship - Task 4

This file is the UI layer only. All computer-vision logic lives in
detector.py (YOLO), tracker.py (multi-object tracking), and
video_processor.py (capture lifecycle, drawing, counting, export).

Author: Fatima
"""

import os
import tempfile

import streamlit as st

from detector import ObjectDetector
from video_processor import VideoProcessor
from utils import FILTERABLE_CLASSES


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="VisionTrack AI",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_NAME = "yolov8n.pt"


# ============================================================
# CACHED MODEL LOADING (loaded exactly once per server process)
# ============================================================
@st.cache_resource(show_spinner="Loading YOLO model (first run downloads weights)...")
def get_detector(model_name: str):
    return ObjectDetector(model_name)


# ============================================================
# SESSION STATE
# ============================================================
def init_session_state():
    defaults = {
        "processor": None,
        "running": False,
        "paused": False,
        "status": "SYSTEM READY",
        "source_type": "Webcam",
        "webcam_index": 0,
        "video_file_path": None,
        "confidence": 0.40,
        "all_classes": True,
        "selected_classes": FILTERABLE_CLASSES.copy(),
        "tracking_enabled": True,
        "show_boxes": True,
        "show_labels": True,
        "show_conf": True,
        "show_ids": True,
        "line_enabled": False,
        "line_ratio": 0.5,
        "last_stats": {},
        "last_message": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if st.session_state.processor is None:
        st.session_state.processor = VideoProcessor(get_detector(MODEL_NAME))


init_session_state()
processor: VideoProcessor = st.session_state.processor


# ============================================================
# CUSTOM CSS - Dark charcoal / warm amber CV monitoring theme
# ============================================================
def load_css():
    st.markdown(
        """
        <style>
        :root {
            --bg-main: #16151a;
            --bg-panel: #1e1c22;
            --bg-panel-alt: #232026;
            --border-subtle: #33303a;
            --amber: #e5942e;
            --amber-dim: #a86a24;
            --text-primary: #ececec;
            --text-secondary: #9a97a3;
            --green: #7aa874;
            --red: #c96a5a;
        }

        .stApp {
            background: var(--bg-main);
        }
        html, body, [class*="css"] {
            font-family: 'Segoe UI', 'Inter', sans-serif;
            color: var(--text-primary);
        }
        /* NOTE: we deliberately do NOT hide Streamlit's header or the
           sidebar-collapse control here - doing so makes the sidebar
           unreachable once collapsed. We only hide the "Made with
           Streamlit" footer, which is purely cosmetic. */
        footer {visibility: hidden;}
        header[data-testid="stHeader"] {
            background: var(--bg-main);
            height: 2.6rem;
        }

        /* ---------- Header ---------- */
        .vt-header {
            background: var(--bg-panel);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 1.1rem 1.5rem;
            margin-bottom: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
        }
        .vt-title {
            font-size: 1.6rem;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: 0.3px;
            margin: 0;
        }
        .vt-title span { color: var(--amber); }
        .vt-subtitle {
            color: var(--text-secondary);
            font-size: 0.88rem;
            margin-top: 0.15rem;
        }
        .status-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            background: var(--bg-panel-alt);
            border: 1px solid var(--border-subtle);
            padding: 0.35rem 0.85rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.6px;
            color: var(--text-secondary);
        }
        .status-dot { width: 8px; height: 8px; border-radius: 50%; }
        .dot-ready { background: var(--text-secondary); }
        .dot-active { background: var(--amber); box-shadow: 0 0 6px var(--amber); }
        .dot-error { background: var(--red); }

        /* ---------- Panels ---------- */
        .vt-panel {
            background: var(--bg-panel);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.9rem;
        }
        .vt-panel-title {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
            color: var(--amber);
            margin-bottom: 0.6rem;
        }

        /* ---------- Feed frame ---------- */
        .feed-frame {
            background: #0c0b0e;
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 0.4rem;
        }
        .feed-empty {
            background: #0c0b0e;
            border: 1px dashed var(--border-subtle);
            border-radius: 10px;
            padding: 4rem 1rem;
            text-align: center;
            color: var(--text-secondary);
        }

        /* ---------- Stat cards ---------- */
        .stat-card {
            background: var(--bg-panel-alt);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 0.7rem 0.6rem;
            text-align: center;
        }
        .stat-value {
            font-size: 1.35rem;
            font-weight: 700;
            color: var(--amber);
        }
        .stat-label {
            font-size: 0.68rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 0.15rem;
        }

        /* ---------- Buttons ---------- */
        div.stButton > button {
            background: var(--bg-panel-alt);
            color: var(--text-primary);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 0.45rem 0.9rem;
            font-weight: 600;
            font-size: 0.85rem;
            transition: all 0.15s ease-in-out;
        }
        div.stButton > button:hover {
            border-color: var(--amber);
            color: var(--amber);
        }
        div.stButton > button:active { transform: scale(0.98); }

        .app-footer {
            text-align: center;
            padding: 1.2rem 0 0.5rem 0;
            color: var(--text-secondary);
            font-size: 0.78rem;
            border-top: 1px solid var(--border-subtle);
            margin-top: 1.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HEADER
# ============================================================
def render_header():
    status = st.session_state.status
    dot_class = "dot-active" if st.session_state.running else "dot-ready"
    if "ERROR" in status:
        dot_class = "dot-error"

    st.markdown(
        f"""
        <div class="vt-header">
            <div>
                <div class="vt-title">◎ Vision<span>Track</span> AI</div>
                <div class="vt-subtitle">Real-Time Object Detection &amp; Tracking</div>
            </div>
            <div class="status-chip"><span class="status-dot {dot_class}"></span> {status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR - CONTROL PANEL
# ============================================================
def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style="margin-bottom: 0.3rem;">
                <div style="font-size:1.15rem; font-weight:800; letter-spacing:0.3px;">
                    ◎ VISIONTRACK AI
                </div>
                <div style="font-size:0.72rem; font-weight:700; letter-spacing:1.2px;
                            color:#e5942e; text-transform:uppercase; margin-top:0.1rem;">
                    Control Panel
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("1️⃣ Choose source → 2️⃣ Configure settings → 3️⃣ Press Start Detection")
        st.markdown("---")

        # ---------- Input Source ----------
        st.markdown('<div class="vt-panel-title">Input Source</div>', unsafe_allow_html=True)
        st.radio(
            "Source", ["Webcam", "Video File"],
            label_visibility="collapsed", key="source_type",
        )

        if st.session_state.source_type == "Webcam":
            st.session_state.webcam_index = st.number_input(
                "Webcam index", min_value=0, max_value=5,
                value=st.session_state.webcam_index, step=1,
            )
        else:
            uploaded = st.file_uploader(
                "Upload a video file", type=["mp4", "avi", "mov"], key="video_uploader"
            )
            if uploaded is not None:
                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, uploaded.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                st.session_state.video_file_path = temp_path
                st.caption(f"Loaded: {uploaded.name}")

        st.markdown("---")

        # ---------- Start / Stop / Pause controls ----------
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶ Start Detection", use_container_width=True, disabled=st.session_state.running):
                _start_detection()
        with col2:
            if st.button("■ Stop", use_container_width=True, disabled=not st.session_state.running):
                _stop_detection()

        col3, col4 = st.columns(2)
        with col3:
            pause_label = "▶ Resume" if st.session_state.paused else "⏸ Pause"
            if st.button(pause_label, use_container_width=True, disabled=not st.session_state.running):
                st.session_state.paused = not st.session_state.paused
                st.rerun()
        with col4:
            if st.button("↺ Reset", use_container_width=True):
                processor.reset_statistics()
                st.session_state.last_message = "Statistics reset."
                st.rerun()

        st.markdown("---")

        # ---------- Detection settings ----------
        st.markdown('<div class="vt-panel-title">Detection Settings</div>', unsafe_allow_html=True)
        st.caption(f"Model: **{MODEL_NAME}** (CPU)")
        st.session_state.confidence = st.slider(
            "Confidence Threshold", min_value=0.20, max_value=0.90,
            value=st.session_state.confidence, step=0.05,
        )

        st.session_state.all_classes = st.checkbox(
            "All Classes", value=st.session_state.all_classes
        )
        if not st.session_state.all_classes:
            st.session_state.selected_classes = st.multiselect(
                "Object Classes", options=FILTERABLE_CLASSES,
                default=st.session_state.selected_classes,
            )

        st.markdown("---")

        # ---------- Tracking ----------
        st.markdown('<div class="vt-panel-title">Tracking</div>', unsafe_allow_html=True)
        st.session_state.tracking_enabled = st.checkbox(
            "Enable Tracking", value=st.session_state.tracking_enabled
        )
        st.session_state.show_ids = st.checkbox(
            "Show Tracking IDs", value=st.session_state.show_ids,
            disabled=not st.session_state.tracking_enabled,
        )

        st.session_state.line_enabled = st.checkbox(
            "Enable Line-Crossing Counter", value=st.session_state.line_enabled
        )
        if st.session_state.line_enabled:
            st.session_state.line_ratio = st.slider(
                "Line Position (vertical %)", 0.1, 0.9, st.session_state.line_ratio, 0.05
            )
        processor.line_enabled = st.session_state.line_enabled
        processor.line_y_ratio = st.session_state.line_ratio

        st.markdown("---")

        # ---------- Display ----------
        st.markdown('<div class="vt-panel-title">Display</div>', unsafe_allow_html=True)
        st.session_state.show_boxes = st.checkbox("Show Bounding Boxes", value=st.session_state.show_boxes)
        st.session_state.show_conf = st.checkbox("Show Confidence", value=st.session_state.show_conf)
        st.session_state.show_labels = st.checkbox("Show Labels", value=st.session_state.show_labels)

        st.markdown("---")

        # ---------- Capture / Recording ----------
        st.markdown('<div class="vt-panel-title">Capture &amp; Export</div>', unsafe_allow_html=True)
        if st.button("📷 Capture Frame", use_container_width=True, disabled=not st.session_state.running):
            path = processor.capture_screenshot()
            if path:
                st.session_state.last_message = f"Screenshot saved: {path}"
            else:
                st.session_state.last_message = "No frame available to capture yet."
            st.rerun()

        if processor.is_recording():
            if st.button("⏹ Stop Recording", use_container_width=True):
                path = processor.stop_recording()
                st.session_state.last_message = f"Processed video saved: {path}"
                st.rerun()
        else:
            if st.button("🔴 Start Recording", use_container_width=True, disabled=not st.session_state.running):
                path = processor.start_recording()
                if path:
                    st.session_state.last_message = "Recording started."
                else:
                    st.session_state.last_message = "Could not start recording."
                st.rerun()

        if st.session_state.last_message:
            st.info(st.session_state.last_message)

        st.markdown("---")
        with st.expander("How VisionTrack AI Works"):
            st.caption(
                "Camera / Video → YOLO → Detection → SORT Tracker → IDs → Visualization\n\n"
                "1. A video frame is captured with OpenCV.\n"
                "2. YOLOv8 analyzes the frame and detects objects.\n"
                "3. Low-confidence detections are filtered out.\n"
                "4. Valid detections are passed to a SORT-style tracker "
                "(Kalman filter + Hungarian assignment + IOU matching).\n"
                "5. The tracker assigns persistent IDs across frames.\n"
                "6. Bounding boxes, labels, and IDs are drawn.\n"
                "7. Statistics update in real time."
            )


def _start_detection():
    if st.session_state.source_type == "Webcam":
        ok = processor.open_webcam(st.session_state.webcam_index)
    else:
        if not st.session_state.video_file_path:
            st.session_state.last_message = "Please upload a video file first."
            st.rerun()
            return
        ok = processor.open_video_file(st.session_state.video_file_path)

    if ok:
        processor.reset_statistics()
        st.session_state.running = True
        st.session_state.paused = False
        st.session_state.status = "TRACKING ACTIVE" if st.session_state.tracking_enabled else "PROCESSING"
        st.session_state.last_message = None
    else:
        st.session_state.status = "ERROR"
        st.session_state.last_message = processor.open_error
    st.rerun()


def _stop_detection():
    processor.release()
    st.session_state.running = False
    st.session_state.paused = False
    st.session_state.status = "STOPPED"
    st.rerun()


# ============================================================
# STATS PANEL
# ============================================================
def stat_card(label, value):
    return f"""<div class="stat-card"><div class="stat-value">{value}</div><div class="stat-label">{label}</div></div>"""


def render_stats(stats: dict):
    cols = st.columns(5)
    cols[0].markdown(stat_card("FPS", stats.get("fps", 0.0)), unsafe_allow_html=True)
    cols[1].markdown(stat_card("Current Objects", stats.get("current_objects", 0)), unsafe_allow_html=True)
    cols[2].markdown(stat_card("Unique Objects", stats.get("unique_objects", 0)), unsafe_allow_html=True)
    cols[3].markdown(stat_card("People", stats.get("people_current", 0)), unsafe_allow_html=True)
    cols[4].markdown(stat_card("Vehicles", stats.get("vehicles_current", 0)), unsafe_allow_html=True)

    if st.session_state.line_enabled:
        st.caption(f"Line Crossing — Entered: **{stats.get('entered', 0)}**  |  Exited: **{stats.get('exited', 0)}**")

    with st.expander("Detailed counts (current frame vs. total unique)"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Currently Detected**")
            counts = stats.get("current_counts", {})
            if counts:
                for k, v in counts.items():
                    st.caption(f"{k.title()}: {v}")
            else:
                st.caption("No objects currently detected.")
        with c2:
            st.markdown("**Total Unique Objects Seen**")
            unique = stats.get("unique_counts", {})
            if unique:
                for k, v in unique.items():
                    st.caption(f"{k.title()}: {v}")
            else:
                st.caption("No objects tracked yet.")


# ============================================================
# MAIN CONTENT - LIVE FEED
# ============================================================
# This section runs as an isolated Streamlit "fragment": Streamlit
# reruns ONLY this function on a timer (run_every), instead of the
# manual whole-script st.rerun() loop used previously. This keeps the
# sidebar, header, and every control panel widget fully responsive and
# interactive at all times, and keeps CPU usage bounded (the fragment
# only ticks a fixed number of times per second rather than looping as
# fast as possible).
FRAME_TICK_SECONDS = 0.05  # ~20 processing attempts/sec ceiling


@st.fragment(run_every=FRAME_TICK_SECONDS)
def live_feed_fragment():
    st.markdown('<div class="vt-panel-title">Live Detection Feed</div>', unsafe_allow_html=True)
    feed_placeholder = st.empty()
    stats_placeholder = st.container()

    if not st.session_state.running:
        with feed_placeholder.container():
            st.markdown(
                '<div class="feed-empty"><b>NO VIDEO SOURCE</b><br>'
                'Choose Webcam or upload a video from the Control Panel, '
                'then press <b>Start Detection</b>.</div>',
                unsafe_allow_html=True,
            )
        with stats_placeholder:
            render_stats(st.session_state.last_stats or {})
        return

    detector = get_detector(MODEL_NAME)
    if not detector.is_ready():
        st.error(f"⚠️ {detector.load_error}")
        st.session_state.running = False
        st.session_state.status = "ERROR"
        st.rerun()
        return

    if st.session_state.paused:
        if processor.last_annotated_frame is not None:
            feed_placeholder.image(
                processor.last_annotated_frame, channels="BGR", use_container_width=True
            )
        with stats_placeholder:
            render_stats(st.session_state.last_stats or {})
        st.caption("⏸ Paused")
        return

    allowed = None if st.session_state.all_classes else st.session_state.selected_classes
    success, frame, stats = processor.process_next_frame(
        confidence_threshold=st.session_state.confidence,
        allowed_classes=allowed,
        show_boxes=st.session_state.show_boxes,
        show_labels=st.session_state.show_labels,
        show_conf=st.session_state.show_conf,
        show_ids=st.session_state.show_ids,
        tracking_enabled=st.session_state.tracking_enabled,
    )

    if not success:
        # Video ended, or webcam/read failure. A full rerun (not just a
        # fragment rerun) is used here on purpose, so the header status
        # chip and the sidebar's Start/Stop button states update too.
        st.session_state.running = False
        st.session_state.status = "STOPPED"
        if stats.get("error"):
            st.session_state.last_message = f"Processing stopped: {stats['error']}"
        else:
            st.session_state.last_message = "Video finished or source disconnected."
        processor.release()
        st.rerun()
        return

    feed_placeholder.image(frame, channels="BGR", use_container_width=True)
    st.session_state.last_stats = stats
    with stats_placeholder:
        render_stats(stats)


def render_footer():
    st.markdown(
        """
        <div class="app-footer">
            VisionTrack AI — Built by Fatima for the CodeAlpha AI Internship (Task 4) &nbsp;|&nbsp;
            YOLOv8 (Ultralytics) + SORT-Style Tracker (Kalman Filter + Hungarian Assignment) &nbsp;|&nbsp; Runs fully on CPU
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ENTRY POINT
# ============================================================
def main():
    load_css()
    render_sidebar()
    render_header()
    live_feed_fragment()
    render_footer()


if __name__ == "__main__":
    main()
