import streamlit as st
import cv2
import time
from collections import deque

st.set_page_config(
    page_title="Study Vision",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background: #0b0d10;
        color: #f4f4f5;
    }

    [data-testid="stSidebar"] {
        background: #111419;
        border-right: 1px solid #252a31;
    }

    .hero {
        padding: 10px 0 18px 0;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -1.5px;
        margin: 0;
    }

    .hero-subtitle {
        color: #9ca3af;
        font-size: 15px;
        margin-top: 5px;
    }

    .metric-card {
        background: #14181e;
        border: 1px solid #272d35;
        border-radius: 16px;
        padding: 18px;
        min-height: 120px;
    }

    .metric-label {
        color: #9ca3af;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 750;
        margin-top: 8px;
    }

    .status {
        display: inline-block;
        padding: 7px 13px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 14px;
        background: #17231b;
        color: #72e59a;
        border: 1px solid #274d35;
    }

    .section-title {
        font-size: 20px;
        font-weight: 750;
        margin: 24px 0 10px 0;
    }

    .tip {
        background: #14181e;
        border: 1px solid #272d35;
        border-radius: 14px;
        padding: 14px;
        color: #c7cbd1;
        font-size: 14px;
    }

    div[data-testid="stImage"] img {
        border-radius: 16px;
        border: 1px solid #272d35;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Session state
# -----------------------------
if "running" not in st.session_state:
    st.session_state.running = False

if "start_time" not in st.session_state:
    st.session_state.start_time = None

if "study_seconds" not in st.session_state:
    st.session_state.study_seconds = 0.0

if "away_seconds" not in st.session_state:
    st.session_state.away_seconds = 0.0

if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=60)

# -----------------------------
# Helpers
# -----------------------------
def format_time(seconds):
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def detect_activity(frame, previous_gray, threshold_value=25):
    """
    Beginner-friendly motion detection.

    Returns:
        active: bool
        processed_frame: frame with detection overlay
        current_gray: grayscale frame for next iteration
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if previous_gray is None:
        return False, frame, gray

    difference = cv2.absdiff(previous_gray, gray)
    _, thresh = cv2.threshold(difference, threshold_value, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    active = False

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 900:
            continue

        active = True
        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 255, 255),
            2,
        )

    return active, frame, gray


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## ⚙️ Study Vision")
    st.caption("Beginner OpenCV productivity tracker")

    st.markdown("---")

    camera_index = st.number_input(
        "Camera index",
        min_value=0,
        max_value=5,
        value=0,
        step=1,
    )

    sensitivity = st.slider(
        "Motion sensitivity",
        min_value=5,
        max_value=60,
        value=25,
        help="Lower values detect smaller movements.",
    )

    st.markdown("---")

    if st.button("▶ Start Session", use_container_width=True):
        st.session_state.running = True
        st.session_state.start_time = time.time()
        st.session_state.study_seconds = 0
        st.session_state.away_seconds = 0
        st.session_state.history.clear()

    if st.button("■ Stop Session", use_container_width=True):
        st.session_state.running = False

    if st.button("↻ Reset", use_container_width=True):
        st.session_state.running = False
        st.session_state.start_time = None
        st.session_state.study_seconds = 0
        st.session_state.away_seconds = 0
        st.session_state.history.clear()

    st.markdown("---")

    st.markdown(
        '<div class="tip">'
        "<b>How it works</b><br><br>"
        "Study Vision currently uses basic OpenCV motion detection. "
        "Later you can replace this with YOLO, face detection, pose estimation, "
        "and phone detection."
        "</div>",
        unsafe_allow_html=True,
    )

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="hero">
    <div class="hero-title">📚 Study Vision</div>
    <div class="hero-subtitle">
        A computer-vision powered study session monitor
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Metrics placeholders
# -----------------------------
metric_cols = st.columns(4)

status_placeholder = metric_cols[0].empty()
session_placeholder = metric_cols[1].empty()
focus_placeholder = metric_cols[2].empty()
activity_placeholder = metric_cols[3].empty()

def render_metrics(status, session_time, focus, activity):
    status_placeholder.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Status</div>
            <div style="margin-top:12px">
                <span class="status">{status}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    session_placeholder.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Session</div>
            <div class="metric-value">{session_time}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    focus_placeholder.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Focus Score</div>
            <div class="metric-value">{focus:.0f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    activity_placeholder.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Activity</div>
            <div class="metric-value">{activity}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Main content
# -----------------------------
camera_col, info_col = st.columns([2.2, 1])

with camera_col:
    st.markdown('<div class="section-title">📷 Live Study Monitor</div>',
                unsafe_allow_html=True)
    camera_placeholder = st.empty()

with info_col:
    st.markdown('<div class="section-title">📊 Session Analytics</div>',
                unsafe_allow_html=True)

    chart_placeholder = st.empty()

    st.markdown(
        '<div class="section-title">💡 Current Mode</div>',
        unsafe_allow_html=True,
    )

    mode_placeholder = st.empty()

# -----------------------------
# Initial state
# -----------------------------
if not st.session_state.running:
    render_metrics("READY", "00:00", 0, "—")

    camera_placeholder.info(
        "Click **▶ Start Session** in the sidebar to activate the webcam."
    )

    mode_placeholder.markdown(
        '<div class="tip">'
        "<b>V1 OpenCV Mode</b><br><br>"
        "The app detects movement inside the camera frame. "
        "This is intentionally simple so you can understand the computer-vision pipeline."
        "</div>",
        unsafe_allow_html=True,
    )

else:
    # -----------------------------
    # Camera loop
    # -----------------------------
    cap = cv2.VideoCapture(int(camera_index))

    if not cap.isOpened():
        st.error(
            "Could not open the webcam. Check camera permissions or try camera index 1."
        )
        st.session_state.running = False
        st.stop()

    previous_gray = None
    last_update = time.time()

    try:
        while st.session_state.running:
            ret, frame = cap.read()

            if not ret:
                st.error("Could not read a frame from the webcam.")
                break

            frame = cv2.flip(frame, 1)

            active, processed_frame, current_gray = detect_activity(
                frame,
                previous_gray,
                int(sensitivity),
            )

            now = time.time()
            elapsed = now - last_update
            last_update = now

            if active:
                st.session_state.study_seconds += elapsed
                status = "STUDYING"
                activity = "Active"
            else:
                st.session_state.away_seconds += elapsed
                status = "INACTIVE"
                activity = "Low"

            previous_gray = current_gray

            total = (
                st.session_state.study_seconds
                + st.session_state.away_seconds
            )

            focus = (
                st.session_state.study_seconds / total * 100
                if total > 0
                else 0
            )

            # Save timeline sample
            st.session_state.history.append({
                "Time": time.strftime("%H:%M:%S"),
                "Focus": focus,
            })

            # Overlay status on camera
            cv2.putText(
                processed_frame,
                status,
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                processed_frame,
                f"Focus: {focus:.0f}%",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # OpenCV uses BGR; Streamlit expects RGB.
            rgb_frame = cv2.cvtColor(
                processed_frame,
                cv2.COLOR_BGR2RGB,
            )

            camera_placeholder.image(
                rgb_frame,
                channels="RGB",
                use_container_width=True,
            )

            render_metrics(
                status,
                format_time(total),
                focus,
                activity,
            )

            if st.session_state.history:
                import pandas as pd

                history_df = pd.DataFrame(
                    list(st.session_state.history)
                ).set_index("Time")

                chart_placeholder.line_chart(
                    history_df,
                    y="Focus",
                    height=280,
                )

            mode_placeholder.markdown(
                '<div class="tip">'
                "<b>OpenCV Motion Detection</b><br><br>"
                "Movement detected → activity timer increases.<br>"
                "Little/no movement → inactive timer increases.<br><br>"
                "<b>Next upgrade:</b> YOLO object detection."
                "</div>",
                unsafe_allow_html=True,
            )

            # Small delay keeps the app responsive.
            time.sleep(0.03)

    finally:
        cap.release()