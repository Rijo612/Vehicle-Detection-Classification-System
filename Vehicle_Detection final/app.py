import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time

from PIL import Image
from ultralytics import YOLO
import tensorflow as tf


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="VehicleVision AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(37,99,235,0.15), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(124,58,237,0.12), transparent 30%),
        #080b14;
    color: #f8fafc;
}

/* Main container */

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}


/* Header */

.hero {
    padding: 30px 35px;
    border-radius: 24px;
    background: linear-gradient(
        135deg,
        rgba(30,41,59,0.95),
        rgba(15,23,42,0.95)
    );
    border: 1px solid rgba(148,163,184,0.15);
    box-shadow: 0 20px 60px rgba(0,0,0,0.35);
    margin-bottom: 25px;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    margin: 0;
}

.hero-title span {
    color: #60a5fa;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 16px;
    margin-top: 8px;
}


/* Cards */

.card {
    padding: 22px;
    border-radius: 18px;
    background: rgba(15,23,42,0.85);
    border: 1px solid rgba(148,163,184,0.12);
    box-shadow: 0 10px 30px rgba(0,0,0,0.20);
}

.card-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 8px;
}

.card-description {
    color: #94a3b8;
    font-size: 14px;
}


/* Metrics */

.metric-card {
    padding: 18px;
    border-radius: 16px;
    background: rgba(30,41,59,0.75);
    border: 1px solid rgba(148,163,184,0.10);
}

.metric-value {
    font-size: 28px;
    font-weight: 800;
}

.metric-label {
    color: #94a3b8;
    font-size: 13px;
}


/* Sidebar */

section[data-testid="stSidebar"] {
    background: #090d18;
    border-right: 1px solid rgba(148,163,184,0.10);
}


/* Buttons */

.stButton > button {
    border-radius: 12px;
    font-weight: 600;
    border: 1px solid rgba(96,165,250,0.35);
    background: #2563eb;
    color: white;
}

.stButton > button:hover {
    background: #1d4ed8;
}


/* File uploader */

[data-testid="stFileUploader"] {
    background: rgba(15,23,42,0.7);
    border-radius: 16px;
}


/* Tabs */

button[data-baseweb="tab"] {
    font-size: 16px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    cnn_model = tf.keras.models.load_model(
        "vehicle_det_new.h5"
    )

    yolo_model = YOLO(
        "yolo11n.pt"
    )

    return cnn_model, yolo_model


with st.spinner("Loading AI models..."):
    cnn_model, yolo_model = load_models()


# =========================================================
# CLASS NAMES
# =========================================================

CLASS_NAMES = [
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "rickshaw",
    "truck"
]

ALLOWED_YOLO_CLASSES = [
    "car",
    "motorcycle",
    "bus",
    "truck",
    "bicycle"
]


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🚗 VehicleVision AI")

    st.markdown(
        "AI-powered vehicle detection and classification."
    )

    st.divider()

    st.markdown("### ⚙️ Detection Settings")

    confidence = st.slider(
        "YOLO Confidence",
        min_value=0.10,
        max_value=0.95,
        value=0.40,
        step=0.05
    )

    cnn_threshold = st.slider(
        "CNN Confidence",
        min_value=0.10,
        max_value=0.95,
        value=0.60,
        step=0.05
    )

    st.divider()

    st.markdown("### 🧠 AI Models")

    st.info(
        """
        **YOLO11n**

        Object detection

        **Custom CNN**

        Vehicle classification
        """
    )

    st.divider()

    st.caption("VehicleVision AI • Computer Vision Project")


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
Vehicle<span>Vision</span> AI 🚘
</div>

<div class="hero-subtitle">
Intelligent vehicle detection and classification using
YOLO + Deep Learning
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# TABS
# =========================================================

tab_image, tab_video = st.tabs(
    ["🖼️ Image Detection", "🎥 Video Detection"]
)


# =========================================================
# IMAGE DETECTION
# =========================================================

with tab_image:

    st.markdown(
        "### 🖼️ Upload an Image"
    )

    st.caption(
        "Upload a vehicle image and the AI will detect and classify the vehicle."
    )

    uploaded_image = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        key="image_uploader"
    )

    if uploaded_image:

        image = Image.open(
            uploaded_image
        ).convert("RGB")

        image_np = np.array(image)

        col1, col2 = st.columns(
            [1, 1],
            gap="large"
        )

        with col1:

            st.markdown(
                "#### Original Image"
            )

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.markdown(
                "#### Detection Result"
            )

            with st.spinner(
                "Analyzing image..."
            ):

                results = yolo_model(
                    image_np,
                    conf=confidence,
                    verbose=False
                )

                result = results[0]

                annotated = result.plot()

                annotated = cv2.cvtColor(
                    annotated,
                    cv2.COLOR_BGR2RGB
                )

            st.image(
                annotated,
                use_container_width=True
            )

        # ---------------------------------------------
        # Detection statistics
        # ---------------------------------------------

        detections = []

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            conf_score = float(
                box.conf[0]
            )

            name = yolo_model.names[
                class_id
            ]

            if name in ALLOWED_YOLO_CLASSES:

                detections.append(
                    (name, conf_score)
                )

        st.divider()

        st.markdown(
            "### 📊 Detection Summary"
        )

        total = len(detections)

        unique = len(
            set(
                d[0]
                for d in detections
            )
        )

        avg_conf = (
            np.mean(
                [d[1] for d in detections]
            )
            if detections
            else 0
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{total}</div>
                    <div class="metric-label">Vehicles Detected</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{unique}</div>
                    <div class="metric-label">Vehicle Types</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">
                        {avg_conf * 100:.1f}%
                    </div>
                    <div class="metric-label">Average Confidence</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        if detections:

            st.markdown(
                "#### 🚘 Detected Vehicles"
            )

            for name, score in detections:

                st.write(
                    f"**{name.title()}** — "
                    f"{score * 100:.1f}% confidence"
                )

    else:

        st.info(
            "👆 Upload an image to start vehicle detection."
        )

# =========================================================
# VIDEO DETECTION
# =========================================================

with tab_video:

    st.markdown("## 🎥 Video Detection")

    st.write(
        "Upload a video and detect vehicles frame-by-frame using YOLO."
    )

    uploaded_video = st.file_uploader(
        "📁 Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload"
    )

    if uploaded_video is not None:

        st.success(
            f"Video uploaded: {uploaded_video.name}"
        )

        # Save uploaded video
        video_bytes = uploaded_video.read()

        input_path = os.path.join(
            tempfile.gettempdir(),
            uploaded_video.name
        )

        with open(input_path, "wb") as f:
            f.write(video_bytes)

        st.markdown("### 🎬 Uploaded Video")

        # Display uploaded video
        st.video(
            video_bytes
        )

        st.divider()

        # Detection button
        if st.button(
            "🚀 Start Vehicle Detection",
            use_container_width=True
        ):

            cap = cv2.VideoCapture(input_path)

            if not cap.isOpened():

                st.error(
                    "❌ Could not open the uploaded video."
                )

                st.stop()

            fps = cap.get(
                cv2.CAP_PROP_FPS
            )

            width = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            height = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            total_frames = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            if fps <= 0:
                fps = 25

            # Output file
            output_path = os.path.join(
                tempfile.gettempdir(),
                "vehicle_detection_result.mp4"
            )

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

            progress_bar = st.progress(0)

            status_text = st.empty()

            preview = st.empty()

            frame_count = 0

            total_detections = 0

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                # YOLO detection
                results = yolo_model(
                    frame,
                    conf=confidence,
                    verbose=False
                )

                result = results[0]

                # Draw detections
                annotated_frame = result.plot()

                # Write frame
                out.write(
                    annotated_frame
                )

                # Count detections
                for box in result.boxes:

                    class_id = int(
                        box.cls[0]
                    )

                    class_name = yolo_model.names[
                        class_id
                    ]

                    if class_name in ALLOWED_YOLO_CLASSES:

                        total_detections += 1

                frame_count += 1

                # Progress
                if total_frames > 0:

                    progress = (
                        frame_count /
                        total_frames
                    )

                    progress_bar.progress(
                        min(progress, 1.0)
                    )

                status_text.write(
                    f"Processing frame "
                    f"{frame_count} / "
                    f"{total_frames}"
                )

                # Show preview
                if frame_count % 10 == 0:

                    preview_frame = cv2.cvtColor(
                        annotated_frame,
                        cv2.COLOR_BGR2RGB
                    )

                    preview.image(
                        preview_frame,
                        channels="RGB",
                        use_container_width=True
                    )

            cap.release()
            out.release()

            progress_bar.progress(1.0)

            status_text.success(
                "✅ Video processing completed!"
            )

            st.divider()

            st.markdown(
                "### 🎬 Detection Result"
            )

            # Read processed video
            with open(
                output_path,
                "rb"
            ) as f:

                processed_video = f.read()

            st.video(
                processed_video
            )

            st.success(
                f"🚗 Total vehicle detections: "
                f"{total_detections}"
            )

            st.download_button(
                label="⬇️ Download Processed Video",
                data=processed_video,
                file_name="vehicle_detection_result.mp4",
                mime="video/mp4",
                use_container_width=True
            )

    else:

        st.info(
            "👆 Please choose a video file above."
        )
# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;color:#64748b;padding:20px;">
        VehicleVision AI • YOLO + Custom CNN • Computer Vision
    </div>
    """,
    unsafe_allow_html=True
)