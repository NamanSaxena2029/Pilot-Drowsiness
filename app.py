"""
PILOT DROWSINESS DETECTION — STREAMLIT APPLICATION
====================================================
Pages:
  1. Live Detection  — Real-time camera feed with detection overlay
  2. Model Details   — Architecture, training config, performance metrics
  3. Dataset Info    — Dataset provenance, pipeline, known issues & fixes

Run:
  streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import time

# Page config must be the first Streamlit call
st.set_page_config(
    page_title = "Pilot Drowsiness System",
    page_icon  = "✈️",
    layout     = "wide"
)

# ==============================================================
# SIDEBAR NAVIGATION
# ==============================================================
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
    width="stretch"
)
st.sidebar.title("✈️ Pilot Drowsiness")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🎥 Live Detection", "📊 Model Details", "📁 Dataset Info"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.caption("CNN-LSTM + MediaPipe | NTHU Dataset | v2.0")


# ==============================================================
# PAGE 1 — LIVE DETECTION
# ==============================================================
if page == "🎥 Live Detection":

    st.title("✈️ Real-Time Pilot Drowsiness Monitor")
    st.markdown(
        "Monitors up to **3 faces** simultaneously using "
        "**CNN-LSTM** (primary) + **MediaPipe geometry** (confirmation). "
        "Alerts fire only after a sustained signal — single blinks and "
        "momentary head turns are intentionally ignored."
    )
    st.markdown("---")

    # Layout: wide video feed | narrow controls + stats
    col_video, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        st.subheader("⚙️ Settings")

        drowsy_time = st.slider(
            "Drowsy alert threshold (seconds)",
            min_value=2.0, max_value=8.0, value=4.0, step=0.5,
            help="How long a drowsy signal must be sustained before an alert fires."
        )
        inatt_time = st.slider(
            "Inattentive alert threshold (seconds)",
            min_value=8.0, max_value=20.0, value=12.0, step=1.0,
            help="How long gaze/head deviation must persist before NOT ATTENTIVE fires."
        )
        max_faces = st.selectbox(
            "Maximum faces to track",
            options=[1, 2, 3], index=2,
            help="Reduce to 1 for single-pilot use (faster inference)."
        )
        cnn_thresh = st.slider(
            "CNN probability threshold",
            min_value=0.40, max_value=0.85, value=0.60, step=0.05,
            help="Drowsy probability above this threshold triggers the drowsy timer."
        )

        st.markdown("---")

        col_start, col_stop = st.columns(2)
        start_btn = col_start.button("▶ Start", use_container_width=True, type="primary")
        stop_btn  = col_stop.button("⏹ Stop",  use_container_width=True)

        st.markdown("---")
        st.subheader("📈 Live Stats")

        metric_faces  = st.empty()
        metric_drowsy = st.empty()
        metric_inatt  = st.empty()
        metric_forced = st.empty()

        st.markdown("---")
        st.subheader("🟢 Face Status")
        status_box = st.empty()

    with col_video:
        frame_placeholder = st.empty()
        info_placeholder  = st.empty()

    # Session state management
    if "cam_running" not in st.session_state:
        st.session_state.cam_running = False

    if start_btn:
        st.session_state.cam_running = True
    if stop_btn:
        st.session_state.cam_running = False

    # ---- Detection loop ----
    if st.session_state.cam_running:
        from drowsy_detection import DrowsinessDetector

        detector = DrowsinessDetector(
            model_path      = "best_model_v2.pth",
            landmarker_path = "face_landmarker.task",
            max_faces       = max_faces
        )
        detector.DROWSY_TIME_SEC  = drowsy_time
        detector.INATTENTIVE_SEC  = inatt_time
        detector.DROWSY_CNN_THRESH = cnn_thresh

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Camera not found. Please check your camera connection.")
            st.session_state.cam_running = False
        else:
            info_placeholder.info(
                "Camera active. Press **Stop** to end session."
            )

            while st.session_state.cam_running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Frame capture failed — retrying...")
                    time.sleep(0.1)
                    continue

                output_frame, face_states = detector.process_frame(frame)

                # Display frame (BGR -> RGB for Streamlit)
                rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb, channels="RGB", width="stretch")

                # Update metrics
                n_total  = len(face_states)
                n_drowsy = sum(1 for s in face_states.values() if s.status == "DROWSY")
                n_inatt  = sum(1 for s in face_states.values() if s.status == "NOT ATTENTIVE")
                n_forced = sum(1 for s in face_states.values() if s.status == "EYES FORCED SHUT")

                metric_faces.metric("Faces Detected", n_total)
                metric_drowsy.metric("Drowsy", n_drowsy,
                                     delta="⚠️" if n_drowsy else None,
                                     delta_color="inverse")
                metric_inatt.metric("Not Attentive", n_inatt)
                metric_forced.metric("Eyes Forced Shut", n_forced)

                # Per-face status badges
                badges = []
                for fid, state in face_states.items():
                    icon = {
                        "DROWSY"           : "🔴",
                        "NOT ATTENTIVE"    : "🟠",
                        "EYES FORCED SHUT" : "🟡",
                        "ACTIVE"           : "🟢",
                    }.get(state.status, "⚪")
                    badges.append(
                        f"{icon} **Face {fid + 1}**: {state.status} "
                        f"&nbsp;&nbsp; CNN: `{state.cnn_prob:.2f}`"
                    )

                status_box.markdown("\n\n".join(badges) if badges else "_No faces detected_")

            cap.release()
            detector.release()
            info_placeholder.success("Camera stopped.")


# ==============================================================
# PAGE 2 — MODEL DETAILS
# ==============================================================
elif page == "📊 Model Details":

    st.title("📊 Model Architecture & Performance")
    st.markdown(
        "This system uses a **CNN-LSTM** architecture where MobileNetV2 "
        "extracts spatial features per frame and an LSTM captures temporal "
        "patterns across a 3-frame sequence. The CNN is the primary decision "
        "maker; geometric signals (EAR, head pose, gaze) serve as confirmation "
        "and edge-case guards only."
    )
    st.markdown("---")

    # ---- Architecture ----
    st.subheader("🧠 Architecture")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**CNN Backbone**")
        st.table({
            "Component"       : ["Backbone", "Pretrained On", "Frozen Layers",
                                  "Trainable Layers", "Feature Output"],
            "Details"         : ["MobileNetV2", "ImageNet (ILSVRC)",
                                  "First 14 feature blocks",
                                  "Last 4 feature blocks + classifier replaced",
                                  "1280-dim vector per frame"]
        })

    with col2:
        st.markdown("**LSTM + Classifier**")
        st.table({
            "Component"  : ["Input", "LSTM Hidden Size", "LSTM Layers",
                             "Output Classes", "Final Activation"],
            "Details"    : ["3 × 1280 = 3840 (sequence)",
                             "128 units", "1",
                             "2  (Not Drowsy / Drowsy)",
                             "Softmax (probability)"]
        })

    st.markdown("**Forward Pass**")
    st.code("""
Input Shape : (Batch, 3 frames, 3 channels, 160px, 160px)
        ↓
Flatten time into batch dimension  →  (B×3, 3, 160, 160)
        ↓
MobileNetV2 feature extraction     →  (B×3, 1280)
        ↓
Reshape back to sequence           →  (B, 3, 1280)
        ↓
LSTM (hidden=128, layers=1)        →  (B, 3, 128)
        ↓
Take last timestep output          →  (B, 128)
        ↓
Fully connected classifier         →  (B, 2)
        ↓
Softmax                            →  P(Not Drowsy), P(Drowsy)
    """, language="text")

    st.markdown("---")

    # ---- Training Config ----
    st.subheader("🏋️ Training Configuration")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Optimizer & Learning Rate**")
        st.table({
            "Parameter Group"       : ["CNN last 4 blocks", "LSTM", "FC layer"],
            "Learning Rate"         : ["1e-5 (fine-tune)", "1e-4", "1e-4"],
            "Optimizer"             : ["Adam", "Adam", "Adam"]
        })

    with col4:
        st.markdown("**Training Settings**")
        st.table({
            "Setting"           : ["Batch Size", "Loss Function", "Class Weights",
                                    "LR Scheduler", "Early Stopping", "Mixed Precision",
                                    "Gradient Clipping", "Max Epochs", "Epochs Trained"],
            "Value"             : ["8", "CrossEntropyLoss", "NotDrowsy=1.0 / Drowsy=1.2",
                                    "StepLR (step=3, γ=0.1)", "Patience=3",
                                    "torch.amp (AMP)", "Max norm=1.0", "15", "12"]
        })

    st.markdown("---")

    # ---- Performance ----
    st.subheader("📈 Test Set Performance")
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Overall Accuracy", "95.0%")
    col6.metric("Drowsy Precision", "95%")
    col7.metric("Drowsy Recall",    "96%")
    col8.metric("Drowsy F1-Score",  "95%")

    st.markdown("**Classification Report**")
    st.table({
        "Class"       : ["Not Drowsy", "Drowsy", "Macro Avg", "Weighted Avg"],
        "Precision"   : ["0.95", "0.95", "0.95", "0.95"],
        "Recall"      : ["0.94", "0.96", "0.95", "0.95"],
        "F1-Score"    : ["0.94", "0.95", "0.95", "0.95"],
        "Support"     : ["2973", "3600", "6573", "6573"]
    })

    st.markdown("---")

    # ---- Detection Logic ----
    st.subheader("🔍 Real-Time Detection Logic")

    st.markdown("**Status Priority (per face)**")
    st.table({
        "Priority" : ["1 (highest)", "2", "3", "4 (default)"],
        "Status"   : ["EYES FORCED SHUT 🟡",
                       "DROWSY 🔴",
                       "NOT ATTENTIVE 🟠",
                       "ACTIVE 🟢"],
        "Trigger"  : [
            "Current EAR < 0.08 AND previous 3-frame EAR > 0.22 (sudden forced shut)",
            "CNN prob ≥ 60% AND eyes closed, sustained for 4 sec  "
            "OR  CNN prob ≥ 82% (instant high-confidence alert)",
            "Head yaw > 0.25 OR pitch > 0.18 OR gaze offset > 12px, "
            "sustained for 12 sec",
            "None of the above conditions met"
        ]
    })

    st.markdown("**Full Pipeline Diagram**")
    st.code("""
Camera Frame (BGR)
       │
       ▼
MediaPipe FaceLandmarker  ──────────────────────────┐
  478 landmarks, up to 3 faces                      │
       │                                             │
       ├─── Face Crop (pad=20%) ───► CNN-LSTM        │
       │         Sequence of 3 frames                │
       │         → MobileNetV2 features              │
       │         → LSTM temporal model               │
       │         → P(Drowsy)                         │
       │                                             │
       ├─── EAR (Eye Aspect Ratio) ◄─────────────────┤
       │         Per-face calibrated threshold        │
       │         Forced-close detection               │
       │                                             │
       ├─── Head Pose (Yaw / Pitch) ◄────────────────┤
       │         Normalized face-width units          │
       │                                             │
       └─── Gaze Offset (Iris vs Eye center) ◄───────┘
                 Average pixel offset both eyes

                          │
                          ▼
               ┌──────────────────────┐
               │  Decision Engine     │
               │  (per face, timed)   │
               └──────────┬───────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   EYES FORCED        DROWSY         NOT ATTENTIVE
      SHUT 🟡          🔴                🟠
   (immediate)      (4 sec)          (12 sec)
                       │                 │
                       └────────┬────────┘
                                ▼
                         Beep Alert 🔔
                         (3 sec cooldown)
    """, language="text")

    st.markdown("---")

    # ---- Thresholds reference ----
    st.subheader("🎛️ Default Threshold Reference")
    st.table({
        "Parameter"                 : [
            "CNN Drowsy Threshold",
            "CNN Instant Alert Threshold",
            "Drowsy Sustained Duration",
            "EAR Closed Floor (absolute)",
            "EAR Closed Ratio (relative)",
            "EAR Forced Shut Threshold",
            "Head Yaw Threshold",
            "Head Pitch Threshold",
            "Gaze Offset Threshold",
            "Inattentive Duration",
            "Beep Cooldown"
        ],
        "Default Value"             : [
            "0.60", "0.82", "4.0 sec",
            "0.18", "75% of open-eye baseline",
            "0.08", "0.25", "0.18",
            "12 px", "12.0 sec", "3.0 sec"
        ],
        "Description"               : [
            "Minimum CNN probability to start drowsy timer",
            "CNN probability above which alert fires instantly",
            "Signal must persist this long before DROWSY alert",
            "EAR below this = eyes closed (absolute fallback)",
            "Calibrated per face — safe for naturally small eyes",
            "Sudden drop below this = deliberate tight-close (separate label)",
            "Normalized left-right head turn",
            "Normalized downward head tilt",
            "Average iris displacement from eye center (pixels)",
            "Inattentive signal must persist this long before alert",
            "Minimum gap between consecutive audio alerts"
        ]
    })


# ==============================================================
# PAGE 3 — DATASET INFO
# ==============================================================
elif page == "📁 Dataset Info":

    st.title("📁 Dataset Information")
    st.markdown(
        "The model was trained on the **NTHU Drowsy Driver Detection Dataset**, "
        "a widely used benchmark for driver drowsiness research collected under "
        "controlled yet realistic conditions."
    )
    st.markdown("---")

    # ---- Dataset Overview ----
    st.subheader("📦 Dataset Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.table({
            "Property"      : ["Name", "Source Institution", "Type",
                                "Recording Environment", "Subjects",
                                "Ethnicities Covered"],
            "Details"       : [
                "NTHU Drowsy Driver Detection Dataset",
                "National Tsing Hua University (NTHU), Taiwan",
                "Frame-level labeled driver video footage",
                "Simulated driving scenario, controlled lighting",
                "Multiple subjects across sessions",
                "Asian (East Asian), Caucasian, South Asian"
            ]
        })

    with col2:
        st.table({
            "Condition"     : ["No glasses", "Glasses", "Sunglasses", "Night / Low light"],
            "Included"      : ["✅ Yes", "✅ Yes", "✅ Yes", "✅ Yes"]
        })

    st.markdown("---")

    # ---- Data counts ----
    st.subheader("📊 Image Counts After Processing")
    col3, col4, col5 = st.columns(3)
    col3.metric("Total Images (after recrop)", "65,756")
    col4.metric("Drowsy Images", "36,012")
    col5.metric("Not Drowsy Images", "29,744")

    st.markdown("---")

    # ---- Data Pipeline ----
    st.subheader("🔄 Data Processing Pipeline")

    st.markdown("**Stage 1 — Raw Dataset**")
    st.code("""
data/train_data/
    drowsy/      36,030 images  (original frames labeled drowsy)
    notdrowsy/   30,491 images  (original frames labeled not drowsy)
    """, language="text")

    st.markdown("**Stage 2 — Initial Crop Attempt (Abandoned)**")
    st.error(
        "**Problem with YOLO-based crop:**  \n"
        "The original notebook used YOLOv8n to detect a person bounding box, "
        "then applied a fixed center-crop (`h//4 : h*3//4, w//4 : w*3//4`).  \n"
        "This frequently cut off the eye region entirely, especially for subjects "
        "closer to or farther from the camera.  \n"
        "As a result, the model learned to classify drowsiness from face texture "
        "and head pose — **not from the eyes themselves**.  \n"
        "Consequence: closed eyes were not detected reliably in real-time use "
        "despite 94% test accuracy on the (also incorrectly cropped) test set."
    )

    st.markdown("**Stage 3 — MediaPipe Recrop (Current Pipeline)**")
    st.success(
        "**Fix: MediaPipe FaceLandmarker with 478 landmarks**  \n"
        "The recrop script (`step1_recrop.py`) re-processes every original image:  \n"
        "1. Run MediaPipe FaceLandmarker (IMAGE mode)  \n"
        "2. Compute tight bounding box from all 478 face landmarks  \n"
        "3. Add 15% padding on all sides — forehead, chin, and full eye region always included  \n"
        "4. Save to `data/cropped_mp/`  \n\n"
        "Result: CNN now learns eye-closure patterns from correctly cropped faces."
    )

    st.code("""
MediaPipe Recrop Results
─────────────────────────────────────────
drowsy      : 36,012 / 36,030  saved   (18 no-face — mostly extreme angles)
notdrowsy   : 29,744 / 30,491  saved   (747 no-face — blurry / partial frames)

Output: data/cropped_mp/
    """, language="text")

    st.markdown("**Stage 4 — Train / Val / Test Split**")
    st.code("""
Strategy   : Stratified split (class balance preserved in every subset)
Train      : 80%  →  52,604 sequences
Validation : 10%  →   6,575 sequences
Test       : 10%  →   6,573 sequences

Sequence length : 3 consecutive frames per sample
    """, language="text")

    st.markdown("---")

    # ---- Known Issues & Fixes ----
    st.subheader("⚠️ Known Dataset Challenges & Mitigations")

    st.markdown("**1. Naturally Small Eyes (East Asian subjects)**")
    st.info(
        "The NTHU dataset includes East Asian subjects whose open-eye EAR is "
        "naturally lower than the Western-centric EAR thresholds (e.g. 0.25) "
        "commonly cited in the literature.  \n\n"
        "**Mitigation:** Per-face EAR baseline calibration.  \n"
        "During the first 30 frames, open-eye EAR samples are collected.  \n"
        "The closed-eye threshold is set to **75% of the individual baseline**, "
        "not a fixed global value.  \n"
        "This prevents false DROWSY alerts for people with naturally narrow eyes."
    )

    st.markdown("**2. Class Imbalance**")
    st.info(
        "Drowsy images (36,012) outnumber Not Drowsy (29,744) by ~21%.  \n\n"
        "**Mitigation:** Weighted CrossEntropyLoss:  \n"
        "Not Drowsy weight = 1.0, Drowsy weight = 1.2  \n"
        "This discourages the model from defaulting to the majority class."
    )

    st.markdown("**3. Temporal Mismatch in Sequences**")
    st.info(
        "Because images are sorted by filename (not true video timestamp), "
        "consecutive triplets in the dataset may span across different video "
        "clips or sessions.  \n\n"
        "**Mitigation:** The LSTM's 1-layer, 128-unit design is intentionally "
        "conservative — it captures short-range temporal patterns without "
        "overfitting to spurious cross-clip correlations."
    )

    st.markdown("**4. Simulated vs Real Drowsiness**")
    st.info(
        "Some drowsy frames in the dataset were acted/simulated by subjects.  \n\n"
        "**Mitigation:** The real-time detection system requires a **sustained "
        "signal** (4 seconds for drowsy, 12 seconds for inattentive) before "
        "triggering an alert. This filters out momentary or exaggerated poses "
        "that do not correspond to genuine drowsiness."
    )

    st.markdown("---")

    # ---- Future Work ----
    st.subheader("🚀 Planned: IR Adaptation")
    st.markdown(
        "The current model was trained on standard RGB camera footage.  \n"
        "The architecture is designed to be easily fine-tuned for "
        "**infrared (IR) camera input** for nighttime / cockpit use:  \n\n"
        "- Collect IR driver footage with drowsy / alert labels  \n"
        "- Fine-tune only the last 4 CNN blocks + LSTM + FC (same architecture)  \n"
        "- Replace `best_model_v2.pth` with IR-tuned weights  \n"
        "- No changes required in the detection pipeline  \n\n"
        "IR images are single-channel (grayscale) — convert to 3-channel by "
        "replicating across R/G/B before passing to the existing transform pipeline."
    )