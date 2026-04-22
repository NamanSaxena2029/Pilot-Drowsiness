"""
PILOT DROWSINESS DETECTION — STREAMLIT APPLICATION  (v3.0 — Low-Light Edition)
====================================================
Pages:
  1. Live Detection  — Real-time camera feed with dual-model adaptive detection
  2. Model Details   — Architecture, training config, performance metrics (best_model_ll)
  3. Dataset Info    — Dataset provenance, pipeline, low-light augmentation & fixes

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
# CUSTOM CSS — cockpit dark theme
# ==============================================================
st.markdown("""
<style>
    /* Dark cockpit theme */
    .stApp { background-color: #0d1117; }
    section[data-testid="stSidebar"] { background-color: #0a0f1a; border-right: 1px solid #1e2d40; }
    .stMetric { background: #111827; border: 1px solid #1e3a5f; border-radius: 8px; padding: 12px; }
    .stMetric label { color: #7eb8f7 !important; font-size: 0.75rem !important; letter-spacing: 0.08em; text-transform: uppercase; }
    .stMetric [data-testid="metric-container"] { color: #e2f0ff; }
    h1, h2, h3 { color: #c8deff !important; }
    p, li, td, th { color: #9baec8 !important; }
    .stTable th { background: #111827 !important; color: #7eb8f7 !important; border-color: #1e3a5f !important; }
    .stTable td { background: #0d1929 !important; border-color: #1e2d40 !important; }
    .stCodeBlock { background: #070d15 !important; border: 1px solid #1e3a5f !important; }
    .status-badge { display: inline-block; padding: 4px 12px; border-radius: 4px; font-size: 0.82rem; font-weight: 600; letter-spacing: 0.04em; margin: 2px; }
    .badge-green  { background: #0d2d18; color: #4ade80; border: 1px solid #166534; }
    .badge-red    { background: #2d0d0d; color: #f87171; border: 1px solid #991b1b; }
    .badge-orange { background: #2d1a0d; color: #fb923c; border: 1px solid #9a3412; }
    .badge-yellow { background: #2d260d; color: #fbbf24; border: 1px solid #92400e; }
    .stAlert { border-radius: 6px !important; }
    div[data-testid="stImage"] img { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ==============================================================
# SIDEBAR NAVIGATION
# ==============================================================
st.sidebar.markdown("""
<div style="
    background: linear-gradient(135deg, #0a1628 0%, #0d2240 50%, #071020 100%);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 20px 16px;
    margin-bottom: 16px;
    text-align: center;
">
    <div style="font-size: 3.2rem; margin-bottom: 6px;">✈️</div>
    <div style="
        font-size: 1.15rem;
        font-weight: 700;
        color: #7eb8f7;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    ">PILOT GUARD</div>
    <div style="
        font-size: 0.7rem;
        color: #4a6a8a;
        letter-spacing: 0.2em;
        margin-top: 4px;
        text-transform: uppercase;
    ">Drowsiness Detection System</div>
    <div style="
        margin-top: 14px;
        padding: 8px;
        background: #070d15;
        border-radius: 6px;
        border: 1px solid #1e2d40;
        display: flex;
        gap: 8px;
        justify-content: center;
        flex-wrap: wrap;
    ">
        <span style="font-size: 0.65rem; color: #4ade80; background: #0d2d18; padding: 2px 8px; border-radius: 3px; border: 1px solid #166534;">CNN-LSTM</span>
        <span style="font-size: 0.65rem; color: #60a5fa; background: #0d1f40; padding: 2px 8px; border-radius: 3px; border: 1px solid #1e3a7f;">MediaPipe</span>
        <span style="font-size: 0.65rem; color: #fb923c; background: #2d1a0d; padding: 2px 8px; border-radius: 3px; border: 1px solid #9a3412;">IR-Adapted</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("")

page = st.sidebar.radio(
    "Navigate",
    ["🎥 Live Detection", "📊 Model Details", "📁 Dataset Info"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="padding: 10px; background: #070d15; border-radius: 6px; border: 1px solid #1e2d40;">
    <div style="font-size: 0.68rem; color: #4a6a8a; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 8px;">System Info</div>
    <div style="font-size: 0.75rem; color: #7eb8f7;">🔬 Model: <span style="color:#c8deff">CNN-LSTM v2.1</span></div>
    <div style="font-size: 0.75rem; color: #7eb8f7; margin-top: 4px;">📦 Dataset: <span style="color:#c8deff">NTHU-DDD</span></div>
    <div style="font-size: 0.75rem; color: #7eb8f7; margin-top: 4px;">🌙 LL Model: <span style="color:#4ade80">Active</span></div>
    <div style="font-size: 0.75rem; color: #7eb8f7; margin-top: 4px;">📈 AUC: <span style="color:#c8deff">0.9746</span></div>
</div>
""", unsafe_allow_html=True)
st.sidebar.caption("v3.0 — Low-Light Edition")


# ==============================================================
# PAGE 1 — LIVE DETECTION
# ==============================================================
if page == "🎥 Live Detection":

    st.title("✈️ Real-Time Pilot Drowsiness Monitor")
    st.markdown(
        "Monitors up to **3 faces** simultaneously using "
        "**CNN-LSTM** (primary) + **MediaPipe geometry** (confirmation). "
        "Automatically switches between **normal** and **low-light** models "
        "based on ambient brightness — no manual configuration required."
    )

    st.markdown("""
    <div style="display:flex; gap:10px; margin-bottom:16px; flex-wrap:wrap;">
        <div style="background:#0d2d18; border:1px solid #166534; border-radius:6px; padding:8px 16px; font-size:0.8rem; color:#4ade80;">🟢 ACTIVE — eyes open, attentive</div>
        <div style="background:#2d0d0d; border:1px solid #991b1b; border-radius:6px; padding:8px 16px; font-size:0.8rem; color:#f87171;">🔴 DROWSY — eyes closed ≥ threshold</div>
        <div style="background:#2d1a0d; border:1px solid #9a3412; border-radius:6px; padding:8px 16px; font-size:0.8rem; color:#fb923c;">🟠 NOT ATTENTIVE — head/gaze away</div>
        <div style="background:#2d260d; border:1px solid #92400e; border-radius:6px; padding:8px 16px; font-size:0.8rem; color:#fbbf24;">🟡 FORCED SHUT — sudden forced close</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

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
            min_value=0.40, max_value=0.85, value=0.7, step=0.05,
            help="Drowsy probability above this threshold triggers the drowsy timer. Lowered to 0.55 for better low-light sensitivity."
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

        cap = cv2.VideoCapture(0)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # IMPORTANT FIX
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)

        # AUTO SETTINGS (IMPORTANT)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)

        detector = None
        current_mode = None
        last_switch_time = 0

        if not cap.isOpened():
            st.error("❌ Camera not found. Please check your camera connection.")
            st.session_state.cam_running = False
        else:
            info_placeholder.info("Camera active. Press **Stop** to end session.")

            while st.session_state.cam_running:
                ret, frame = cap.read()

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = np.percentile(gray, 75)

                # adaptive correction
                if brightness < 60:
                    frame = cv2.convertScaleAbs(frame, alpha=1.4, beta=25)

                elif brightness > 180:
                    frame = cv2.convertScaleAbs(frame, alpha=0.7, beta=-30)
                if not ret:
                    st.warning("⚠️ Frame capture failed — retrying...")
                    time.sleep(0.1)
                    continue

                # SMART brightness
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                h, w = gray.shape
                # better brightness estimation
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # histogram based brightness (much accurate)
                brightness = np.percentile(gray, 75)
                # NEW thresholds
                if brightness < 70:
                    mode = "low_light"
                    model_path = "best_model_ll.pth"
                else:
                    mode = "normal"
                    model_path = "best_model_v2.pth"

                # 🔥 SWITCH MODEL (only when needed + cooldown)
                if (detector is None or mode != current_mode) and (time.time() - last_switch_time > 2):
                    detector = DrowsinessDetector(
                        model_path=model_path,
                        landmarker_path="face_landmarker.task",
                        max_faces=max_faces
                    )

                    detector.DROWSY_TIME_SEC   = drowsy_time
                    detector.INATTENTIVE_SEC   = inatt_time
                    detector.DROWSY_CNN_THRESH = cnn_thresh

                    current_mode     = mode
                    last_switch_time = time.time()

                    st.toast(f"🚀 Switched to {'🌙 Low-Light' if mode == 'low_light' else '☀️ Normal'} model")

                # ---- PROCESS FRAME ----

                output_frame, face_states = detector.process_frame(frame)

                # Display
                rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb, channels="RGB", width="stretch")

                # 🔥 MODE DISPLAY
                mode_icon  = "🌙" if current_mode == "low_light" else "☀️"
                mode_label = "Low-Light Model" if current_mode == "low_light" else "Normal Model"
                info_placeholder.markdown(
                    f"{mode_icon} **Active:** {mode_label} &nbsp;|&nbsp; "
                    f"Brightness: `{int(brightness)}` &nbsp;|&nbsp; "
                    f"Threshold: `{'<75' if current_mode == 'low_light' else '>90'}`"
                )

                # ---- METRICS ----
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

                # ---- STATUS ----
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
        "patterns across a 3-frame sequence. The model was originally trained "
        "on RGB data (`best_model_v2`) and then **fine-tuned on low-light "
        "enhanced data** (`best_model_ll`) for robust real-world cockpit performance."
    )
    st.markdown("---")

    # ---- Architecture ----
    st.subheader("🧠 Architecture")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**CNN Backbone**")
        st.table({
            "Component"  : ["Backbone", "Pretrained On", "Frozen Layers",
                             "Trainable Layers", "Feature Output"],
            "Details"    : ["MobileNetV2", "ImageNet (ILSVRC)",
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
    st.subheader("🏋️ Training & Fine-Tuning Configuration")

    tab1, tab2 = st.tabs(["📦 Base Model (best_model_v2)", "🌙 Fine-Tuned LL Model (best_model_ll)"])

    with tab1:
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Optimizer & Learning Rate**")
            st.table({
                "Parameter Group"   : ["CNN last 4 blocks", "LSTM", "FC layer"],
                "Learning Rate"     : ["1e-5 (fine-tune)", "1e-4", "1e-4"],
                "Optimizer"         : ["Adam", "Adam", "Adam"]
            })
        with col4:
            st.markdown("**Training Settings**")
            st.table({
                "Setting"           : ["Batch Size", "Loss Function", "Class Weights",
                                        "LR Scheduler", "Early Stopping", "Max Epochs"],
                "Value"             : ["8", "CrossEntropyLoss", "NotDrowsy=1.0 / Drowsy=1.2",
                                        "StepLR (step=3, γ=0.1)", "Patience=3", "15"]
            })

    with tab2:
        st.success(
            "**Fine-tuning strategy:** Loaded `best_model_v2.pth` weights → adapted to "
            "low-light enhanced dataset (`data/cropped_mp_ll/`) with lower learning rates "
            "to preserve learned features while adapting to IR-like visual patterns."
        )
        col5, col6 = st.columns(2)
        with col5:
            st.markdown("**Fine-Tune Learning Rates (reduced)**")
            st.table({
                "Parameter Group"   : ["CNN last 4 blocks", "LSTM", "FC layer"],
                "Learning Rate"     : ["5e-6 (very conservative)", "5e-5", "5e-5"],
                "Optimizer"         : ["Adam", "Adam", "Adam"]
            })
        with col6:
            st.markdown("**Low-Light Augmentations Added**")
            st.table({
                "Augmentation"      : ["GaussianBlur(3)", "ColorJitter(0.3, 0.3)",
                                        "CLAHE preprocessing", "Histogram equalization",
                                        "Brightness/contrast boost"],
                "Purpose"           : ["IR noise simulation", "Lighting variation",
                                        "Low-light enhancement", "IR-style normalization",
                                        "Extreme dark recovery"]
            })
        st.code("""
Fine-Tune Data  : data/cropped_mp_ll/   (low-light enhanced crops)
Epochs          : 10
Best Val Acc    : saved as best_model_ll.pth
Grayscale Fix   : cv2.COLOR_GRAY2RGB    (handles single-channel IR frames)
        """, language="text")

    st.markdown("---")

    # ---- Performance ----
    st.subheader("📈 Test Set Performance — Both Models")

    perf_tab1, perf_tab2 = st.tabs(["☀️ best_model_v2  (Non-IR / RGB)", "🌙 best_model_ll  (Low-Light Fine-Tuned)"])

    # ── Non-IR Model ─────────────────────────────────────────────
    with perf_tab1:
        st.info(
            "Results from `eval_results.py` on the **held-out test set** using `best_model_v2.pth`.  \n"
            "Trained on standard RGB MediaPipe-cropped data (`data/cropped_mp/`).  \n"
            "This is the base model — strong in normal lighting, struggles in low light."
        )
        col_v1, col_v2, col_v3, col_v4, col_v5 = st.columns(5)
        col_v1.metric("Overall Accuracy", "95%")
        col_v2.metric("Drowsy Precision", "95%")
        col_v3.metric("Drowsy Recall",    "96%")
        col_v4.metric("Drowsy F1-Score",  "95%")
        col_v5.metric("AUC (ROC)",        "~0.98+")

        st.markdown("**Classification Report — best_model_v2**")
        st.table({
            "Class"      : ["Not Drowsy", "Drowsy", "Macro Avg", "Weighted Avg"],
            "Precision"  : ["0.95", "0.95", "0.95", "0.95"],
            "Recall"     : ["0.94", "0.96", "0.95", "0.95"],
            "F1-Score"   : ["0.94", "0.95", "0.95", "0.95"],
            "Support"    : ["2973", "3600", "6573", "6573"]
        })

        st.markdown("**Confusion Matrix — best_model_v2**")
        st.table({
            "Actual \\ Predicted"  : ["Not Drowsy (Actual)", "Drowsy (Actual)"],
            "→ Not Drowsy"         : ["2794 ✅ (True Not Drowsy)", "144 ❌ (Missed Drowsy)"],
            "→ Drowsy"             : ["179 ⚠️ (False Alarm)",    "3456 ✅ (True Drowsy)"]
        })

        st.markdown("**Confusion Matrix Interpretation — best_model_v2**")
        st.table({
            "Metric"                          : ["True Not Drowsy", "False Alarm (False Positive)",
                                                  "Missed Drowsy (False Negative)", "True Drowsy"],
            "Count"                           : ["2794", "179", "144", "3456"],
            "Notes"                           : [
                "Alert pilot correctly identified ✅",
                "Alert pilot wrongly flagged — safety-acceptable",
                "⚠️ Drowsy pilot missed by CNN — EAR path catches these",
                "Drowsy pilot correctly caught ✅ — highest priority"
            ]
        })

        st.success(
            "**best_model_v2 is the stronger base model** — 95% accuracy, fewer false negatives (144 vs 188).  \n"
            "Used automatically when ambient brightness > 90 (good lighting conditions)."
        )

    # ── Low-Light Model ──────────────────────────────────────────
    with perf_tab2:
        st.info(
            "Results from `eval_results.py` on the **held-out test set** using `best_model_ll.pth`.  \n"
            "Fine-tuned from `best_model_v2` on low-light enhanced data (`data/cropped_mp_ll/`).  \n"
            "Optimized for dark / cockpit lighting. AUC = 0.9746 confirms strong discriminative power."
        )
        col_l1, col_l2, col_l3, col_l4, col_l5 = st.columns(5)
        col_l1.metric("Overall Accuracy", "92%")
        col_l2.metric("Drowsy Precision", "91%")
        col_l3.metric("Drowsy Recall",    "95%")
        col_l4.metric("Drowsy F1-Score",  "93%")
        col_l5.metric("AUC (ROC)",        "0.9746")

        st.markdown("**Classification Report — best_model_ll**")
        st.table({
            "Class"      : ["Not Drowsy", "Drowsy", "Macro Avg", "Weighted Avg"],
            "Precision"  : ["0.93", "0.91", "0.92", "0.92"],
            "Recall"     : ["0.89", "0.95", "0.92", "0.92"],
            "F1-Score"   : ["0.91", "0.93", "0.92", "0.92"],
            "Support"    : ["2973", "3600", "6573", "6573"]
        })

        st.markdown("**Confusion Matrix — best_model_ll**")
        st.table({
            "Actual \\ Predicted"  : ["Not Drowsy (Actual)", "Drowsy (Actual)"],
            "→ Not Drowsy"         : ["2650 ✅ (True Not Drowsy)", "188 ❌ (Missed Drowsy)"],
            "→ Drowsy"             : ["323 ⚠️ (False Alarm)",    "3412 ✅ (True Drowsy)"]
        })

        st.markdown("**Confusion Matrix Interpretation — best_model_ll**")
        st.table({
            "Metric"                          : ["True Not Drowsy", "False Alarm (False Positive)",
                                                  "Missed Drowsy (False Negative)", "True Drowsy"],
            "Count"                           : ["2650", "323", "188", "3412"],
            "Notes"                           : [
                "Alert pilot correctly identified ✅",
                "Alert pilot wrongly flagged — higher than v2, but still safety-acceptable",
                "⚠️ Drowsy pilot missed by CNN — EAR Path A catches these in real-time",
                "Drowsy pilot correctly caught ✅ — 95% recall is strong"
            ]
        })

        st.success(
            "**Safety-first design:** EAR-only Path A acts as a fallback — "
            "sustained eye closure ≥ 4 sec fires DROWSY alert even if CNN confidence is low.  \n"
            "This minimizes the real-world impact of the 188 CNN false-negatives.  \n"
            "Used automatically when ambient brightness < 75 (dark / cockpit conditions)."
        )

    st.markdown("---")
    st.subheader("📊 Model Comparison")
    st.table({
        "Metric"             : ["Overall Accuracy", "Drowsy Precision", "Drowsy Recall",
                                 "Drowsy F1", "Not Drowsy Recall", "False Alarms",
                                 "Missed Drowsy", "AUC", "Best used in"],
        "best_model_v2 ☀️"  : ["95%", "0.95", "0.96", "0.95", "0.94",
                                 "179", "144", "~0.98+", "Normal light (brightness > 90)"],
        "best_model_ll 🌙"  : ["92%", "0.91", "0.95", "0.93", "0.89",
                                 "323", "188", "0.9746", "Low light (brightness < 75)"]
    })

    st.markdown("---")

    # ---- Adaptive Model Switching ----
    st.subheader("🌙 Adaptive Model Switching (NEW in v3.0)")

    st.success(
        "**Dynamic Model Switching Enabled**  \n"
        "The system automatically selects the optimal model based on ambient brightness:  \n\n"
        "- ☀️ Bright scenes (brightness > 90) → `best_model_v2.pth` (RGB trained)  \n"
        "- 🌙 Low-light scenes (brightness < 75) → `best_model_ll.pth` (fine-tuned)  \n"
        "- 🔄 Hysteresis zone (75–90) → stays on current model to prevent flickering  \n"
        "- ⏱️ 2-second cooldown between switches prevents rapid toggling  \n\n"
        "This approach ensures maximum accuracy in both daytime and nighttime/cockpit conditions."
    )

    st.code("""
Brightness Detection:
    gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()

Hysteresis Model Selection:
    if brightness < 75:
        model = best_model_ll.pth   # Low-light fine-tuned
    elif brightness > 90:
        model = best_model_v2.pth   # RGB trained
    else:
        model = current_model       # Stay — avoid flicker

Cooldown guard: switch only if time.time() - last_switch > 2 sec
    """, language="python")

    st.markdown("---")

    # ---- Detection Logic ----
    st.subheader("🔍 Real-Time Detection Logic")

    st.markdown("**Status Priority (per face)**")
    st.table({
        "Priority"  : ["1 (highest)", "2", "3", "4 (default)"],
        "Status"    : ["EYES FORCED SHUT 🟡", "DROWSY 🔴", "NOT ATTENTIVE 🟠", "ACTIVE 🟢"],
        "Trigger"   : [
            "Current EAR < 0.08 AND previous 3-frame EAR > 0.22 (sudden forced shut)",
            "Path A: Eyes closed ≥ 4 sec (EAR-only, no CNN needed)  |  "
            "Path B: CNN prob ≥ 60% AND eyes closed sustained  |  "
            "Path B instant: CNN prob ≥ 82%",
            "Head yaw > 0.25 OR pitch > 0.18 OR gaze offset > 12px, sustained 12 sec",
            "None of the above"
        ]
    })

    st.markdown("**Full Pipeline Diagram**")
    st.code("""
Camera Frame (BGR)
       │
       ▼
Low-Light Enhancement  ←  CLAHE + brightness boost + IR simulation
       │
       ▼
MediaPipe FaceLandmarker  ───────────────────────────┐
  478 landmarks, up to 3 faces                       │
       │                                             │
       ├─── Face Crop (pad=20%) ───► CNN-LSTM        │
       │         Enhanced crop → histogram eq        │
       │         Sequence of 3 frames                │
       │         → MobileNetV2 features              │
       │         → LSTM temporal model               │
       │         → P(Drowsy)                         │
       │                                             │
       ├─── EAR (Eye Aspect Ratio) ◄─────────────────┤
       │         Per-face calibrated threshold       │
       │         Path A: EAR timer (independent)     │
       │                                             │
       ├─── Head Pose (Yaw / Pitch) ◄────────────────┤
       │         Normalized face-width units         │
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
        "Parameter"              : [
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
            "Beep Cooldown",
            "Low-Light Switch Threshold",
            "Normal-Light Switch Threshold",
        ],
        "Default Value"          : [
            "0.55 (reduced for LL)", "0.82", "4.0 sec",
            "0.15", "68% of open-eye baseline",
            "0.07", "0.40", "0.30",
            "25 px", "12.0 sec", "3.0 sec",
            "brightness < 75", "brightness > 90"
        ],
        "Description"            : [
            "Minimum CNN probability to start drowsy timer — lowered for LL sensitivity",
            "CNN probability above which alert fires instantly",
            "Signal must persist this long before DROWSY alert",
            "EAR below this = eyes closed (absolute fallback)",
            "Calibrated per face — safe for naturally small eyes",
            "Sudden drop below this = deliberate tight-close (separate label)",
            "Normalized left-right head turn",
            "Normalized downward head tilt",
            "Average iris displacement from eye center (pixels)",
            "Inattentive signal must persist this long before alert",
            "Minimum gap between consecutive audio alerts",
            "Frame brightness below which low-light model activates",
            "Frame brightness above which normal model activates"
        ]
    })


# ==============================================================
# PAGE 3 — DATASET INFO
# ==============================================================
elif page == "📁 Dataset Info":

    st.title("📁 Dataset Information & Project Pipeline")
    st.markdown(
        "Complete journey from raw NTHU dataset → RGB model → Low-Light fine-tuning → "
        "Adaptive real-time detection system. Scroll down for detailed dataset stats, "
        "pipeline stages, known issues and fixes."
    )
    st.markdown("---")

    # ============================================================
    # COMPLETE PROJECT FLOW DIAGRAM
    # ============================================================
    st.subheader("🗺️ Complete Project Flow")

    st.markdown("#### 📦 Phase 0 — Raw Data")
    st.code("""
                    ┌─────────────────────────────────────────────┐
                    │        NTHU Drowsy Driver Dataset           │
                    │  66,521 labeled frames  |  RGB camera       │
                    │  drowsy: 36,030   |   notdrowsy: 30,491     │
                    └─────────────────────────────────────────────┘
                                          │
                                          ▼
""", language="text")

    st.error("**❌ Phase 1 — First Crop Attempt (Abandoned)**")
    st.code("""
  ┌──────────────────────┐       ┌────────────────────────┐       ┌──────────────────┐
  │  YOLOv8n Detection   │  ──►  │ 94% Test Acc           │  ──►  │  Real-time FAIL  │
  │  Fixed center-crop   │       │ (Misleading)           │       │                  │
  │  h//4 : h*3//4       │       │ Learned texture/pose   │       │  Eyes closed NOT │
  │  cut off eyes ❌    │       │ NOT eye closure ❌     │       │  detected ❌    │
  └──────────────────────┘       └────────────────────────┘       └──────────────────┘

  Problem: Fixed crop cut off eye region → model learned face texture not eye state
  Fix    : Scrapped. Restarted with MediaPipe landmark-based crop
                                          │
                                          ▼
""", language="text")

    st.markdown("#### ☀️ Phase 2 — Non-IR (RGB) Pipeline")
    st.code("""
  Raw NTHU Frames     MediaPipe FaceLandmarker    Face Crop       Train/Val/Test Split
  (66,521 images)     478 landmarks               pad = 15%       80% / 10% / 10%
       │              step1_recrop.py             65,756 saved     stratified
       ▼                    │                         │                   │
  ┌─────────┐        ┌──────────────┐          ┌──────────┐       ┌──────────────┐
  │  data/  │ ──────►│  Detect face │ ────────►│  data/   │──────►│  Sequence    │
  │ train_  │        │  on each img │          │ cropped_ │       │  Dataset     │
  │  data/  │        │  landmarks   │          │    mp/   │       │  3 frames    │
  └─────────┘        └──────────────┘          └──────────┘       └──────┬───────┘
                                                                          │
                                                                          ▼
  ┌───────────────────────────────────────────────────────────────────────────────────┐
  │                     CNN-LSTM Training  (step2_retrain.py)                         │
  │                                                                                   │
  │  MobileNetV2 (ImageNet) → Last 4 blocks trainable → LSTM(128) → FC(2)             │
  │  Optimizer: Adam  |  LR: 1e-5/1e-4  |  Loss: CrossEntropy  |  Epochs: 12          │
  └──────────────────────────────────────┬────────────────────────────────────────────┘
                                         │
                                         ▼
                          ┌───────────────────────────────┐
                          │      best_model_v2.pth        │
                          │  Accuracy      : 95%          │
                          │  Drowsy F1     : 0.95         │
                          │  Drowsy Recall : 96%          │
                          │  Works in      : Normal ☀️    │
                          │  Fails in      : Low light ❌ │
                          └───────────────────────────────┘
                                         │
                                         ▼
             Problem: Dark frames → MediaPipe fails → no crop → no CNN → no alert
""", language="text")

    st.markdown("#### 🌙 Phase 3 — Low-Light / IR Pipeline (Fine-Tune)")
    st.code("""
  Same Raw NTHU        CLAHE Enhancement      IR Simulation         Face Crop
  Frames (reused)      clipLimit=4.0          Fallback              pad=20%
  66,521 images        alpha=1.4, beta=20     equalizeHist          100% saved ✅
       │               step1_recrop_lowlight  gray → BGR
       ▼                    │                      │                      │
  ┌─────────┐        ┌──────────────┐       ┌─────────────┐       ┌──────────────┐
  │  data/  │ ──────►│  Enhance     │──────►│  MediaPipe  │──────►│  data/       │
  │ train_  │        │  low-light   │       │  try 1:     │       │  cropped_    │
  │  data/  │        │  frame       │       │  enhanced   │       │  mp_ll/      │
  └─────────┘        └──────────────┘       │  try 2: IR  │       └──────┬───────┘
                                            └─────────────┘              │
                                                              36,030 + 30,491 (100% ✅)
                                                                          │
                                                                          ▼
  ┌───────────────────────────────────────────────────────────────────────────────────┐
  │                     Fine-Tune  (step2_finetune_ll.py)                             │
  │                                                                                   │
  │  Start from : best_model_v2.pth weights  (NOT from scratch)                       │
  │  Dataset    : data/cropped_mp_ll/  (low-light enhanced crops)                     │
  │  LR         : 5e-6 (CNN) / 5e-5 (LSTM+FC) — conservative                          │
  │  Augment    : GaussianBlur(3) + ColorJitter(0.3,0.3) → IR noise simulation        │
  │  Epochs     : 10  |  Best val acc saved automatically                             │
  └──────────────────────────────────────┬────────────────────────────────────────────┘
                                         │
                                         ▼
                          ┌───────────────────────────────┐
                          │      best_model_ll.pth        │
                          │  Accuracy      : 92%          │
                          │  AUC           : 0.9746       │
                          │  Drowsy Recall : 95%          │
                          │  Works in      : Low light 🌙 │
                          └───────────────────────────────┘
""", language="text")

    st.markdown("#### 🚀 Phase 4 — Real-Time Adaptive Detection System")
    st.code("""
  Live Camera Frame (BGR)
          │
          ▼
  ┌─────────────────────┐
  │   Brightness Check  │    gray.mean() < 75  →  LOW LIGHT  →  best_model_ll
  │     every frame     │    gray.mean() > 90  →  NORMAL     →  best_model_v2
  └──────────┬──────────┘    75–90             →  stay on current (hysteresis)
             │                                    2 sec cooldown between switches
             ▼
  ┌──────────────────────────────────────────────────┐
  │        Low-Light Enhancement (always on)         │
  │   CLAHE (LAB L-channel) → brightness boost       │
  │   convertScaleAbs(alpha=2.0, beta=40)            │
  └──────────────────────┬───────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────┐
  │           MediaPipe FaceLandmarker               │
  │  478 landmarks | up to 3 faces | conf=0.3        │
  │  Fallback: IR simulation if face not found       │
  └──────┬───────────────────────────────────────────┘
         │
         ├─────────────────────────────┐
         ▼                             ▼
  ┌──────────────┐             ┌───────────────────────┐
  │  Face Crop   │             │   Geometry Signals    │
  │  enhance +   │             │   EAR  Eye Aspect     │
  │  hist eq     │             │        Ratio          │
  └──────┬───────┘             │   Yaw  head turn      │
         ▼                     │   Pitch head down     │
  ┌──────────────┐             │   Gaze iris offset    │
  │  CNN-LSTM    │             └──────────┬────────────┘
  │  3-frame seq │                        │
  │  P(Drowsy)   │                        │
  └──────┬───────┘                        │
         └───────────────┬────────────────┘
                         ▼
  ┌──────────────────────────────────────────────────┐
  │               Decision Engine                    │
  │                                                  │
  │  Path A (EAR-only): eyes closed ≥ 4s → DROWSY    │
  │  Path B (CNN+EAR) : CNN ≥ 0.55 + closed → timer  │
  │  Path B instant   : CNN ≥ 0.82 → DROWSY          │
  │  Forced shut      : sudden EAR drop < 0.07       │
  │  Inattentive      : yaw/pitch/gaze for 12s       │
  └────────────────────┬─────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
     DROWSY 🔴   NOT ATTENTIVE   FORCED SHUT 🟡
     (4 sec)       🟠 (12 sec)    (immediate)
        │              │
        └──────┬────────┘
               ▼
        Beep Alert 🔔 (3s cooldown)
               │
               ▼
      Streamlit Dashboard
      Live feed + metrics + face badges
""", language="text")

    st.markdown("**📄 Scripts Reference**")
    st.table({
        "Script"    : [
            "step1_recrop.py",
            "step1_recrop_lowlight.py",
            "step2_retrain.py",
            "step2_finetune_ll.py",
            "eval_results.py",
            "drowsy_detection.py",
            "app.py"
        ],
        "Purpose"   : [
            "MediaPipe face crop on raw NTHU dataset → data/cropped_mp/",
            "CLAHE + IR-fallback crop → data/cropped_mp_ll/ (Phase 3)",
            "Full CNN-LSTM training on cropped_mp → best_model_v2.pth",
            "Fine-tune best_model_v2 on cropped_mp_ll → best_model_ll.pth",
            "Generate confusion matrix + ROC curve for any saved model",
            "Core detection engine: MediaPipe + CNN-LSTM + EAR + alerts",
            "Streamlit UI: Live detection + Model Details + Dataset Info"
        ]
    })

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
    col3.metric("Total Images (cropped_mp)", "65,756")
    col4.metric("Drowsy Images", "36,012")
    col5.metric("Not Drowsy Images", "29,744")

    st.markdown("**Low-Light Enhanced Dataset (cropped_mp_ll)**")
    col6, col7, col8 = st.columns(3)
    col6.metric("Total LL Images", "66,521")
    col7.metric("Drowsy (LL)", "36,030  (100%)")
    col8.metric("Not Drowsy (LL)", "30,491  (100%)")

    st.success(
        "**100% detection rate** in both classes during low-light recrop — "
        "the CLAHE + IR fallback pipeline successfully detected faces in all images."
    )

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

    st.markdown("**Stage 3 — MediaPipe Recrop (Standard Pipeline)**")
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
MediaPipe Recrop Results (cropped_mp)
─────────────────────────────────────────
drowsy      : 36,012 / 36,030  saved   (18 no-face — mostly extreme angles)
notdrowsy   : 29,744 / 30,491  saved   (747 no-face — blurry / partial frames)

Output: data/cropped_mp/
    """, language="text")

    st.markdown("**Stage 4 — Low-Light Enhanced Recrop (NEW)**")
    st.success(
        "**`step1_recrop_lowlight.py`** — Enhanced pipeline for dark/IR-like frames:  \n"
        "1. Apply CLAHE enhancement (clipLimit=4.0, adaptive per brightness)  \n"
        "2. Brightness boost: `alpha=1.4, beta=20`  \n"
        "3. Try MediaPipe on enhanced frame  \n"
        "4. **Fallback:** IR simulation (`equalizeHist` on grayscale) if face not found  \n"
        "5. Save to `data/cropped_mp_ll/` — original images kept separately  \n\n"
        "Achieved **100% detection rate** on both classes."
    )

    st.code("""
Low-Light Recrop Results (cropped_mp_ll)
─────────────────────────────────────────
drowsy      : 36,030 / 36,030  saved   (100% ✅ — CLAHE + IR fallback)
notdrowsy   : 30,491 / 30,491  saved   (100% ✅ — CLAHE + IR fallback)

Speed       : ~62.84 it/s (drowsy), ~66.60 it/s (notdrowsy)
Total time  : ~17 minutes

Output: data/cropped_mp_ll/
    """, language="text")

    st.markdown("**Stage 5 — Train / Val / Test Split**")
    st.code("""
Strategy   : Stratified split (class balance preserved in every subset)
Train      : 80%  →  ~53,216 sequences
Validation : 10%  →   ~6,652 sequences
Test       : 10%  →   ~6,573 sequences

Sequence length : 3 consecutive frames per sample
Sorted by filename to maintain temporal ordering within clips
    """, language="text")

    st.markdown("---")

    # ---- Low-Light Augmentation ----
    st.subheader("🌙 Low-Light Data Augmentation Pipeline (NEW)")

    st.info(
        "To improve robustness in dark environments, a dedicated preprocessing and "
        "augmentation pipeline was introduced for both dataset preparation and inference:  \n\n"
        "**During training (transform augmentations):**  \n"
        "- `GaussianBlur(3)` — simulates IR sensor noise  \n"
        "- `ColorJitter(brightness=0.3, contrast=0.3)` — lighting variation  \n"
        "- `RandomHorizontalFlip()` + `RandomRotation(8°)` — pose variation  \n\n"
        "**During inference (runtime enhancement):**  \n"
        "- CLAHE on L-channel (LAB space) — adaptive contrast  \n"
        "- Hard brightness boost: `alpha=2.0, beta=40`  \n"
        "- IR fallback: `equalizeHist` on grayscale  \n"
        "- Crop enhancement: `alpha=1.3, beta=15` + histogram equalization  \n\n"
        "This eliminates the **domain gap** between training and inference — "
        "both now use the same enhancement pipeline."
    )

    st.code("""
Low-Light Enhancement Flow (runtime):

frame
  → check brightness (gray.mean())
  → if dark: CLAHE (clipLimit=3.0–5.0 adaptive)
  → hard boost: convertScaleAbs(alpha=2.0, beta=40)
  → MediaPipe detection
  → if fail: IR simulation (equalizeHist → gray → BGR)
  → MediaPipe detection (fallback)
  → crop face
  → crop enhance: convertScaleAbs(alpha=1.3, beta=15)
  → histogram equalize crop
  → CNN-LSTM inference
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
        "During the first 60 frames, open-eye EAR samples are collected.  \n"
        "The closed-eye threshold is set to **68% of the individual baseline**, "
        "not a fixed global value.  \n"
        "For very low baselines (< 0.22), ratio is further reduced to 58%."
    )

    st.markdown("**2. Class Imbalance**")
    st.info(
        "Drowsy images (36,030) outnumber Not Drowsy (30,491) by ~18%.  \n\n"
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

    st.markdown("**5. Low-Light Domain Gap (Resolved)**")
    st.warning(
        "**Original problem:** Model trained on well-lit RGB data failed in dark/cockpit conditions.  \n"
        "- MediaPipe could not detect faces in dark frames → no crop → no CNN input  \n"
        "- Even when detected, dark crops caused CNN to output low-confidence scores  \n\n"
        "**Resolution (multi-step):**  \n"
        "1. Created `cropped_mp_ll` dataset with CLAHE + IR fallback preprocessing  \n"
        "2. Fine-tuned `best_model_ll` on this enhanced dataset (10 epochs, low LR)  \n"
        "3. Added runtime CLAHE + hard brightness boost before MediaPipe detection  \n"
        "4. Added crop-level enhancement before CNN inference  \n"
        "5. Implemented adaptive model switching based on ambient brightness  \n\n"
        "Result: Stable detection in low-light conditions without IR camera hardware."
    )

    st.markdown("---")

    # ---- Real-World Deployment Notes ----
    st.subheader("💡 Real-World Deployment Notes")

    st.warning(
        "Performance depends significantly on lighting conditions.  \n\n"
        "- **Extreme darkness** (no visible face outline) → detection fails for any model  \n"
        "- **Low light** (face silhouette visible) → low-light model handles well ✅  \n"
        "- **Balanced lighting** → optimal performance for both models ✅  \n"
        "- **Overexposure** → feature loss, reduced accuracy  \n\n"
        "Camera settings applied at startup: `BRIGHTNESS=130, GAIN=40, EXPOSURE=-6`  \n"
        "These values balance sensitivity and overexposure for typical indoor lighting."
    )

    st.markdown("**For production / real cockpit use:**")
    col_dep1, col_dep2 = st.columns(2)

    with col_dep1:
        st.info(
            "**Software improvements (immediate):**  \n"
            "- Tune camera exposure per lighting environment  \n"
            "- Add YOLO face fallback for extreme pose angles  \n"
            "- WebSocket streaming for lower latency  \n"
            "- Collect in-cockpit labeled data for domain fine-tuning"
        )
    with col_dep2:
        st.info(
            "**Hardware improvements (recommended):**  \n"
            "- IR camera (~₹1500–5000) for true night-vision detection  \n"
            "- Dedicated GPU inference (NVIDIA Jetson or similar)  \n"
            "- Dual-camera setup (IR + RGB) for best-of-both-worlds  \n"
            "- Helmet/headset-mounted camera for stable face framing"
        )

    st.markdown("---")

    # ---- Future Work ----
    st.subheader("🚀 Future Work")

    col_f1, col_f2 = st.columns(2)

    with col_f1:
        st.markdown("**Model Improvements**")
        st.markdown(
            "- True IR dataset training (NTHU-DDD2 IR version)  \n"
            "- Larger LSTM (256 units) or Transformer temporal model  \n"
            "- Knowledge distillation for edge deployment  \n"
            "- ONNX export for cross-platform inference"
        )
    with col_f2:
        st.markdown("**System Improvements**")
        st.markdown(
            "- React + FastAPI web interface for browser deployment  \n"
            "- Mobile app (iOS/Android) integration  \n"
            "- CAN bus integration for vehicle systems  \n"
            "- Multi-modal fusion (physiological + visual signals)"
        )