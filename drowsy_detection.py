"""
PILOT DROWSINESS DETECTION — CORE ENGINE  (v2.1 — Fixed)
=========================================
Fixes over v2.0:
  - Low-light preprocessing (CLAHE on L-channel) so MediaPipe detects face in dim light
  - EAR-only drowsy path: if eyes stay closed ≥ DROWSY_TIME_SEC → DROWSY (no CNN needed)
  - CNN + EAR combined path (original) still works alongside
  - Inattentive timer = 10 sec (head/gaze away, eyes can be open)
  - Removed winsound dependency issue (graceful fallback)
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarkerOptions, FaceLandmarker
from torchvision import transforms, models
from collections import deque
import time

# Beep — Windows only, safe fallback
try:
    import winsound
    _WINSOUND = True
except ImportError:
    _WINSOUND = False


# ==============================================================
# LOW-LIGHT PREPROCESSING
# Apply CLAHE on the L-channel (LAB space) to brighten dark frames
# so MediaPipe can detect faces in dim/cockpit lighting
# ==============================================================
def enhance_low_light(frame: np.ndarray) -> np.ndarray:
    """
    Boost contrast in dark frames using CLAHE on LAB L-channel.
    Only applied when mean brightness < 80 (out of 255).
    Returns enhanced BGR frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(gray.mean())

    if mean_brightness >= 80:
        return frame  # Good light — no change

    lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Stronger clip limit for very dark frames
    clip = 3.0 if mean_brightness > 50 else 5.0
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l_eq  = clahe.apply(l)

    lab_eq  = cv2.merge([l_eq, a, b])
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return enhanced


# ==============================================================
# MODEL DEFINITION
# ==============================================================
class CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = models.mobilenet_v2(weights=None)
        self.cnn.classifier = nn.Identity()
        for param in self.cnn.features[:-4].parameters():
            param.requires_grad = False
        self.lstm = nn.LSTM(1280, 128, num_layers=1, batch_first=True)
        self.fc   = nn.Linear(128, 2)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x        = x.view(B * T, C, H, W)
        features = self.cnn(x)
        features = features.view(B, T, -1).contiguous()
        out, _   = self.lstm(features)
        return self.fc(out[:, -1, :])


# ==============================================================
# MEDIAPIPE LANDMARK INDICES
# ==============================================================
LEFT_EYE   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE  = [33,  160, 158, 133, 153, 144]
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
NOSE_TIP     = 1
CHIN         = 152
LEFT_EAR_PT  = 234
RIGHT_EAR_PT = 454
LEFT_EYE_C   = 159
RIGHT_EYE_C  = 386


# ==============================================================
# GEOMETRY HELPERS
# ==============================================================
def compute_ear(landmarks, eye_indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C + 1e-6)


def compute_head_pose(landmarks, w, h):
    nose  = np.array([landmarks[NOSE_TIP].x,    landmarks[NOSE_TIP].y])
    l_ear = np.array([landmarks[LEFT_EAR_PT].x, landmarks[LEFT_EAR_PT].y])
    r_ear = np.array([landmarks[RIGHT_EAR_PT].x, landmarks[RIGHT_EAR_PT].y])
    l_eye = np.array([landmarks[LEFT_EYE_C].x,  landmarks[LEFT_EYE_C].y])

    face_width    = np.linalg.norm(r_ear - l_ear) + 1e-6
    face_center_x = (l_ear[0] + r_ear[0]) / 2.0

    yaw   = (nose[0] - face_center_x) / face_width
    pitch = (nose[1] - l_eye[1]) / face_width
    return float(yaw), float(pitch)


def compute_gaze_offset(landmarks, w, h):
    def center(indices):
        pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])
        return pts.mean(axis=0)

    l_iris  = center(LEFT_IRIS)
    r_iris  = center(RIGHT_IRIS)
    l_eye_c = center(LEFT_EYE)
    r_eye_c = center(RIGHT_EYE)
    return float((np.linalg.norm(l_iris - l_eye_c) + np.linalg.norm(r_iris - r_eye_c)) / 2.0)


def crop_face(frame, landmarks, w, h, pad=0.20):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1 = max(0, int(min(xs) - pad * w))
    y1 = max(0, int(min(ys) - pad * h))
    x2 = min(w, int(max(xs) + pad * w))
    y2 = min(h, int(max(ys) + pad * h))
    crop = frame[y1:y2, x1:x2]
    valid = crop is not None and crop.size > 0
    return (crop if valid else None), (x1, y1, x2, y2)


# ==============================================================
# PER-FACE STATE TRACKER
# ==============================================================
class FaceState:
    def __init__(self, face_id: int):
        self.face_id     = face_id
        self.seq_buffer  = deque(maxlen=3)
        self.prob_buffer = deque(maxlen=20)

        # Drowsy state (CNN + EAR combined)
        self.drowsy_start = None
        self.is_drowsy    = False

        # Pure EAR drowsy path (eyes closed timer — independent of CNN)
        self.ear_closed_start = None  # when eyes first closed this streak
        self.ear_drowsy       = False  # fired by EAR alone

        # Inattentive state
        self.inatt_start = None
        self.is_inatt    = False

        self.ear_history = deque(maxlen=6)
        self.status      = "ACTIVE"
        self.cnn_prob    = 0.0

    def reset_drowsy(self):
        self.drowsy_start = None
        self.is_drowsy    = False

    def reset_ear_drowsy(self):
        self.ear_closed_start = None
        self.ear_drowsy       = False

    def reset_inatt(self):
        self.inatt_start = None
        self.is_inatt    = False


# ==============================================================
# MAIN DETECTOR CLASS
# ==============================================================
class DrowsinessDetector:
    """
    Real-time multi-face drowsiness and attention monitoring.

    Two drowsy paths:
      Path A (EAR-only)  : Eyes stay closed ≥ DROWSY_TIME_SEC → DROWSY
                           Works even when CNN buffer is not full yet.
      Path B (CNN+EAR)   : CNN prob ≥ thresh AND eyes closed, sustained
                           OR CNN instant threshold

    Both paths fire the same DROWSY alert and beep.
    """

    def __init__(
        self,
        model_path      = "best_model_v2.pth",
        landmarker_path = "face_landmarker.task",
        max_faces       = 3
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Detector running on: {self.device}")

        self.model = CNN_LSTM().to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        print(f"[INFO] Model loaded: {model_path}")

        options = FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=landmarker_path),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=max_faces,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            running_mode=mp_vision.RunningMode.IMAGE
        )
        self.landmarker  = FaceLandmarker.create_from_options(options)
        self.max_faces   = max_faces
        self.face_states = {}
        print(f"[INFO] MediaPipe FaceLandmarker ready (max faces: {max_faces})")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self._ear_baseline_samples = {}
        self._ear_thresholds       = {}

        # ============================================================
        # THRESHOLDS
        # ============================================================
        self.DROWSY_CNN_THRESH    = 0.78   # CNN prob to start CNN drowsy timer
        self.DROWSY_CNN_INSTANT   = 0.92   # CNN instant alert (no timer needed)
        self.DROWSY_TIME_SEC      = 4.5    # Sustained closed-eye duration → DROWSY

        self.EAR_CLOSED_FLOOR     = 0.15   # Lower floor — glasses reduce EAR naturally
        self.EAR_FORCED_THRESHOLD = 0.07   # Sudden forced-close threshold
        self.EAR_CLOSED_RATIO     = 0.68   # 62% of baseline = closed (was 72%, glasses pehnte ho)

        # How many open-eye samples to collect before calibrating threshold
        # 60 samples = ~2 sec at 30fps — enough to get stable baseline
        self.EAR_BASELINE_SAMPLES = 60

        # Minimum EAR to count as "open eye" during baseline collection
        # Lowered to 0.17 so glasses users' frames are not skipped
        self.EAR_BASELINE_MIN     = 0.17

        self.YAW_THRESH           = 0.40   # Normalized head left-right turn
        self.PITCH_THRESH         = 0.30   # Normalized head down tilt
        self.GAZE_THRESH          = 25.0   # Iris offset (px) for sideways gaze

        self.INATTENTIVE_SEC      = 15.0   # Sustained away-look → NOT ATTENTIVE

        self.BEEP_COOLDOWN_SEC    = 3.0
        self._last_beep_time      = 0.0

    # ----------------------------------------------------------
    def _beep(self):
        now = time.time()
        if now - self._last_beep_time >= self.BEEP_COOLDOWN_SEC:
            if _WINSOUND:
                try:
                    winsound.Beep(1000, 500)
                except Exception:
                    pass
            self._last_beep_time = now

    # ----------------------------------------------------------
    def _update_ear_baseline(self, face_id, ear_val):
        if face_id not in self._ear_baseline_samples:
            self._ear_baseline_samples[face_id] = []
        samples = self._ear_baseline_samples[face_id]

        # Collect open-eye samples — lower min threshold handles glasses users
        if len(samples) < self.EAR_BASELINE_SAMPLES and ear_val > self.EAR_BASELINE_MIN:
            samples.append(ear_val)

        if len(samples) == self.EAR_BASELINE_SAMPLES and face_id not in self._ear_thresholds:
            baseline = float(np.mean(samples))

            # If baseline itself is very low (e.g. thick glasses), be extra lenient
            # Use 58% ratio instead of 62% to avoid false closed detection
            ratio = self.EAR_CLOSED_RATIO if baseline > 0.22 else 0.58

            threshold = max(self.EAR_CLOSED_FLOOR, baseline * ratio)
            self._ear_thresholds[face_id] = threshold
            print(f"[INFO] Face {face_id} EAR baseline={baseline:.3f}, "
                  f"ratio={ratio}, closed threshold={threshold:.3f}")

    def _eyes_closed(self, face_id, ear_val):
        threshold = self._ear_thresholds.get(face_id, self.EAR_CLOSED_FLOOR)
        return ear_val < threshold

    def _forced_close(self, face_id, ear_val):
        state = self.face_states.get(face_id)
        if state is None or len(state.ear_history) < 3:
            return False
        recent_avg = float(np.mean(list(state.ear_history)[-3:]))
        return ear_val < self.EAR_FORCED_THRESHOLD and recent_avg > 0.22

    # ----------------------------------------------------------
    def process_frame(self, frame: np.ndarray):
        """
        Process a single BGR camera frame with low-light enhancement.
        Returns: (annotated_frame, face_states_dict)
        """
        h, w = frame.shape[:2]

        # --- Low-light enhancement before MediaPipe ---
        enhanced = enhance_low_light(frame)

        rgb    = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_img)

        output = frame.copy()  # Draw on original (not enhanced) for natural look
        now    = time.time()
        active = set()

        for idx, face_lms in enumerate(result.face_landmarks):
            if idx >= self.max_faces:
                break

            face_id = idx
            active.add(face_id)

            if face_id not in self.face_states:
                self.face_states[face_id] = FaceState(face_id)
            state = self.face_states[face_id]

            # ---- Geometry ----
            ear_l   = compute_ear(face_lms, LEFT_EYE,  w, h)
            ear_r   = compute_ear(face_lms, RIGHT_EYE, w, h)
            ear_avg = (ear_l + ear_r) / 2.0
            state.ear_history.append(ear_avg)
            self._update_ear_baseline(face_id, ear_avg)

            yaw, pitch = compute_head_pose(face_lms, w, h)
            try:
                gaze = compute_gaze_offset(face_lms, w, h)
            except Exception:
                gaze = 0.0

            # ---- CNN-LSTM inference ----
            crop, bbox = crop_face(enhanced, face_lms, w, h, pad=0.20)
            x1, y1, x2, y2 = bbox
            cnn_prob = 0.0

            if crop is not None:
                try:
                    img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    img_t   = self.transform(img_rgb)
                    state.seq_buffer.append(img_t)

                    if len(state.seq_buffer) == 3:
                        seq = torch.stack(list(state.seq_buffer)).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            logits   = self.model(seq)
                            cnn_prob = torch.softmax(logits, dim=1)[0][1].item()
                        state.prob_buffer.append(cnn_prob)
                        cnn_prob = float(np.mean(state.prob_buffer))
                        state.cnn_prob = cnn_prob
                except Exception:
                    pass

            # ============================================================
            # DECISION LOGIC
            # ============================================================

            forced = self._forced_close(face_id, ear_avg)
            closed = self._eyes_closed(face_id, ear_avg) and not forced

            # --- PATH A: Pure EAR drowsy (eyes closed timer) ---
            # Works from frame 1, does NOT need CNN buffer to be full
            if closed and not forced:
                if state.ear_closed_start is None:
                    state.ear_closed_start = now
                ear_elapsed = now - state.ear_closed_start
                if ear_elapsed >= self.DROWSY_TIME_SEC:
                    state.ear_drowsy = True
            else:
                state.reset_ear_drowsy()

            # --- PATH B: CNN + EAR drowsy ---
            drowsy_cnn_signal = (
                (cnn_prob >= self.DROWSY_CNN_THRESH and closed and ear_avg < 0.20) or
                (cnn_prob >= self.DROWSY_CNN_INSTANT and ear_avg < 0.20)
            ) and not forced

            if drowsy_cnn_signal:
                if state.drowsy_start is None:
                    state.drowsy_start = now
                if (now - state.drowsy_start) >= self.DROWSY_TIME_SEC:
                    state.is_drowsy = True
            else:
                state.reset_drowsy()

            # Combined drowsy = either path fires
            is_drowsy_final = state.ear_drowsy or (
                state.is_drowsy and state.ear_closed_start is not None
            )

            # --- NOT ATTENTIVE ---
            # Head turned OR gaze away, eyes can be open OR closed
            looking_away = (
                (abs(yaw) > self.YAW_THRESH and pitch > 0.15) or
                (gaze > self.GAZE_THRESH and abs(yaw) > 0.20)
            )

            if looking_away and not is_drowsy_final and not forced:
                if state.inatt_start is None:
                    state.inatt_start = now
                if (now - state.inatt_start) >= self.INATTENTIVE_SEC:
                    state.is_inatt = True
            else:
                state.reset_inatt()

            # --- Final status ---
            if forced:
                state.status = "EYES FORCED SHUT"
                box_color    = (0, 165, 255)
            elif is_drowsy_final:
                state.status = "DROWSY"
                box_color    = (0, 0, 255)
                self._beep()
            elif state.is_inatt:
                state.status = "NOT ATTENTIVE"
                box_color    = (0, 200, 255)
                self._beep()
            else:
                state.status = "ACTIVE"
                box_color    = (0, 220, 0)

            # ============================================================
            # DRAWING
            # ============================================================
            cv2.rectangle(output, (x1, y1), (x2, y2), box_color, 2)

            label_y = y1 - 12 if y1 > 40 else y2 + 28
            cv2.putText(output, f"Face {face_id + 1}: {state.status}",
                        (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2)

            bar_x  = x1
            bar_y  = y2 + 6
            bar_w  = max(1, x2 - x1)
            filled = int(bar_w * cnn_prob)
            cv2.rectangle(output, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8),
                          (60, 60, 60), -1)
            cv2.rectangle(output, (bar_x, bar_y), (bar_x + filled, bar_y + 8),
                          box_color, -1)
            cv2.putText(output, f"CNN: {cnn_prob:.2f}",
                        (bar_x, bar_y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

            # EAR closed timer (Path A)
            if state.ear_closed_start is not None:
                ear_el = now - state.ear_closed_start
                cv2.putText(output,
                            f"Eyes closed: {ear_el:.1f}s / {self.DROWSY_TIME_SEC}s",
                            (bar_x, bar_y + 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 160, 255), 1)

            # CNN drowsy timer (Path B)
            elif state.drowsy_start is not None:
                elapsed = now - state.drowsy_start
                cv2.putText(output,
                            f"Drowsy timer: {elapsed:.1f}s / {self.DROWSY_TIME_SEC}s",
                            (bar_x, bar_y + 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 180, 255), 1)

            if state.inatt_start is not None:
                elapsed = now - state.inatt_start
                cv2.putText(output,
                            f"Inattentive: {elapsed:.1f}s / {self.INATTENTIVE_SEC}s",
                            (bar_x, bar_y + 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 220, 180), 1)

            cv2.putText(output,
                        f"EAR:{ear_avg:.2f}(thr:{self._ear_thresholds.get(face_id, self.EAR_CLOSED_FLOOR):.2f})  YAW:{yaw:.2f}  PITCH:{pitch:.2f}  GAZE:{gaze:.1f}",
                        (bar_x, bar_y + 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)

        # Remove stale face states
        for fid in list(self.face_states.keys()):
            if fid not in active:
                del self.face_states[fid]

        # Summary bar
        n_drowsy = sum(1 for s in self.face_states.values() if s.status == "DROWSY")
        n_inatt  = sum(1 for s in self.face_states.values() if s.status == "NOT ATTENTIVE")
        n_forced = sum(1 for s in self.face_states.values() if s.status == "EYES FORCED SHUT")
        n_active = len(active)

        # Low light indicator
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        light_tag  = "  🌙 LOW LIGHT" if brightness < 80 else ""

        summary = (f"Faces: {n_active}   "
                   f"Active: {n_active - n_drowsy - n_inatt - n_forced}   "
                   f"Drowsy: {n_drowsy}   "
                   f"Inattentive: {n_inatt}   "
                   f"Forced Shut: {n_forced}"
                   f"{light_tag}")
        cv2.rectangle(output, (0, 0), (w, 30), (20, 20, 20), -1)
        cv2.putText(output, summary, (10, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        return output, self.face_states

    # ----------------------------------------------------------
    def release(self):
        self.landmarker.close()
        print("[INFO] Detector released.")


# ==============================================================
# STANDALONE USAGE
# ==============================================================
if __name__ == "__main__":
    detector = DrowsinessDetector(
        model_path      = "best_model_v2.pth",
        landmarker_path = "face_landmarker.task",
        max_faces       = 3
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        exit()

    print("[INFO] Camera started. Press ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_frame, states = detector.process_frame(frame)
        cv2.imshow("Pilot Drowsiness Detection", output_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.release()