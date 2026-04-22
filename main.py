import asyncio, base64, cv2, numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from drowsy_detection import DrowsinessDetector
import json, time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/detect")
async def detect(websocket: WebSocket):
    await websocket.accept()

    # Config sent as first message from React
    config = await websocket.receive_json()

    detector = DrowsinessDetector(
        model_path      = config.get("model_path", "best_model_v2.pth"),
        landmarker_path = "face_landmarker.task",
        max_faces       = config.get("max_faces", 3)
    )
    detector.DROWSY_TIME_SEC   = config.get("drowsy_time", 4.0)
    detector.INATTENTIVE_SEC   = config.get("inatt_time", 12.0)
    detector.DROWSY_CNN_THRESH = config.get("cnn_thresh", 0.70)

    current_mode     = config.get("model_path", "best_model_v2.pth")
    last_switch_time = 0

    try:
        while True:
            # Receive frame as base64 from React
            data    = await websocket.receive_text()
            payload = json.loads(data)

            img_b64 = payload["frame"].split(",")[1]
            img_arr = np.frombuffer(base64.b64decode(img_b64), np.uint8)
            frame   = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # Adaptive model switching (same logic as your Streamlit app)
            gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(np.percentile(gray, 75))

            new_model = "best_model_ll.pth" if brightness < 70 else "best_model_v2.pth"
            if new_model != current_mode and (time.time() - last_switch_time > 2):
                detector = DrowsinessDetector(
                    model_path      = new_model,
                    landmarker_path = "face_landmarker.task",
                    max_faces       = config.get("max_faces", 3)
                )
                detector.DROWSY_TIME_SEC   = config.get("drowsy_time", 4.0)
                detector.INATTENTIVE_SEC   = config.get("inatt_time", 12.0)
                detector.DROWSY_CNN_THRESH = config.get("cnn_thresh", 0.70)
                current_mode     = new_model
                last_switch_time = time.time()

            # Run your existing engine
            output_frame, face_states = detector.process_frame(frame)

            # Encode output frame back to base64
            _, buf     = cv2.imencode(".jpg", output_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            out_b64    = base64.b64encode(buf).decode("utf-8")

            # Build face state payload
            states_payload = {
                fid: {
                    "status"   : s.status,
                    "cnn_prob" : round(s.cnn_prob, 3),
                    "ear_timer": round(time.time() - s.ear_closed_start, 1)
                               if s.ear_closed_start else None,
                }
                for fid, s in face_states.items()
            }

            await websocket.send_json({
                "frame"      : f"data:image/jpeg;base64,{out_b64}",
                "states"     : states_payload,
                "brightness" : round(brightness, 1),
                "mode"       : "low_light" if "ll" in current_mode else "normal",
            })

    except WebSocketDisconnect:
        detector.release()