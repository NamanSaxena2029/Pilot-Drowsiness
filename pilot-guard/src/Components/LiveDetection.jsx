import { useState, useRef } from "react";
import { useDetector } from "../hooks/useDetector";

const STATUS_STYLE = {
  DROWSY:             { bg: "#2d0d0d", border: "#991b1b", text: "#f87171", icon: "🔴", cls: "drowsy" },
  "NOT ATTENTIVE":    { bg: "#2d1a0d", border: "#9a3412", text: "#fb923c", icon: "🟠", cls: "" },
  "EYES FORCED SHUT": { bg: "#2d260d", border: "#92400e", text: "#fbbf24", icon: "🟡", cls: "" },
  ACTIVE:             { bg: "#0d2d18", border: "#166534", text: "#4ade80", icon: "🟢", cls: "" },
};

function Metric({ label, value, highlight }) {
  return (
    <div className="metric-card" style={highlight ? { borderColor: "#991b1b" } : {}}>
      <div className="metric-label">{label}</div>
      <div className="metric-value" style={highlight && value > 0 ? { color: "#f87171" } : {}}>
        {value}
      </div>
    </div>
  );
}

export default function LiveDetection() {
  const [config, setConfig] = useState({
    drowsy_time : 4.0,
    inatt_time  : 12.0,
    max_faces   : 3,
    cnn_thresh  : 0.70,
  });

  const camRef    = useRef(null);
  const canvasRef = useRef(null);

  const { running, frame, faceStates, brightness, mode, error, alert, start, stop } =
    useDetector(config);

  const counts = Object.values(faceStates).reduce((acc, s) => {
    acc[s.status] = (acc[s.status] || 0) + 1;
    return acc;
  }, {});

  const handleStart = () => start(camRef, canvasRef);
  const handleStop  = () => stop(camRef);

  return (
    <div className="detection-layout">

      {/* Hidden elements for camera capture */}
      <video   ref={camRef}    style={{ display: "none" }} muted playsInline />
      <canvas  ref={canvasRef} style={{ display: "none" }} />

      {/* ── VIDEO PANEL ── */}
      <div className="video-panel">

        <h2 className="page-title" style={{ marginBottom: 8 }}>✈️ Real-Time Monitor</h2>
        <p className="page-sub" style={{ marginBottom: 16 }}>
          Monitors up to <strong>3 faces</strong> simultaneously using <strong>CNN-LSTM</strong> + <strong>MediaPipe geometry</strong>.
          Automatically switches between <strong>normal</strong> and <strong>low-light</strong> models based on ambient brightness.
        </p>

        {/* Status legend */}
        <div className="status-legend">
          <span className="badge-active">🟢 ACTIVE</span>
          <span className="badge-drowsy">🔴 DROWSY</span>
          <span className="badge-inattentive">🟠 NOT ATTENTIVE</span>
          <span className="badge-forced">🟡 FORCED SHUT</span>
        </div>

        {/* Video feed */}
        <div className="video-container">
          {/* HUD corners */}
          <div className="hud-tl hud-corner" />
          <div className="hud-tr hud-corner" />
          <div className="hud-bl hud-corner" />
          <div className="hud-br hud-corner" />

          {running && <div className="scan-line" />}

          {frame ? (
            <img src={frame} alt="Detection feed" className="video-feed" />
          ) : (
            <div className="video-placeholder">
              <div className="cam-icon">📷</div>
              <p>Camera inactive — press Start to begin</p>
              {error && (
                <p style={{ color: "#f87171", fontSize: "0.78rem", marginTop: 6, textAlign: "center", padding: "0 20px" }}>
                  ⚠️ {error}
                </p>
              )}
            </div>
          )}
        </div>

        {/* Mode bar */}
        {brightness !== null && (
          <div className="mode-bar">
            <span>{mode === "low_light" ? "🌙" : "☀️"}</span>
            <strong style={{ color: "#c8deff" }}>
              {mode === "low_light" ? "Low-Light Model" : "Normal Model"}
            </strong>
            <span style={{ color: "#4a6a8a" }}>|</span>
            <span>Brightness: <code>{brightness}</code></span>
            <span style={{ color: "#4a6a8a" }}>|</span>
            <span>Threshold: <code>{mode === "low_light" ? "< 70" : "> 70"}</code></span>
          </div>
        )}

        {/* Drowsy alert overlay */}
        {alert && (
          <div style={{
            marginTop: 12,
            background: alert === "DROWSY" ? "#2d0d0d" : "#2d1a0d",
            border: `1px solid ${alert === "DROWSY" ? "#991b1b" : "#9a3412"}`,
            borderRadius: 8,
            padding: "12px 18px",
            color: alert === "DROWSY" ? "#f87171" : "#fb923c",
            fontWeight: 700,
            fontSize: "1rem",
            display: "flex",
            alignItems: "center",
            gap: 10,
            animation: "pulse-border 1s ease"
          }}>
            {alert === "DROWSY" ? "🔴 DROWSINESS DETECTED — WAKE UP!" : "🟠 PILOT NOT ATTENTIVE!"}
          </div>
        )}

        {/* Metrics row */}
        <div className="metrics-row" style={{ marginTop: 16 }}>
          <Metric label="Faces Detected"  value={Object.keys(faceStates).length} />
          <Metric label="Drowsy"          value={counts["DROWSY"] || 0}          highlight />
          <Metric label="Not Attentive"   value={counts["NOT ATTENTIVE"] || 0} />
          <Metric label="Forced Shut"     value={counts["EYES FORCED SHUT"] || 0} />
        </div>
      </div>

      {/* ── CONTROL PANEL ── */}
      <div className="ctrl-panel">

        {/* Settings */}
        <div className="card">
          <div className="ctrl-section-title">⚙️ Settings</div>

          <div className="slider-label">
            Drowsy threshold <span>{config.drowsy_time}s</span>
          </div>
          <input type="range" min={2} max={8} step={0.5}
            value={config.drowsy_time} disabled={running}
            onChange={e => setConfig(c => ({ ...c, drowsy_time: +e.target.value }))} />

          <div className="slider-label">
            Inattentive threshold <span>{config.inatt_time}s</span>
          </div>
          <input type="range" min={8} max={20} step={1}
            value={config.inatt_time} disabled={running}
            onChange={e => setConfig(c => ({ ...c, inatt_time: +e.target.value }))} />

          <div className="slider-label">
            CNN threshold <span>{config.cnn_thresh}</span>
          </div>
          <input type="range" min={0.4} max={0.85} step={0.05}
            value={config.cnn_thresh} disabled={running}
            onChange={e => setConfig(c => ({ ...c, cnn_thresh: +e.target.value }))} />

          <div className="slider-label" style={{ marginBottom: 6 }}>Max faces</div>
          <select value={config.max_faces} disabled={running}
            onChange={e => setConfig(c => ({ ...c, max_faces: +e.target.value }))}>
            <option value={1}>1 face (fastest)</option>
            <option value={2}>2 faces</option>
            <option value={3}>3 faces</option>
          </select>

          <p style={{ fontSize: "0.68rem", color: "#4a6a8a", marginTop: -8 }}>
            Settings locked while running. Stop first to change.
          </p>
        </div>

        {/* Start / Stop */}
        <div className="btn-row">
          <button className="btn btn-start" onClick={handleStart} disabled={running}>
            ▶ Start
          </button>
          <button className="btn btn-stop" onClick={handleStop} disabled={!running}>
            ⏹ Stop
          </button>
        </div>

        {/* Face status cards */}
        <div className="card">
          <div className="ctrl-section-title">🟢 Face Status</div>

          {Object.keys(faceStates).length === 0 ? (
            <p style={{ color: "#4a6a8a", fontSize: "0.78rem" }}>No faces detected</p>
          ) : (
            Object.entries(faceStates).map(([fid, s]) => {
              const st = STATUS_STYLE[s.status] || STATUS_STYLE.ACTIVE;
              return (
                <div key={fid}
                  className={`face-card ${st.cls}`}
                  style={{ background: st.bg, border: `1px solid ${st.border}` }}>
                  <div className="face-card-title" style={{ color: st.text }}>
                    {st.icon} Face {+fid + 1}: {s.status}
                  </div>
                  <div className="face-card-sub" style={{ color: st.text }}>
                    CNN: {s.cnn_prob}
                    {s.ear_timer !== null && s.ear_timer !== undefined &&
                      ` | Eyes: ${s.ear_timer}s`}
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Tips */}
        <div className="alert alert-info" style={{ fontSize: "0.72rem" }}>
          💡 <strong>Tips:</strong> Ensure your face is well-lit and centred in frame.
          The system calibrates your eye baseline in the first ~2 seconds.
          Low-light model activates automatically below brightness 70.
        </div>
      </div>
    </div>
  );
}