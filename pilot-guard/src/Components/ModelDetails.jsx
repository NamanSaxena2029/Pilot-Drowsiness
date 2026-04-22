import { useState } from "react";

function Table({ headers, rows }) {
  return (
    <div style={{ overflowX: "auto", marginBottom: 16 }}>
      <table className="data-table">
        <thead>
          <tr>{headers.map(h => <th key={h}>{h}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>{row.map((cell, j) => <td key={j}>{cell}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function ModelDetails() {
  const [trainTab, setTrainTab] = useState(0);
  const [perfTab,  setPerfTab]  = useState(0);

  return (
    <div className="page">
      <h1 className="page-title">📊 Model Architecture & Performance</h1>
      <p className="page-sub">
        This system uses a <strong>CNN-LSTM</strong> architecture where MobileNetV2 extracts spatial
        features per frame and an LSTM captures temporal patterns across a 3-frame sequence. The model
        was originally trained on RGB data (<code style={{color:"#7eb8f7"}}>best_model_v2</code>) and
        then <strong>fine-tuned on low-light enhanced data</strong> (<code style={{color:"#7eb8f7"}}>best_model_ll</code>) for
        robust real-world cockpit performance.
      </p>
      <hr className="divider" />

      {/* ── ARCHITECTURE ── */}
      <h2 className="section-heading">🧠 Architecture</h2>
      <div className="grid-2">
        <div>
          <p style={{ color: "#7eb8f7", fontWeight: 700, marginBottom: 8, fontSize: "0.82rem" }}>CNN Backbone</p>
          <Table
            headers={["Component", "Details"]}
            rows={[
              ["Backbone",          "MobileNetV2"],
              ["Pretrained On",     "ImageNet (ILSVRC)"],
              ["Frozen Layers",     "First 14 feature blocks"],
              ["Trainable Layers",  "Last 4 blocks + replaced classifier"],
              ["Feature Output",    "1280-dim vector per frame"],
            ]}
          />
        </div>
        <div>
          <p style={{ color: "#7eb8f7", fontWeight: 700, marginBottom: 8, fontSize: "0.82rem" }}>LSTM + Classifier</p>
          <Table
            headers={["Component", "Details"]}
            rows={[
              ["Input",             "3 × 1280 = 3840 (sequence)"],
              ["LSTM Hidden Size",  "128 units"],
              ["LSTM Layers",       "1"],
              ["Output Classes",    "2 (Not Drowsy / Drowsy)"],
              ["Final Activation",  "Softmax (probability)"],
            ]}
          />
        </div>
      </div>

      <p style={{ color: "#7eb8f7", fontWeight: 700, marginBottom: 6, fontSize: "0.82rem" }}>Forward Pass</p>
      <div className="code-block">{`Input Shape : (Batch, 3 frames, 3 channels, 160px, 160px)
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
Softmax                            →  P(Not Drowsy), P(Drowsy)`}</div>

      <hr className="divider" />

      {/* ── TRAINING CONFIG ── */}
      <h2 className="section-heading">🏋️ Training & Fine-Tuning Configuration</h2>

      <div className="tabs">
        {["📦 Base Model (best_model_v2)", "🌙 Fine-Tuned LL Model (best_model_ll)"].map((t, i) => (
          <button key={t} className={`tab-btn ${trainTab === i ? "active" : ""}`} onClick={() => setTrainTab(i)}>{t}</button>
        ))}
      </div>

      {trainTab === 0 && (
        <div>
          <div className="grid-2">
            <div>
              <p style={{ color: "#7eb8f7", fontWeight: 700, marginBottom: 8, fontSize: "0.82rem" }}>Optimizer & Learning Rate</p>
              <Table
                headers={["Parameter Group", "Learning Rate", "Optimizer"]}
                rows={[
                  ["CNN last 4 blocks", "1e-5 (fine-tune)", "Adam"],
                  ["LSTM",              "1e-4",             "Adam"],
                  ["FC layer",          "1e-4",             "Adam"],
                ]}
              />
            </div>
            <div>
              <p style={{ color: "#7eb8f7", fontWeight: 700, marginBottom: 8, fontSize: "0.82rem" }}>Training Settings</p>
              <Table
                headers={["Setting", "Value"]}
                rows={[
                  ["Batch Size",      "8"],
                  ["Loss Function",   "CrossEntropyLoss"],
                  ["Class Weights",   "NotDrowsy=1.0 / Drowsy=1.2"],
                  ["LR Scheduler",    "StepLR (step=3, γ=0.1)"],
                  ["Early Stopping",  "Patience = 3"],
                  ["Max Epochs",      "15"],
                ]}
              />
            </div>
          </div>
        </div>
      )}

      {trainTab === 1 && (
        <div>
          <div className="alert alert-success">
            <strong>Fine-tuning strategy:</strong> Loaded <code>best_model_v2.pth</code> weights →
            adapted to low-light enhanced dataset (<code>data/cropped_mp_ll/</code>) with lower
            learning rates to preserve learned features while adapting to IR-like visual patterns.
          </div>
          <div className="grid-2">
            <div>
              <p style={{ color: "#7eb8f7", fontWeight: 700, marginBottom: 8, fontSize: "0.82rem" }}>Fine-Tune Learning Rates (reduced)</p>
              <Table
                headers={["Parameter Group", "Learning Rate", "Optimizer"]}
                rows={[
                  ["CNN last 4 blocks", "5e-6 (very conservative)", "Adam"],
                  ["LSTM",              "5e-5",                     "Adam"],
                  ["FC layer",          "5e-5",                     "Adam"],
                ]}
              />
            </div>
            <div>
              <p style={{ color: "#7eb8f7", fontWeight: 700, marginBottom: 8, fontSize: "0.82rem" }}>Low-Light Augmentations Added</p>
              <Table
                headers={["Augmentation", "Purpose"]}
                rows={[
                  ["GaussianBlur(3)",          "IR noise simulation"],
                  ["ColorJitter(0.3, 0.3)",    "Lighting variation"],
                  ["CLAHE preprocessing",       "Low-light enhancement"],
                  ["Histogram equalization",    "IR-style normalization"],
                  ["Brightness/contrast boost", "Extreme dark recovery"],
                ]}
              />
            </div>
          </div>
          <div className="code-block">{`Fine-Tune Data  : data/cropped_mp_ll/   (low-light enhanced crops)
Epochs          : 10
Best Val Acc    : saved as best_model_ll.pth
Grayscale Fix   : cv2.COLOR_GRAY2RGB    (handles single-channel IR frames)`}</div>
        </div>
      )}

      <hr className="divider" />

      {/* ── PERFORMANCE ── */}
      <h2 className="section-heading">📈 Test Set Performance — Both Models</h2>

      <div className="tabs">
        {["☀️ best_model_v2 (RGB)", "🌙 best_model_ll (Low-Light)"].map((t, i) => (
          <button key={t} className={`tab-btn ${perfTab === i ? "active" : ""}`} onClick={() => setPerfTab(i)}>{t}</button>
        ))}
      </div>

      {perfTab === 0 && (
        <div>
          <div className="alert alert-info">
            Results on the held-out test set using <code>best_model_v2.pth</code>. Trained on standard
            RGB MediaPipe-cropped data. Strong in normal lighting — used when brightness &gt; 70.
          </div>
          <div className="metrics-row">
            {[["Accuracy","95%"],["Drowsy Precision","95%"],["Drowsy Recall","96%"],["Drowsy F1","95%"],["AUC (ROC)","~0.98+"]].map(([l,v]) => (
              <div className="metric-card" key={l}>
                <div className="metric-label">{l}</div>
                <div className="metric-value" style={{ fontSize: "1.3rem" }}>{v}</div>
              </div>
            ))}
          </div>
          <Table
            headers={["Class", "Precision", "Recall", "F1-Score", "Support"]}
            rows={[
              ["Not Drowsy",   "0.95", "0.94", "0.94", "2973"],
              ["Drowsy",       "0.95", "0.96", "0.95", "3600"],
              ["Macro Avg",    "0.95", "0.95", "0.95", "6573"],
              ["Weighted Avg", "0.95", "0.95", "0.95", "6573"],
            ]}
          />
          <Table
            headers={["Actual \\ Predicted", "→ Not Drowsy", "→ Drowsy"]}
            rows={[
              ["Not Drowsy (Actual)", "2794 ✅ True Not Drowsy", "179 ⚠️ False Alarm"],
              ["Drowsy (Actual)",     "144 ❌ Missed Drowsy",    "3456 ✅ True Drowsy"],
            ]}
          />
          <div className="alert alert-success">
            <strong>best_model_v2 is the stronger base model</strong> — 95% accuracy, fewer false
            negatives (144 vs 188). Used automatically when ambient brightness &gt; 70.
          </div>
        </div>
      )}

      {perfTab === 1 && (
        <div>
          <div className="alert alert-info">
            Results using <code>best_model_ll.pth</code>. Fine-tuned from best_model_v2 on
            low-light enhanced data. Optimized for dark / cockpit lighting. AUC = 0.9746.
          </div>
          <div className="metrics-row">
            {[["Accuracy","92%"],["Drowsy Precision","91%"],["Drowsy Recall","95%"],["Drowsy F1","93%"],["AUC (ROC)","0.9746"]].map(([l,v]) => (
              <div className="metric-card" key={l}>
                <div className="metric-label">{l}</div>
                <div className="metric-value" style={{ fontSize: "1.3rem" }}>{v}</div>
              </div>
            ))}
          </div>
          <Table
            headers={["Class", "Precision", "Recall", "F1-Score", "Support"]}
            rows={[
              ["Not Drowsy",   "0.93", "0.89", "0.91", "2973"],
              ["Drowsy",       "0.91", "0.95", "0.93", "3600"],
              ["Macro Avg",    "0.92", "0.92", "0.92", "6573"],
              ["Weighted Avg", "0.92", "0.92", "0.92", "6573"],
            ]}
          />
          <Table
            headers={["Actual \\ Predicted", "→ Not Drowsy", "→ Drowsy"]}
            rows={[
              ["Not Drowsy (Actual)", "2650 ✅ True Not Drowsy", "323 ⚠️ False Alarm"],
              ["Drowsy (Actual)",     "188 ❌ Missed Drowsy",    "3412 ✅ True Drowsy"],
            ]}
          />
          <div className="alert alert-success">
            <strong>Safety-first design:</strong> EAR-only Path A acts as fallback — sustained eye
            closure ≥ 4 sec fires DROWSY even if CNN confidence is low. This minimizes real-world
            impact of the 188 CNN false-negatives. Used when brightness &lt; 70.
          </div>
        </div>
      )}

      <hr className="divider" />

      {/* ── MODEL COMPARISON ── */}
      <h2 className="section-heading">📊 Model Comparison</h2>
      <Table
        headers={["Metric", "best_model_v2 ☀️", "best_model_ll 🌙"]}
        rows={[
          ["Overall Accuracy",   "95%",                          "92%"],
          ["Drowsy Precision",   "0.95",                         "0.91"],
          ["Drowsy Recall",      "0.96",                         "0.95"],
          ["Drowsy F1",          "0.95",                         "0.93"],
          ["Not Drowsy Recall",  "0.94",                         "0.89"],
          ["False Alarms",       "179",                          "323"],
          ["Missed Drowsy",      "144",                          "188"],
          ["AUC",                "~0.98+",                       "0.9746"],
          ["Best used in",       "Normal light (brightness > 70)", "Low light (brightness < 70)"],
        ]}
      />

      <hr className="divider" />

      {/* ── DETECTION LOGIC ── */}
      <h2 className="section-heading">🔍 Real-Time Detection Logic</h2>

      <Table
        headers={["Priority", "Status", "Trigger"]}
        rows={[
          ["1 (highest)", "EYES FORCED SHUT 🟡", "Current EAR < 0.08 AND previous 3-frame EAR > 0.22"],
          ["2",           "DROWSY 🔴",            "Path A: Eyes closed ≥ 4 sec (EAR-only) | Path B: CNN ≥ 60% + eyes closed sustained | Instant: CNN ≥ 82%"],
          ["3",           "NOT ATTENTIVE 🟠",     "Head yaw > 0.25 OR pitch > 0.18 OR gaze > 12px, sustained 12 sec"],
          ["4 (default)", "ACTIVE 🟢",            "None of the above"],
        ]}
      />

      <hr className="divider" />

      {/* ── THRESHOLDS ── */}
      <h2 className="section-heading">🎛️ Default Threshold Reference</h2>
      <Table
        headers={["Parameter", "Default Value", "Description"]}
        rows={[
          ["CNN Drowsy Threshold",        "0.55",              "Min CNN prob to start drowsy timer"],
          ["CNN Instant Alert Threshold", "0.82",              "Above this → alert fires instantly"],
          ["Drowsy Sustained Duration",   "4.0 sec",           "Signal must persist before DROWSY"],
          ["EAR Closed Floor",            "0.15",              "Absolute closed-eye threshold"],
          ["EAR Closed Ratio",            "68% of baseline",   "Calibrated per face"],
          ["EAR Forced Shut Threshold",   "0.07",              "Sudden drop = deliberate tight-close"],
          ["Head Yaw Threshold",          "0.40",              "Normalized left-right head turn"],
          ["Head Pitch Threshold",        "0.30",              "Normalized downward tilt"],
          ["Gaze Offset Threshold",       "25 px",             "Avg iris displacement from eye center"],
          ["Inattentive Duration",        "12.0 sec",          "Must persist before NOT ATTENTIVE"],
          ["Beep Cooldown",               "3.0 sec",           "Min gap between audio alerts"],
          ["Low-Light Switch Threshold",  "brightness < 70",   "Activates low-light model"],
        ]}
      />
    </div>
  );
}