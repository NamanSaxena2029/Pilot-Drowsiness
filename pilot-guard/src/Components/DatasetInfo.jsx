export default function DatasetInfo() {
  return (
    <div className="page">
      <h1 className="page-title">📁 Dataset Information & Project Pipeline</h1>
      <p className="page-sub">
        Complete journey from raw NTHU dataset → RGB model → Low-Light fine-tuning →
        Adaptive real-time detection system.
      </p>
      <hr className="divider" />

      {/* ── OVERVIEW ── */}
      <h2 className="section-heading">📦 Dataset Overview</h2>
      <div className="grid-2">
        <div style={{ overflowX: "auto" }}>
          <table className="data-table">
            <thead><tr><th>Property</th><th>Details</th></tr></thead>
            <tbody>
              {[
                ["Name",                    "NTHU Drowsy Driver Detection Dataset"],
                ["Source",                  "National Tsing Hua University (NTHU), Taiwan"],
                ["Type",                    "Frame-level labeled driver video footage"],
                ["Recording",               "Simulated driving, controlled lighting"],
                ["Ethnicities",             "Asian, Caucasian, South Asian"],
              ].map(([k,v]) => <tr key={k}><td>{k}</td><td>{v}</td></tr>)}
            </tbody>
          </table>
        </div>
        <div style={{ overflowX: "auto" }}>
          <table className="data-table">
            <thead><tr><th>Condition</th><th>Included</th></tr></thead>
            <tbody>
              {[["No glasses","✅"],["Glasses","✅"],["Sunglasses","✅"],["Night / Low light","✅"]].map(([k,v]) => (
                <tr key={k}><td>{k}</td><td style={{color:"#4ade80"}}>{v}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <hr className="divider" />

      {/* ── IMAGE COUNTS ── */}
      <h2 className="section-heading">📊 Image Counts After Processing</h2>
      <div className="metrics-row">
        {[
          ["Total (cropped_mp)",  "65,756"],
          ["Drowsy Images",       "36,012"],
          ["Not Drowsy Images",   "29,744"],
        ].map(([l,v]) => (
          <div className="metric-card" key={l}>
            <div className="metric-label">{l}</div>
            <div className="metric-value" style={{ fontSize: "1.2rem" }}>{v}</div>
          </div>
        ))}
      </div>
      <p style={{ color: "#7eb8f7", fontWeight: 700, marginBottom: 8, fontSize: "0.82rem" }}>Low-Light Enhanced Dataset (cropped_mp_ll)</p>
      <div className="metrics-row">
        {[
          ["Total LL Images",   "66,521"],
          ["Drowsy (LL)",       "36,030 (100%)"],
          ["Not Drowsy (LL)",   "30,491 (100%)"],
        ].map(([l,v]) => (
          <div className="metric-card" key={l}>
            <div className="metric-label">{l}</div>
            <div className="metric-value" style={{ fontSize: "1.1rem", color: "#4ade80" }}>{v}</div>
          </div>
        ))}
      </div>
      <div className="alert alert-success">
        <strong>100% detection rate</strong> in both classes during low-light recrop — the
        CLAHE + IR fallback pipeline successfully detected faces in all images.
      </div>

      <hr className="divider" />

      {/* ── PIPELINE ── */}
      <h2 className="section-heading">🗺️ Complete Project Pipeline</h2>

      <p style={{ color: "#7eb8f7", fontWeight: 700, marginBottom: 6, fontSize: "0.88rem" }}>❌ Phase 1 — First Crop Attempt (Abandoned)</p>
      <div className="alert alert-error">
        <strong>Problem with YOLO-based crop:</strong> Used YOLOv8n + fixed center-crop
        (<code>h//4 : h*3//4</code>) which frequently cut off the eye region. The model
        learned face texture and head pose — <strong>not eye closure</strong>. Despite 94%
        test accuracy, closed eyes were not detected in real-time use.
        <br /><strong>Fix:</strong> Scrapped. Restarted with MediaPipe landmark-based crop.
      </div>

      <p style={{ color: "#7eb8f7", fontWeight: 700, marginBottom: 6, fontSize: "0.88rem" }}>☀️ Phase 2 — Non-IR (RGB) Pipeline</p>
      <div className="alert alert-success">
        <strong>Fix: MediaPipe FaceLandmarker with 478 landmarks</strong><br />
        Each image: detect face → compute tight bounding box from all landmarks → add 15%
        padding (full eye region always included) → save to <code>data/cropped_mp/</code>.
        CNN now learns eye-closure from correctly cropped faces.
      </div>
      <div className="code-block">{`MediaPipe Recrop Results (cropped_mp)
─────────────────────────────────────────
drowsy      : 36,012 / 36,030  saved   (18 no-face — extreme angles)
notdrowsy   : 29,744 / 30,491  saved   (747 no-face — blurry/partial)

Output: data/cropped_mp/`}</div>

      <p style={{ color: "#7eb8f7", fontWeight: 700, marginBottom: 6, fontSize: "0.88rem" }}>🌙 Phase 3 — Low-Light Enhanced Recrop</p>
      <div className="alert alert-success">
        <strong>step1_recrop_lowlight.py</strong> — Enhanced pipeline for dark/IR frames:<br />
        1. Apply CLAHE (clipLimit=4.0, adaptive) &nbsp; 2. Brightness boost alpha=1.4, beta=20 &nbsp;
        3. Try MediaPipe on enhanced frame &nbsp; 4. <strong>Fallback:</strong> IR simulation
        (equalizeHist) if face not found &nbsp; 5. Save to <code>data/cropped_mp_ll/</code>
        <br /><strong>Result: 100% detection rate on both classes.</strong>
      </div>
      <div className="code-block">{`Low-Light Recrop Results (cropped_mp_ll)
─────────────────────────────────────────
drowsy      : 36,030 / 36,030  saved   (100% ✅ — CLAHE + IR fallback)
notdrowsy   : 30,491 / 30,491  saved   (100% ✅ — CLAHE + IR fallback)

Speed       : ~62.84 it/s (drowsy), ~66.60 it/s (notdrowsy)
Total time  : ~17 minutes

Output: data/cropped_mp_ll/`}</div>

      <p style={{ color: "#7eb8f7", fontWeight: 700, marginBottom: 6, fontSize: "0.88rem" }}>Train / Val / Test Split</p>
      <div className="code-block">{`Strategy   : Stratified split (class balance preserved in every subset)
Train      : 80%  →  ~53,216 sequences
Validation : 10%  →   ~6,652 sequences
Test       : 10%  →   ~6,573 sequences

Sequence length : 3 consecutive frames per sample
Sorted by filename to maintain temporal ordering within clips`}</div>

      <hr className="divider" />

      {/* ── LOW LIGHT AUGMENTATION ── */}
      <h2 className="section-heading">🌙 Low-Light Augmentation Pipeline</h2>
      <div className="alert alert-info">
        <strong>During training (transform augmentations):</strong><br />
        GaussianBlur(3) — IR noise simulation &nbsp;|&nbsp;
        ColorJitter(0.3, 0.3) — lighting variation &nbsp;|&nbsp;
        RandomHorizontalFlip + RandomRotation(8°)<br /><br />
        <strong>During inference (runtime enhancement):</strong><br />
        CLAHE on L-channel (LAB space) → hard brightness boost (alpha=2.0, beta=40) →
        IR fallback (equalizeHist) → crop-level histogram equalization
      </div>
      <div className="code-block">{`Low-Light Enhancement Flow (runtime):

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
  → CNN-LSTM inference`}</div>

      <hr className="divider" />

      {/* ── KNOWN ISSUES ── */}
      <h2 className="section-heading">⚠️ Known Challenges & Mitigations</h2>

      {[
        {
          title: "1. Naturally Small Eyes (East Asian subjects)",
          cls: "alert-info",
          text: `NTHU includes East Asian subjects whose open-eye EAR is naturally lower than Western-centric thresholds (e.g. 0.25).
Mitigation: Per-face EAR baseline calibration over first 60 frames. Closed-eye threshold = 68% of individual baseline (58% for very low baselines < 0.22).`
        },
        {
          title: "2. Class Imbalance",
          cls: "alert-info",
          text: "Drowsy images (36,030) outnumber Not Drowsy (30,491) by ~18%.\nMitigation: Weighted CrossEntropyLoss — Not Drowsy weight = 1.0, Drowsy weight = 1.2."
        },
        {
          title: "3. Temporal Mismatch in Sequences",
          cls: "alert-info",
          text: "Images sorted by filename (not true video timestamp) — consecutive triplets may span different clips.\nMitigation: LSTM is intentionally conservative (1 layer, 128 units) to avoid overfitting to spurious cross-clip correlations."
        },
        {
          title: "4. Simulated vs Real Drowsiness",
          cls: "alert-info",
          text: "Some drowsy frames were acted by subjects.\nMitigation: Real-time system requires sustained signal — 4 sec for drowsy, 12 sec for inattentive — filtering out momentary exaggerated poses."
        },
        {
          title: "5. Low-Light Domain Gap (Resolved ✅)",
          cls: "alert-warning",
          text: `Problem: Model trained on well-lit RGB data failed in dark conditions. MediaPipe couldn't detect faces → no crop → no CNN input.
Resolution: Created cropped_mp_ll with CLAHE + IR fallback → fine-tuned best_model_ll → runtime CLAHE enhancement → crop-level boost → adaptive model switching.
Result: Stable detection in low-light without IR camera hardware.`
        },
      ].map(({ title, cls, text }) => (
        <div key={title}>
          <p style={{ color: "#c8deff", fontWeight: 700, marginBottom: 4, fontSize: "0.85rem" }}>{title}</p>
          <div className={`alert ${cls}`} style={{ whiteSpace: "pre-line" }}>{text}</div>
        </div>
      ))}

      <hr className="divider" />

      {/* ── SCRIPTS REFERENCE ── */}
      <h2 className="section-heading">📄 Scripts Reference</h2>
      <div style={{ overflowX: "auto" }}>
        <table className="data-table">
          <thead><tr><th>Script</th><th>Purpose</th></tr></thead>
          <tbody>
            {[
              ["step1_recrop.py",           "MediaPipe face crop on raw NTHU → data/cropped_mp/"],
              ["step1_recrop_lowlight.py",  "CLAHE + IR-fallback crop → data/cropped_mp_ll/"],
              ["step2_retrain.py",          "Full CNN-LSTM training → best_model_v2.pth"],
              ["step2_finetune_ll.py",      "Fine-tune on low-light data → best_model_ll.pth"],
              ["eval_results.py",           "Confusion matrix + ROC curve for any saved model"],
              ["drowsy_detection.py",       "Core engine: MediaPipe + CNN-LSTM + EAR + alerts"],
              ["main.py",                   "FastAPI WebSocket backend (replaces app.py)"],
              ["pilot-guard/",              "React frontend: Live + Model Details + Dataset Info"],
            ].map(([s,p]) => <tr key={s}><td style={{ fontFamily: "var(--font-mono)", color: "#7eb8f7" }}>{s}</td><td>{p}</td></tr>)}
          </tbody>
        </table>
      </div>

      <hr className="divider" />

      {/* ── DEPLOYMENT NOTES ── */}
      <h2 className="section-heading">💡 Real-World Deployment Notes</h2>
      <div className="alert alert-warning">
        Performance depends significantly on lighting. Extreme darkness → detection fails.
        Low light (face silhouette visible) → low-light model handles well. Overexposure → feature loss.
      </div>
      <div className="grid-2">
        <div className="alert alert-info">
          <strong>Software improvements:</strong><br />
          Tune camera exposure per environment &nbsp;|&nbsp; Add YOLO face fallback for extreme angles &nbsp;|&nbsp;
          Collect in-cockpit labeled data for domain fine-tuning
        </div>
        <div className="alert alert-info">
          <strong>Hardware improvements:</strong><br />
          IR camera (~₹1500–5000) for true night-vision &nbsp;|&nbsp;
          NVIDIA Jetson or similar for edge inference &nbsp;|&nbsp;
          Dual-camera setup (IR + RGB) &nbsp;|&nbsp;
          Helmet-mounted camera for stable framing
        </div>
      </div>
    </div>
  );
}