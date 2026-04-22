import { useState } from "react";
import LiveDetection from "./Components/LiveDetection";
import ModelDetails from "./Components/ModelDetails";
import DatasetInfo from "./Components/DatasetInfo";
import "./App.css";

const NAV = [
  { icon: "🎥", label: "Live Detection" },
  { icon: "📊", label: "Model Details" },
  { icon: "📁", label: "Dataset Info" },
];

export default function App() {
  const [page, setPage] = useState(0);

  return (
    <div className="app-shell">
      {/* SIDEBAR */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="brand-icon">✈️</div>
          <div className="brand-title">PILOT GUARD</div>
          <div className="brand-sub">Drowsiness Detection System</div>
          <div className="brand-tags">
            <span className="tag tag-green">CNN-LSTM</span>
            <span className="tag tag-blue">MediaPipe</span>
            <span className="tag tag-orange">IR-Adapted</span>
          </div>
        </div>

        <nav className="sidebar-nav">
          {NAV.map((n, i) => (
            <button
              key={n.label}
              className={`nav-item ${page === i ? "active" : ""}`}
              onClick={() => setPage(i)}
            >
              <span className="nav-icon">{n.icon}</span>
              <span>{n.label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-info">
          <div className="info-row">🔬 Model: <span>CNN-LSTM v2.1</span></div>
          <div className="info-row">📦 Dataset: <span>NTHU-DDD</span></div>
          <div className="info-row">🌙 LL Model: <span className="green">Active</span></div>
          <div className="info-row">📈 AUC: <span>0.9746</span></div>
          <div className="version">v3.0 — React Edition</div>
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main className="main-content">
        {page === 0 && <LiveDetection />}
        {page === 1 && <ModelDetails />}
        {page === 2 && <DatasetInfo />}
      </main>
    </div>
  );
}