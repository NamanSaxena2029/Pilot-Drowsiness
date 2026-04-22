import { useState, useRef, useCallback, useEffect } from "react";

export function useDetector(config) {
  const [running, setRunning]       = useState(false);
  const [frame, setFrame]           = useState(null);
  const [faceStates, setFaceStates] = useState({});
  const [brightness, setBrightness] = useState(null);
  const [mode, setMode]             = useState("normal");
  const [error, setError]           = useState(null);
  const [alert, setAlert]           = useState(null);

  const wsRef     = useRef(null);
  const loopRef   = useRef(null);
  const alertRef  = useRef(null);

  // Audio context for beep alerts (cross-platform, replaces winsound)
  const beep = useCallback((freq = 880, duration = 400) => {
    try {
      const ctx  = new (window.AudioContext || window.webkitAudioContext)();
      const osc  = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.frequency.value = freq;
      osc.type = "sine";
      gain.gain.setValueAtTime(0.3, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration / 1000);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + duration / 1000);
    } catch (e) {
      console.warn("Audio beep failed:", e);
    }
  }, []);

  const start = useCallback(async (camRef, canvasRef) => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" }
      });
      camRef.current.srcObject = stream;
      await camRef.current.play();

      const ws = new WebSocket("ws://localhost:8000/ws/detect");
      wsRef.current = ws;

      ws.onopen = () => {
        // Send config as first message
        ws.send(JSON.stringify({
          drowsy_time : config.drowsy_time,
          inatt_time  : config.inatt_time,
          max_faces   : config.max_faces,
          cnn_thresh  : config.cnn_thresh,
          model_path  : "best_model_v2.pth",
        }));
        setRunning(true);

        // Frame sending loop
        const tick = () => {
          if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
          const canvas = canvasRef.current;
          const video  = camRef.current;
          if (!canvas || !video || video.readyState < 2) {
            loopRef.current = setTimeout(tick, 100);
            return;
          }
          const ctx = canvas.getContext("2d");
          canvas.width  = 640;
          canvas.height = 480;
          ctx.drawImage(video, 0, 0, 640, 480);
          const b64 = canvas.toDataURL("image/jpeg", 0.75);
          try {
            ws.send(JSON.stringify({ frame: b64 }));
          } catch (e) { /* ws may have closed */ }
          loopRef.current = setTimeout(tick, 120); // ~8fps — plenty for detection
        };
        tick();
      };

      ws.onmessage = (e) => {
        const data = JSON.parse(e.data);
        setFrame(data.frame);
        setFaceStates(data.states || {});
        setBrightness(data.brightness);
        setMode(data.mode || "normal");

        // Trigger browser alert + beep if backend signals drowsy/inattentive
        if (data.alert) {
          const now = Date.now();
          if (!alertRef.current || now - alertRef.current > 3000) {
            alertRef.current = now;
            setAlert(data.alert);
            beep(data.alert === "DROWSY" ? 880 : 660);
            setTimeout(() => setAlert(null), 2500);
          }
        }
      };

      ws.onerror = () => {
        setError("WebSocket error — is the backend running on port 8000?");
        setRunning(false);
      };

      ws.onclose = () => {
        setRunning(false);
        clearTimeout(loopRef.current);
      };

    } catch (err) {
      if (err.name === "NotAllowedError") {
        setError("Camera permission denied. Please allow camera access.");
      } else if (err.name === "NotFoundError") {
        setError("No camera found. Please connect a webcam.");
      } else {
        setError(`Camera error: ${err.message}`);
      }
    }
  }, [config, beep]);

  const stop = useCallback((camRef) => {
    clearTimeout(loopRef.current);
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (camRef?.current?.srcObject) {
      camRef.current.srcObject.getTracks().forEach(t => t.stop());
      camRef.current.srcObject = null;
    }
    setRunning(false);
    setFrame(null);
    setFaceStates({});
    setBrightness(null);
    setAlert(null);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      clearTimeout(loopRef.current);
      wsRef.current?.close();
    };
  }, []);

  return { running, frame, faceStates, brightness, mode, error, alert, start, stop };
}