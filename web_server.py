from __future__ import annotations

import argparse
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template_string, request
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "best.pt"  # Assumes user copies best YOLOv9 weights here
MODEL_LOCK = Lock()


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Garbage Bin Detector (YOLOv9)</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #07111c;
      --panel: rgba(10, 18, 31, 0.78);
      --panel-border: rgba(255, 255, 255, 0.12);
      --text: #eff6ff;
      --muted: #9fb2c9;
      --accent: #5eead4;
      --accent-2: #f59e0b;
      --danger: #fb7185;
      --shadow: 0 30px 80px rgba(0, 0, 0, 0.35);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(94, 234, 212, 0.2), transparent 28%),
        radial-gradient(circle at top right, rgba(245, 158, 11, 0.18), transparent 24%),
        linear-gradient(180deg, #05101a 0%, #0b1624 45%, #050a12 100%);
      color: var(--text);
    }

    .wrap {
      max-width: 1200px;
      margin: 0 auto;
      padding: 28px 18px 40px;
    }

    .hero {
      display: grid;
      gap: 14px;
      margin-bottom: 18px;
      animation: rise 0.7s ease both;
    }

    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      width: fit-content;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.07);
      color: var(--muted);
      border: 1px solid var(--panel-border);
      backdrop-filter: blur(10px);
    }

    h1 {
      margin: 0;
      font-size: clamp(2rem, 4vw, 4rem);
      line-height: 0.96;
      letter-spacing: -0.05em;
      max-width: 10ch;
    }

    .sub {
      margin: 0;
      max-width: 68ch;
      color: var(--muted);
      font-size: 1rem;
      line-height: 1.6;
    }

    .grid {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 18px;
      align-items: start;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px);
      overflow: hidden;
    }

    .panel-inner {
      padding: 18px;
    }

    .stage {
      position: relative;
      aspect-ratio: 4 / 3;
      background: #02060c;
      border-bottom: 1px solid var(--panel-border);
      overflow: hidden;
    }

    video, img {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
      background: #02060c;
    }

    .overlay-badge {
      position: absolute;
      left: 14px;
      top: 14px;
      padding: 8px 10px;
      border-radius: 999px;
      background: rgba(2, 6, 12, 0.7);
      border: 1px solid rgba(255, 255, 255, 0.12);
      color: #d7e7ff;
      font-size: 0.875rem;
      backdrop-filter: blur(10px);
    }

    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }

    button {
      appearance: none;
      border: 0;
      border-radius: 14px;
      padding: 12px 16px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      transition: transform 0.15s ease, opacity 0.15s ease, background 0.15s ease;
    }

    button:hover { transform: translateY(-1px); }
    button:disabled { opacity: 0.45; cursor: not-allowed; transform: none; }

    .primary { background: linear-gradient(135deg, var(--accent), #36cfc2); color: #03201d; }
    .secondary { background: rgba(255, 255, 255, 0.09); color: var(--text); border: 1px solid var(--panel-border); }

    .status {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-top: 14px;
      color: var(--muted);
      line-height: 1.5;
      min-height: 48px;
    }

    .dot {
      width: 12px;
      height: 12px;
      border-radius: 999px;
      background: var(--danger);
      box-shadow: 0 0 0 6px rgba(251, 113, 133, 0.15);
      flex: 0 0 auto;
    }

    .dot.live {
      background: var(--accent);
      box-shadow: 0 0 0 6px rgba(94, 234, 212, 0.14);
    }

    .cards {
      display: grid;
      gap: 12px;
    }

    .card {
      padding: 16px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .card h2 {
      margin: 0 0 8px;
      font-size: 1rem;
      letter-spacing: -0.02em;
    }

    .card p {
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
      font-size: 0.95rem;
    }

    .hint {
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.6;
    }

    @keyframes rise {
      from { opacity: 0; transform: translateY(12px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @media (max-width: 920px) {
      .grid { grid-template-columns: 1fr; }
      h1 { max-width: 100%; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="eyebrow">LAN camera inference on <strong>best.pt</strong></div>
      <h1>Garbage bin detection (YOLOv9) from any phone or laptop camera.</h1>
      <p class="sub">
        Open this page from a local device, grant camera permission, and the browser will stream frames to the Flask server for YOLO detection.
      </p>
    </section>

    <div class="grid">
      <section class="panel">
        <div class="stage">
          <video id="camera" playsinline autoplay muted></video>
          <img id="preview" alt="Annotated inference output" />
          <div class="overlay-badge" id="overlayBadge">Preview waiting for camera</div>
        </div>
        <div class="panel-inner">
          <div class="controls">
            <button class="primary" id="startBtn">Start camera</button>
            <button class="secondary" id="stopBtn" disabled>Stop</button>
          </div>
          <div class="status">
            <span class="dot" id="statusDot"></span>
            <span id="statusText">Idle. Camera access on phones usually requires HTTPS or localhost.</span>
          </div>
        </div>
      </section>

      <aside class="cards">
        <article class="card">
          <h2>How it works</h2>
          <p>The browser uses its own camera, captures frames, sends them to <code>/api/predict</code>, and displays the annotated result.</p>
        </article>
        <article class="card">
          <h2>Access from other devices</h2>
          <p>Start the server on <code>0.0.0.0</code>, then open the PC's LAN IP from each device. If the camera is blocked, run with HTTPS.</p>
        </article>
        <article class="card">
          <h2>Model</h2>
          <p>Loads the weights from the local <code>best.pt</code> file by default. You can point it at another checkpoint with <code>--model</code>.</p>
        </article>
        <article class="card">
          <h2>Tip</h2>
          <p>Use the rear camera on phones for cleaner framing. The app throttles requests so the server stays responsive on a LAN.</p>
        </article>
      </aside>
    </div>
  </div>

  <canvas id="capture" hidden></canvas>
  <script>
    const video = document.getElementById("camera");
    const preview = document.getElementById("preview");
    const capture = document.getElementById("capture");
    const startBtn = document.getElementById("startBtn");
    const stopBtn = document.getElementById("stopBtn");
    const statusText = document.getElementById("statusText");
    const statusDot = document.getElementById("statusDot");
    const overlayBadge = document.getElementById("overlayBadge");

    let stream = null;
    let running = false;
    let requestInFlight = false;
    let previewUrl = null;
    let delayMs = 180;

    function setStatus(message, live = false) {
      statusText.textContent = message;
      statusDot.classList.toggle("live", live);
      overlayBadge.textContent = message;
    }

    function revokePreviewUrl() {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
      }
    }

    async function startCamera() {
      if (!navigator.mediaDevices?.getUserMedia) {
        setStatus("This browser does not expose getUserMedia.", false);
        return;
      }

      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: "environment" } },
          audio: false,
        });
      } catch (error) {
        setStatus(`Camera permission failed: ${error.message}. Use HTTPS or localhost.`, false);
        return;
      }

      video.srcObject = stream;
      running = true;
      startBtn.disabled = true;
      stopBtn.disabled = false;
      setStatus("Camera live. Sending frames to the server...", true);
      loop();
    }

    function stopCamera() {
      running = false;
      requestInFlight = false;
      revokePreviewUrl();

      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        stream = null;
      }

      video.srcObject = null;
      preview.removeAttribute("src");
      startBtn.disabled = false;
      stopBtn.disabled = true;
      setStatus("Idle. Camera stopped.", false);
    }

    function captureFrame() {
      const width = video.videoWidth;
      const height = video.videoHeight;
      if (!width || !height) {
        return null;
      }

      capture.width = width;
      capture.height = height;
      const context = capture.getContext("2d", { willReadFrequently: false });
      context.drawImage(video, 0, 0, width, height);

      return new Promise((resolve) => {
        capture.toBlob((blob) => resolve(blob), "image/jpeg", 0.82);
      });
    }

    async function loop() {
      if (!running) {
        return;
      }

      if (requestInFlight || !video.videoWidth) {
        window.setTimeout(loop, 40);
        return;
      }

      requestInFlight = true;

      try {
        const blob = await captureFrame();
        if (!blob) {
          setStatus("Waiting for the camera to initialize...", true);
          requestInFlight = false;
          window.setTimeout(loop, 80);
          return;
        }

        const formData = new FormData();
        formData.append("frame", blob, "frame.jpg");

        const response = await fetch("/api/predict", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          const payload = await response.json().catch(() => ({}));
          throw new Error(payload.error || `Server returned ${response.status}`);
        }

        const output = await response.blob();
        revokePreviewUrl();
        previewUrl = URL.createObjectURL(output);
        preview.src = previewUrl;
        setStatus("Inference running. Keep the subject in frame.", true);
      } catch (error) {
        setStatus(`Inference error: ${error.message}`, false);
      } finally {
        requestInFlight = false;
        window.setTimeout(loop, delayMs);
      }
    }

    startBtn.addEventListener("click", startCamera);
    stopBtn.addEventListener("click", stopCamera);

    window.addEventListener("beforeunload", stopCamera);
  </script>
</body>
</html>
"""


def create_app(model_path: Path, device: str | int, imgsz: int, conf: float) -> Flask:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    app = Flask(__name__)
    model = YOLO(str(model_path))

    @app.get("/")
    def index() -> str:
        return render_template_string(HTML_TEMPLATE)

    @app.get("/health")
    def health() -> Response:
        return jsonify(
            status="ok",
            model=str(model_path),
            device=str(device),
            imgsz=imgsz,
            conf=conf,
        )

    @app.post("/api/predict")
    def predict() -> Response:
        uploaded = request.files.get("frame")
        if uploaded is None:
            return jsonify(error="Missing 'frame' upload."), 400

        raw = np.frombuffer(uploaded.read(), dtype=np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify(error="Could not decode the uploaded image."), 400

        with MODEL_LOCK:
            results = model.predict(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)

        annotated = results[0].plot()
        success, encoded = cv2.imencode(
            ".jpg",
            annotated,
            [int(cv2.IMWRITE_JPEG_QUALITY), 85],
        )
        if not success:
            return jsonify(error="Could not encode annotated image."), 500

        return Response(encoded.tobytes(), mimetype="image/jpeg")

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run best.pt on a LAN-accessible web server.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to the YOLOv9 weights file.")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind for LAN access.")
    parser.add_argument("--port", type=int, default=5000, help="Port to serve on.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    parser.add_argument("--device", default=None, help="YOLO device, for example cpu, 0, or 0,1.")
    parser.add_argument(
        "--https",
        action="store_true",
        help="Run with a temporary self-signed certificate so mobile browsers can access the camera.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device is None:
        device: str | int = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    app = create_app(Path(args.model).expanduser().resolve(), device=device, imgsz=args.imgsz, conf=args.conf)

    scheme = "https" if args.https else "http"
    print(f"Model: {Path(args.model).expanduser().resolve()}")
    print(f"Server: {scheme}://{args.host}:{args.port}")
    if not args.https:
        print("Note: many mobile browsers require HTTPS for camera access.")

    app.run(host=args.host, port=args.port, debug=False, ssl_context="adhoc" if args.https else None, threaded=True)


if __name__ == "__main__":
    main()