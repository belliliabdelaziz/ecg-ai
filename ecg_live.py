"""
ECG Live Classifier
────────────────────────────────────────────────────────
- Reads raw ECG from serial (AD8232 / STM32)
- Detects R-peaks with Pan-Tompkins
- Extracts 187-point beat windows → normalizes
- Classifies with Keras model ('ecgclassifier')
- Grad-CAM heatmap saved for Flask UI
- Matplotlib window shows live waveform + BPM + label
- Flask web UI shows heatmap, confidence, history
────────────────────────────────────────────────────────
"""

import threading
import time
import sys
import os
from collections import deque

import numpy as np
import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import cv2
import tensorflow as tf
from flask import Flask, render_template_string, jsonify

# ─── CONFIG ───────────────────────────────────────────────────────────────────
PORT        = None          # None = auto-detect, or set e.g. "/dev/ttyACM0"
BAUD        = 115200
SAMPLE_RATE = 360           # Hz
WINDOW_SIZE = 500           # samples shown in live plot
MODEL_PATH  = "ecgclassifier"
FLASK_PORT  = 5000
# ──────────────────────────────────────────────────────────────────────────────

CLASSES = {
    0: ("Normal Beat",                  "#00ff88"),
    1: ("Supraventricular Ectopic",     "#ffdd00"),
    2: ("Ventricular Ectopic",          "#ff6644"),
    3: ("Fusion Beat",                  "#aa88ff"),
    4: ("Unknown Beat",                 "#888888"),
}

# ─── SHARED STATE (thread-safe via locks) ─────────────────────────────────────
lock           = threading.Lock()
ecg_buffer     = deque([0] * WINDOW_SIZE, maxlen=WINDOW_SIZE)
leads_off_buf  = deque([False] * WINDOW_SIZE, maxlen=WINDOW_SIZE)

latest = {
    "label":      "Waiting…",
    "color":      "#888888",
    "confidence": 0.0,
    "bpm":        0,
    "beat_idx":   [],          # sample indices of detected R-peaks in current window
    "heatmap_ready": False,
}

beat_history   = []            # list of dicts for Flask history table
r_peak_times   = deque(maxlen=10)   # timestamps of last N R-peaks for BPM

# ─── SERIAL ───────────────────────────────────────────────────────────────────
def find_port():
    for p in serial.tools.list_ports.comports():
        if any(k in p.description for k in ("STM", "CDC", "ACM", "Arduino")):
            print(f"[AUTO] {p.device} — {p.description}")
            return p.device
    ports = serial.tools.list_ports.comports()
    if ports:
        print(f"[AUTO] fallback: {ports[0].device}")
        return ports[0].device
    print("[ERROR] No serial port found.")
    sys.exit(1)

def parse_line(line: str):
    try:
        line = line.strip()
        if line.startswith("data :"):
            return int(line.split(":")[1].strip())
    except (ValueError, IndexError):
        pass
    return None

# ─── MODEL ────────────────────────────────────────────────────────────────────
print("[MODEL] Skipped — no ECG sensor connected.")
model = None
last_conv_name = None

os.makedirs("static", exist_ok=True)

# ─── GRAD-CAM ─────────────────────────────────────────────────────────────────
def compute_grad_cam(beat_187):
    """beat_187: 1-D numpy array of 187 samples (already normalized)"""
    if model is None or last_conv_name is None:
        return None, 0.0, 0

    try:
        arr = beat_187.reshape(1, 187, 1).astype(np.float32)
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(arr)
            target = int(np.argmax(preds[0]))
            loss   = preds[:, target]

        grads      = tape.gradient(loss, conv_out)
        pooled     = tf.reduce_mean(grads, axis=(0, 1)).numpy()
        cam        = conv_out.numpy()[0]
        for i in range(pooled.shape[-1]):
            cam[:, i] *= pooled[i]
        heatmap = np.mean(cam, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        big = cv2.resize(heatmap.reshape(1, -1), dsize=(187, 100),
                         interpolation=cv2.INTER_CUBIC)

        norm_beat = beat_187 / (np.max(np.abs(beat_187)) + 1e-9)

        fig, ax = plt.subplots(figsize=(9, 3))
        fig.patch.set_facecolor("#0d0d0d")
        ax.set_facecolor("#0d0d0d")
        ax.imshow(big, cmap="seismic", interpolation="lanczos",
                  extent=[0, 187, 0, 100], aspect="auto")
        ax.plot((norm_beat * 40) + 30, color="white", linewidth=2)
        ax.set_xlim(0, 187); ax.set_ylim(0, 100)
        conf  = float(np.max(preds[0]))
        label = CLASSES[target][0]
        ax.set_title(f"{label}  —  {conf*100:.1f}% confidence",
                     color="white", fontsize=11)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig("static/gradcam.png", dpi=120, facecolor="#0d0d0d")
        plt.close(fig)

        return big, conf, target
    except Exception as e:
        print(f"[WARN] Grad-CAM failed: {e}")
        return None, 0.0, 0

# ─── PAN-TOMPKINS (simplified) ────────────────────────────────────────────────
class PanTompkins:
    """Lightweight R-peak detector suitable for real-time use."""
    def __init__(self, fs=360):
        self.fs          = fs
        self.buf         = deque(maxlen=fs * 4)   # 4-second ring buffer
        self.last_r      = -999
        self.refractory  = int(0.2 * fs)          # 200 ms refractory period
        self._sq_buf     = deque(maxlen=int(0.15 * fs))
        self.threshold   = 0.0
        self._sig_level  = 0.0
        self._noise_level= 0.0

    def _bandpass(self, x):
        """Single-sample first-order IIR approximation of 5-15 Hz bandpass."""
        # Derivative + squaring as in Pan-Tompkins
        return x

    def add_sample(self, val, idx):
        """Returns beat_window (187 pts) if R-peak detected, else None."""
        self.buf.append(val)
        n = len(self.buf)
        if n < 30:
            return None

        buf = list(self.buf)
        # Simple derivative-squared energy
        d  = buf[-1] - buf[-3] if n >= 3 else 0
        sq = d * d
        self._sq_buf.append(sq)
        energy = np.mean(self._sq_buf)

        # Adaptive threshold
        if energy > self._sig_level:
            self._sig_level = 0.125 * energy + 0.875 * self._sig_level
        else:
            self._noise_level = 0.125 * energy + 0.875 * self._noise_level
        self.threshold = self._noise_level + 0.25 * (self._sig_level - self._noise_level)

        # Peak detection with refractory period
        if (energy > self.threshold and
                idx - self.last_r > self.refractory and
                n >= 187):
            # Check local maximum
            window3 = buf[-3:]
            if buf[-2] == max(window3):
                self.last_r = idx
                # Extract 187-point window centred on R-peak
                centre = n - 2
                start  = max(0, centre - 93)
                end    = start + 187
                if end > n:
                    start = n - 187
                    end   = n
                segment = np.array(buf[start:end], dtype=np.float32)
                if len(segment) == 187:
                    return segment
        return None

detector = PanTompkins(fs=SAMPLE_RATE)

# ─── SERIAL READER THREAD ─────────────────────────────────────────────────────
sample_idx = 0

def serial_thread(port_name):
    global sample_idx
    try:
        ser = serial.Serial(port_name, BAUD, timeout=1)
        print(f"[SERIAL] Connected to {port_name}")
    except serial.SerialException as e:
        print(f"[SERIAL ERROR] {e}")
        sys.exit(1)

    while True:
        try:
            raw = ser.readline().decode("utf-8", errors="ignore")
        except Exception:
            continue

        val = parse_line(raw)
        if val is None:
            continue

        leads_off = (val == -1)
        if leads_off:
            val = 0

        with lock:
            ecg_buffer.append(val)
            leads_off_buf.append(leads_off)

        if not leads_off:
            segment = detector.add_sample(val, sample_idx)
            if segment is not None:
                # Normalize same way as training data
                seg_norm = (segment - segment.mean()) / (segment.std() + 1e-9)
                # Classify + Grad-CAM in a separate thread to not block serial
                threading.Thread(target=classify_beat,
                                 args=(seg_norm, sample_idx),
                                 daemon=True).start()

        sample_idx += 1

def classify_beat(seg_norm, idx):
    if model is None:
        # No model loaded—use placeholder
        label, color = "No Model", "#888888"
        conf = 0.0
    else:
        try:
            # Predict class
            arr = seg_norm.reshape(1, 187, 1).astype(np.float32)
            preds = model.predict(arr, verbose=0)
            target = int(np.argmax(preds[0]))
            conf = float(np.max(preds[0]))
            label, color = CLASSES[target]
            
            # Grad-CAM skipped for SavedModel (structure not exposed)
            # Save a placeholder heatmap
            fig, ax = plt.subplots(figsize=(9, 3))
            fig.patch.set_facecolor("#0d0d0d")
            ax.set_facecolor("#0d0d0d")
            norm_beat = seg_norm / (np.max(np.abs(seg_norm)) + 1e-9)
            ax.plot(norm_beat * 40 + 50, color="white", linewidth=2)
            ax.set_xlim(0, 187); ax.set_ylim(0, 100)
            ax.set_title(f"{label}  —  {conf*100:.1f}% confidence",
                         color="white", fontsize=11)
            ax.axis("off")
            fig.tight_layout()
            fig.savefig("static/gradcam.png", dpi=120, facecolor="#0d0d0d")
            plt.close(fig)
        except Exception as e:
            print(f"[WARN] Prediction failed: {e}")
            label, color = "Error", "#888888"
            conf = 0.0
    
    now = time.time()
    r_peak_times.append(now)

    # BPM from last N R-peaks
    bpm = 0
    if len(r_peak_times) >= 2:
        intervals = np.diff(list(r_peak_times))
        bpm = int(60.0 / np.mean(intervals))

    with lock:
        latest["label"]        = label
        latest["color"]        = color
        latest["confidence"]   = conf
        latest["bpm"]          = bpm
        latest["heatmap_ready"]= True
        beat_history.insert(0, {
            "time":  time.strftime("%H:%M:%S"),
            "label": label,
            "conf":  f"{conf*100:.1f}%",
            "bpm":   bpm,
            "color": color,
        })
        if len(beat_history) > 50:
            beat_history.pop()

# ─── MATPLOTLIB LIVE WINDOW ───────────────────────────────────────────────────
def run_matplotlib():
    fig = plt.figure(figsize=(14, 5), facecolor="#0d0d0d")
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_ecg   = fig.add_subplot(gs[0])
    ax_info  = fig.add_subplot(gs[1])

    for ax in (ax_ecg, ax_info):
        ax.set_facecolor("#0d0d0d")
        for spine in ax.spines.values():
            spine.set_visible(False)

    line_ecg,  = ax_ecg.plot([], [], color="#00ff88", linewidth=1.2)
    ax_ecg.set_xlim(0, WINDOW_SIZE)
    ax_ecg.set_ylim(-100, 4200)
    ax_ecg.set_ylabel("ADC", color="#555")
    ax_ecg.tick_params(colors="#444", labelbottom=False)

    status_txt = ax_ecg.text(0.01, 0.95, "", transform=ax_ecg.transAxes,
                             color="white", fontsize=10, va="top")
    label_txt  = ax_ecg.text(0.5,  0.95, "", transform=ax_ecg.transAxes,
                             color="#00ff88", fontsize=13, va="top", ha="center",
                             fontweight="bold")

    ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1)
    ax_info.axis("off")
    bpm_txt   = ax_info.text(0.02, 0.5, "BPM: —",
                             color="#00ccff", fontsize=14, va="center", fontweight="bold")
    conf_txt  = ax_info.text(0.25, 0.5, "Conf: —",
                             color="#ffdd00", fontsize=12, va="center")
    flask_txt = ax_info.text(0.75, 0.5, f"Web UI → localhost:{FLASK_PORT}",
                             color="#555", fontsize=9, va="center")

    def update(_frame):
        with lock:
            y     = list(ecg_buffer)
            lo    = list(leads_off_buf)
            lbl   = latest["label"]
            col   = latest["color"]
            conf  = latest["confidence"]
            bpm   = latest["bpm"]
            lo_now= lo[-1] if lo else False

        x = list(range(len(y)))
        line_ecg.set_data(x, y)

        if lo_now:
            status_txt.set_text("⚠  Leads off")
            status_txt.set_color("#ff4444")
            line_ecg.set_color("#333")
        else:
            status_txt.set_text("●  Leads on")
            status_txt.set_color("#00ff88")
            line_ecg.set_color("#00ff88")

        label_txt.set_text(lbl)
        label_txt.set_color(col)
        bpm_txt.set_text(f"♥  {bpm} BPM" if bpm else "BPM: —")
        conf_txt.set_text(f"Confidence: {conf*100:.1f}%" if conf else "Conf: —")

        return line_ecg, status_txt, label_txt, bpm_txt, conf_txt

    ani = animation.FuncAnimation(fig, update, interval=30,
                                  blit=True, cache_frame_data=False)
    plt.tight_layout()
    plt.show()

# ─── FLASK WEB UI ─────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="2">
<title>ECG Live Classifier</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg:    #080c10;
    --panel: #0d1318;
    --border:#1a2a1a;
    --green: #00ff88;
    --cyan:  #00ccff;
    --yellow:#ffdd00;
    --red:   #ff4444;
    --dim:   #334;
  }
  body {
    background: var(--bg);
    color: #ccc;
    font-family: 'Rajdhani', sans-serif;
    min-height: 100vh;
    padding: 24px;
  }
  h1 {
    font-family: 'Share Tech Mono', monospace;
    color: var(--green);
    font-size: 1.4rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 12px;
    margin-bottom: 24px;
  }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 20px;
  }
  .panel h2 {
    font-size: 0.75rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--dim);
    margin-bottom: 14px;
    font-family: 'Share Tech Mono', monospace;
  }
  .stat-row { display: flex; gap: 24px; margin-bottom: 20px; flex-wrap: wrap; }
  .stat {
    background: #0a1208;
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 12px 20px;
    min-width: 120px;
  }
  .stat .val {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
  }
  .stat .key {
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #556;
    margin-top: 4px;
    font-family: 'Share Tech Mono', monospace;
  }
  .label-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 2px;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 1px;
    border: 1px solid currentColor;
    margin-bottom: 16px;
  }
  .heatmap img { width: 100%; border-radius: 3px; border: 1px solid var(--border); }
  table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
  th {
    text-align: left;
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #446;
    padding: 6px 8px;
    border-bottom: 1px solid var(--border);
    font-family: 'Share Tech Mono', monospace;
  }
  td { padding: 7px 8px; border-bottom: 1px solid #0f1a0f; }
  tr:hover td { background: #0f1a0f; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
  @media(max-width:700px){ .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<h1>⬡ ECG Live Classifier</h1>

<div class="stat-row">
  <div class="stat">
    <div class="val" style="color:{{ color }}">{{ bpm }}</div>
    <div class="key">BPM</div>
  </div>
  <div class="stat">
    <div class="val" style="color:{{ color }}">{{ conf }}</div>
    <div class="key">Confidence</div>
  </div>
  <div class="stat" style="flex:1">
    <div class="label-badge" style="color:{{ color }}">{{ label }}</div>
    <div class="key">Last Classification</div>
  </div>
</div>

<div class="grid">
  <div class="panel heatmap">
    <h2>Grad-CAM — Model Attention</h2>
    {% if heatmap %}
    <img src="/static/gradcam.png?t={{ ts }}" alt="Grad-CAM">
    {% else %}
    <p style="color:#334;font-family:monospace;font-size:0.85rem">Waiting for first beat…</p>
    {% endif %}
  </div>
  <div class="panel">
    <h2>Beat History</h2>
    <table>
      <thead><tr><th>Time</th><th>Classification</th><th>Conf</th><th>BPM</th></tr></thead>
      <tbody>
      {% for b in history %}
      <tr>
        <td style="color:#445;font-family:monospace">{{ b.time }}</td>
        <td><span class="dot" style="background:{{ b.color }}"></span>{{ b.label }}</td>
        <td style="color:{{ b.color }}">{{ b.conf }}</td>
        <td style="color:#00ccff">{{ b.bpm }}</td>
      </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>
</div>
</body>
</html>
"""

flask_app = Flask(__name__)

@flask_app.route("/")
def index():
    with lock:
        lbl   = latest["label"]
        col   = latest["color"]
        conf  = f'{latest["confidence"]*100:.1f}%' if latest["confidence"] else "—"
        bpm   = latest["bpm"] or "—"
        ready = latest["heatmap_ready"]
        hist  = list(beat_history[:20])
    return render_template_string(HTML,
        label=lbl, color=col, conf=conf, bpm=bpm,
        heatmap=ready, history=hist,
        ts=int(time.time()))

@flask_app.route("/api/latest")
def api_latest():
    with lock:
        return jsonify(latest)

def run_flask():
    flask_app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False)

# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port_name = PORT or find_port()

    # Serial reader thread
    t_serial = threading.Thread(target=serial_thread, args=(port_name,), daemon=True)
    t_serial.start()

    # Flask thread
    t_flask = threading.Thread(target=run_flask, daemon=True)
    t_flask.start()

    print(f"[WEB] Flask UI → http://localhost:{FLASK_PORT}")
    print("[PLOT] Opening matplotlib window…")

    # Matplotlib runs on main thread (required on most OS)
    run_matplotlib()
