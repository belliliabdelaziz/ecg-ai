# ECG Live Classifier

Real-time ECG signal streaming, R-peak detection, and beat classification with explainability.

## Quick Start

### Option 1: Conda (Recommended)

```bash
# Create environment
conda create -n ecg_live python=3.10 -y
conda activate ecg_live

# Install dependencies
pip install -r requirements.txt

# Run
python ecg_live.py
```

### Option 2: Venv

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run
python ecg_live.py
```

## What Happens

- **Matplotlib window** opens → live ECG waveform + BPM + label
- **Web UI** → http://localhost:5000 → beat history + signal info
- Waits for ECG data on serial port (configurable in `ecg_live.py`, line 36)

## Configuration (Lines 35-40 in `ecg_live.py`)

```python
PORT        = None              # None = auto-detect, or set "/dev/ttyACM0"
BAUD        = 115200            # Serial baud rate
SAMPLE_RATE = 360               # Hz
WINDOW_SIZE = 500               # Samples shown in live plot
MODEL_PATH  = "ecgclassifier"   # Path to model (when available)
FLASK_PORT  = 5000              # Web UI port
```

## Architecture

```
Serial Input (360 Hz ECG data)
      ↓
[Pan-Tompkins R-peak Detector]
      ↓
[Extract 187-point beat window]
      ↓
[Normalize: zero mean, unit std]
      ↓
[Keras Model] (optional)
      ↓
├── Matplotlib: live waveform + BPM
└── Flask UI (port 5000): beat history + visualization
```

## Adding a Trained Model

When you have a trained ECG classifier:

1. Export it as TensorFlow SavedModel: `model.save('ecgclassifier/')`
2. Place the folder in the project root
3. Update `MODEL_PATH = "ecgclassifier"` in line 39
4. Restart the app — classification will auto-enable

Expected input: 187-point ECG beat (normalized float32)  
Expected output: 5-class probabilities (softmax)

## Hardware Setup (When Ready)

- **Sensor**: AD8232 ECG module or compatible
- **Microcontroller**: STM32, Arduino, or similar
- **Data Format**: Plain text `data: <value>` on serial port at 115200 baud
- **Sampling Rate**: 360 Hz (configurable)

## Notes

- Beat detection runs in background thread — serial stream never blocks
- Web page auto-refreshes every 2 seconds
- Works without a trained model (demo mode)
- Grad-CAM visualization enabled when Conv1D layer is available
# ecg-ai
