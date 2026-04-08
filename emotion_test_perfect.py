#!/usr/bin/env python3
"""
📷 Camera + Emotion Recognition Test
Standalone script — no Gemini, no Whisper, no TTS.
Just opens the webcam and runs fer_3stage_fp16.tflite in real-time.

Press Q to quit.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import cv2
import numpy as np
import time
from pathlib import Path

# ── TFLite ──────────────────────────────────────────────────────────────────
try:
    from ai_edge_litert.interpreter import Interpreter
    print("✅ ai-edge-litert loaded")
except ImportError:
    print("❌ ai-edge-litert not installed. Run: pip install ai-edge-litert")
    sys.exit(1)

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH     = "fer_3stage_fp16.tflite"
IMG_SIZE       = 224
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

EMOTION_COLORS = {
    'Angry':    (0,   0,   255),   # Red
    'Disgust':  (0,   128,  0),    # Dark Green
    'Fear':     (128,  0,  128),   # Purple
    'Happy':    (0,   255, 255),   # Yellow
    'Neutral':  (255, 255, 255),   # White
    'Sad':      (255, 128,   0),   # Blue-ish
    'Surprise': (255,   0,  255),  # Magenta
}

# ── Load model ───────────────────────────────────────────────────────────────
script_dir = Path(__file__).parent
model_path = str(script_dir / MODEL_PATH)

if not Path(model_path).exists():
    print(f"❌ Model not found: {model_path}")
    sys.exit(1)

print(f"📥 Loading TFLite model: {model_path}")
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"   Input : {input_details[0]['shape']}  dtype={input_details[0]['dtype']}")
print(f"   Output: {output_details[0]['shape']}")
print("✅ Model ready!\n")

# ── Load face detector ───────────────────────────────────────────────────────
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade  = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    print("❌ Failed to load face cascade!")
    sys.exit(1)
print("✅ Face detector ready!")

# ── Helper functions ─────────────────────────────────────────────────────────
def preprocess(face_roi):
    face = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.expand_dims(face, axis=0)   # (1, 224, 224, 3), raw 0-255

def predict(face_input):
    interpreter.set_tensor(input_details[0]['index'], face_input)
    interpreter.invoke()
    raw = interpreter.get_tensor(output_details[0]['index'])[0]
    # softmax if logits
    if raw.min() < 0 or abs(raw.sum() - 1.0) > 0.01:
        e = np.exp(raw - raw.max())
        raw = e / e.sum()
    idx = int(np.argmax(raw))
    return EMOTION_LABELS[idx], float(raw[idx]), raw

def draw_bars(frame, probs, x, y):
    """Draw probability bars to the right of the face box."""
    bw, bh, gap = 110, 14, 4
    for i, (label, p) in enumerate(zip(EMOTION_LABELS, probs)):
        yp = y + i * (bh + gap)
        if yp + bh > frame.shape[0]:
            break
        cv2.rectangle(frame, (x, yp), (x + bw, yp + bh), (40, 40, 40), -1)
        pw = int(p * bw)
        cv2.rectangle(frame, (x, yp), (x + pw, yp + bh), EMOTION_COLORS[label], -1)
        cv2.putText(frame, f"{label[:3]} {p*100:4.1f}%",
                    (x + bw + 4, yp + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1)

# ── Open camera ──────────────────────────────────────────────────────────────
print("\n🎥 Opening camera...")
cap = None
for idx in [0, 1, 2]:
    c = cv2.VideoCapture(idx)
    if c.isOpened():
        ret, frame = c.read()
        if ret and frame is not None:
            cap = c
            print(f"   ✅ Camera {idx} opened")
            break
        c.release()

if cap is None:
    print("❌ Could not open any camera.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("✅ Running — press Q in the window to quit.\n")

# ── Main loop ────────────────────────────────────────────────────────────────
frame_n    = 0
fps_time   = time.time()
fps        = 0.0
last_emotion    = "Neutral"
last_confidence = 0.0
last_probs      = np.ones(7) / 7

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_n += 1
    if frame_n % 30 == 0:
        fps = 30 / max(time.time() - fps_time, 1e-6)
        fps_time = time.time()

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))

    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        inp = preprocess(roi)
        emotion, conf, probs = predict(inp)

        last_emotion    = emotion
        last_confidence = conf
        last_probs      = probs

        color = EMOTION_COLORS[emotion]

        # Face bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Label above the box
        label  = f"{emotion}  {conf*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw + 8, y), color, -1)
        cv2.putText(frame, label, (x + 4, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Probability bars (right side of face)
        draw_bars(frame, probs, x + w + 8, y)

    # ── HUD overlay ──────────────────────────────────────────────────────────
    h_frame = frame.shape[0]
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (270, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, "Emotion Test",
                (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}   Faces: {len(faces)}",
                (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"{last_emotion}  {last_confidence*100:.1f}%",
                (16, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                EMOTION_COLORS[last_emotion], 2)

    cv2.putText(frame, "Press Q to quit",
                (8, h_frame - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    cv2.imshow("Emotion Test — fer_3stage_fp16", frame)

    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✅ Done.")
