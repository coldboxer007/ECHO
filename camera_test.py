#!/usr/bin/env python3
"""
ECHO Robot — Standalone Camera & TFLite Sentiment Test
======================================================
Shows the live camera feed on the 800x480 display with:
  - Face detection bounding boxes (green)
  - Emotion label + confidence above each face
  - Per-emotion confidence bar chart (right panel)
  - FPS counter and model info
  - Helps you position and aim the camera properly

Usage:
    python camera_test.py          # Normal (fullscreen on 800x480 display)
    python camera_test.py --window  # Windowed mode (for testing on desktop)

Press 'q' or ESC to quit.
"""

import sys
import time
import numpy as np
import cv2

# ── Config imports ──
from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    SENTIMENT_MODEL_PATH, SENTIMENT_LABELS,
    SENTIMENT_CONFIDENCE_THRESHOLD,
    DISPLAY_WIDTH, DISPLAY_HEIGHT,
    EMOTION_COLORS,
)

# ── Display layout ──
# Camera is 640x480, display is 800x480
# Left: camera feed (640x480) — exact height match
# Right: info panel (160x480)
CAM_W, CAM_H = CAMERA_WIDTH, CAMERA_HEIGHT    # 640, 480
PANEL_X = CAM_W                                # 640 — where right panel starts
PANEL_W = DISPLAY_WIDTH - CAM_W                # 160

# Bar chart geometry inside the right panel
BAR_LEFT = PANEL_X + 10
BAR_WIDTH = PANEL_W - 20        # 140px
BAR_HEIGHT = 22
BAR_GAP = 6
BAR_TOP = 120                   # Start bars below model info

WINDOW_NAME = "ECHO Camera Test"

# Emotion bar colors (BGR) — derived from config's RGB tuples
BAR_COLORS = {}
for label, rgb in EMOTION_COLORS.items():
    BAR_COLORS[label] = (rgb[2], rgb[1], rgb[0])  # RGB → BGR


# ═══════════════════════════════════════════
# TFLite loader (standalone — no ECHO class)
# ═══════════════════════════════════════════

def load_tflite_model():
    """Load TFLite model and return (interpreter, input_details, output_details) or Nones."""
    import os
    interpreter = None
    input_details = None
    output_details = None

    # Try tflite_runtime first (lightweight, preferred on RPi)
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        try:
            import tensorflow.lite as tflite
        except ImportError:
            print("[ERROR] No TFLite runtime found.")
            print("  Install: pip install tflite-runtime")
            return None, None, None

    if not os.path.exists(SENTIMENT_MODEL_PATH):
        print(f"[ERROR] Model not found: {SENTIMENT_MODEL_PATH}")
        return None, None, None

    try:
        interpreter = tflite.Interpreter(model_path=SENTIMENT_MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        shape = input_details[0]['shape']
        dtype = input_details[0]['dtype']
        print(f"[OK] TFLite model loaded: {SENTIMENT_MODEL_PATH}")
        print(f"     Input shape: {shape}  dtype: {dtype}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None, None, None

    return interpreter, input_details, output_details


def load_face_detector():
    """Load Haar cascade for face detection."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        print("[ERROR] Failed to load Haar cascade!")
        return None
    print("[OK] Haar cascade face detector loaded")
    return cascade


def run_inference(interpreter, input_details, output_details, face_roi):
    """Run TFLite inference on a face ROI. Returns softmax probabilities array."""
    input_shape = input_details[0]['shape']  # e.g. [1, 48, 48, 1]
    target_h, target_w = input_shape[1], input_shape[2]
    channels = input_shape[3]

    face_resized = cv2.resize(face_roi, (target_w, target_h))

    if channels == 1:
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        face_input = face_gray.astype(np.float32) / 255.0
        face_input = np.expand_dims(face_input, axis=-1)
    else:
        face_input = face_resized.astype(np.float32) / 255.0

    face_input = np.expand_dims(face_input, axis=0)  # batch dim

    input_dtype = input_details[0]['dtype']
    face_input = face_input.astype(input_dtype)

    interpreter.set_tensor(input_details[0]['index'], face_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    # Softmax
    raw = output[0].astype(np.float64)
    shifted = raw - np.max(raw)
    exp_vals = np.exp(shifted)
    probs = exp_vals / np.sum(exp_vals)
    return probs


# ═══════════════════════════════════════════
# Drawing helpers
# ═══════════════════════════════════════════

def draw_face_boxes(canvas, faces, labels, confidences):
    """Draw green rectangles around faces with emotion labels."""
    for i, (x, y, w, h) in enumerate(faces):
        # Box
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Label above box
        if i < len(labels):
            text = f"{labels[i]} {confidences[i]:.0%}"
            # Black background for readability
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(canvas, (x, y - th - 8), (x + tw + 4, y - 2), (0, 0, 0), -1)
            cv2.putText(canvas, text, (x + 2, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def draw_right_panel(canvas, ema_scores, fps, model_shape, face_count, top_emotion, top_conf):
    """Draw the info panel on the right 160px strip."""
    # Dark background
    cv2.rectangle(canvas, (PANEL_X, 0), (DISPLAY_WIDTH, DISPLAY_HEIGHT), (20, 20, 20), -1)
    # Separator line
    cv2.line(canvas, (PANEL_X, 0), (PANEL_X, DISPLAY_HEIGHT), (60, 60, 60), 1)

    # ── Header ──
    cv2.putText(canvas, "ECHO", (PANEL_X + 8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(canvas, "Camera Test", (PANEL_X + 8, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # ── Stats ──
    cv2.putText(canvas, f"FPS: {fps:.1f}", (PANEL_X + 8, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.putText(canvas, f"Faces: {face_count}", (PANEL_X + 8, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    if model_shape is not None:
        shape_str = f"Model: {model_shape[1]}x{model_shape[2]}"
        cv2.putText(canvas, shape_str, (PANEL_X + 8, 106),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)

    # ── Confidence bars ──
    y = BAR_TOP
    for label in SENTIMENT_LABELS:
        score = ema_scores.get(label, 0.0)
        color = BAR_COLORS.get(label, (180, 180, 180))

        # Label text
        cv2.putText(canvas, label[:7], (BAR_LEFT, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
        y += 14

        # Bar background
        cv2.rectangle(canvas, (BAR_LEFT, y), (BAR_LEFT + BAR_WIDTH, y + BAR_HEIGHT),
                      (50, 50, 50), -1)
        # Filled portion
        fill_w = int(BAR_WIDTH * min(score, 1.0))
        if fill_w > 0:
            cv2.rectangle(canvas, (BAR_LEFT, y), (BAR_LEFT + fill_w, y + BAR_HEIGHT),
                          color, -1)
        # Score text
        cv2.putText(canvas, f"{score:.0%}", (BAR_LEFT + BAR_WIDTH - 38, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        y += BAR_HEIGHT + BAR_GAP

    # ── Threshold line ──
    thresh_x = BAR_LEFT + int(BAR_WIDTH * SENTIMENT_CONFIDENCE_THRESHOLD)
    # Draw threshold indicator on each bar position
    y_start = BAR_TOP + 14
    for _ in SENTIMENT_LABELS:
        cv2.line(canvas, (thresh_x, y_start), (thresh_x, y_start + BAR_HEIGHT),
                 (0, 0, 200), 1)
        y_start += 14 + BAR_HEIGHT + BAR_GAP

    # Threshold label
    cv2.putText(canvas, f"Thresh: {SENTIMENT_CONFIDENCE_THRESHOLD:.0%}",
                (PANEL_X + 8, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1)

    # ── Current emotion (large) ──
    if top_emotion and top_conf > 0:
        y_bottom = DISPLAY_HEIGHT - 50
        color = BAR_COLORS.get(top_emotion, (180, 180, 180))
        cv2.putText(canvas, top_emotion.upper(), (PANEL_X + 8, y_bottom),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(canvas, f"{top_conf:.0%}", (PANEL_X + 8, y_bottom + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    else:
        y_bottom = DISPLAY_HEIGHT - 50
        cv2.putText(canvas, "NO FACE", (PANEL_X + 8, y_bottom),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 2)

    # ── Quit hint ──
    cv2.putText(canvas, "q/ESC: quit", (PANEL_X + 8, DISPLAY_HEIGHT - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)


def draw_crosshair(canvas, cam_w, cam_h):
    """Draw center crosshair to help aim the camera."""
    cx, cy = cam_w // 2, cam_h // 2
    size = 20
    color = (0, 255, 255)  # Cyan
    cv2.line(canvas, (cx - size, cy), (cx + size, cy), color, 1)
    cv2.line(canvas, (cx, cy - size), (cx, cy + size), color, 1)


# ═══════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════

def main():
    fullscreen = "--window" not in sys.argv

    print("=" * 50)
    print("  ECHO — Camera & TFLite Sentiment Test")
    print("=" * 50)
    print(f"Display: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
    print(f"Camera:  {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps")
    print(f"Model:   {SENTIMENT_MODEL_PATH}")
    print()

    # Load model
    interpreter, input_details, output_details = load_tflite_model()
    model_loaded = interpreter is not None
    model_shape = input_details[0]['shape'] if model_loaded else None

    # Load face detector
    face_cascade = load_face_detector()
    if face_cascade is None:
        print("[FATAL] Cannot run without face detector. Exiting.")
        sys.exit(1)

    # Open camera
    print(f"[...] Opening camera index {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[FATAL] Cannot open camera! Check USB connection.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[OK] Camera opened: {actual_w}x{actual_h}")
    print()
    print("Press 'q' or ESC to quit.")
    print()

    # Create window
    if fullscreen:
        cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    # EMA smoothing state
    ema_alpha = 0.4
    ema_scores = {label: 0.0 for label in SENTIMENT_LABELS}

    # FPS tracking
    frame_times = []
    fps = 0.0

    try:
        while True:
            t_start = time.time()

            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame capture failed, retrying...")
                time.sleep(0.1)
                continue

            # If camera resolution differs from expected, resize
            if frame.shape[1] != CAM_W or frame.shape[0] != CAM_H:
                frame = cv2.resize(frame, (CAM_W, CAM_H))

            # ── Create 800x480 canvas ──
            canvas = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
            # Place camera feed on the left
            canvas[0:CAM_H, 0:CAM_W] = frame

            # ── Face detection ──
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
            )
            faces = list(faces)

            # ── Run inference on each face ──
            face_labels = []
            face_confs = []

            if model_loaded and len(faces) > 0:
                # Sort by area (largest first)
                faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

                for (x, y, w, h) in faces_sorted:
                    face_roi = frame[y:y+h, x:x+w]
                    probs = run_inference(interpreter, input_details, output_details, face_roi)

                    # EMA update (only for largest face)
                    if (x, y, w, h) == tuple(faces_sorted[0]):
                        for i, label in enumerate(SENTIMENT_LABELS):
                            if i < len(probs):
                                ema_scores[label] = (
                                    ema_alpha * float(probs[i]) +
                                    (1 - ema_alpha) * ema_scores[label]
                                )

                    # Raw top prediction for this face
                    max_idx = int(np.argmax(probs))
                    face_labels.append(SENTIMENT_LABELS[max_idx])
                    face_confs.append(float(probs[max_idx]))
            elif len(faces) > 0:
                # No model — still show face boxes
                for _ in faces:
                    face_labels.append("no model")
                    face_confs.append(0.0)

            # Smoothed top emotion (from EMA)
            top_emotion = max(ema_scores, key=ema_scores.get)
            top_conf = ema_scores[top_emotion]
            if top_conf < SENTIMENT_CONFIDENCE_THRESHOLD:
                top_emotion = "neutral"
                top_conf = 0.0

            # Decay EMA when no face (so bars slowly drop to zero)
            if len(faces) == 0:
                for label in SENTIMENT_LABELS:
                    ema_scores[label] *= 0.95

            # ── Draw ──
            draw_face_boxes(canvas, faces, face_labels, face_confs)
            draw_crosshair(canvas, CAM_W, CAM_H)
            draw_right_panel(canvas, ema_scores, fps, model_shape,
                             len(faces), top_emotion, top_conf)

            # Show
            cv2.imshow(WINDOW_NAME, canvas)

            # FPS
            t_end = time.time()
            frame_times.append(t_end - t_start)
            if len(frame_times) > 30:
                frame_times.pop(0)
            if len(frame_times) > 1:
                fps = len(frame_times) / sum(frame_times)

            # Quit on 'q' or ESC
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[OK] Camera released. Done.")


if __name__ == "__main__":
    main()
