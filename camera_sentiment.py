"""
ECHO Robot — Camera & Sentiment Analysis
==========================================
Captures frames from USB webcam and runs local TFLite sentiment inference.
The TFLite model is expected to take a face image and output emotion probabilities.

Flow:
  1. OpenCV captures frame from USB camera
  2. Face detection (Haar cascade) finds faces
  3. Cropped face is preprocessed and fed to TFLite model
  4. Model outputs probability vector → mapped to emotion label
"""

import os
import time
import logging
import threading
import numpy as np
import cv2

logger = logging.getLogger("echo.camera")

# Try TFLite runtimes in order of preference:
#   1. ai-edge-litert (Google's official replacement for tflite-runtime)
#   2. tflite-runtime  (legacy, still common)
#   3. Full TensorFlow  (heavy, last resort)
try:
    from ai_edge_litert.interpreter import Interpreter as _TFLiteInterpreter
    TFLITE_AVAILABLE = True
    logger.info("Using ai-edge-litert")
except (ImportError, AttributeError):
    try:
        import tflite_runtime.interpreter as _tflite_mod
        _TFLiteInterpreter = _tflite_mod.Interpreter
        TFLITE_AVAILABLE = True
        logger.info("Using tflite_runtime")
    except (ImportError, AttributeError):
        try:
            import tensorflow.lite as _tflite_mod
            _TFLiteInterpreter = _tflite_mod.Interpreter
            TFLITE_AVAILABLE = True
            logger.info("Using tensorflow.lite")
        except (ImportError, AttributeError):
            _TFLiteInterpreter = None
            TFLITE_AVAILABLE = False
            logger.warning(
                "No TFLite runtime available — sentiment will always be 'neutral'. "
                "Install with: pip install ai-edge-litert"
            )

from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    SENTIMENT_MODEL_PATH, SENTIMENT_LABELS,
    SENTIMENT_CONFIDENCE_THRESHOLD, SENTIMENT_INTERVAL,
)


class CameraSentiment:
    """USB webcam capture + TFLite-based facial emotion detection."""

    def __init__(self):
        self._cap = None
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._face_cascade = None
        self._current_emotion = "neutral"
        self._current_confidence = 0.0
        self._current_frame = None
        self._running = False
        self._lock = threading.Lock()

        # Cached face detection results (avoid running Haar cascade twice per cycle)
        self._cached_faces = []
        self._cached_faces_frame_id = -1
        self._frame_counter = 0

        # Sentiment backoff: skip analysis when no face seen for a while
        self._no_face_streak = 0
        self._backoff_interval = SENTIMENT_INTERVAL  # Current interval (grows with backoff)

        # Emotion temporal smoothing: EMA (exponential moving average)
        self._emotion_scores = {label: 0.0 for label in SENTIMENT_LABELS}
        self._ema_alpha = 0.4  # Blending factor: 0=all history, 1=only latest

        self._init_camera()
        self._init_model()
        self._init_face_detector()
        logger.info("CameraSentiment initialized")

    def _init_camera(self):
        """Open USB webcam."""
        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self._cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            # Minimize buffer to get fresh frames (reduces 200-300ms lag in follow mode)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            logger.info(f"Camera opened: index={CAMERA_INDEX}")
        else:
            logger.error("Failed to open USB camera!")

    def _init_model(self):
        """Load TFLite sentiment model."""
        if not TFLITE_AVAILABLE:
            return

        if not os.path.exists(SENTIMENT_MODEL_PATH):
            logger.error(f"TFLite model not found: {SENTIMENT_MODEL_PATH}")
            return

        try:
            self._interpreter = _TFLiteInterpreter(model_path=SENTIMENT_MODEL_PATH)
            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()

            input_shape = self._input_details[0]['shape']
            logger.info(f"TFLite model loaded. Input shape: {input_shape}")
        except Exception as e:
            logger.error(f"Failed to load TFLite model: {e}")
            self._interpreter = None

    def _init_face_detector(self):
        """Load Haar cascade for face detection."""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._face_cascade = cv2.CascadeClassifier(cascade_path)
        if self._face_cascade.empty():
            logger.error("Failed to load face cascade!")
        else:
            logger.info("Face detector loaded")

    def capture_frame(self) -> np.ndarray:
        """Capture a single frame from the webcam. Returns BGR numpy array or None."""
        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        if not ret:
            return None

        self._frame_counter += 1
        with self._lock:
            # Store reference directly — only copy when get_current_frame() is called
            self._current_frame = frame

        return frame

    def get_current_frame(self) -> np.ndarray:
        """Get the most recent captured frame (thread-safe)."""
        with self._lock:
            return self._current_frame.copy() if self._current_frame is not None else None

    def detect_faces(self, frame: np.ndarray) -> list:
        """Detect faces in frame. Returns list of (x, y, w, h) rectangles.
        Results are cached per frame to avoid running Haar cascade twice
        (once for sentiment, once for follow mode)."""
        if self._face_cascade is None or frame is None:
            return []

        # Return cached result if same frame
        if self._frame_counter == self._cached_faces_frame_id:
            return self._cached_faces

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
        )
        result = list(faces)

        # Cache for this frame
        self._cached_faces = result
        self._cached_faces_frame_id = self._frame_counter

        return result

    def analyze_sentiment(self, frame: np.ndarray = None) -> tuple:
        """
        Run sentiment analysis on the current or given frame.
        Returns (emotion_label: str, confidence: float).
        """
        if frame is None:
            frame = self.capture_frame()
        if frame is None:
            return "neutral", 0.0

        # Detect faces
        faces = self.detect_faces(frame)
        if len(faces) == 0:
            # Decay EMA scores when no face visible (matching Mac test script)
            # Prevents stale emotions from persisting after a face disappears
            for label in self._emotion_scores:
                self._emotion_scores[label] *= 0.95
            self._current_emotion = "neutral"
            self._current_confidence = 0.0
            return "neutral", 0.0

        # Use the largest face
        faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces_sorted[0]

        # Crop face region
        face_roi = frame[y:y+h, x:x+w]

        if self._interpreter is None:
            return "neutral", 0.0

        try:
            # Preprocessing matched exactly to emotion_test_perfect.py.
            input_shape = self._input_details[0]['shape']  # e.g., [1, 224, 224, 3]
            target_h, target_w = input_shape[1], input_shape[2]
            face_resized = cv2.resize(face_roi, (target_w, target_h))
            face_input = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
            face_input = np.expand_dims(face_input, axis=0)  # Add batch dim

            self._interpreter.set_tensor(self._input_details[0]['index'], face_input)
            self._interpreter.invoke()
            raw = self._interpreter.get_tensor(self._output_details[0]['index'])[0]

            raw = raw.astype(np.float64)
            if raw.min() < 0 or abs(raw.sum() - 1.0) > 0.01:
                e = np.exp(raw - raw.max())
                raw = e / e.sum()
            probabilities = raw

            # Temporal smoothing via EMA — reduces emotion flicker from noisy frames
            for i, label in enumerate(SENTIMENT_LABELS):
                if i < len(probabilities):
                    self._emotion_scores[label] = (
                        self._ema_alpha * float(probabilities[i]) +
                        (1 - self._ema_alpha) * self._emotion_scores[label]
                    )

            # Pick emotion from smoothed scores
            smoothed_emotion = max(self._emotion_scores, key=self._emotion_scores.get)
            smoothed_confidence = self._emotion_scores[smoothed_emotion]

            max_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[max_idx])
            logger.debug(
                f"Sentiment probs: {probabilities.tolist()}, "
                f"best: idx={max_idx} conf={confidence:.3f}, "
                f"smoothed: {smoothed_emotion} ({smoothed_confidence:.3f})"
            )

            if smoothed_confidence >= SENTIMENT_CONFIDENCE_THRESHOLD:
                emotion = smoothed_emotion
                confidence = smoothed_confidence
            else:
                emotion = "neutral"
                confidence = 0.0

            with self._lock:
                self._current_emotion = emotion
                self._current_confidence = confidence

            return emotion, confidence

        except Exception as e:
            logger.error(f"Sentiment inference error: {e}")
            return "neutral", 0.0

    def get_face_center(self, frame: np.ndarray = None) -> tuple:
        """
        Get the center position of the largest detected face.
        Returns (center_x, center_y) or None if no face found.
        Used for person-following mode.
        """
        if frame is None:
            frame = self.capture_frame()
        if frame is None:
            return None

        faces = self.detect_faces(frame)
        if len(faces) == 0:
            return None

        faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces_sorted[0]
        return (x + w // 2, y + h // 2)

    def get_frame_jpeg(self, frame=None) -> bytes:
        """
        Get the current or given frame as JPEG bytes.
        Used for Gemini API fallback sentiment analysis.
        """
        if frame is None:
            frame = self.get_current_frame()
        if frame is None:
            return None

        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if success:
            return buffer.tobytes()
        return None

    @property
    def current_emotion(self) -> str:
        with self._lock:
            return self._current_emotion

    @property
    def current_confidence(self) -> float:
        with self._lock:
            return self._current_confidence

    # ── Background Loop ──

    def start_analysis(self):
        """Start background sentiment analysis loop."""
        self._running = True
        self._thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self._thread.start()
        logger.info("Sentiment analysis started")

    def stop_analysis(self):
        """Stop background analysis."""
        self._running = False
        logger.info("Sentiment analysis stopped")

    def _analysis_loop(self):
        """Continuously capture and analyze sentiment.
        Uses adaptive backoff: slows down when no face is visible,
        speeds back up when a face appears."""
        while self._running:
            try:
                frame = self.capture_frame()
                if frame is not None:
                    emotion, conf = self.analyze_sentiment(frame)
                    if conf > 0:
                        logger.debug(f"Emotion: {emotion} ({conf:.2f})")
                        # Face found — reset backoff
                        self._no_face_streak = 0
                        self._backoff_interval = SENTIMENT_INTERVAL
                    else:
                        # No face — increase backoff (up to 5x the base interval)
                        self._no_face_streak += 1
                        if self._no_face_streak > 5:
                            self._backoff_interval = min(
                                SENTIMENT_INTERVAL * 5.0,
                                self._backoff_interval * 1.5
                            )
            except Exception as e:
                logger.error(f"Analysis loop error: {e}")
            time.sleep(self._backoff_interval)

    # ── Cleanup ──

    def cleanup(self):
        """Release camera and stop analysis."""
        self.stop_analysis()
        if self._cap is not None:
            self._cap.release()
        logger.info("CameraSentiment cleaned up")
