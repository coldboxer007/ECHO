#!/usr/bin/env python3
"""
ECHO — Debug Main (Verbose Terminal Output)
=============================================
Same interaction loop as main.py but prints ALL inputs, outputs,
sensor data, and system state to the terminal for debugging.

ECHODebug inherits from ECHO (main.py) and overrides handlers to add
verbose color-coded terminal output. All core logic lives in the base
class — this file only adds debug printing and stats tracking.

Run:
    python3 debug_main.py

Everything the robot hears, thinks, says, and senses is printed
with timestamps and color-coded tags.
"""

import time
import logging
import threading
import traceback
from datetime import datetime

import cv2
import numpy as np

# ─── Logging Setup (DEBUG level for max detail) ───
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
# Quiet down noisy libraries
for noisy in ["httpx", "urllib3", "google_genai", "faster_whisper", "httpcore"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("echo.debug")

# Import the base ECHO class
from main import ECHO
from config import (
    WAKE_WORD_ENABLED, WAKE_WORD_PHRASES, CAMERA_WIDTH, CAMERA_HEIGHT,
    SENTIMENT_LABELS, EMOTION_COLORS,
)


# ═══════════════════════════════════════════════════
# Terminal Helpers
# ═══════════════════════════════════════════════════

# ANSI color codes
C_RESET  = "\033[0m"
C_BOLD   = "\033[1m"
C_RED    = "\033[91m"
C_GREEN  = "\033[92m"
C_YELLOW = "\033[93m"
C_BLUE   = "\033[94m"
C_MAGENTA= "\033[95m"
C_CYAN   = "\033[96m"
C_WHITE  = "\033[97m"
C_DIM    = "\033[2m"


def ts():
    """Current timestamp string."""
    return datetime.now().strftime("%H:%M:%S")


def banner(msg):
    """Print a section banner."""
    print(f"\n{C_BOLD}{'═' * 60}")
    print(f"  [{ts()}]  {msg}")
    print(f"{'═' * 60}{C_RESET}")


def dp(tag, msg, color=C_WHITE):
    """Debug print with tag and color."""
    print(f"  {color}[{ts()}] [{tag:>12}]{C_RESET}  {msg}")


def box(lines):
    """Print lines in a box."""
    width = max(len(line) for line in lines) + 4
    print(f"  ┌{'─' * width}┐")
    for line in lines:
        print(f"  │  {line:<{width - 2}}│")
    print(f"  └{'─' * width}┘")


# ═══════════════════════════════════════════════════
# Debug ECHO Controller (inherits from ECHO)
# ═══════════════════════════════════════════════════

class ECHODebug(ECHO):
    """
    Inherits all functionality from ECHO, adding verbose terminal output
    for every step. All inputs (speech, emotion, sensors) and outputs
    (responses, motor commands) are printed in real-time.
    """

    def __init__(self):
        banner("🤖 ECHO Robot — DEBUG MODE")

        # Initialize the base ECHO class (all subsystems)
        super().__init__()

        # Debug-specific stats
        self._loop_count = 0
        self._listen_count = 0
        self._speak_count = 0
        self._cmd_counts = {"chat": 0, "move": 0, "follow": 0, "stop": 0, "goodbye": 0}

        banner("✅ All subsystems initialized — DEBUG MODE")

    # ═══════════════════════════════════════════
    # Start / Main Loop
    # ═══════════════════════════════════════════

    def start(self):
        """Start all services and enter the debug main loop."""
        self._running = True

        # Start background services (same as base class)
        self.sensors.start_monitoring(interval=0.2)
        self.camera.start_analysis()
        self.face.start()

        # Debug-specific: sensor monitor thread
        self._sensor_thread = threading.Thread(
            target=self._sensor_debug_loop, daemon=True
        )
        self._sensor_thread.start()

        # Debug-specific: camera preview + confidence meter window
        self._camera_thread = threading.Thread(
            target=self._camera_debug_loop, daemon=True
        )
        self._camera_thread.start()

        # Startup greeting
        time.sleep(1)
        self.face.set_emotion("happy")

        dp("SPEAK", "Saying startup greeting...", C_MAGENTA)
        self.speech.speak(
            "Hello! I'm Echo in debug mode. All systems are being monitored.",
            emotion="happy"
        )
        self._speak_count += 1
        self.face.set_emotion("neutral")

        print()
        box([
            "🤖 ECHO DEBUG MODE — ACTIVE",
            "",
            "All inputs and outputs are printed below.",
            "Sensor data printed every 5 seconds.",
            "Camera preview window with confidence meter.",
            "Press 'q' in camera window or Ctrl+C to stop.",
        ])
        print()

        try:
            self._main_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def _sensor_debug_loop(self):
        """Print sensor readings periodically."""
        while self._running:
            try:
                dist = self.sensors.last_distance
                ir = self.sensors.last_ir_blocked
                obstacle = self.sensors.is_obstacle_ahead()
                emo = self.camera.current_emotion
                emo_conf = self.camera.current_confidence

                dp("SENSOR",
                   f"Dist: {dist:6.1f}cm │ IR: {'OBSTACLE' if ir else 'clear':>8s} │ "
                   f"Blocked: {'YES' if obstacle else 'no':<3s} │ "
                   f"Cam emotion: {emo} ({emo_conf:.0%})",
                   C_BLUE)
            except Exception:
                pass
            time.sleep(5)

    def _camera_debug_loop(self):
        """Show OpenCV window with live camera feed, face detection boxes,
        emotion label, and per-emotion confidence bar chart.
        Runs at ~10fps to keep CPU light. Fails silently on headless setups."""
        WINDOW_NAME = "ECHO Debug — Camera + Confidence"
        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, 640, 480)
        except Exception:
            dp("CAMERA", "Cannot open debug window (headless?). Skipping camera preview.", C_YELLOW)
            return

        # BGR colors for each emotion (derived from config RGB, swapped to BGR)
        emo_bgr = {}
        for label, rgb in EMOTION_COLORS.items():
            emo_bgr[label] = (rgb[2], rgb[1], rgb[0])

        while self._running:
            try:
                frame = self.camera.get_current_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue

                # ── Draw face detection boxes ──
                faces = self.camera.detect_faces(frame)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # ── Overlay current emotion label + confidence ──
                emo = self.camera.current_emotion
                conf = self.camera.current_confidence
                label_text = f"{emo.upper()} ({conf:.0%})"
                color = emo_bgr.get(emo, (200, 200, 200))
                cv2.putText(frame, label_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

                # ── Per-emotion confidence bar chart (bottom of frame) ──
                bar_h = 12          # Height of each bar
                bar_max_w = 150     # Max bar width in pixels
                bar_x = 10          # Left edge
                bar_y_start = frame.shape[0] - (len(SENTIMENT_LABELS) * (bar_h + 4)) - 10

                # Semi-transparent background for readability
                overlay_y1 = max(0, bar_y_start - 5)
                overlay_y2 = frame.shape[0]
                overlay = frame[overlay_y1:overlay_y2, 0:bar_x + bar_max_w + 80].copy()
                cv2.rectangle(frame, (0, overlay_y1), (bar_x + bar_max_w + 80, overlay_y2),
                              (0, 0, 0), -1)
                # Blend: 60% black overlay, 40% original
                frame[overlay_y1:overlay_y2, 0:bar_x + bar_max_w + 80] = \
                    cv2.addWeighted(frame[overlay_y1:overlay_y2, 0:bar_x + bar_max_w + 80],
                                    0.6, overlay, 0.4, 0)

                scores = self.camera._emotion_scores
                for i, label in enumerate(SENTIMENT_LABELS):
                    score = scores.get(label, 0.0)
                    y_pos = bar_y_start + i * (bar_h + 4)
                    bar_w = int(score * bar_max_w)
                    bar_color = emo_bgr.get(label, (200, 200, 200))

                    # Label text
                    cv2.putText(frame, f"{label[:3]}", (bar_x, y_pos + bar_h - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                    # Bar background
                    cv2.rectangle(frame, (bar_x + 30, y_pos),
                                  (bar_x + 30 + bar_max_w, y_pos + bar_h),
                                  (40, 40, 40), -1)
                    # Bar fill
                    if bar_w > 0:
                        cv2.rectangle(frame, (bar_x + 30, y_pos),
                                      (bar_x + 30 + bar_w, y_pos + bar_h),
                                      bar_color, -1)
                    # Percentage
                    cv2.putText(frame, f"{score:.0%}",
                                (bar_x + 30 + bar_max_w + 4, y_pos + bar_h - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(100) & 0xFF  # ~10fps, also pumps event queue
                if key == ord('q'):
                    dp("CAMERA", "Debug camera window closed by user (q)", C_YELLOW)
                    break

            except Exception:
                time.sleep(0.5)

        try:
            cv2.destroyWindow(WINDOW_NAME)
        except Exception:
            pass

    def _main_loop(self):
        """Core debug loop — same as ECHO but with print statements."""
        while self._running:
            try:
                self._loop_count += 1

                # Skip if speaking
                if self.speech.is_speaking:
                    time.sleep(0.1)
                    continue

                # ── Step 1: Listen ──
                dp("LISTEN", "🎤 Waiting for speech...", C_YELLOW)
                user_text = self.speech.listen()

                if not user_text:
                    dp("LISTEN", "(silence — no speech detected)", C_DIM)
                    time.sleep(0.1)
                    continue

                self._listen_count += 1
                dp("INPUT", f"👤 User said: \"{user_text}\"", C_GREEN)

                # ── Wake word gate (optional) ──
                if WAKE_WORD_ENABLED:
                    text_lower = user_text.lower().strip()
                    matched_phrase = None
                    for phrase in WAKE_WORD_PHRASES:
                        if text_lower.startswith(phrase):
                            matched_phrase = phrase
                            break

                    if matched_phrase is None:
                        dp("WAKE", f"No wake word — ignoring: \"{user_text}\"", C_DIM)
                        continue

                    user_text = user_text[len(matched_phrase):].strip(" ,!.")
                    dp("WAKE", f"Wake word '{matched_phrase}' detected", C_GREEN)
                    if not user_text:
                        dp("WAKE", "Just wake word, no command — saying 'Yes?'", C_YELLOW)
                        self.speech.speak("Yes?", emotion="neutral")
                        self._speak_count += 1
                        continue

                # ── Step 2: Get emotion ──
                emotion = self.camera.current_emotion
                confidence = self.camera.current_confidence
                dp("EMOTION", f"Local model → {emotion} ({confidence:.0%})", C_CYAN)

                # Gemini fallback for low-confidence
                if emotion == "neutral" and confidence < 0.3:
                    dp("EMOTION", "Low confidence — trying Gemini vision fallback...", C_YELLOW)
                    frame_jpeg = self.camera.get_frame_jpeg()
                    if frame_jpeg:
                        try:
                            emotion, confidence = self.brain.analyze_emotion_from_image(frame_jpeg)
                            dp("EMOTION", f"Gemini fallback → {emotion} ({confidence:.0%})", C_CYAN)
                        except Exception as e:
                            dp("EMOTION", f"Gemini fallback failed: {e}", C_RED)

                # ── Step 2b: Update face gaze to track detected face ──
                face_center = self.camera.get_face_center()
                if face_center is not None:
                    cx, cy = face_center
                    gaze_x = (cx / (CAMERA_WIDTH / 2)) - 1.0
                    gaze_y = (cy / (CAMERA_HEIGHT / 2)) - 1.0
                    self.face.set_gaze(gaze_x, gaze_y)
                    dp("GAZE", f"Face at ({cx},{cy}) → gaze ({gaze_x:.2f},{gaze_y:.2f})", C_DIM)

                # ── Step 3: Parse command ──
                command = self.brain.interpret_command(user_text)
                cmd_type = command['type']

                # Hybrid NLP: if local match returns 'chat', try Gemini NLP
                # classification for natural language movement phrases
                if cmd_type == 'chat':
                    nlp_command = self.brain.interpret_command_nlp(user_text)
                    if nlp_command['type'] != 'chat':
                        command = nlp_command
                        cmd_type = command['type']
                        dp("NLP", f"🧠 Reclassified '{user_text}' → {cmd_type}", C_GREEN)

                cmd_dir = command.get('direction', '')
                dp("COMMAND", f"Type: {cmd_type}" + (f" → {cmd_dir}" if cmd_dir else ""), C_MAGENTA)
                self._cmd_counts[cmd_type] = self._cmd_counts.get(cmd_type, 0) + 1

                # ── Step 4: Execute (uses overridden handlers with debug output) ──
                if cmd_type == 'move':
                    self._handle_move(command)
                elif cmd_type == 'keep_moving':
                    self._handle_keep_moving(command)
                elif cmd_type == 'safe_move':
                    self._handle_safe_move(command)
                elif cmd_type == 'patrol':
                    self._handle_patrol()
                elif cmd_type == 'follow':
                    self._handle_follow()
                elif cmd_type == 'stop':
                    self._handle_stop()
                elif cmd_type == 'goodbye':
                    self._handle_goodbye()
                elif cmd_type == 'clear_history':
                    self._handle_clear_history()
                elif cmd_type == 'look':
                    self._handle_look()
                elif cmd_type == 'volume':
                    self._handle_volume(command)
                elif cmd_type == 'chat':
                    self._handle_chat(user_text, emotion, confidence)

                # ── Stats line ──
                dp("STATS",
                   f"Loop #{self._loop_count} │ Heard: {self._listen_count} │ "
                   f"Spoke: {self._speak_count} │ "
                   f"Cmds: {self._cmd_counts}",
                   C_DIM)
                print()

            except Exception as e:
                dp("ERROR", f"Main loop error: {e}", C_RED)
                traceback.print_exc()
                time.sleep(1)

    # ═══════════════════════════════════════════
    # Command Handlers (override base with debug output)
    # ═══════════════════════════════════════════

    def _handle_move(self, command: dict):
        """Handle movement with debug output."""
        direction = command['direction']
        duration = command.get('duration', None)
        dp("MOVE", f"Executing movement: {direction}" + (f" for {duration}s" if duration else ""), C_CYAN)

        # Use base class handler (which speaks ack + executes move)
        super()._handle_move(command)
        self._speak_count += 1

        success = not self.sensors.is_obstacle_ahead()  # Approximate check for debug output
        dp("MOVE", f"Result: {'✅ SUCCESS' if success else '❌ BLOCKED'}", C_GREEN if success else C_RED)

    def _handle_keep_moving(self, command: dict):
        """Handle continuous movement with debug output."""
        direction = command.get('direction', 'forward')
        dp("MOVE", f"Starting continuous {direction} movement", C_CYAN)
        super()._handle_keep_moving(command)
        self._speak_count += 1

    def _handle_safe_move(self, command: dict):
        """Handle obstacle-aware movement with debug output."""
        direction = command.get('direction', 'forward')
        dp("MOVE", f"Safe move: {direction} with obstacle checking", C_CYAN)
        super()._handle_safe_move(command)
        self._speak_count += 1
        dp("MOVE", "Safe forward running in background", C_GREEN)

    def _handle_patrol(self):
        """Handle patrol mode with debug output."""
        dp("MOVE", "Starting patrol mode", C_MAGENTA)
        super()._handle_patrol()
        self._speak_count += 1

    def _handle_follow(self):
        """Handle follow mode with debug output."""
        dp("FOLLOW", "Starting follow mode", C_MAGENTA)
        super()._handle_follow()
        self._speak_count += 1
        dp("SPEAK", "🔊 Follow mode announcement", C_MAGENTA)

    def _handle_stop(self):
        """Handle stop with debug output."""
        dp("STOP", "🛑 Emergency stop!", C_RED)
        super()._handle_stop()
        self._speak_count += 1

    def _handle_goodbye(self):
        """Handle goodbye / shutdown with debug output."""
        dp("GOODBYE", "👋 Goodbye command — initiating shutdown", C_YELLOW)
        super()._handle_goodbye()

    def _handle_clear_history(self):
        """Handle clear history with debug output."""
        dp("BRAIN", "Clearing conversation history", C_MAGENTA)
        super()._handle_clear_history()
        self._speak_count += 1

    def _handle_look(self):
        """Handle look/scene description with debug output."""
        dp("VISION", "Capturing frame for Gemini scene analysis...", C_CYAN)
        super()._handle_look()
        self._speak_count += 1

    def _handle_volume(self, command: dict):
        """Handle volume with debug output."""
        direction = command.get('direction', 'up')
        dp("VOLUME", f"Volume {direction} requested (current: {self.speech.volume:.2f}x)", C_CYAN)
        super()._handle_volume(command)
        self._speak_count += 1
        dp("VOLUME", f"Volume now: {self.speech.volume:.2f}x", C_GREEN)

    def _handle_chat(self, user_text: str, emotion: str, confidence: float):
        """Handle conversation with streaming think→TTS pipeline + debug output."""
        self.face.set_emotion(emotion)

        dp("BRAIN", f"Sending to Gemini (streaming)...", C_MAGENTA)
        dp("BRAIN", f"  Input: \"{user_text}\"", C_MAGENTA)
        dp("BRAIN", f"  Emotion context: {emotion} ({confidence:.0%})", C_MAGENTA)

        # "Thinking" indicator
        self.face.set_talking(True)

        # Play brief audio cue so user knows ECHO heard them
        self.speech.play_thinking_cue()
        dp("SPEAK", "♪ Thinking cue played", C_DIM)

        # Stream Gemini response sentence-by-sentence
        full_response = ""
        first_sentence = True
        response_emotion = emotion  # default until first sentence arrives
        sentence_count = 0

        for sentence in self.brain.think_stream(user_text, emotion, confidence):
            sentence_count += 1
            if first_sentence:
                self.face.set_talking(False)  # Stop thinking indicator
                first_sentence = False

                # Determine response emotion from first sentence
                response_emotion = self.brain.determine_response_emotion(sentence, emotion)
                dp("EMOTION", f"Response emotion: {response_emotion}", C_CYAN)
                self.face.set_emotion(response_emotion)

            full_response += sentence + " "
            dp("STREAM", f"  [{sentence_count}] \"{sentence}\"", C_GREEN)

            # Speak each sentence as it arrives
            self.face.set_talking(True)
            self.speech.speak(sentence, emotion=response_emotion)
            self._speak_count += 1

        # If no sentences came through (empty response)
        if not full_response.strip():
            self.face.set_talking(False)
            full_response = "Hmm, I'm not sure what to say about that."
            response_emotion = "neutral"
            dp("BRAIN", "⚠️  No response from Gemini, using default", C_YELLOW)
            self.face.set_emotion(response_emotion)
            self.face.set_talking(True)
            self.speech.speak(full_response, emotion=response_emotion)
            self._speak_count += 1

        dp("OUTPUT", f"🤖 ECHO says: \"{full_response.strip()}\"", C_GREEN)
        dp("STREAM", f"Total sentences streamed: {sentence_count}", C_DIM)

        self.face.set_talking(False)
        time.sleep(0.5)
        self.face.set_emotion("neutral")

    # ═══════════════════════════════════════════
    # Shutdown
    # ═══════════════════════════════════════════

    def shutdown(self):
        """Clean shutdown with summary."""
        banner("🤖 ECHO — Shutting Down (DEBUG)")

        # Close camera debug window if open
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        # Call base class shutdown (cleans up all subsystems)
        super().shutdown()

        # Print session summary
        print()
        box([
            "📊 DEBUG SESSION SUMMARY",
            "",
            f"Total loops:       {self._loop_count}",
            f"Speech inputs:     {self._listen_count}",
            f"Speech outputs:    {self._speak_count}",
            f"Chat commands:     {self._cmd_counts.get('chat', 0)}",
            f"Move commands:     {self._cmd_counts.get('move', 0)}",
            f"Follow commands:   {self._cmd_counts.get('follow', 0)}",
            f"Stop commands:     {self._cmd_counts.get('stop', 0)}",
            f"Goodbye commands:  {self._cmd_counts.get('goodbye', 0)}",
        ])
        print()
        banner("✅ ECHO shutdown complete")


# ═══════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════

def main():
    echo = ECHODebug()
    echo.start()


if __name__ == "__main__":
    main()
