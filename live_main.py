#!/usr/bin/env python3
"""
ECHO — Live Mode (Gemini Live API — Full Integration)
=======================================================
Uses the Gemini Live API for real-time bidirectional voice conversation
with full robot hardware control via function calling.

Instead of separate Whisper STT + Gemini Chat + Gemini TTS, this uses a
single streaming connection that handles speech input, audio output,
AND robot control through function calling.

Benefits over main.py:
  - No local STT model needed (saves ~200MB RAM + 2s startup)
  - Lower latency (streaming vs request/response cycle)
  - Natural conversation with interruption support
  - Gemini handles voice activity detection
  - Function calling lets Gemini directly control motors, face, sensors
  - Reconnection with exponential backoff for reliability

Run:
    source venv/bin/activate
    python3 live_main.py
"""

import time
import signal
import logging
import threading

# ─── Logging Setup ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("echo.live_main")

# ─── ANSI Colors for Terminal Output ───
class C:
    INIT    = "\033[96m"   # Cyan
    LISTEN  = "\033[92m"   # Green
    SPEAK   = "\033[93m"   # Yellow
    MOVE    = "\033[95m"   # Magenta
    SENSOR  = "\033[90m"   # Gray
    FUNC    = "\033[94m"   # Blue
    ERROR   = "\033[91m"   # Red
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"

# ─── Import Subsystems ───
from motor_controller import MotorController
from sensor_controller import SensorController
from camera_sentiment import CameraSentiment
from face_display import FaceDisplay
from navigation import NavigationController
from gemini_live import GeminiLiveEngine
from config import CAMERA_WIDTH, CAMERA_HEIGHT


class ECHOLive:
    """
    ECHO robot running in Live API mode with full hardware integration.

    The Gemini Live API handles all voice I/O through a single streaming
    connection. Function calling gives Gemini direct control of:
      - Motors (move, turn, stop)
      - Navigation (follow, patrol, safe_move)
      - Face display (emotions)
      - Sensors (distance, IR)
      - Volume control
      - Shutdown

    Camera frames are sent periodically for visual context.
    Detected facial emotions are sent as text context.
    """

    def __init__(self):
        print(f"\n{C.BOLD}{C.INIT}{'='*55}")
        print(f"  ECHO Robot — Live Mode (Full Integration)")
        print(f"{'='*55}{C.RESET}\n")

        self._running = False

        # Initialize hardware subsystems
        print(f"{C.INIT}[INIT]{C.RESET} Initializing motors...")
        self.motors = MotorController()

        print(f"{C.INIT}[INIT]{C.RESET} Initializing sensors...")
        self.sensors = SensorController()

        print(f"{C.INIT}[INIT]{C.RESET} Initializing camera...")
        self.camera = CameraSentiment()

        print(f"{C.INIT}[INIT]{C.RESET} Initializing face display...")
        self.face = FaceDisplay()

        print(f"{C.INIT}[INIT]{C.RESET} Initializing navigation...")
        self.nav = NavigationController(self.motors, self.sensors, self.camera)

        # Initialize Gemini Live voice engine with all callbacks
        print(f"{C.INIT}[INIT]{C.RESET} Initializing Gemini Live API (with function calling)...")
        self.voice = GeminiLiveEngine(
            on_turn_start=self._on_gemini_speaking,
            on_turn_end=self._on_gemini_done,
            on_text=self._on_gemini_text,
            on_function_call=self._on_function_call,
            on_input_transcript=self._on_user_transcript,
        )

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Safety: stop all motors
        self.motors.stop()
        print(f"\n{C.BOLD}{C.INIT}All subsystems initialized!{C.RESET}\n")

    # ═══════════════════════════════════════════
    # Gemini Live Callbacks
    # ═══════════════════════════════════════════

    def _on_gemini_speaking(self):
        """Called when Gemini starts responding with audio."""
        self.face.set_talking(True)
        print(f"{C.SPEAK}[SPEAK]{C.RESET} ECHO is speaking...")

    def _on_gemini_done(self):
        """Called when Gemini finishes a response turn."""
        self.face.set_talking(False)
        print(f"{C.LISTEN}[LISTEN]{C.RESET} Listening...")

    def _on_gemini_text(self, text: str):
        """Called when Gemini provides response text."""
        print(f"{C.SPEAK}[TEXT]{C.RESET} {text}")

    def _on_user_transcript(self, text: str):
        """Called with the user's transcribed speech."""
        print(f"{C.LISTEN}[USER]{C.RESET} \"{text}\"")

    def _on_function_call(self, name: str, args: dict) -> dict:
        """
        Execute a function call from Gemini to control robot hardware.
        This runs in a thread (called via asyncio.to_thread) so it's
        safe to do blocking operations like motor sleep.

        Returns a result dict that gets sent back to Gemini.
        """
        print(f"{C.FUNC}[FUNC]{C.RESET} {name}({args})")

        try:
            if name == "move_robot":
                return self._func_move(args)
            elif name == "stop_robot":
                return self._func_stop()
            elif name == "start_follow_mode":
                return self._func_follow()
            elif name == "start_patrol_mode":
                return self._func_patrol()
            elif name == "safe_move_forward":
                return self._func_safe_move(args)
            elif name == "set_face_emotion":
                return self._func_set_emotion(args)
            elif name == "get_sensor_data":
                return self._func_get_sensors()
            elif name == "get_camera_emotion":
                return self._func_get_emotion()
            elif name == "set_volume":
                return self._func_set_volume(args)
            elif name == "shutdown_robot":
                return self._func_shutdown()
            else:
                return {"status": "error", "message": f"Unknown function: {name}"}
        except Exception as e:
            logger.error(f"Function call '{name}' error: {e}")
            return {"status": "error", "message": str(e)}

    # ═══════════════════════════════════════════
    # Function Call Implementations
    # ═══════════════════════════════════════════

    def _func_move(self, args: dict) -> dict:
        """Move the robot in a direction."""
        direction = args.get("direction", "forward")
        duration = args.get("duration", 1.0)
        duration = max(0.5, min(10.0, float(duration)))

        print(f"{C.MOVE}[MOVE]{C.RESET} {direction} for {duration}s")
        self.face.set_emotion("neutral")

        if direction == "forward":
            self.motors.forward(duration=duration)
        elif direction == "backward":
            self.motors.backward(duration=duration)
        elif direction == "left":
            self.motors.turn_left(duration=duration)
        elif direction == "right":
            self.motors.turn_right(duration=duration)
        else:
            return {"status": "error", "message": f"Unknown direction: {direction}"}

        return {"status": "ok", "message": f"Moving {direction} for {duration}s"}

    def _func_stop(self) -> dict:
        """Stop all movement."""
        print(f"{C.MOVE}[STOP]{C.RESET} All movement stopped")
        self.nav.stop_continuous()
        self.nav.emergency_stop()
        self.face.set_emotion("neutral")
        return {"status": "ok", "message": "All movement stopped"}

    def _func_follow(self) -> dict:
        """Start follow mode."""
        print(f"{C.MOVE}[FOLLOW]{C.RESET} Starting follow mode")
        self.face.set_emotion("happy")
        self.nav.start_follow()
        return {"status": "ok", "message": "Following person. Say stop to halt."}

    def _func_patrol(self) -> dict:
        """Start patrol mode."""
        print(f"{C.MOVE}[PATROL]{C.RESET} Starting patrol mode")
        self.face.set_emotion("happy")
        self.nav.start_patrol()
        return {"status": "ok", "message": "Patrolling. Say stop to halt."}

    def _func_safe_move(self, args: dict) -> dict:
        """Move forward with obstacle detection."""
        duration = args.get("duration", 8.0)
        duration = max(1.0, min(15.0, float(duration)))
        print(f"{C.MOVE}[SAFE]{C.RESET} Safe forward for {duration}s")
        self.face.set_emotion("neutral")
        self.nav.safe_forward(duration=duration)
        return {"status": "ok", "message": f"Moving carefully for {duration}s with obstacle detection"}

    def _func_set_emotion(self, args: dict) -> dict:
        """Change the face display emotion."""
        emotion = args.get("emotion", "neutral")
        valid = ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]
        if emotion not in valid:
            emotion = "neutral"
        self.face.set_emotion(emotion)
        return {"status": "ok", "message": f"Face set to {emotion}"}

    def _func_get_sensors(self) -> dict:
        """Read sensor data."""
        distance = self.sensors.last_distance
        ir_blocked = self.sensors.last_ir_blocked
        obstacle = self.sensors.is_obstacle_ahead()
        result = {
            "status": "ok",
            "distance_cm": round(distance, 1),
            "ir_obstacle": ir_blocked,
            "obstacle_ahead": obstacle,
        }
        print(f"{C.SENSOR}[SENSOR]{C.RESET} dist={distance:.1f}cm ir={'blocked' if ir_blocked else 'clear'} obstacle={'YES' if obstacle else 'no'}")
        return result

    def _func_get_emotion(self) -> dict:
        """Get current detected facial emotion."""
        emotion = self.camera.current_emotion
        confidence = self.camera.current_confidence
        result = {
            "status": "ok",
            "emotion": emotion,
            "confidence": round(confidence, 2),
        }
        print(f"{C.SENSOR}[EMOTION]{C.RESET} {emotion} ({confidence:.0%})")
        return result

    def _func_set_volume(self, args: dict) -> dict:
        """Adjust volume."""
        direction = args.get("direction", "up")
        delta = 0.25 if direction == "up" else -0.25
        new_vol = self.voice.adjust_volume(delta)
        pct = int(new_vol / 2.0 * 100)
        print(f"{C.FUNC}[VOLUME]{C.RESET} {direction} → {new_vol:.2f}x ({pct}%)")
        return {"status": "ok", "message": f"Volume {direction}: now at {pct}%"}

    def _func_shutdown(self) -> dict:
        """Initiate graceful shutdown."""
        print(f"{C.ERROR}[SHUTDOWN]{C.RESET} Shutdown requested by Gemini")
        self._running = False
        return {"status": "ok", "message": "Shutting down. Goodbye!"}

    # ═══════════════════════════════════════════
    # Main
    # ═══════════════════════════════════════════

    def start(self):
        """Start all services."""
        self._running = True

        # Start background services
        self.sensors.start_monitoring(interval=0.2)
        self.camera.start_analysis()
        self.face.start()

        # Start Gemini Live voice engine
        self.voice.start()

        # Wait for connection
        time.sleep(2)
        self.face.set_emotion("happy")
        time.sleep(1)
        self.face.set_emotion("neutral")

        print(f"\n{C.BOLD}{C.LISTEN}ECHO is running in LIVE mode (full integration)!{C.RESET}")
        print(f"{C.LISTEN}  Speak naturally — Gemini handles everything.{C.RESET}")
        print(f"{C.LISTEN}  Gemini can control motors, face, and sensors via function calling.{C.RESET}")
        print(f"{C.LISTEN}  Press Ctrl+C to stop.{C.RESET}\n")
        print(f"{C.LISTEN}[LISTEN]{C.RESET} Listening...")

        try:
            self._main_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def _main_loop(self):
        """
        Main loop — periodically sends camera frames and emotion context
        to give Gemini visual awareness, and updates face gaze tracking.
        """
        last_frame_time = 0
        last_emotion_text_time = 0
        last_emotion_sent = "neutral"
        FRAME_INTERVAL = 5.0        # Send a camera frame every 5 seconds
        EMOTION_TEXT_INTERVAL = 10.0  # Send emotion context text every 10 seconds

        while self._running and self.voice.is_running:
            now = time.time()

            # ── Send camera frame periodically ──
            if now - last_frame_time > FRAME_INTERVAL:
                try:
                    frame_jpeg = self.camera.get_frame_jpeg()
                    if frame_jpeg:
                        self.voice.send_image(frame_jpeg)
                except Exception:
                    pass
                last_frame_time = now

            # ── Send emotion context as text when it changes ──
            # This helps Gemini adapt its tone even without being asked
            if now - last_emotion_text_time > EMOTION_TEXT_INTERVAL:
                try:
                    emotion = self.camera.current_emotion
                    confidence = self.camera.current_confidence
                    if emotion != last_emotion_sent and confidence > 0.4:
                        self.voice.send_text(
                            f"[CONTEXT: The person's facial expression is {emotion} "
                            f"(confidence: {confidence:.0%}). Adapt your tone accordingly.]"
                        )
                        last_emotion_sent = emotion
                        # Also update the face to mirror detected emotion
                        self.face.set_emotion(emotion)
                except Exception:
                    pass
                last_emotion_text_time = now

            # ── Update face gaze to track detected face ──
            try:
                face_center = self.camera.get_face_center()
                if face_center is not None:
                    cx, cy = face_center
                    gaze_x = (cx / (CAMERA_WIDTH / 2)) - 1.0
                    gaze_y = (cy / (CAMERA_HEIGHT / 2)) - 1.0
                    self.face.set_gaze(gaze_x, gaze_y)
            except Exception:
                pass

            time.sleep(0.1)

    # ═══════════════════════════════════════════
    # Shutdown
    # ═══════════════════════════════════════════

    def _signal_handler(self, signum, frame):
        print(f"\n{C.ERROR}Signal {signum} received — shutting down...{C.RESET}")
        self._running = False

    def shutdown(self):
        print(f"\n{C.BOLD}{C.INIT}{'='*55}")
        print(f"  ECHO Robot — Shutting Down...")
        print(f"{'='*55}{C.RESET}")

        self._running = False

        self.face.set_emotion("sad")
        time.sleep(0.5)

        # Cleanup in reverse order
        self.voice.cleanup()
        self.nav.cleanup()
        self.face.cleanup()
        self.camera.cleanup()
        self.sensors.cleanup()
        self.motors.cleanup()

        print(f"{C.BOLD}{C.INIT}ECHO shutdown complete. Goodbye!{C.RESET}")


def main():
    echo = ECHOLive()
    echo.start()


if __name__ == "__main__":
    main()
