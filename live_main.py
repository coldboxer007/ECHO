#!/usr/bin/env python3
"""
ECHO — Live Mode (Gemini Live API)
=====================================
Uses the Gemini Live API for real-time bidirectional voice conversation.
Instead of separate Whisper STT + Gemini Chat + Gemini TTS, this uses a
single streaming connection that handles both speech input AND audio output.

Benefits:
  • No local STT model needed (saves RAM + startup time)
  • Lower latency (streaming vs request/response)
  • Natural conversation with interruption support
  • Gemini handles voice activity detection

Run:
    source venv/bin/activate
    python3 live_main.py
"""

import sys
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

# ─── ANSI Colors for Debug Output ───
class C:
    INIT    = "\033[96m"   # Cyan
    LISTEN  = "\033[92m"   # Green
    SPEAK   = "\033[93m"   # Yellow
    MOVE    = "\033[95m"   # Magenta
    SENSOR  = "\033[90m"   # Gray
    ERROR   = "\033[91m"   # Red
    RESET   = "\033[0m"
    BOLD    = "\033[1m"

# ─── Import Subsystems ───
from motor_controller import MotorController
from sensor_controller import SensorController
from camera_sentiment import CameraSentiment
from face_display import FaceDisplay
from navigation import NavigationController
from gemini_live import GeminiLiveEngine


class ECHOLive:
    """
    ECHO robot running in Live API mode.
    The Gemini Live API handles all voice I/O through a single streaming connection.
    Camera, motors, sensors, and face display still work as before.
    """

    def __init__(self):
        print(f"\n{C.BOLD}{C.INIT}{'='*50}")
        print(f"  🤖 ECHO Robot — Live Mode Starting Up...")
        print(f"{'='*50}{C.RESET}\n")

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

        # Initialize Gemini Live voice engine
        print(f"{C.INIT}[INIT]{C.RESET} Initializing Gemini Live API voice engine...")
        self.voice = GeminiLiveEngine(
            on_turn_start=self._on_gemini_speaking,
            on_turn_end=self._on_gemini_done,
            on_text=self._on_gemini_text,
        )

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Safety: stop all motors
        self.motors.stop()
        print(f"\n{C.BOLD}{C.INIT}✅ All subsystems initialized!{C.RESET}\n")

    # ═══════════════════════════════════════════
    # Gemini Live Callbacks
    # ═══════════════════════════════════════════

    def _on_gemini_speaking(self):
        """Called when Gemini starts responding with audio."""
        self.face.set_talking(True)
        print(f"{C.SPEAK}[SPEAK]{C.RESET} 🤖 ECHO is speaking...")

    def _on_gemini_done(self):
        """Called when Gemini finishes a response turn."""
        self.face.set_talking(False)
        self.face.set_emotion("neutral")
        print(f"{C.LISTEN}[LISTEN]{C.RESET} 🎤 Listening...")

    def _on_gemini_text(self, text: str):
        """Called when Gemini provides text (transcription or response)."""
        print(f"{C.SPEAK}[TEXT]{C.RESET} 📝 {text}")

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

        print(f"\n{C.BOLD}{C.LISTEN}🟢 ECHO is running in LIVE mode!{C.RESET}")
        print(f"{C.LISTEN}   Speak naturally — Gemini handles everything.{C.RESET}")
        print(f"{C.LISTEN}   Press Ctrl+C to stop.{C.RESET}\n")
        print(f"{C.LISTEN}[LISTEN]{C.RESET} 🎤 Listening...")

        # Periodically send camera frames for visual context
        try:
            self._main_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def _main_loop(self):
        """
        Main loop — periodically sends camera frames to give Gemini
        visual context, and monitors sensors for obstacle avoidance.
        """
        last_frame_time = 0
        last_sensor_time = 0
        FRAME_INTERVAL = 3.0    # Send a camera frame every 3 seconds
        SENSOR_INTERVAL = 1.0   # Check sensors every second

        while self._running and self.voice.is_running:
            now = time.time()

            # Send camera frame periodically
            if now - last_frame_time > FRAME_INTERVAL:
                try:
                    frame_jpeg = self.camera.get_frame_jpeg()
                    if frame_jpeg:
                        self.voice.send_image(frame_jpeg)
                except Exception:
                    pass
                last_frame_time = now

            # Update face with detected emotion
            try:
                emotion = self.camera.current_emotion
                if emotion and emotion != "neutral":
                    self.face.set_emotion(emotion)
            except Exception:
                pass

            # Check sensors
            if now - last_sensor_time > SENSOR_INTERVAL:
                try:
                    distance = self.sensors.read_distance()
                    ir_obstacle = self.sensors.read_ir()
                    if distance is not None and distance < 15:
                        self.motors.stop()
                except Exception:
                    pass
                last_sensor_time = now

            time.sleep(0.1)

    # ═══════════════════════════════════════════
    # Shutdown
    # ═══════════════════════════════════════════

    def _signal_handler(self, signum, frame):
        print(f"\n{C.ERROR}Signal {signum} received — shutting down...{C.RESET}")
        self._running = False

    def shutdown(self):
        print(f"\n{C.BOLD}{C.INIT}{'='*50}")
        print(f"  🤖 ECHO Robot — Shutting Down...")
        print(f"{'='*50}{C.RESET}")

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

        print(f"{C.BOLD}{C.INIT}✅ ECHO shutdown complete. Goodbye!{C.RESET}")


def main():
    echo = ECHOLive()
    echo.start()


if __name__ == "__main__":
    main()
