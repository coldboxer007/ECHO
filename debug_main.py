#!/usr/bin/env python3
"""
ECHO — Debug Main (Verbose Terminal Output)
=============================================
Same interaction loop as main.py but prints ALL inputs, outputs,
sensor data, and system state to the terminal for debugging.

Run:
    python3 debug_main.py

Everything the robot hears, thinks, says, and senses is printed
with timestamps and color-coded tags.
"""

import sys
import time
import signal
import logging
import threading
import traceback
from datetime import datetime

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

# ─── Import Subsystems ───
from config import FOLLOW_MODE_ENABLED
from motor_controller import MotorController
from sensor_controller import SensorController
from camera_sentiment import CameraSentiment
from speech_engine import SpeechEngine
from gemini_brain import GeminiBrain
from face_display import FaceDisplay
from navigation import NavigationController


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
# Debug ECHO Controller
# ═══════════════════════════════════════════════════

class ECHODebug:
    """
    Same as ECHO but with verbose terminal output for every step.
    All inputs (speech, emotion, sensors) and outputs (responses,
    motor commands) are printed in real-time.
    """

    def __init__(self):
        banner("🤖 ECHO Robot — DEBUG MODE")

        self._running = False

        # ── Initialize all subsystems ──
        dp("INIT", "Initializing motors...", C_CYAN)
        self.motors = MotorController()
        dp("INIT", "✅ Motors ready", C_GREEN)

        dp("INIT", "Initializing sensors...", C_CYAN)
        self.sensors = SensorController()
        dp("INIT", "✅ Sensors ready", C_GREEN)

        dp("INIT", "Initializing camera & sentiment...", C_CYAN)
        self.camera = CameraSentiment()
        dp("INIT", "✅ Camera ready", C_GREEN)

        dp("INIT", "Initializing speech engine...", C_CYAN)
        self.speech = SpeechEngine()
        dp("INIT", "✅ Speech engine ready", C_GREEN)

        dp("INIT", "Initializing AI brain...", C_CYAN)
        self.brain = GeminiBrain()
        dp("INIT", "✅ AI brain ready", C_GREEN)

        dp("INIT", "Initializing face display...", C_CYAN)
        self.face = FaceDisplay()
        dp("INIT", "✅ Face display ready", C_GREEN)

        dp("INIT", "Initializing navigation...", C_CYAN)
        self.nav = NavigationController(self.motors, self.sensors, self.camera)
        dp("INIT", "✅ Navigation ready", C_GREEN)

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Safety stop
        self.motors.stop()

        # Stats
        self._loop_count = 0
        self._listen_count = 0
        self._speak_count = 0
        self._cmd_counts = {"chat": 0, "move": 0, "follow": 0, "stop": 0}

        banner("✅ All subsystems initialized — DEBUG MODE")

    # ═══════════════════════════════════════════
    # Start / Main Loop
    # ═══════════════════════════════════════════

    def start(self):
        """Start all services and enter the debug main loop."""
        self._running = True

        # Start background services
        self.sensors.start_monitoring(interval=0.2)
        self.camera.start_analysis()
        self.face.start()

        # Start sensor monitor thread
        self._sensor_thread = threading.Thread(
            target=self._sensor_debug_loop, daemon=True
        )
        self._sensor_thread.start()

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
            "Press Ctrl+C to stop.",
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
                dist = self.sensors.distance_cm
                ir = self.sensors.ir_obstacle
                obstacle = self.sensors.obstacle_detected
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

    def _main_loop(self):
        """Core debug loop — same as main.py but with print statements."""
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

                # ── Step 3: Parse command ──
                command = self.brain.interpret_command(user_text)
                cmd_type = command['type']
                cmd_dir = command.get('direction', '')
                dp("COMMAND", f"Type: {cmd_type}" + (f" → {cmd_dir}" if cmd_dir else ""), C_MAGENTA)
                self._cmd_counts[cmd_type] = self._cmd_counts.get(cmd_type, 0) + 1

                # ── Step 4: Execute ──
                if cmd_type == 'move':
                    self._handle_move(command)
                elif cmd_type == 'follow':
                    self._handle_follow()
                elif cmd_type == 'stop':
                    self._handle_stop()
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
    # Command Handlers (with debug output)
    # ═══════════════════════════════════════════

    def _handle_move(self, command: dict):
        """Handle movement with debug output."""
        direction = command['direction']
        dp("MOVE", f"Executing movement: {direction}", C_CYAN)

        self.face.set_emotion("neutral")

        ack = {
            'forward':  "Moving forward!",
            'backward': "Going backward!",
            'left':     "Turning left!",
            'right':    "Turning right!",
        }.get(direction, "Moving!")

        # Speak in thread
        t = threading.Thread(target=self.speech.speak, args=(ack,), daemon=True)
        t.start()
        self._speak_count += 1
        dp("SPEAK", f"🔊 \"{ack}\"", C_MAGENTA)

        success = self.nav.execute_move(direction)
        dp("MOVE", f"Result: {'✅ SUCCESS' if success else '❌ BLOCKED'}", C_GREEN if success else C_RED)

        if not success:
            self.speech.speak("I can't move that way, there's something in front of me!")
            self._speak_count += 1
            self.face.set_emotion("surprise")
            time.sleep(1)
            self.face.set_emotion("neutral")

    def _handle_follow(self):
        """Handle follow mode with debug output."""
        dp("FOLLOW", "Starting follow mode", C_MAGENTA)
        self.face.set_emotion("happy")
        self.speech.speak("I'll follow you! Say stop when you want me to stop.", emotion="happy")
        self._speak_count += 1
        dp("SPEAK", "🔊 Follow mode announcement", C_MAGENTA)
        self.nav.start_follow()

    def _handle_stop(self):
        """Handle stop with debug output."""
        dp("STOP", "🛑 Emergency stop!", C_RED)
        self.nav.emergency_stop()
        self.face.set_emotion("neutral")
        self.speech.speak("Stopping!", emotion="neutral")
        self._speak_count += 1

    def _handle_chat(self, user_text: str, emotion: str, confidence: float):
        """Handle conversation with full debug output."""
        self.face.set_emotion(emotion)

        dp("BRAIN", f"Sending to Gemini...", C_MAGENTA)
        dp("BRAIN", f"  Input: \"{user_text}\"", C_MAGENTA)
        dp("BRAIN", f"  Emotion context: {emotion} ({confidence:.0%})", C_MAGENTA)

        response = self.brain.think(user_text, emotion, confidence)

        if not response:
            response = "Hmm, I'm not sure what to say about that."
            dp("BRAIN", "⚠️  No response from Gemini, using default", C_YELLOW)

        dp("OUTPUT", f"🤖 ECHO says: \"{response}\"", C_GREEN)

        # Determine response emotion
        response_emotion = self.brain.determine_response_emotion(response, emotion)
        dp("EMOTION", f"Response emotion: {response_emotion}", C_CYAN)

        # Speak with emotion
        self.face.set_emotion(response_emotion)
        self.face.set_talking(True)

        dp("SPEAK", f"🔊 Speaking with emotion={response_emotion}...", C_MAGENTA)
        self.speech.speak(response, emotion=response_emotion)
        self._speak_count += 1

        self.face.set_talking(False)
        time.sleep(0.5)
        self.face.set_emotion("neutral")

    # ═══════════════════════════════════════════
    # Shutdown
    # ═══════════════════════════════════════════

    def _signal_handler(self, signum, frame):
        dp("SIGNAL", f"Signal {signum} received — shutting down", C_RED)
        self._running = False

    def shutdown(self):
        """Clean shutdown with summary."""
        banner("🤖 ECHO — Shutting Down (DEBUG)")

        self._running = False

        self.face.set_emotion("sad")
        try:
            self.speech.speak("Goodbye from debug mode!", emotion="sad")
        except Exception:
            pass

        # Cleanup all subsystems
        self.nav.cleanup()
        self.face.cleanup()
        self.brain.cleanup()
        self.speech.cleanup()
        self.camera.cleanup()
        self.sensors.cleanup()
        self.motors.cleanup()

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
