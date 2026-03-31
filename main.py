#!/usr/bin/env python3
"""
ECHO — Emotionally Connected Humanoid Observer
=================================================
Main orchestrator that connects all subsystems:
  • Camera + TFLite sentiment analysis
  • Faster-Whisper speech-to-text
  • Gemini AI brain (conversation + emotion-aware responses)
  • Gemini TTS (expressive text-to-speech)
  • L298N motor control (voice-commanded + follow mode)
  • Ultrasonic + IR obstacle avoidance
  • Pygame animated face display

Run on Raspberry Pi 4B:
    source venv/bin/activate
    python3 main.py
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
logger = logging.getLogger("echo.main")

# ─── Import Subsystems ───
from config import FOLLOW_MODE_ENABLED
from motor_controller import MotorController
from sensor_controller import SensorController
from camera_sentiment import CameraSentiment
from speech_engine import SpeechEngine
from gemini_brain import GeminiBrain
from face_display import FaceDisplay
from navigation import NavigationController


class ECHO:
    """
    Main robot controller. Wires up all subsystems and runs the
    listen → think → speak → act loop.
    """

    def __init__(self):
        logger.info("=" * 50)
        logger.info("  🤖 ECHO Robot — Starting Up...")
        logger.info("=" * 50)

        self._running = False

        # Initialize subsystems (order matters)
        logger.info("Initializing motors...")
        self.motors = MotorController()

        logger.info("Initializing sensors...")
        self.sensors = SensorController()

        logger.info("Initializing camera & sentiment...")
        self.camera = CameraSentiment()

        logger.info("Initializing speech engine...")
        self.speech = SpeechEngine()

        logger.info("Initializing AI brain...")
        self.brain = GeminiBrain()

        logger.info("Initializing face display...")
        self.face = FaceDisplay()

        logger.info("Initializing navigation...")
        self.nav = NavigationController(self.motors, self.sensors, self.camera)

        # Register signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Safety: ensure all motors are stopped before starting
        self.motors.stop()
        logger.info("✅ All subsystems initialized!")

    def start(self):
        """Start all background services and enter the main loop."""
        self._running = True

        # Start background services
        self.sensors.start_monitoring(interval=0.2)
        self.camera.start_analysis()
        self.face.start()

        # Startup greeting
        time.sleep(1)
        self.face.set_emotion("happy")
        self.speech.speak(
            "Hello! I'm Echo, your companion robot. How can I help you today?",
            emotion="happy"
        )
        self.face.set_emotion("neutral")

        logger.info("🟢 ECHO is running! Listening for commands...")

        # Main interaction loop
        try:
            self._main_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def _main_loop(self):
        """
        Core loop:
        1. Listen for voice input
        2. Analyze sentiment from camera
        3. Interpret command (move vs chat)
        4. For chat: send to Gemini with emotion context → speak response
        5. For move: execute movement
        6. Update face display
        """
        while self._running:
            try:
                # Skip listening while speaking or in follow mode
                if self.speech.is_speaking:
                    time.sleep(0.1)
                    continue

                # ── Step 1: Listen ──
                user_text = self.speech.listen()

                if not user_text:
                    time.sleep(0.1)
                    continue

                logger.info(f"👤 User said: '{user_text}'")

                # ── Step 2: Get current emotion from camera ──
                # Use local TFLite model only (skip Gemini API fallback to save ~4s latency)
                emotion = self.camera.current_emotion
                confidence = self.camera.current_confidence
                logger.info(f"😊 Detected emotion: {emotion} ({confidence:.0%})")

                # ── Step 3: Interpret command ──
                command = self.brain.interpret_command(user_text)
                logger.info(f"🎯 Command type: {command['type']}")

                if command['type'] == 'move':
                    self._handle_move(command)

                elif command['type'] == 'keep_moving':
                    self._handle_keep_moving(command)

                elif command['type'] == 'safe_move':
                    self._handle_safe_move(command)

                elif command['type'] == 'patrol':
                    self._handle_patrol()

                elif command['type'] == 'follow':
                    self._handle_follow()

                elif command['type'] == 'stop':
                    self._handle_stop()

                elif command['type'] == 'chat':
                    self._handle_chat(user_text, emotion, confidence)

            except Exception as e:
                logger.error(f"Main loop error: {e}", exc_info=True)
                time.sleep(1)

    # ═══════════════════════════════════════════
    # Command Handlers
    # ═══════════════════════════════════════════

    def _handle_move(self, command: dict):
        """Handle movement voice commands."""
        direction = command['direction']
        duration = command.get('duration', None)
        logger.info(f"🚗 Executing move: {direction}{f' for {duration}s' if duration else ''}")

        # Update face
        self.face.set_emotion("neutral")

        # Execute movement FIRST (don't wait for TTS)
        success = self.nav.execute_move(direction, duration=duration)

        if not success:
            # Only speak on failure (obstacle blocked)
            self.speech._speak_fallback("Obstacle ahead!")
            self.face.set_emotion("surprise")
            time.sleep(0.5)
            self.face.set_emotion("neutral")

    def _handle_keep_moving(self, command: dict):
        """Handle continuous movement commands (keep going until stop)."""
        direction = command.get('direction', 'forward')
        logger.info(f"🔄 Starting continuous {direction} movement")
        self.face.set_emotion("neutral")
        self.speech._speak_fallback(f"Moving {direction}. Say stop to halt.")
        self.nav.start_continuous_move(direction)

    def _handle_safe_move(self, command: dict):
        """Handle obstacle-aware careful movement."""
        direction = command.get('direction', 'forward')
        logger.info(f"🛡️ Safe move: {direction} with obstacle checking")
        self.face.set_emotion("neutral")
        self.speech._speak_fallback(f"Moving carefully. I'll stop if I see an obstacle.")
        success = self.nav.safe_forward(duration=8.0)
        if not success:
            self.speech._speak_fallback("I stopped because I detected an obstacle ahead.")
            self.face.set_emotion("surprise")
            time.sleep(0.5)
            self.face.set_emotion("neutral")
        else:
            self.speech._speak_fallback("Done! Path was clear.")

    def _handle_patrol(self):
        """Handle patrol / back-and-forth movement."""
        logger.info("🔄 Starting patrol mode")
        self.face.set_emotion("happy")
        self.speech._speak_fallback("Patrolling! Say stop when you want me to halt.")
        self.nav.start_patrol()

    def _handle_follow(self):
        """Handle follow-me command."""
        self.face.set_emotion("happy")
        self.speech.speak("I'll follow you! Say stop when you want me to stop.", emotion="happy")
        self.nav.start_follow()

    def _handle_stop(self):
        """Handle stop command."""
        self.nav.stop_continuous()  # Stop continuous/patrol modes too
        self.nav.emergency_stop()
        self.face.set_emotion("neutral")
        self.speech._speak_fallback("Stopped!")

    def _handle_chat(self, user_text: str, emotion: str, confidence: float):
        """Handle conversational input."""
        # Show detected user emotion while thinking
        self.face.set_emotion(emotion)

        # Get Gemini response
        response = self.brain.think(user_text, emotion, confidence)

        if not response:
            response = "Hmm, I'm not sure what to say about that."

        # Determine response emotion based on what ECHO says
        response_emotion = self.brain.determine_response_emotion(response, emotion)
        logger.info(f"🎭 Response emotion: {response_emotion}")

        # Switch face to response emotion and start talking
        self.face.set_emotion(response_emotion)
        self.face.set_talking(True)

        # Speak the response
        self.speech.speak(response, emotion=response_emotion)

        # Stop talking animation
        self.face.set_talking(False)

        # Return face to a gentle neutral after speaking
        time.sleep(0.5)
        self.face.set_emotion("neutral")

    # ═══════════════════════════════════════════
    # Shutdown
    # ═══════════════════════════════════════════

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C and SIGTERM gracefully."""
        logger.info(f"Signal {signum} received — shutting down...")
        self._running = False

    def shutdown(self):
        """Clean up all subsystems."""
        logger.info("=" * 50)
        logger.info("  🤖 ECHO Robot — Shutting Down...")
        logger.info("=" * 50)

        self._running = False

        # Goodbye
        self.face.set_emotion("sad")
        try:
            self.speech.speak("Goodbye! See you soon.", emotion="sad")
        except Exception:
            pass

        # Cleanup in reverse order
        self.nav.cleanup()
        self.face.cleanup()
        self.brain.cleanup()
        self.speech.cleanup()
        self.camera.cleanup()
        self.sensors.cleanup()
        self.motors.cleanup()

        logger.info("✅ ECHO shutdown complete. Goodbye!")


# ═══════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════

def main():
    """Create and start ECHO."""
    echo = ECHO()
    echo.start()


if __name__ == "__main__":
    main()
