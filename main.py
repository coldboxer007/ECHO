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
                emotion = self.camera.current_emotion
                confidence = self.camera.current_confidence
                logger.info(f"😊 Detected emotion: {emotion} ({confidence:.0%})")

                # ── Step 3: Interpret command ──
                command = self.brain.interpret_command(user_text)
                logger.info(f"🎯 Command type: {command['type']}")

                if command['type'] == 'move':
                    self._handle_move(command)

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

        # Update face
        self.face.set_emotion("neutral")

        # Acknowledge
        ack_messages = {
            'forward':  "Moving forward!",
            'backward': "Going backward!",
            'left':     "Turning left!",
            'right':    "Turning right!",
        }
        ack = ack_messages.get(direction, "Moving!")

        # Speak acknowledgment in a thread so movement starts quickly
        speak_thread = threading.Thread(
            target=self.speech.speak, args=(ack,), daemon=True
        )
        speak_thread.start()

        # Execute movement
        success = self.nav.execute_move(direction)

        if not success:
            self.speech.speak("I can't move that way, there's something in front of me!")
            self.face.set_emotion("surprise")
            time.sleep(1)
            self.face.set_emotion("neutral")

    def _handle_follow(self):
        """Handle follow-me command."""
        self.face.set_emotion("happy")
        self.speech.speak("I'll follow you! Say stop when you want me to stop.", emotion="happy")
        self.nav.start_follow()

    def _handle_stop(self):
        """Handle stop command."""
        self.nav.emergency_stop()
        self.face.set_emotion("neutral")
        self.speech.speak("Stopping!", emotion="neutral")

    def _handle_chat(self, user_text: str, emotion: str, confidence: float):
        """Handle conversational input."""
        # Update face to match detected emotion
        self.face.set_emotion(emotion)

        # Get Gemini response
        response = self.brain.think(user_text, emotion, confidence)

        if not response:
            response = "Hmm, I'm not sure what to say about that."

        # Determine response emotion (could be different from detected)
        # Simple heuristic: match the user's emotion for empathy
        response_emotion = emotion

        # Start talking animation
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
