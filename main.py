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

import time
import signal
import logging
import threading
import queue

# ─── Logging Setup ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("echo.main")

# ─── Import Subsystems ───
from motor_controller import MotorController
from sensor_controller import SensorController
from camera_sentiment import CameraSentiment
from speech_engine import SpeechEngine
from gemini_brain import GeminiBrain
from face_display import FaceDisplay
from navigation import NavigationController
from config import WAKE_WORD_ENABLED, WAKE_WORD_PHRASES, CAMERA_WIDTH, CAMERA_HEIGHT


class ECHO:
    """
    Main robot controller. Wires up all subsystems and runs the
    listen → think → speak → act loop.
    """

    # Minimum seconds between Gemini vision emotion fallback API calls
    _EMOTION_FALLBACK_COOLDOWN = 10.0

    # Words/phrases that indicate the input is conversation, not a movement command.
    # Used to skip expensive Gemini NLP classification for obvious chat.
    _CHAT_INDICATORS = frozenset({
        'what', 'why', 'how', 'when', 'where', 'who',
        'tell', 'explain', 'think', 'feel', 'know',
        'like', 'love', 'hate', 'want', 'need',
        'can you', 'do you', 'are you', 'is it',
        'have you', "what's", "how's", "who's",
    })

    def __init__(self):
        logger.info("=" * 50)
        logger.info("  ECHO Robot — Starting Up...")
        logger.info("=" * 50)

        self._running = False
        self._last_emotion_fallback = 0.0  # monotonic timestamp of last Gemini emotion call

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

        # ── Wire up mouth sync callback ──
        # When speech_engine starts/stops playing audio, it fires this callback
        # so the face display can precisely track when the mouth should move.
        self.speech.set_talking_callback(self._on_talking_changed)

        # Safety: ensure all motors are stopped before starting
        self.motors.stop()
        logger.info("All subsystems initialized!")

    def start(self):
        """Start all background services and enter the main loop."""
        self._running = True

        # Start background services
        self.sensors.start_monitoring(interval=0.2)
        self.camera.start_analysis()
        self.face.start()

        # Startup greeting
        time.sleep(0.5)  # Brief pause for background services to initialize
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
                self.face.set_state("listening")
                user_text = self.speech.listen()
                self.face.set_state(None)

                if not user_text:
                    time.sleep(0.1)
                    continue

                # ── Wake word gate (optional) ──
                # When enabled, discard speech that doesn't start with a wake phrase.
                if WAKE_WORD_ENABLED:
                    text_lower = user_text.lower().strip()
                    matched_phrase = None
                    for phrase in WAKE_WORD_PHRASES:
                        if text_lower.startswith(phrase):
                            matched_phrase = phrase
                            break

                    if matched_phrase is None:
                        logger.debug(f"Wake word not detected, ignoring: '{user_text}'")
                        continue

                    # Strip the wake phrase from the command
                    user_text = user_text[len(matched_phrase):].strip(" ,!.")
                    if not user_text:
                        # Just the wake word with nothing after — acknowledge and listen again
                        self.speech.speak("Yes?", emotion="neutral")
                        continue

                logger.info(f"User said: '{user_text}'")

                # ── Step 2: Get current emotion from camera ──
                # Use local TFLite model; fall back to Gemini vision if low confidence
                # (rate-limited to avoid wasteful API calls every listen cycle)
                emotion = self.camera.current_emotion
                confidence = self.camera.current_confidence
                if emotion == "neutral" and confidence < 0.3:
                    now = time.monotonic()
                    if (now - self._last_emotion_fallback) >= self._EMOTION_FALLBACK_COOLDOWN:
                        frame_jpeg = self.camera.get_frame_jpeg()
                        if frame_jpeg:
                            try:
                                emotion, confidence = self.brain.analyze_emotion_from_image(frame_jpeg)
                                self._last_emotion_fallback = now
                            except Exception as e:
                                logger.debug(f"Gemini emotion fallback failed: {e}")
                logger.info(f"😊 Detected emotion: {emotion} ({confidence:.0%})")

                # ── Step 2b: Update face gaze to track detected face ──
                face_center = self.camera.get_face_center()
                if face_center is not None:
                    cx, cy = face_center
                    gaze_x = (cx / (CAMERA_WIDTH / 2)) - 1.0
                    gaze_y = (cy / (CAMERA_HEIGHT / 2)) - 1.0
                    self.face.set_gaze(gaze_x, gaze_y)

                # ── Step 3: Interpret command ──
                # Fast local keyword matching first; if it returns 'chat',
                # try NLP classification for natural language movement phrases
                # (e.g. "come ahead", "move closer") before treating as conversation.
                command = self.brain.interpret_command(user_text)

                if command['type'] == 'chat':
                    # Hybrid NLP: ask Gemini only for SHORT ambiguous phrases
                    # that might be natural-language movement commands.
                    # Skip NLP for obvious conversation (questions, long sentences)
                    # to save 1-2s of API latency per interaction.
                    words = user_text.split()
                    text_lower_strip = user_text.lower().strip()
                    is_likely_chat = (
                        len(words) > 6  # Long sentences are almost never movement
                        or any(text_lower_strip.startswith(w) for w in self._CHAT_INDICATORS)
                        or '?' in user_text  # Questions are conversation
                    )
                    if not is_likely_chat:
                        nlp_command = self.brain.interpret_command_nlp(user_text)
                        if nlp_command['type'] != 'chat':
                            command = nlp_command
                            logger.info(f"🧠 NLP reclassified '{user_text}' → {command['type']}")

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

                elif command['type'] == 'goodbye':
                    self._handle_goodbye()

                elif command['type'] == 'clear_history':
                    self._handle_clear_history()

                elif command['type'] == 'look':
                    self._handle_look()

                elif command['type'] == 'volume':
                    self._handle_volume(command)

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

        # Speak acknowledgment in background (non-blocking so movement starts fast)
        ack = {
            'forward':  "Moving forward!",
            'backward': "Going backward!",
            'left':     "Turning left!",
            'right':    "Turning right!",
        }.get(direction, "Moving!")
        ack_thread = threading.Thread(
            target=self.speech.speak, args=(ack,),
            kwargs={'emotion': 'neutral'}, daemon=True,
        )
        ack_thread.start()

        # Execute movement
        success = self.nav.execute_move(direction, duration=duration)

        if not success:
            # Speak on failure (obstacle blocked)
            self.speech.speak("Obstacle ahead!", emotion="surprise")
            self.face.set_emotion("surprise")
            # Face transitions back to neutral after a brief display;
            # the FaceDisplay class handles smooth transitions internally.
            self.face.set_emotion("neutral")

    def _handle_keep_moving(self, command: dict):
        """Handle continuous movement commands (keep going until stop)."""
        direction = command.get('direction', 'forward')
        logger.info(f"🔄 Starting continuous {direction} movement")
        self.face.set_emotion("neutral")
        self.speech.speak(f"Moving {direction}. Say stop to halt.", emotion="neutral")
        self.nav.start_continuous_move(direction)

    def _handle_safe_move(self, command: dict):
        """Handle obstacle-aware careful movement."""
        direction = command.get('direction', 'forward')
        logger.info(f"🛡️ Safe move: {direction} with obstacle checking")
        self.face.set_emotion("neutral")
        self.speech.speak(f"Moving carefully. I'll stop if I see an obstacle.", emotion="neutral")
        # safe_forward now runs in background — non-blocking so robot stays responsive
        self.nav.safe_forward(duration=8.0)

    def _handle_patrol(self):
        """Handle patrol / back-and-forth movement."""
        logger.info("🔄 Starting patrol mode")
        self.face.set_emotion("happy")
        self.speech.speak("Patrolling! Say stop when you want me to halt.", emotion="happy")
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
        self.speech.speak("Stopped!", emotion="neutral")

    def _handle_goodbye(self):
        """Handle goodbye command — triggers graceful full shutdown."""
        logger.info("👋 Goodbye command received — initiating shutdown")
        self.nav.stop_continuous()
        self.nav.emergency_stop()
        self._running = False  # Breaks main loop → triggers shutdown() in start()

    def _handle_clear_history(self):
        """Handle clear conversation history command."""
        self.brain.clear_history()
        self.face.set_emotion("neutral")
        self.speech.speak("Conversation cleared! Let's start fresh.", emotion="neutral")
        logger.info("Conversation history cleared by voice command")

    def _handle_look(self):
        """Handle 'what do you see' — sends camera frame to Gemini for scene description."""
        self.face.set_emotion("surprise")
        self.speech.speak("Let me take a look!", emotion="surprise")

        frame_jpeg = self.camera.get_frame_jpeg()
        if frame_jpeg is None:
            self.speech.speak("I can't see anything right now. My camera might be off.", emotion="sad")
            self.face.set_emotion("sad")
            self.face.set_emotion("neutral")
            return

        self.face.set_state("thinking")  # Thinking indicator while analyzing
        description = self.brain.analyze_scene(frame_jpeg)
        self.face.set_state(None)

        if description:
            response_emotion = self.brain.determine_response_emotion(description, "neutral")
            self.face.set_emotion(response_emotion)
            self.speech.speak(description, emotion=response_emotion)
        else:
            self.speech.speak("I looked but I'm having trouble describing what I see.", emotion="neutral")

        self.face.set_emotion("neutral")

    def _handle_volume(self, command: dict):
        """Handle volume up/down voice commands."""
        direction = command.get('direction', 'up')
        delta = 0.25 if direction == 'up' else -0.25
        new_vol = self.speech.adjust_volume(delta)
        pct = int(new_vol / 2.0 * 100)  # 2.0 is max → 100%
        if direction == 'up':
            self.speech.speak(f"Volume up! Now at {pct} percent.", emotion="neutral")
        else:
            self.speech.speak(f"Volume down. Now at {pct} percent.", emotion="neutral")
        logger.info(f"Volume adjusted {direction}: {new_vol:.2f}x ({pct}%)")

    def _handle_chat(self, user_text: str, emotion: str, confidence: float):
        """Handle conversational input with pipelined think→TTS→play pipeline.

        Architecture (producer-consumer pattern):
        1. Show "thinking" face state + play audio cue
        2. Producer thread: reads sentences from Gemini stream, generates TTS
           audio for each, and puts (sentence, audio_bytes, rate) into a queue
        3. Consumer (this thread): pulls from queue and plays audio in order
        4. Mouth animation is synced to actual audio playback via callback

        Because TTS generation (~1-3s) overlaps with audio playback (~1-3s),
        sentences after the first play with near-zero gap between them.
        """
        # Show detected user emotion while thinking
        self.face.set_emotion(emotion)

        # "Thinking" state — distinct from talking (eyes look up, processing look)
        self.face.set_state("thinking")

        # Play brief audio cue so user knows ECHO heard them
        self.speech.play_thinking_cue()

        # ── Producer-consumer TTS pipeline ──
        # Queue holds: (sentence, audio_bytes, sample_rate) or None as sentinel
        tts_queue = queue.Queue(maxsize=4)
        response_emotion_ref = [emotion]  # Mutable ref so producer can read consumer's update

        def _tts_producer():
            """Read sentences from Gemini stream, generate TTS, enqueue results."""
            first = True
            try:
                for sentence in self.brain.think_stream(user_text, emotion, confidence):
                    if first:
                        # Determine response emotion from first sentence
                        resp_emo = self.brain.determine_response_emotion(sentence, emotion)
                        response_emotion_ref[0] = resp_emo

                    # Generate TTS audio (this is the expensive step: 1-3s)
                    audio_bytes, audio_rate = self.speech.generate_tts_audio(
                        sentence, emotion=response_emotion_ref[0]
                    )
                    tts_queue.put((sentence, audio_bytes, audio_rate, first))
                    first = False
            except Exception as e:
                logger.error(f"TTS producer error: {e}")
            finally:
                tts_queue.put(None)  # Sentinel: stream exhausted

        # Start producer thread
        producer = threading.Thread(target=_tts_producer, daemon=True)
        producer.start()

        # ── Consumer: play audio in order ──
        full_response = ""
        first_sentence = True
        response_emotion = emotion

        while True:
            try:
                item = tts_queue.get(timeout=30)  # 30s timeout for safety
            except queue.Empty:
                logger.warning("TTS pipeline timeout — no more sentences after 30s")
                break

            if item is None:
                # Sentinel — stream exhausted
                break

            sentence, audio_bytes, audio_rate, is_first = item

            if is_first and first_sentence:
                self.face.set_state(None)  # Clear thinking state
                first_sentence = False
                response_emotion = response_emotion_ref[0]
                logger.info(f"Response emotion: {response_emotion}")
                self.face.set_emotion(response_emotion)

            full_response += sentence + " "

            if audio_bytes:
                # Play pre-generated audio (mouth sync handled by callback)
                logger.info(f"Playing TTS: '{sentence[:50]}...'")
                self.speech.play_audio(audio_bytes, audio_rate)
            else:
                # TTS generation failed — fall back to synchronous speak
                logger.warning(f"TTS pipeline miss, falling back: '{sentence[:40]}...'")
                self.speech.speak(sentence, emotion=response_emotion)

        # Wait for producer to finish
        producer.join(timeout=5)

        # If no sentences came through (empty response)
        if not full_response.strip():
            self.face.set_state(None)
            response = "Hmm, I'm not sure what to say about that."
            response_emotion = "neutral"
            self.face.set_emotion(response_emotion)
            self.speech.speak(response, emotion=response_emotion)

        # Ensure mouth stops and face returns to neutral
        self.face.set_talking(False)
        self.face.set_emotion("neutral")

    # ═══════════════════════════════════════════
    # Shutdown
    # ═══════════════════════════════════════════

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C and SIGTERM gracefully."""
        logger.info(f"Signal {signum} received — shutting down...")
        self._running = False

    def _on_talking_changed(self, talking: bool, duration: float = 0.0):
        """Mouth sync callback from SpeechEngine.
        Called when audio playback starts (talking=True, duration=seconds)
        and when it ends (talking=False)."""
        self.face.set_talking(talking)

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
