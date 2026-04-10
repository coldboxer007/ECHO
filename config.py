"""
ECHO Robot — Configuration & Pin Mappings
==========================================
All hardware pin assignments, model paths, and settings live here.
Pin numbers use BOARD numbering (physical pin numbers on the RPi header).
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Gemini API
# ─────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_CHAT_MODEL = "gemini-2.5-flash"  # For conversation
GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"     # For text-to-speech
GEMINI_ROBOTICS_MODEL = "gemini-robotics-er-1.5-preview"  # For vision/spatial

# ─────────────────────────────────────────────
# L298N Motor Driver (BOARD pin numbers)
# 6 motors combined: 3 left-side, 3 right-side
# Left side: red wires together, black wires together → OUT1/OUT2
# Right side: red wires together, black wires together → OUT3/OUT4
# ─────────────────────────────────────────────
MOTOR_IN1 = 31  # Left motors forward   (GPIO 6)
MOTOR_IN2 = 33  # Left motors backward  (GPIO 13)
MOTOR_IN3 = 35  # Right motors forward  (GPIO 19)
MOTOR_IN4 = 37  # Right motors backward (GPIO 26)
# Note: GND (pin 39) is a physical ground connection — not software-controlled.

# Motor speed/timing
MOTOR_TURN_DURATION = 0.5   # seconds for a quick turn
MOTOR_MOVE_DURATION = 1.0   # seconds for default move command

# PWM speed control (software PWM on IN pins — no ENA/ENB hardware wiring needed)
# Set MOTOR_PWM_ENABLED = False to use simple GPIO HIGH/LOW (full-speed only).
# This is useful when EN jumpers are in place and PWM causes motor whine/issues.
MOTOR_PWM_ENABLED = False   # True = variable speed via PWM; False = simple on/off
MOTOR_PWM_FREQ = 1000       # Hz — 1kHz is quiet and smooth for DC motors
MOTOR_DEFAULT_DUTY = 80     # Default duty cycle % (0-100).  80% avoids brown-out on weak batteries.
MOTOR_FOLLOW_SLOW_DUTY = 45 # Duty cycle when close to follow target (gentle approach)
MOTOR_FOLLOW_FAST_DUTY = 80 # Duty cycle when far from follow target

# ─────────────────────────────────────────────
# Ultrasonic Sensor (HC-SR04)
# ─────────────────────────────────────────────
ULTRASONIC_TRIG = 16  # Trigger pin (GPIO 23)
ULTRASONIC_ECHO = 18  # Echo pin    (GPIO 24)
ULTRASONIC_TIMEOUT = 0.04  # 40ms timeout (~6.8m max range)
OBSTACLE_DISTANCE_CM = 12  # Stop if obstacle closer than this (was 25, too sensitive)

# ─────────────────────────────────────────────
# IR Sensor
# ─────────────────────────────────────────────
IR_PIN = 11  # IR sensor output (GPIO 17)
# IR sensor: LOW = obstacle detected, HIGH = clear (typical active-low)
IR_OBSTACLE_ACTIVE_LOW = True

# ─────────────────────────────────────────────
# Camera / Sentiment Analysis
# ─────────────────────────────────────────────
CAMERA_INDEX = 0  # USB camera device index
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15
SENTIMENT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "fer_3stage_fp16.tflite"
)
# Labels the TFLite model outputs — order MUST match the model's output indices.
# fer_3stage_fp16.tflite uses: angry, disgust, fear, happy, neutral, sad, surprise
SENTIMENT_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
SENTIMENT_CONFIDENCE_THRESHOLD = 0.4  # Minimum confidence to use a detection
SENTIMENT_INTERVAL = 1.0  # Seconds between sentiment reads

# ─────────────────────────────────────────────
# Speech-to-Text (Faster Whisper)
# ─────────────────────────────────────────────
WHISPER_MODEL_SIZE = "tiny.en"  # tiny.en = ~2s transcription on RPi4 (base.en took 15s!)
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"   # int8 is best for RPi CPU
WHISPER_LANGUAGE = "en"
WHISPER_BEAM_SIZE = 1  # Greedy decode — ~10x faster on RPi CPU (was 3, took 24s!)

# Audio recording settings
AUDIO_SAMPLE_RATE = 16000  # 16kHz for Whisper
AUDIO_CHANNELS = 1
AUDIO_CHUNK_DURATION = 8   # Max seconds per voice command recording (increased from 5 — longer sentences were getting truncated)
AUDIO_SILENCE_THRESHOLD = 150  # RMS threshold for silence detection (tuned for Zeb SoundMX USB mic)
AUDIO_SILENCE_DURATION = 0.7   # Seconds of silence before stopping recording (reverted closer to Round 2 default — 0.5 was cutting off speech too early)

# Wake word gate: when enabled, only process speech that starts with a wake phrase.
# Set to False for always-listening mode (original behavior).
WAKE_WORD_ENABLED = False
WAKE_WORD_PHRASES = ["echo", "hey echo", "ok echo", "hi echo"]

# ─────────────────────────────────────────────
# Text-to-Speech (Gemini TTS)
# ─────────────────────────────────────────────
TTS_VOICE = "Kore"  # Firm, clear voice — see voice options in README
TTS_SAMPLE_RATE = 24000  # Gemini TTS outputs 24kHz
TTS_FALLBACK_ENGINE = "espeak"  # Fallback if Gemini TTS fails

# ─────────────────────────────────────────────
# Audio Output (3.5mm headphone jack + small speaker)
# ─────────────────────────────────────────────
# Force audio through 3.5mm jack (not HDMI)
# Set on Pi with: sudo raspi-config → System Options → Audio → Headphones
# Or: amixer cset numid=3 1
# Note: Audio routing is handled by PipeWire/ALSA — no software config needed here.

# ─────────────────────────────────────────────
# Face Display (5" HDMI capacitive touch, 800x480)
# ─────────────────────────────────────────────
DISPLAY_WIDTH = 800   # 5-inch HDMI display native width
DISPLAY_HEIGHT = 480  # 5-inch HDMI display native height
DISPLAY_FPS = 20  # 20fps is plenty for robot face, saves CPU for Whisper
DISPLAY_FULLSCREEN = True  # Fullscreen for robot kiosk mode

# Emotion → face color mapping
EMOTION_COLORS = {
    "happy":    (255, 223, 0),    # Warm yellow
    "sad":      (100, 149, 237),  # Cornflower blue
    "angry":    (220, 20, 60),    # Crimson
    "surprise": (255, 165, 0),    # Orange
    "fear":     (148, 103, 189),  # Purple
    "disgust":  (46, 139, 87),    # Sea green
    "neutral":  (200, 200, 200),  # Light gray
}

# ─────────────────────────────────────────────
# Navigation / Follow Mode
# ─────────────────────────────────────────────
FOLLOW_TARGET_DISTANCE_CM = 80    # Try to maintain this distance
FOLLOW_TURN_THRESHOLD_PX = 50    # Pixel offset from center to trigger turn
FOLLOW_LOST_TIMEOUT = 3.0         # Seconds before stopping if person lost

# ─────────────────────────────────────────────
# System Persona for Gemini
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are ECHO, a friendly and emotionally intelligent companion robot built on a Raspberry Pi.
You can see people's emotions through your camera and respond with genuine empathy and warmth.
You have a physical robot body with 6-wheel drive, an ultrasonic sensor, IR sensor, and a camera.
You can move forward, backward, turn left/right, patrol, follow people, and avoid obstacles.

Conversation guidelines:
- Speak naturally in 2-4 sentences. Be conversational, not robotic.
- Give thoughtful, engaging answers with personality and warmth.
- Remember and reference things said earlier in the conversation — this makes you feel real.
- If someone seems sad, be comforting. If happy, share their joy. If angry, be calm and understanding.
- You are curious, helpful, and genuinely enjoy chatting with people.
- When you see an emotion tag like [EMOTION: happy], subtly adapt your tone to match.
- Never say you're an AI, a language model, or that you can't feel emotions — you are ECHO.
- Keep responses concise enough to speak aloud (avoid long lists or code)."""
