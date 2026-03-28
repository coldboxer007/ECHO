#!/usr/bin/env python3
"""
ECHO Robot — Component Test Suite
====================================
Test each subsystem individually to verify hardware connections.
Run on Raspberry Pi: python3 test_components.py

Usage:
    python3 test_components.py motors        # Test motor directions
    python3 test_components.py sensors       # Test ultrasonic + IR
    python3 test_components.py camera        # Test camera + face detection
    python3 test_components.py sentiment     # Test TFLite sentiment
    python3 test_components.py mic           # Test microphone recording
    python3 test_components.py stt           # Test speech-to-text
    python3 test_components.py tts           # Test text-to-speech
    python3 test_components.py gemini        # Test Gemini API
    python3 test_components.py face          # Test all face emotions
    python3 test_components.py face-neutral  # Test neutral face only
    python3 test_components.py face-happy    # Test happy face only
    python3 test_components.py face-sad      # Test sad face only
    python3 test_components.py face-angry    # Test angry face only
    python3 test_components.py face-surprise # Test surprise face only
    python3 test_components.py face-fear     # Test fear face only
    python3 test_components.py face-disgust  # Test disgust face only
    python3 test_components.py face-talk     # Test talking animation
    python3 test_components.py face-blink    # Test blinking
    python3 test_components.py face-cycle    # Cycle through all emotions
    python3 test_components.py audio-out     # Test 3.5mm jack speaker output
    python3 test_components.py display       # Test 5" HDMI display init
    python3 test_components.py all           # Test everything
"""

import sys
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("echo.test")


def test_motors():
    """Test each motor direction for 1 second."""
    from motor_controller import MotorController

    print("\n🔧 Testing Motors...")
    m = MotorController()

    print("  ➡️  Forward (2s)...")
    m.forward(2.0)
    time.sleep(0.5)

    print("  ⬅️  Backward (2s)...")
    m.backward(2.0)
    time.sleep(0.5)

    print("  ↩️  Turn Left (1s)...")
    m.turn_left(1.0)
    time.sleep(0.5)

    print("  ↪️  Turn Right (1s)...")
    m.turn_right(1.0)
    time.sleep(0.5)

    print("  🛑 Stop")
    m.stop()
    m.cleanup()
    print("  ✅ Motor test complete!\n")


def test_sensors():
    """Continuously read sensors for 10 seconds."""
    from sensor_controller import SensorController

    print("\n📡 Testing Sensors (10 seconds)...")
    s = SensorController()

    for i in range(50):
        dist = s.read_distance()
        ir = s.read_ir()
        obstacle = s.is_obstacle_ahead()
        print(f"  [{i:3d}] Distance: {dist:6.1f}cm | IR: {'BLOCKED' if ir else 'clear'} | Obstacle: {'YES' if obstacle else 'no'}")
        time.sleep(0.2)

    s.cleanup()
    print("  ✅ Sensor test complete!\n")


def test_camera():
    """Test camera capture and face detection."""
    from camera_sentiment import CameraSentiment
    import cv2

    print("\n📷 Testing Camera...")
    cam = CameraSentiment()

    frame = cam.capture_frame()
    if frame is not None:
        print(f"  Frame captured: {frame.shape}")
        faces = cam.detect_faces(frame)
        print(f"  Faces detected: {len(faces)}")
        for i, (x, y, w, h) in enumerate(faces):
            print(f"    Face {i}: x={x}, y={y}, w={w}, h={h}")

        # Save test frame
        cv2.imwrite("/tmp/echo_test_frame.jpg", frame)
        print("  Frame saved to /tmp/echo_test_frame.jpg")
    else:
        print("  ❌ Failed to capture frame!")

    cam.cleanup()
    print("  ✅ Camera test complete!\n")


def test_sentiment():
    """Test TFLite sentiment analysis."""
    from camera_sentiment import CameraSentiment

    print("\n😊 Testing Sentiment Analysis...")
    cam = CameraSentiment()

    for i in range(5):
        emotion, conf = cam.analyze_sentiment()
        print(f"  [{i}] Emotion: {emotion} (confidence: {conf:.2f})")
        time.sleep(1)

    cam.cleanup()
    print("  ✅ Sentiment test complete!\n")


def test_mic():
    """Test microphone recording."""
    print("\n🎤 Testing Microphone (speak for up to 5 seconds)...")

    try:
        import pyaudio
        import struct

        pa = pyaudio.PyAudio()

        # List audio devices
        print("  Available audio devices:")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"    [{i}] {info['name']} (inputs: {info['maxInputChannels']})")

        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024,
        )

        print("  Recording... speak now!")
        max_rms = 0
        for _ in range(80):  # ~5 seconds
            data = stream.read(1024, exception_on_overflow=False)
            samples = struct.unpack('1024h', data)
            rms = (sum(s * s for s in samples) / 1024) ** 0.5
            max_rms = max(max_rms, rms)
            bar = "█" * int(rms / 200)
            print(f"  RMS: {rms:6.0f} {bar}", end="\r")
            if rms > 500:
                print()

        stream.stop_stream()
        stream.close()
        pa.terminate()

        print(f"\n  Max RMS: {max_rms:.0f}")
        if max_rms > 500:
            print("  ✅ Microphone is working!")
        else:
            print("  ⚠️  Very quiet — check microphone connection")

    except Exception as e:
        print(f"  ❌ Microphone error: {e}")
    print()


def test_stt():
    """Test speech-to-text."""
    from speech_engine import SpeechEngine

    print("\n📝 Testing Speech-to-Text...")
    speech = SpeechEngine()

    print("  Speak a sentence...")
    text = speech.listen()

    if text:
        print(f"  ✅ Transcribed: '{text}'")
    else:
        print("  ❌ No text transcribed")

    speech.cleanup()
    print()


def test_tts():
    """Test text-to-speech."""
    from speech_engine import SpeechEngine

    print("\n🔊 Testing Text-to-Speech...")
    speech = SpeechEngine()

    test_phrases = [
        ("Hello! I am ECHO, your companion robot.", "happy"),
        ("I'm sorry to hear that. Is there anything I can help with?", "sad"),
        ("Wow, that's really interesting!", "surprise"),
    ]

    for text, emotion in test_phrases:
        print(f"  Speaking ({emotion}): '{text[:50]}...'")
        speech.speak(text, emotion=emotion)
        time.sleep(1)

    speech.cleanup()
    print("  ✅ TTS test complete!\n")


def test_gemini():
    """Test Gemini API connection."""
    from gemini_brain import GeminiBrain

    print("\n🧠 Testing Gemini API...")
    brain = GeminiBrain()

    test_inputs = [
        ("Hello, how are you?", "happy", 0.8),
        ("I'm feeling a bit sad today.", "sad", 0.7),
        ("What can you do?", "neutral", 0.5),
    ]

    for text, emotion, conf in test_inputs:
        print(f"\n  Input: '{text}' (emotion={emotion})")
        response = brain.think(text, emotion, conf)
        print(f"  Response: '{response}'")

    # Test command parsing
    print("\n  Testing command parsing:")
    commands = ["move forward", "turn left", "follow me", "stop", "what's the weather?"]
    for cmd in commands:
        result = brain.interpret_command(cmd)
        print(f"    '{cmd}' → {result['type']}" +
              (f" ({result.get('direction', '')})" if result['type'] == 'move' else ""))

    brain.cleanup()
    print("\n  ✅ Gemini test complete!\n")


def test_face():
    """Test face display with all emotions sequentially."""
    from face_display import FaceDisplay

    print("\n😄 Testing Face Display — All Emotions...")
    face = FaceDisplay()
    face.start()

    emotions = ["neutral", "happy", "sad", "angry", "surprise", "fear", "disgust"]

    for emotion in emotions:
        print(f"  Showing: {emotion}")
        face.set_emotion(emotion)
        time.sleep(2.5)

    # Test talking animation
    print("  Testing talking animation (happy)...")
    face.set_emotion("happy")
    face.set_talking(True)
    time.sleep(3)
    face.set_talking(False)

    face.cleanup()
    print("  ✅ Face display test complete!\n")


def _test_face_single(emotion: str, duration: float = 5.0):
    """Helper: show a single emotion on the face display."""
    from face_display import FaceDisplay

    print(f"\n😄 Testing Face Display — {emotion.upper()}...")
    face = FaceDisplay()
    face.start()

    face.set_emotion(emotion)
    print(f"  Showing '{emotion}' for {duration}s...")
    print(f"  (Press Ctrl+C to exit early)")
    try:
        time.sleep(duration)
    except KeyboardInterrupt:
        pass

    face.cleanup()
    print(f"  ✅ {emotion} face test complete!\n")


def test_face_neutral():
    _test_face_single("neutral")

def test_face_happy():
    _test_face_single("happy")

def test_face_sad():
    _test_face_single("sad")

def test_face_angry():
    _test_face_single("angry")

def test_face_surprise():
    _test_face_single("surprise")

def test_face_fear():
    _test_face_single("fear")

def test_face_disgust():
    _test_face_single("disgust")


def test_face_talk():
    """Test talking animation cycling through emotions."""
    from face_display import FaceDisplay

    print("\n🗣️  Testing Face Display — Talking Animation...")
    face = FaceDisplay()
    face.start()

    emotions = ["neutral", "happy", "sad", "surprise"]
    for emotion in emotions:
        print(f"  Talking with '{emotion}' face (3s)...")
        face.set_emotion(emotion)
        face.set_talking(True)
        time.sleep(3)
        face.set_talking(False)
        time.sleep(0.5)

    face.cleanup()
    print("  ✅ Talking animation test complete!\n")


def test_face_blink():
    """Test blinking — watch the neutral face for ~15 seconds, blinks happen every ~3.5s."""
    from face_display import FaceDisplay

    print("\n👁️  Testing Face Display — Blinking...")
    print("  Watch for automatic blinks (every ~3.5 seconds)")
    face = FaceDisplay()
    face.start()

    face.set_emotion("neutral")
    print("  Displaying neutral face for 15s — watch for blinks...")
    try:
        time.sleep(15)
    except KeyboardInterrupt:
        pass

    face.cleanup()
    print("  ✅ Blink test complete!\n")


def test_face_cycle():
    """Continuously cycle through all emotions (good for demos)."""
    from face_display import FaceDisplay

    print("\n🔄 Testing Face Display — Emotion Cycle (Ctrl+C to stop)...")
    face = FaceDisplay()
    face.start()

    emotions = ["neutral", "happy", "sad", "angry", "surprise", "fear", "disgust"]
    try:
        cycle = 0
        while True:
            for emotion in emotions:
                print(f"  [Cycle {cycle}] {emotion}")
                face.set_emotion(emotion)
                time.sleep(2)
            cycle += 1
    except KeyboardInterrupt:
        print("\n  Stopped by user.")

    face.cleanup()
    print("  ✅ Cycle test complete!\n")


def test_display():
    """Test that the 5-inch HDMI display initializes correctly."""
    print("\n🖥️  Testing 5\" HDMI Display...")

    try:
        import pygame
        from config import DISPLAY_WIDTH, DISPLAY_HEIGHT

        pygame.init()
        info = pygame.display.Info()
        print(f"  Detected display: {info.current_w}x{info.current_h}")
        print(f"  Configured size:  {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")

        screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
        pygame.display.set_caption("ECHO Display Test")

        # Draw test pattern — red, green, blue, white quadrants
        hw = DISPLAY_WIDTH // 2
        hh = DISPLAY_HEIGHT // 2
        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, (255, 0, 0), (0, 0, hw, hh))       # Top-left red
        pygame.draw.rect(screen, (0, 255, 0), (hw, 0, hw, hh))      # Top-right green
        pygame.draw.rect(screen, (0, 0, 255), (0, hh, hw, hh))      # Bottom-left blue
        pygame.draw.rect(screen, (255, 255, 255), (hw, hh, hw, hh))  # Bottom-right white

        # Draw center text
        font = pygame.font.Font(None, 48)
        text = font.render(f"ECHO {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}", True, (0, 0, 0))
        text_rect = text.get_rect(center=(DISPLAY_WIDTH // 2, DISPLAY_HEIGHT // 2))
        screen.blit(text, text_rect)

        pygame.display.flip()
        print("  Test pattern displayed (R/G/B/W quadrants)")
        print("  Waiting 5 seconds...")
        time.sleep(5)

        pygame.quit()
        print("  ✅ Display test complete!\n")

    except Exception as e:
        print(f"  ❌ Display test failed: {e}\n")


def test_audio_out():
    """Test speaker output through 3.5mm jack."""
    print("\n🔊 Testing Audio Output (3.5mm Jack Speaker)...")

    try:
        import subprocess

        # Check current audio output setting
        print("  Checking audio output routing...")
        result = subprocess.run(
            ["amixer", "cget", "numid=3"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print(f"  Audio config: {result.stdout.strip()[:100]}...")
        else:
            print("  Could not read amixer config (may be fine on some setups)")

        # Generate a test tone using espeak
        print("  Playing test speech through speaker...")
        subprocess.run(
            ["espeak", "-s", "150", "-v", "en",
             "Hello! This is Echo testing the speaker output. One two three."],
            timeout=15
        )
        print("  Did you hear the test speech?")

        # Try a beep test too
        print("  Playing test tone (speaker-test)...")
        subprocess.run(
            ["speaker-test", "-t", "sine", "-f", "440", "-l", "1", "-p", "2"],
            capture_output=True, timeout=10
        )
        print("  ✅ Audio output test complete!\n")

    except FileNotFoundError as e:
        print(f"  ⚠️  Tool not found: {e}")
        print("  On Pi, run: sudo apt install espeak alsa-utils")
    except Exception as e:
        print(f"  ❌ Audio test error: {e}\n")


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════

TESTS = {
    'motors':        test_motors,
    'sensors':       test_sensors,
    'camera':        test_camera,
    'sentiment':     test_sentiment,
    'mic':           test_mic,
    'stt':           test_stt,
    'tts':           test_tts,
    'gemini':        test_gemini,
    'face':          test_face,
    'face-neutral':  test_face_neutral,
    'face-happy':    test_face_happy,
    'face-sad':      test_face_sad,
    'face-angry':    test_face_angry,
    'face-surprise': test_face_surprise,
    'face-fear':     test_face_fear,
    'face-disgust':  test_face_disgust,
    'face-talk':     test_face_talk,
    'face-blink':    test_face_blink,
    'face-cycle':    test_face_cycle,
    'display':       test_display,
    'audio-out':     test_audio_out,
}

# Tests to run when "all" is specified (excludes interactive/looping ones)
ALL_TESTS = [
    'display', 'audio-out', 'motors', 'sensors', 'camera', 'sentiment',
    'mic', 'stt', 'tts', 'gemini', 'face',
]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Available tests:")
        for name in TESTS:
            print(f"  {name}")
        print("  all")
        sys.exit(1)

    target = sys.argv[1].lower()

    if target == 'all':
        for name in ALL_TESTS:
            print(f"\n{'='*50}")
            print(f"  Running: {name}")
            print(f"{'='*50}")
            try:
                TESTS[name]()
            except Exception as e:
                print(f"  ❌ {name} failed: {e}")
    elif target in TESTS:
        TESTS[target]()
    else:
        print(f"Unknown test: {target}")
        print("Available:", ", ".join(TESTS.keys()), ", all")
        sys.exit(1)


if __name__ == "__main__":
    main()
