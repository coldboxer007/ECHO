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
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("echo.test")


# ═══════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════

def timestamp():
    """Return current time as a formatted string."""
    return datetime.now().strftime("%H:%M:%S")


def banner(title: str, icon: str = "🔧"):
    """Print a clear test section banner with timestamp."""
    ts = timestamp()
    width = 55
    print()
    print("═" * width)
    print(f"  {icon}  {title}")
    print(f"  Started: {ts}")
    print("═" * width)


def step(msg: str):
    """Print a timestamped test step."""
    print(f"  [{timestamp()}]  {msg}")


def result(passed: bool, msg: str = ""):
    """Print a pass/fail result."""
    icon = "✅ PASS" if passed else "❌ FAIL"
    suffix = f" — {msg}" if msg else ""
    print(f"  [{timestamp()}]  {icon}{suffix}")
    return passed


# ═══════════════════════════════════════════════════
# Individual Tests
# ═══════════════════════════════════════════════════

def test_motors():
    """Test each motor direction for 2 seconds each."""
    from motor_controller import MotorController

    banner("Motor Test (L298N — 6 motors, 2 sides)", "⚙️")
    m = MotorController()
    passed = True

    try:
        step("Ensuring all motors stopped first...")
        m.stop()
        time.sleep(0.5)

        step("Forward (2s) — both sides drive forward...")
        m.forward(2.0)
        time.sleep(0.5)

        step("Backward (2s) — both sides drive backward...")
        m.backward(2.0)
        time.sleep(0.5)

        step("Turn Left (1s) — left back, right forward...")
        m.turn_left(1.0)
        time.sleep(0.5)

        step("Turn Right (1s) — left forward, right back...")
        m.turn_right(1.0)
        time.sleep(0.5)

        step("Stop — all motors off")
        m.stop()

    except Exception as e:
        passed = False
        step(f"Error: {e}")

    m.cleanup()
    return result(passed, "Motor test complete")


def test_sensors():
    """Continuously read sensors for 10 seconds."""
    from sensor_controller import SensorController

    banner("Sensor Test (HC-SR04 Ultrasonic + IR)", "📡")
    s = SensorController()
    passed = True
    obstacle_seen = False

    try:
        step("Reading sensors for 10 seconds (50 samples)...")
        print(f"  {'─' * 50}")

        for i in range(50):
            dist = s.read_distance()
            ir = s.read_ir()
            obstacle = s.is_obstacle_ahead()

            if obstacle:
                obstacle_seen = True

            ir_str = "BLOCKED" if ir else "clear  "
            obs_str = "⚠ YES" if obstacle else "  no "
            bar = "█" * min(40, int(dist / 5))
            print(
                f"  [{i+1:3d}/50] Dist: {dist:6.1f}cm  IR: {ir_str}  Obstacle: {obs_str}  {bar}",
                end="\r" if i < 49 else "\n"
            )
            time.sleep(0.2)

        print(f"  {'─' * 50}")
        step(f"Obstacle detected at any point: {'Yes' if obstacle_seen else 'No'}")

    except Exception as e:
        passed = False
        step(f"Error: {e}")

    s.cleanup()
    return result(passed, "Sensor test complete")


def test_camera():
    """Test camera capture and face detection."""
    from camera_sentiment import CameraSentiment
    import cv2

    banner("Camera Test (USB Webcam + Face Detection)", "📷")
    cam = CameraSentiment()
    passed = True

    try:
        step("Capturing frame from USB camera...")
        frame = cam.capture_frame()

        if frame is not None:
            step(f"Frame captured: {frame.shape[1]}x{frame.shape[0]} ({frame.shape[2]}ch)")
            step("Running face detection (Haar cascade)...")
            faces = cam.detect_faces(frame)
            step(f"Faces detected: {len(faces)}")

            for i, (x, y, w, h) in enumerate(faces):
                step(f"  Face {i}: pos=({x},{y}) size={w}x{h}")

            # Save test frame
            path = "/tmp/echo_test_frame.jpg"
            cv2.imwrite(path, frame)
            step(f"Test frame saved to {path}")
        else:
            passed = False
            step("Failed to capture frame!")

    except Exception as e:
        passed = False
        step(f"Error: {e}")

    cam.cleanup()
    return result(passed, "Camera test complete")


def test_sentiment():
    """Test TFLite sentiment analysis on live frames."""
    from camera_sentiment import CameraSentiment

    banner("Sentiment Analysis (TFLite FER Model)", "😊")
    cam = CameraSentiment()
    passed = True
    detected_emotions = []

    try:
        step("Running 5 sentiment analysis cycles (1s apart)...")
        for i in range(5):
            emotion, conf = cam.analyze_sentiment()
            detected_emotions.append(emotion)
            conf_bar = "█" * int(conf * 20)
            step(f"  [{i+1}/5] Emotion: {emotion:10s} Confidence: {conf:.2f} {conf_bar}")
            time.sleep(1)

        unique = set(detected_emotions)
        step(f"Unique emotions detected: {unique}")

    except Exception as e:
        passed = False
        step(f"Error: {e}")

    cam.cleanup()
    return result(passed, "Sentiment test complete")


def test_mic():
    """Test microphone recording and RMS levels."""
    banner("Microphone Test (USB Mic)", "🎤")
    passed = True

    try:
        import pyaudio
        import struct

        pa = pyaudio.PyAudio()

        step("Scanning audio input devices...")
        input_count = 0
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                input_count += 1
                step(f"  [{i}] {info['name']} (inputs: {info['maxInputChannels']})")

        step(f"Total input devices found: {input_count}")

        # Find a working sample rate
        stream = None
        selected_rate = None
        candidate_rates = []

        try:
            default_in = pa.get_default_input_device_info()
            default_rate = int(default_in.get('defaultSampleRate', 16000))
            if default_rate > 0:
                candidate_rates.append(default_rate)
        except Exception:
            pass

        for rate in [16000, 48000, 44100, 32000]:
            if rate not in candidate_rates:
                candidate_rates.append(rate)

        for rate in candidate_rates:
            try:
                stream = pa.open(
                    format=pyaudio.paInt16, channels=1, rate=rate,
                    input=True, frames_per_buffer=1024,
                )
                selected_rate = rate
                break
            except Exception:
                continue

        if stream is None:
            raise RuntimeError("Could not open microphone at any supported sample rate")

        step(f"Recording at {selected_rate} Hz — speak now! (5 seconds)")
        max_rms = 0
        for _ in range(int(5 * selected_rate / 1024)):
            data = stream.read(1024, exception_on_overflow=False)
            samples = struct.unpack('1024h', data)
            rms = (sum(s * s for s in samples) / 1024) ** 0.5
            max_rms = max(max_rms, rms)
            bar = "█" * min(40, int(rms / 150))
            print(f"  RMS: {rms:6.0f} {bar}", end="\r")
            if rms > 500:
                print()

        stream.stop_stream()
        stream.close()
        pa.terminate()

        print()
        step(f"Peak RMS: {max_rms:.0f}")

        if max_rms > 500:
            step("Microphone is working — audio detected")
        else:
            step("Very quiet — check mic connection")
            passed = False

    except Exception as e:
        passed = False
        step(f"Error: {e}")

    return result(passed, "Microphone test complete")


def test_stt():
    """Test speech-to-text (Faster Whisper)."""
    from speech_engine import SpeechEngine

    banner("Speech-to-Text (Faster Whisper)", "📝")
    speech = SpeechEngine()
    passed = True

    try:
        step("Say something (up to 5 seconds, will stop on silence)...")
        text = speech.listen()

        if text:
            step(f"Transcribed: '{text}'")
        else:
            step("No text transcribed — try speaking louder")
            passed = False

    except Exception as e:
        passed = False
        step(f"Error: {e}")

    speech.cleanup()
    return result(passed, "STT test complete")


def test_tts():
    """Test text-to-speech (Gemini TTS + fallback)."""
    from speech_engine import SpeechEngine

    banner("Text-to-Speech (Gemini TTS → espeak fallback)", "🔊")
    speech = SpeechEngine()
    passed = True

    test_phrases = [
        ("Hello! I am ECHO, your companion robot.", "happy"),
        ("I'm sorry to hear that. Is there anything I can help with?", "sad"),
        ("Wow, that's really interesting!", "surprise"),
    ]

    try:
        for i, (text, emotion) in enumerate(test_phrases, 1):
            step(f"[{i}/{len(test_phrases)}] Speaking ({emotion}): '{text[:50]}...'")
            speech.speak(text, emotion=emotion)
            time.sleep(0.5)

    except Exception as e:
        passed = False
        step(f"Error: {e}")

    speech.cleanup()
    return result(passed, "TTS test complete")


def test_gemini():
    """Test Gemini API connection and command parsing."""
    from gemini_brain import GeminiBrain

    banner("Gemini AI Brain (API + Command Parsing)", "🧠")
    brain = GeminiBrain()
    passed = True

    try:
        # Test conversation
        test_inputs = [
            ("Hello, how are you?", "happy", 0.8),
            ("I'm feeling a bit sad today.", "sad", 0.7),
            ("What can you do?", "neutral", 0.5),
        ]

        step("Testing Gemini conversation...")
        for text, emotion, conf in test_inputs:
            step(f"  Input: '{text}' (emotion={emotion})")
            response = brain.think(text, emotion, conf)
            if response:
                step(f"  Reply: '{response[:80]}...'")
            else:
                step("  No response — check API key")
                passed = False

        # Test command parsing
        step("Testing command parsing...")
        commands = [
            ("move forward", "move/forward"),
            ("turn left", "move/left"),
            ("follow me", "follow"),
            ("stop", "stop"),
            ("what's the weather?", "chat"),
        ]
        for cmd, expected in commands:
            r = brain.interpret_command(cmd)
            actual = r['type'] + (f"/{r.get('direction', '')}" if r['type'] == 'move' else '')
            match = "✓" if actual == expected else "✗"
            step(f"  '{cmd}' → {actual} {match}")

        # Test emotion fallback
        step("Testing response emotion analysis...")
        test_responses = [
            ("I'm so happy to help you!", "neutral"),
            ("I'm sorry to hear that.", "neutral"),
            ("Wow, that's amazing!", "neutral"),
        ]
        for resp, user_em in test_responses:
            em = brain.determine_response_emotion(resp, user_em)
            step(f"  '{resp[:40]}...' → {em}")

    except Exception as e:
        passed = False
        step(f"Error: {e}")

    brain.cleanup()
    return result(passed, "Gemini test complete")


def test_face():
    """Test face display with all emotions sequentially."""
    from face_display import FaceDisplay

    banner("Face Display — All Emotions", "😄")
    face = FaceDisplay()
    face.start()
    passed = True

    emotions = ["neutral", "happy", "sad", "angry", "surprise", "fear", "disgust"]

    try:
        for i, emotion in enumerate(emotions, 1):
            step(f"[{i}/{len(emotions)}] Showing: {emotion} (2.5s)")
            face.set_emotion(emotion)
            time.sleep(2.5)

        # Test talking
        step("Testing talking animation (happy, 3s)...")
        face.set_emotion("happy")
        face.set_talking(True)
        time.sleep(3)
        face.set_talking(False)

    except Exception as e:
        passed = False
        step(f"Error: {e}")

    face.cleanup()
    return result(passed, "Face display test complete")


def _test_face_single(emotion: str, duration: float = 5.0):
    """Helper: show a single emotion on the face display."""
    from face_display import FaceDisplay

    banner(f"Face Display — {emotion.upper()}", "😄")
    face = FaceDisplay()
    face.start()

    step(f"Showing '{emotion}' for {duration}s (Ctrl+C to exit early)...")
    face.set_emotion(emotion)
    try:
        time.sleep(duration)
    except KeyboardInterrupt:
        step("Interrupted by user")

    face.cleanup()
    return result(True, f"{emotion} face test complete")


def test_face_neutral():
    return _test_face_single("neutral")

def test_face_happy():
    return _test_face_single("happy")

def test_face_sad():
    return _test_face_single("sad")

def test_face_angry():
    return _test_face_single("angry")

def test_face_surprise():
    return _test_face_single("surprise")

def test_face_fear():
    return _test_face_single("fear")

def test_face_disgust():
    return _test_face_single("disgust")


def test_face_talk():
    """Test talking animation cycling through emotions."""
    from face_display import FaceDisplay

    banner("Face Display — Talking Animation", "🗣️")
    face = FaceDisplay()
    face.start()
    passed = True

    emotions = ["neutral", "happy", "sad", "surprise"]

    try:
        for i, emotion in enumerate(emotions, 1):
            step(f"[{i}/{len(emotions)}] Talking with '{emotion}' face (3s)...")
            face.set_emotion(emotion)
            face.set_talking(True)
            time.sleep(3)
            face.set_talking(False)
            time.sleep(0.5)

    except Exception as e:
        passed = False
        step(f"Error: {e}")

    face.cleanup()
    return result(passed, "Talking animation test complete")


def test_face_blink():
    """Test blinking — watch the neutral face for ~15 seconds."""
    from face_display import FaceDisplay

    banner("Face Display — Blinking", "👁️")
    face = FaceDisplay()
    face.start()

    step("Showing neutral face for 15s — blinks happen every ~3.5 seconds")
    face.set_emotion("neutral")
    try:
        time.sleep(15)
    except KeyboardInterrupt:
        step("Interrupted by user")

    face.cleanup()
    return result(True, "Blink test complete")


def test_face_cycle():
    """Continuously cycle through all emotions."""
    from face_display import FaceDisplay

    banner("Face Display — Emotion Cycle (Ctrl+C to stop)", "🔄")
    face = FaceDisplay()
    face.start()

    emotions = ["neutral", "happy", "sad", "angry", "surprise", "fear", "disgust"]
    try:
        cycle = 0
        while True:
            for emotion in emotions:
                step(f"[Cycle {cycle}] {emotion}")
                face.set_emotion(emotion)
                time.sleep(2)
            cycle += 1
    except KeyboardInterrupt:
        step("Stopped by user")

    face.cleanup()
    return result(True, "Cycle test complete")


def test_display():
    """Test that the 5-inch HDMI display initializes correctly."""
    banner("5\" HDMI Display Init", "🖥️")
    passed = True

    try:
        import pygame
        from config import DISPLAY_WIDTH, DISPLAY_HEIGHT

        pygame.init()
        info = pygame.display.Info()
        step(f"Detected display: {info.current_w}x{info.current_h}")
        step(f"Configured size:  {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")

        screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
        pygame.display.set_caption("ECHO Display Test")

        # Draw test pattern
        hw = DISPLAY_WIDTH // 2
        hh = DISPLAY_HEIGHT // 2
        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, (255, 0, 0), (0, 0, hw, hh))
        pygame.draw.rect(screen, (0, 255, 0), (hw, 0, hw, hh))
        pygame.draw.rect(screen, (0, 0, 255), (0, hh, hw, hh))
        pygame.draw.rect(screen, (255, 255, 255), (hw, hh, hw, hh))

        font = pygame.font.Font(None, 48)
        text = font.render(f"ECHO {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}", True, (0, 0, 0))
        text_rect = text.get_rect(center=(DISPLAY_WIDTH // 2, DISPLAY_HEIGHT // 2))
        screen.blit(text, text_rect)

        pygame.display.flip()
        step("Test pattern displayed (R/G/B/W quadrants)")
        step("Waiting 5 seconds...")
        time.sleep(5)

        pygame.quit()

    except Exception as e:
        passed = False
        step(f"Error: {e}")

    return result(passed, "Display test complete")


def test_audio_out():
    """Test speaker output through 3.5mm jack."""
    banner("Audio Output (3.5mm Jack Speaker)", "🔊")
    passed = True

    try:
        import subprocess

        step("Checking audio output routing...")
        r = subprocess.run(
            ["amixer", "cget", "numid=3"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            step(f"Audio config: {r.stdout.strip()[:100]}...")
        else:
            step("Could not read amixer config (may be fine on some setups)")

        step("Playing test speech through speaker (espeak)...")
        subprocess.run(
            ["espeak", "-s", "150", "-v", "en",
             "Hello! This is Echo testing the speaker output. One two three."],
            timeout=15
        )
        step("Test speech played — did you hear it?")

        step("Playing 440Hz test tone (2 seconds)...")
        subprocess.run(
            ["speaker-test", "-t", "sine", "-f", "440", "-l", "1", "-p", "2"],
            capture_output=True, timeout=10
        )

    except FileNotFoundError as e:
        step(f"Tool not found: {e}")
        step("On Pi, run: sudo apt install espeak alsa-utils")
        passed = False
    except Exception as e:
        passed = False
        step(f"Error: {e}")

    return result(passed, "Audio output test complete")


# ═══════════════════════════════════════════════════
# Test Registry
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

# Ordered list for "all" (excludes interactive/looping ones)
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
        # Run all tests with progress and summary
        total = len(ALL_TESTS)
        results = {}

        print()
        print("╔" + "═" * 55 + "╗")
        print("║  🤖 ECHO Robot — Full Test Suite                      ║")
        print(f"║  Started: {timestamp()}                               ║")
        print(f"║  Tests:   {total}                                         ║")
        print("╚" + "═" * 55 + "╝")

        for i, name in enumerate(ALL_TESTS, 1):
            print(f"\n  ▶ Test {i}/{total}: {name}")
            try:
                passed = TESTS[name]()
                results[name] = passed
            except Exception as e:
                print(f"  ❌ {name} CRASHED: {e}")
                results[name] = False

        # Print summary
        print()
        print("╔" + "═" * 55 + "╗")
        print("║  📊 TEST SUMMARY                                      ║")
        print("╠" + "═" * 55 + "╣")

        pass_count = 0
        for name in ALL_TESTS:
            passed = results.get(name, False)
            if passed:
                pass_count += 1
            icon = "✅" if passed else "❌"
            print(f"║  {icon}  {name:20s}                              ║")

        print("╠" + "═" * 55 + "╣")
        ratio = f"{pass_count}/{total}"
        pct = int(pass_count / total * 100) if total > 0 else 0
        status = "ALL PASSED! 🎉" if pass_count == total else f"{ratio} passed ({pct}%)"
        print(f"║  Result: {status:44s} ║")
        print(f"║  Finished: {timestamp()}                             ║")
        print("╚" + "═" * 55 + "╝")

    elif target in TESTS:
        TESTS[target]()
    else:
        print(f"Unknown test: {target}")
        print("Available:", ", ".join(TESTS.keys()), ", all")
        sys.exit(1)


if __name__ == "__main__":
    main()
