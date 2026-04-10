# 🤖 ECHO — Emotionally Connected Humanoid Observer

<p align="center">
  <strong>A Raspberry Pi 4B 8GB RAM companion robot that sees, hears, speaks, moves, and feels.</strong>
</p>

---

## Table of Contents

- [What Is ECHO?](#what-is-echo)
- [System Architecture](#system-architecture)
- [Hardware Bill of Materials](#hardware-bill-of-materials)
- [Pin Wiring & Connections](#pin-wiring--connections)
- [Software Stack](#software-stack)
- [Project Structure](#project-structure)
- [How It Works — End to End](#how-it-works--end-to-end)
  - [The Main Loop](#the-main-loop)
  - [Speech-to-Text Pipeline](#speech-to-text-pipeline)
  - [AI Brain & Command Routing](#ai-brain--command-routing)
  - [Text-to-Speech Pipeline](#text-to-speech-pipeline)
  - [Motor Control & Navigation](#motor-control--navigation)
  - [Sensor Feedback & Obstacle Avoidance](#sensor-feedback--obstacle-avoidance)
  - [Face Display & Emotions](#face-display--emotions)
  - [Camera & Sentiment Analysis](#camera--sentiment-analysis)
- [Command Routing — Priority System](#command-routing--priority-system)
- [Movement Modes](#movement-modes)
- [Run Modes](#run-modes)
- [Setup Instructions](#setup-instructions)
  - [1. Raspberry Pi OS Setup](#1-raspberry-pi-os-setup)
  - [2. Clone & Install](#2-clone--install)
  - [3. Environment Variables](#3-environment-variables)
  - [4. Audio Configuration](#4-audio-configuration)
  - [5. First Run](#5-first-run)
- [Auto-Start on Boot (One Startup)](#auto-start-on-boot-one-startup)
- [Testing Individual Components](#testing-individual-components)
- [Debugging & Troubleshooting](#debugging--troubleshooting)
- [Key Design Decisions & Lessons Learned](#key-design-decisions--lessons-learned)
- [Future Directions](#future-directions)
- [Credits](#credits)

---

## What Is ECHO?

ECHO is a **fully autonomous companion robot** built on a Raspberry Pi 4B 8GB RAM. It combines:

- **Voice conversation** — powered by Google Gemini 3.1 Flash Lite for intelligent, emotionally-aware dialogue
- **Speech recognition** — Gemini Cloud STT (primary) with local Faster-Whisper (tiny.en) fallback
- **Expressive speech** — Gemini TTS with emotional voice styling (happy, sad, surprised, etc.)
- **Animated robot face** — full-screen pygame display with smooth eye animations, blinking, pupil wandering, breathing, and emotion transitions
- **Motor control** — 6 DC motors (3 per side) via L298N driver for tank-steer movement
- **Obstacle avoidance** — HC-SR04 ultrasonic + IR sensors for real-time safety
- **Camera sentiment analysis** — USB webcam with Haar cascade face detection + FER TFLite emotion model (224x224 RGB, raw 0-255 pixel values)
- **Complex movement commands** — "go forward carefully", "keep moving", "patrol", "stop when there's an obstacle"
- **Person following** — camera-based face tracking with motor steering

The robot understands natural language, sees facial emotions via camera, responds with empathy, displays animated expressions, and navigates its environment — all running on a single-board computer.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ECHO Robot Architecture                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────┐    ┌──────────────┐    ┌──────────────────┐    │
│  │ USB Mic │───▶│ SpeechEngine │───▶│   GeminiBrain    │    │
│  │(Zeb USB)│    │(Gemini+Whisp)│    │(Gemini 3.1 Flash │    │
│  └─────────┘    └──────────────┘    │  + cmd routing)  │    │
│                                      └────────┬─────────┘    │
│                        ┌──────────────────────┼───────┐      │
│                        │                      │       │      │
│                        ▼                      ▼       ▼      │
│               ┌──────────────┐     ┌──────────┐ ┌────────┐  │
│               │ SpeechEngine │     │Navigation│ │  Face  │  │
│               │ (Gemini TTS) │     │Controller│ │Display │  │
│               └──────┬───────┘     └────┬─────┘ │(pygame)│  │
│                      │                  │       └────────┘  │
│                      ▼                  ▼                    │
│               ┌──────────────┐  ┌──────────────┐            │
│               │  USB Speaker │  │   L298N Motor│            │
│               │  (pw-play)   │  │   Driver     │            │
│               └──────────────┘  └──────┬───────┘            │
│                                        │                     │
│                                ┌───────┴────────┐           │
│                                │ 6 DC Motors    │           │
│                                │ (3L + 3R)      │           │
│                                └────────────────┘           │
│                                                              │
│  ┌──────────────┐    ┌──────────────────┐                   │
│  │  USB Camera  │───▶│ CameraSentiment  │                   │
│  │  (640x480)   │    │ (Haar + TFLite)  │                   │
│  └──────────────┘    └──────────────────┘                   │
│                                                              │
│  ┌──────────────┐    ┌──────────────────┐                   │
│  │ HC-SR04      │───▶│ SensorController │                   │
│  │ Ultrasonic   │    │ (obstacle detect)│                   │
│  │ + IR Sensor  │    └──────────────────┘                   │
│  └──────────────┘                                           │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            5" HDMI Display (800x480)                  │   │
│  │    Animated robot face — eyes, mouth, emotions       │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│              Raspberry Pi 4B (8GB RAM)                       │
│              Raspberry Pi OS (Bookworm 64-bit)               │
│              PipeWire 1.4.2 audio server                     │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User speaks → Mic captures audio (PyAudio @ 44100Hz)
   → Silence detection stops recording (0.7s silence threshold)
   → Gemini Cloud STT transcribes (primary — sends WAV to gemini-3.1-flash-lite-preview, ~1-2s)
   → Falls back to local Faster-Whisper tiny.en if Gemini STT fails
   → Hallucination filter strips known Whisper artifacts ("thank you for watching", etc.)
   → Wake word gate (optional — strips "Echo"/"Hey Echo" prefix, discards non-matching)
  → GeminiBrain.interpret_command() routes to handler:
      ├─ If 'chat' → interpret_command_nlp() Gemini fallback for natural language
      │   (e.g. "come ahead" → reclassified as move/forward)
      ├─ Movement → NavigationController → MotorController → GPIO → L298N → Motors
      ├─ Safe move → Continuous obstacle checking while moving
      ├─ Keep moving → Background thread, auto-pause on obstacle
      ├─ Patrol → Back-and-forth loop with obstacle awareness
      ├─ Stop → Emergency stop all motors
      ├─ Goodbye → Graceful shutdown (stops motors, speaks farewell, cleans up subsystems)
      ├─ Follow → Camera face tracking + variable-speed motor steering
      ├─ Look → Captures camera frame → Gemini vision analysis → speaks description
      ├─ Volume → Adjusts TTS playback volume (louder/quieter)
      ├─ Clear history → Resets conversation memory
      └─ Chat → Gemini 3.1 Flash Lite streams response sentence-by-sentence
               → FaceDisplay set_state("listening") during mic capture
               → play_thinking_cue() (440Hz→660Hz beep, ~200ms, direct PyAudio playback)
               → FaceDisplay set_state("thinking") while waiting for Gemini
               → think_stream() yields sentences as they complete (splits on . ! ? and \\n)
               → determine_response_emotion() picks face emotion from first sentence
               → FaceDisplay shows emotion (shape-morphing transition)
               → Producer-consumer TTS pipeline: producer thread generates TTS audio
                 for sentence N+1 while consumer plays sentence N (queue-based, maxsize=4)
               → Mouth sync via callback: _notify_talking(True/False) fires when audio plays/stops
               → pw-play outputs through USB speaker
  → Face gaze tracks detected face position (set_gaze from camera)
```

---

## Hardware Bill of Materials

| Component | Specification | Purpose |
|---|---|---|
| **Raspberry Pi 4B** | 8GB RAM, 64-bit Bookworm OS | Main computer |
| **L298N Motor Driver** | Dual H-bridge, 5-35V | Drives 6 DC motors |
| **6x DC Motors** | Small gear motors | 3 left-side + 3 right-side (tank steer) |
| **HC-SR04** | Ultrasonic distance sensor | Obstacle detection (2cm - 4m range) |
| **IR Sensor** | Digital obstacle sensor | Close-range obstacle detection |
| **USB Webcam** | 640x480 @ 15fps | Face detection + sentiment analysis |
| **USB Speaker/Mic** | Zebronics SoundMX (card 3) | Audio input + output |
| **5" HDMI Display** | 800x480 capacitive touch | Animated robot face |
| **Robot Chassis** | 6-wheel platform | Physical body |
| **Power Supply** | 5V 3A (Pi) + motor battery pack | Dual power rails |
| **Jumper Wires** | Male-to-female | GPIO connections |

---

## Pin Wiring & Connections

All pin numbers use **BOARD numbering** (physical pin positions on the 40-pin header).

### L298N Motor Driver

The L298N controls 6 motors organized as two groups of 3 (left side + right side). Each group's red wires are tied together and black wires are tied together, creating a differential-drive (tank-steer) system.

| L298N Pin | RPi BOARD Pin | RPi GPIO | Function |
|---|---|---|---|
| **IN1** | **31** | GPIO 6 | Left motors — backward* |
| **IN2** | **33** | GPIO 13 | Left motors — forward* |
| **IN3** | **35** | GPIO 19 | Right motors — backward* |
| **IN4** | **37** | GPIO 26 | Right motors — forward* |
| **GND** | **39** | Ground | Common ground |

> **⚠️ Important — Inverted Wiring:** The physical motor wires are swapped relative to the L298N labels. In software, "forward" drives IN2+IN4 (not IN1+IN3). This was discovered during testing and corrected in `motor_controller.py`. If your wiring is different, swap the HIGH/LOW values in the `forward()` and `backward()` methods.

### HC-SR04 Ultrasonic Sensor

| HC-SR04 Pin | RPi BOARD Pin | RPi GPIO | Function |
|---|---|---|---|
| **TRIG** | **16** | GPIO 23 | Trigger pulse output |
| **ECHO** | **18** | GPIO 24 | Echo pulse input |
| **VCC** | 5V pin | — | Power (5V) |
| **GND** | GND pin | — | Ground |

> **⚠️ Voltage Divider:** The HC-SR04 ECHO pin outputs 5V but RPi GPIO only accepts 3.3V. Use a voltage divider (1kΩ + 2kΩ resistors) on the ECHO line, or use a 3.3V-compatible HC-SR04 module.

### IR Obstacle Sensor

| IR Pin | RPi BOARD Pin | RPi GPIO | Function |
|---|---|---|---|
| **OUT** | **11** | GPIO 17 | Digital output (active-low) |
| **VCC** | 3.3V pin | — | Power |
| **GND** | GND pin | — | Ground |

> **Active-Low:** LOW = obstacle detected, HIGH = clear. Configurable via `IR_OBSTACLE_ACTIVE_LOW` in `config.py`.

### Full Pin Map Visual

```
                    Raspberry Pi 4B — 40-Pin Header
                    ================================
                     3.3V  [1]  [2]  5V
                    SDA.1  [3]  [4]  5V
                    SCL.1  [5]  [6]  GND
                  GPIO  4  [7]  [8]  GPIO 14
                      GND  [9]  [10] GPIO 15
             IR OUT → [11] [12] GPIO 18
                   GPIO 27 [13] [14] GND
                   GPIO 22 [15] [16] ← HC-SR04 TRIG
                      3.3V [17] [18] ← HC-SR04 ECHO
                   GPIO 10 [19] [20] GND
                    GPIO 9 [21] [22] GPIO 25
                   GPIO 11 [23] [24] GPIO 8
                      GND  [25] [26] GPIO 7
                    GPIO 0 [27] [28] GPIO 1
                    GPIO 5 [29] [30] GND
      L298N IN1 →  GPIO 6 [31] [32] GPIO 12
      L298N IN2 → GPIO 13 [33] [34] GND
      L298N IN3 → GPIO 19 [35] [36] GPIO 16
      L298N IN4 → GPIO 26 [37] [38] GPIO 20
      L298N GND →     GND [39] [40] GPIO 21
```

### USB Devices

| Device | Port | Index | Notes |
|---|---|---|---|
| USB Webcam | Any USB port | /dev/video0 (index 0) | 640x480 @ 15fps |
| USB Speaker/Mic (Zeb SoundMX) | Any USB port | ALSA card 3, PipeWire sink 58, PyAudio device 2/3 | Combined mic + speaker |
| 5" HDMI Display | HDMI port 0 | — | 800x480, fullscreen |

---

## Software Stack

| Layer | Technology | Purpose |
|---|---|---|
| **OS** | Raspberry Pi OS Bookworm 64-bit | Base operating system |
| **Audio Server** | PipeWire 1.4.2 + WirePlumber | Low-latency audio routing |
| **Python** | Python 3.13 | Runtime |
| **STT (primary)** | Gemini 3.1 Flash Lite (cloud) | Cloud speech-to-text (~1-2s, more accurate) |
| **STT (fallback)** | Faster-Whisper (tiny.en, int8) | Local speech-to-text (~2-3s on RPi4) |
| **AI Brain** | Google Gemini 3.1 Flash Lite | Conversation + command understanding + response emotion classification |
| **TTS** | Gemini 2.5 Flash Preview TTS | Emotional text-to-speech (voice: Kore) |
| **Cloud STT** | Gemini 3.1 Flash Lite (thinking disabled) | Audio transcription with zero thinking overhead |
| **Live API** | Gemini 2.5 Flash Native Audio Preview | Alternative: bidirectional audio streaming |
| **Face Display** | Pygame 2.6.1 (SDL2) | Animated robot face rendering |
| **Vision** | OpenCV 4.x + Haar cascades | Face detection |
| **Sentiment** | AI Edge LiteRT / TFLite (FER 3-stage FP16) | Facial emotion recognition (224x224 RGB, raw 0-255 pixel values — matched to emotion_test_perfect.py) |
| **GPIO** | RPi.GPIO (lgpio backend) | Motor and sensor control |
| **Audio I/O** | PyAudio (capture) + pw-play (output) | Mic input + speaker output |
| **Fallback TTS** | espeak | Offline TTS when Gemini is unavailable |

---

## Project Structure

```
ECHOtest/
├── main.py                  # Main entry point — the listen→think→speak→act loop
├── debug_main.py            # Verbose debug mode (inherits from ECHO, color-coded output)
├── live_main.py             # Full Gemini Live API mode with function calling + all hardware integration
├── config.py                # ALL settings, pin assignments, model paths, prompts, MOTOR_PWM_ENABLED
├── motor_controller.py      # L298N motor driver — PWM/GPIO dual mode, watchdog, atexit safety
├── sensor_controller.py     # HC-SR04 ultrasonic (median-filtered) + IR obstacle detection
├── camera_sentiment.py      # USB webcam + TFLite facial emotion analysis (adaptive backoff, EMA)
├── camera_test.py           # Standalone camera + TFLite test utility (face detection + emotion display)
├── speech_engine.py         # Faster-Whisper STT (in-memory) + streaming Gemini TTS + thinking cue + volume
├── gemini_brain.py          # Gemini AI conversation (streaming) + hybrid NLP command interpretation + vision
├── gemini_live.py           # Gemini Live API engine (bidirectional audio, function calling, reconnection)
├── face_display.py          # Pygame animated robot face (morphing emotions, gaze tracking, reactions)
├── navigation.py            # High-level movement: manual, follow (variable speed), patrol, safe move
├── battery_monitor.py       # Battery voltage monitoring stub (ready for ADS1115 ADC hardware)
├── test_components.py       # Individual component test suite
├── setup.sh                 # System dependency installer
├── requirements.txt         # Python package requirements
├── fer_3stage_fp16.tflite   # TFLite facial emotion recognition model
├── .env                     # API keys (not committed — create from template)
├── assets/
│   └── sounds/              # Sound assets (future use)
└── models/                  # Model storage directory
```

---

## How It Works — End to End

### The Main Loop

The core of ECHO lives in `main.py` in the `ECHO._main_loop()` method. It runs continuously:

```
while running:
    1. Listen for voice input (SpeechEngine.listen())
    2. Get camera emotion (CameraSentiment.current_emotion)
    2b. Update face gaze from camera face position
    3. Interpret command (local keywords → NLP fallback if 'chat')
    4. Route to handler based on command type
    5. Execute action (move, or streaming think→TTS for chat)
    6. Loop back to listening
```

The loop skips listening while the robot is speaking (to avoid hearing its own voice). For chat commands, the streaming pipeline plays a thinking cue, then speaks each sentence as it arrives from Gemini — reducing perceived latency by 1-3 seconds compared to waiting for the full response.

### Speech-to-Text Pipeline

**File:** `speech_engine.py` → `_record_audio()` + `_transcribe()` + `_transcribe_gemini()` + `_transcribe_whisper()`

1. **Microphone capture:** PyAudio opens the USB mic at its native sample rate (44100Hz for Zeb SoundMX through PipeWire). Records in 1024-sample chunks.

2. **Silence detection:** Each chunk's RMS energy is computed. If RMS > 150 (configurable threshold), it's considered speech. Recording stops after 0.7 seconds of silence following speech, or after 8 seconds maximum.

3. **Gemini Cloud STT (primary):** The recorded PCM audio is wrapped in WAV format and sent to `gemini-3.1-flash-lite-preview` via the Gemini API with `thinkingBudget=0` (no thinking overhead). Typically returns in ~1-2s. Prompt instructs exact transcription with `[SILENCE]` marker for empty audio. Strips quotes/backticks that Gemini sometimes wraps results in.

4. **Local Whisper STT (fallback):** If Gemini Cloud STT fails (network error, API error), falls back to local Faster-Whisper. The recorded PCM audio is converted to a float32 numpy array in memory and passed directly to Faster-Whisper (no temp WAV file I/O). If the source sample rate differs from 16kHz, it is resampled using scipy sinc interpolation (falls back to numpy linear interpolation if scipy is unavailable). Settings:
   - Model: `tiny.en` (39MB — chosen for speed on RPi4 CPU, ~2-3s per transcription)
   - Compute type: `int8` (quantized for ARM CPU)
   - Beam size: `1` (greedy decode — 10x faster than beam_size=3)
   - VAD filter: enabled with tuned parameters (`min_silence_duration_ms=300`, `speech_pad_ms=200`, `threshold=0.35`)
   - Language: English only

5. **Hallucination filtering:** Whisper's tiny.en model sometimes produces phantom transcriptions on silence or noise — phrases like "thank you for watching", "subscribe", single repeated words, etc. A post-transcription filter (`_HALLUCINATION_PATTERNS`) matches these known artifacts and returns empty text, preventing false command triggers.

6. **ALSA error suppression:** Before importing PyAudio, we install a custom ALSA error handler using ctypes to suppress harmless "PCM plugin" warnings that interfere with PipeWire's ALSA compatibility layer.

7. **Wake word gate (optional):** When `WAKE_WORD_ENABLED=True` in config, transcribed text must start with a wake phrase ("echo", "hey echo", "ok echo", "hi echo"). The wake phrase is stripped from the text before command interpretation. Non-matching speech is silently discarded. Implemented as a post-transcription text filter rather than a pre-audio gate — STT already runs fast enough, and adding a separate wake word model would increase latency and RAM usage on RPi 4B.

8. **STT toggle:** `GEMINI_STT_ENABLED` in config.py controls which STT engine is primary. When True (default), Gemini Cloud STT is tried first. When False, local Whisper is used directly.

**Performance evolution:**

| Model | Beam Size | Transcription Time | Notes |
|---|---|---|---|
| base.en | 3 | ~24 seconds | Original — unusable |
| base.en | 1 | ~15 seconds | Still too slow |
| **tiny.en** | **1** | **~2-3 seconds** | Final choice |

### AI Brain & Command Routing

**File:** `gemini_brain.py` → `interpret_command()` + `think()` + `analyze_scene()`

The brain has multiple functions:

1. **Command interpretation** (`interpret_command`): Parses user text to determine if it's a movement command, utility command, or conversation. Uses a **priority-based keyword matching system** (see [Command Routing](#command-routing--priority-system) below). Returns one of 11 command types: `move`, `keep_moving`, `safe_move`, `patrol`, `follow`, `stop`, `goodbye`, `clear_history`, `look`, `volume`, `chat`.

2. **Hybrid NLP fallback** (`interpret_command_nlp`): When local keyword matching returns `chat`, a Gemini API call classifies the phrase as a potential movement command. This catches natural language like "come ahead", "move closer", or "go that way" that local keywords miss. Only called as a fallback to avoid unnecessary API calls. Results are cached for 2 seconds to avoid duplicate API calls for the same text.

3. **Conversation — blocking** (`think`): Sends the user's text + detected emotion to Gemini 2.5 Flash and returns the complete response. Used as fallback when streaming is not needed.

4. **Conversation — streaming** (`think_stream`): Generator that yields response sentences as they complete from Gemini. Uses `_SENTENCE_RE` regex to split on sentence boundaries (`.`, `!`, `?`, `\n`). Enables the streaming think→TTS pipeline where the first sentence is spoken while subsequent ones are still generating. Conversation history is capped at 14 entries (7 exchanges) to keep API calls lean while providing enough context for coherent multi-turn dialogue.

5. **Scene analysis** (`analyze_scene`): Sends a camera frame JPEG to `gemini-robotics-er-1.5-preview` for description (with 15s timeout). If the robotics model times out or fails, falls back to the chat model with thinking disabled. Triggered by "what do you see", "look around", etc.

6. **Emotion determination** (`determine_response_emotion`): Classifies the emotion of ECHO's response text to choose the appropriate face animation. Uses a 3-tier system: (1) **Gemini AI classification** — asks `gemini-3.1-flash-lite-preview` with `thinkingBudget=0` and `max_output_tokens=8` to return a single emotion word (fast, ~200-500ms). (2) **Expanded keyword fallback** — ~90+ keywords across all 7 emotions if Gemini fails. (3) **Neutral default**. ECHO reacts like a real human — with its OWN emotional response, not by mirroring the user's emotion. For example, if the user is sad, ECHO responds with warmth/comfort; if the user is angry, ECHO stays calm.

7. **History management** (`clear_history`): Resets conversation memory on voice command.

### Text-to-Speech Pipeline

**File:** `speech_engine.py` → `speak()` + `_speak_gemini()` + `_play_audio_bytes()`

1. **Gemini TTS (streaming):** The response text is sent to `gemini-2.5-flash-preview-tts` using `generate_content_stream()`. Audio chunks are collected as they arrive, concatenated, and played. Falls back to non-streaming if the stream fails. Settings:
   - Voice: "Kore" (firm, clear voice)
   - Emotion direction prepended (e.g., "Say this warmly and cheerfully with a smile in your voice:")
   - Output: raw PCM audio at 24kHz

2. **Producer-consumer TTS pipeline:** In `_handle_chat`, a producer thread reads sentences from `think_stream()` and calls `generate_tts_audio()` to pre-generate audio bytes for each sentence. These are placed into a `queue.Queue(maxsize=4)`. The consumer (main thread) pulls from the queue and plays each sentence via `play_audio()`. This overlaps TTS generation for sentence N+1 with playback of sentence N, saving 1-3 seconds per sentence after the first.

3. **Mouth sync callback:** `SpeechEngine` fires a `_on_talking_changed` callback when audio actually starts/stops playing. `_notify_talking(True, duration)` is called in `_play_audio_bytes()` before playback begins, and `_notify_talking(False, 0.0)` after playback ends. The main ECHO class registers a callback that calls `face.set_talking()`, ensuring mouth animation is perfectly synchronized with actual audio output rather than being estimated.

4. **Thinking audio cue:** Before streaming begins, `play_thinking_cue()` generates a brief ascending two-tone beep (440Hz → 660Hz, ~200ms) using direct PyAudio playback (no temp file or subprocess overhead). Falls back to `_play_audio_bytes` if direct PyAudio fails. Lets the user know ECHO heard them and is processing.

5. **Volume boost + user volume:** The returned audio gets a base +6dB boost (x2.0 amplitude) because Gemini TTS output tends to be quiet. On top of this, a user-adjustable volume multiplier (0.25x – 2.0x) is applied. Volume can be adjusted by voice ("louder", "volume up", "quieter", "volume down") in 25% steps.

6. **Playback:** Audio is written to a temp WAV file and played via:
   - **Primary:** `pw-play` (PipeWire native — most reliable on RPi with PipeWire)
   - **Fallback 1:** `aplay` (ALSA)
   - **Fallback 2:** PyAudio stream (last resort)

7. **Espeak fallback:** If Gemini TTS fails entirely, `espeak` generates offline speech, piped through `pw-play`.

> **Why not PyAudio for playback?** On RPi with PipeWire, PyAudio playback was silent — PipeWire's ALSA compatibility layer didn't route PyAudio's output to the USB speaker. Subprocess-based `pw-play` communicates directly with PipeWire's native protocol and works reliably.

### Motor Control & Navigation

**Files:** `motor_controller.py` + `navigation.py`

**Motor Controller** (low-level):
- Controls 4 GPIO pins connected to L298N motor driver
- **Dual mode:** PWM speed control (software PWM @ 1000Hz) or simple GPIO on/off, controlled by `MOTOR_PWM_ENABLED` flag in config.py (defaults to `False` / GPIO mode)
- In GPIO mode: pins are driven HIGH/LOW for full-speed on/off — simpler, works on all setups
- In PWM mode: each pin uses `GPIO.PWM` with configurable duty cycle (default 80%) for variable speed
- All 6 motors are in 2 groups: left (IN1/IN2) and right (IN3/IN4)
- Differential drive: same direction = straight, opposite = turn in place
- `set_speed()` is a no-op when PWM is disabled
- **Watchdog thread:** Automatically stops motors after 30 seconds of continuous operation (safety)
- **atexit handler:** Ensures motors stop on program exit, even on crashes
- **Non-blocking timed moves:** Duration-based moves run in background threads, so the robot stays responsive to "stop" commands
- GPIO pins initialized with `initial=GPIO.LOW` to prevent motor spin during setup
- `GPIO.cleanup()` is **intentionally NOT called** — it resets pins to INPUT (floating/high-impedance), which the L298N reads as HIGH and spins the motors

**Navigation Controller** (high-level):
- Wraps motor controller with sensor feedback
- Checks obstacles before forward movement
- All flag reads/writes (`_follow_mode`, `_continuous_mode`, `_running`) are protected by a threading lock
- `safe_forward` runs in a background thread (non-blocking)
- Provides multiple movement modes (see [Movement Modes](#movement-modes))
- **Follow mode** uses variable PWM speed: slow duty (45%) for fine adjustments, fast duty (80%) for approach

### Sensor Feedback & Obstacle Avoidance

**File:** `sensor_controller.py`

- **HC-SR04 ultrasonic:** Sends 10us trigger pulse, measures echo return time, calculates distance. Readings below 2cm are ignored as sensor noise. Uses **median filtering** (buffer of 3 readings) to debounce noisy readings.
- **IR sensor:** Digital output (active-low). LOW = obstacle detected.
- **Combined check:** `is_obstacle_ahead()` returns True if either sensor detects an obstacle within 12cm.
- **Background monitoring:** Runs in a daemon thread at 5Hz (every 200ms), continuously updating sensor readings.
- **Obstacle distance threshold:** 12cm (tuned from original 25cm which was too sensitive for indoor use).

### Face Display & Emotions

**File:** `face_display.py`

A full-screen pygame animation running at 20fps on the HDMI display:

**Features:**
- **Two large eyes** positioned at 25% and 75% of screen width
- **Centered mouth** at 78% of screen height
- **7 emotions:** happy, sad, angry, surprise, fear, disgust, neutral — each with unique eye shapes, mouth curves, and color schemes
- **Happy eyes:** Full round white eyes with iris/pupil/gleam and a curved lower-eyelid squint, giving a warm smiling look (not half-moon arcs)
- **Tear drops:** Sad emotion draws small falling tear ellipses below the eyes
- **Gleam highlights:** All emotions now have a small white gleam dot on the pupil for liveliness
- **Fear idle mouth:** Wavy trembling mouth animation driven by idle phase (not talk phase), so it animates even when silent
- **Scan line overlay:** Drawn on top of all face elements (eyes, mouth, eyebrows) for consistent CRT effect
- **Angry vein lines:** Made more visible with brighter accent color
- **Shape-morphing transitions:** Per-emotion geometry defined in `_EYE_PARAMS` and `_MOUTH_PARAMS` dicts, with smooth interpolation via `_emotion_blend` factor (~0.33 second morph)
- **Emotion-specific eyebrows:** Neutral gets subtle flat lines, happy gets raised arcs, sad droops, angry slants inward, surprise lifts high, fear angles up, disgust furrows asymmetrically
- **Camera-directed gaze tracking:** `set_gaze(x, y)` API accepts -1.0 to 1.0 coordinates from face detection. Overrides random pupil wander. Auto-expires after 3 seconds of no updates (resumes wander)
- **Emotion-specific pupil behavior:** Fear pupils dart rapidly, sad droop downward, angry constrict toward center
- **All emotion mouths animate during speech:** Sad mouth trembles, angry shows teeth bared, fear has wavy opening, disgust wiggles tongue, surprise pulses O shape
- **Reaction animations:** Bounce (vertical displacement, 400ms decay) for surprise/fear/happy/neutral, shake (horizontal) for angry/disgust — triggered on emotion change
- **Curved eyelid blinks:** Ellipse + arc edge for natural curved lid appearance (replaced rectangular blinks)
- **Brighter blush:** For happy/surprise emotions — increased from (45,22,22) to (90,35,40) with larger 44x20 size
- **Random blinking:** Every 2-5.5 seconds, eyes smoothly close and open (triangle wave)
- **Pupil wandering:** Pupils drift randomly with lerp smoothing during idle (overridden by gaze tracking when active)
- **Breathing animation:** Subtle vertical oscillation of eye positions
- **Talk animation:** Multi-frequency mouth movement (3 layered sine waves) when speaking
- **Listening state indicator:** When ECHO is listening for voice input, `set_state("listening")` activates pulsing ear arc indicators on the sides of the face and centers the pupils for an attentive look
- **Thinking state indicator:** When ECHO is processing/waiting for Gemini, `set_state("thinking")` drifts the eyes upward and shows animated processing dots (3 staggered pulsing dots) below the mouth
- **Status bar:** Shows current state (LISTENING/THINKING/SPEAKING) at the bottom of the display
- **Idle micro-movements:** Gentle sway when neutral

**Performance optimizations:**
- Scan line overlay pre-rendered once at startup (was creating new surface every frame)
- Status font cached at init (was allocating a new `pygame.font.Font` every frame)
- All glow effects use direct dim-color drawing instead of per-frame SRCALPHA surface allocations
- Reduced from 30fps to 20fps (saves CPU for Whisper)
- `pygame.display.init()` + `pygame.font.init()` instead of `pygame.init()` (avoids pygame stealing the audio device from PipeWire)
- `pygame.QUIT` events are ignored (RPi window manager sends spurious QUIT events that would crash the display)

### Camera & Sentiment Analysis

**File:** `camera_sentiment.py`

1. **USB webcam** captures at 640x480 @ 15fps via OpenCV (`CAP_PROP_BUFFERSIZE=1` for fresh frames)
2. **Haar cascade** (`haarcascade_frontalface_default.xml`) detects faces in grayscale. Results are **cached per frame** to avoid running detection twice (once for sentiment, once for follow mode).
3. **TFLite model** (`fer_3stage_fp16.tflite`) classifies facial emotion via a 3-level runtime fallback chain (AI Edge LiteRT → tflite-runtime → tensorflow.lite):
   - Input: 224x224 RGB face crop, raw 0-255 float32 pixel values (no normalization — matched exactly to `emotion_test_perfect.py`)
   - BGR→RGB conversion applied before inference (OpenCV captures BGR natively)
   - Output: 7-class probabilities (angry, disgust, fear, happy, neutral, sad, surprise)
   - Softmax applied only if output is raw logits (`raw.min() < 0 or abs(raw.sum() - 1.0) > 0.01`)
   - Confidence threshold: 40%
   - **Emotion temporal smoothing** via exponential moving average (alpha=0.4) to reduce flickering
4. **Adaptive backoff:** When no face is detected, analysis interval slows to 5x the normal rate to reduce CPU usage
5. **Background loop** runs at 1Hz (with adaptive backoff), updating `current_emotion` and `current_confidence`
6. **Gemini API fallback** available if TFLite runtime is not installed

---

## Command Routing — Priority System

The `interpret_command()` function uses a **strict priority ordering** to prevent false matches. This was redesigned after discovering that phrases like "move forward until you can stop at your obstacle" were being incorrectly classified as "stop" commands.

```
Priority 1: Complex / continuous movement
    ├── keep_moving  → "keep going", "continue moving", "don't stop"
    ├── safe_move    → "carefully", "stop when obstacle", "watch out"
    └── patrol       → "patrol", "back and forth", "explore"

Priority 2a: Goodbye / shutdown
    └── goodbye   → "goodbye", "bye bye", "goodnight", "shut down", "power off",
                     "go to sleep", "see you later"
        Checked BEFORE stop so "shut down" triggers full shutdown, not motor stop

Priority 2b: Explicit stop (with disambiguation)
    └── stop         → "stop", "halt", "freeze"
        BUT ONLY IF no movement words are also present
        (prevents "stop when obstacle" from triggering stop)

Priority 3: Duration extraction
    └── Regex: "for N seconds" → attaches duration to move command

Priority 4: Standard directional movement
    ├── forward  → "move forward", "go ahead", "straight"
    ├── backward → "go back", "reverse"
    ├── left     → "turn left"
    ├── right    → "turn right"
    └── spin     → "spin", "turn around", "360"

Priority 5: Follow mode
    └── follow → "follow me", "come here", "come with me"

Priority 6: Utility commands
    ├── clear_history → "clear history", "forget everything", "start over"
    ├── look          → "what do you see", "look around", "describe"
    └── volume        → "louder", "volume up", "quieter", "volume down"

Priority 7: Chat (default)
    └─ Everything else → sent to Gemini for conversation
        BUT FIRST: NLP skip heuristic checks if phrase is obvious chat
        (>6 words, starts with question word, contains '?')
        If ambiguous: interpret_command_nlp() asks Gemini to classify
        the phrase as a movement command (catches "come ahead",
        "move closer", etc. that keywords miss)
```

---

## Movement Modes

| Mode | Trigger Phrase | Behavior |
|---|---|---|
| **Single move** | "move forward", "turn right" | Moves for default duration (1.0s move, 0.5s turn) at configurable PWM speed |
| **Timed move** | "go forward for 3 seconds" | Moves for specified duration (non-blocking — responds to stop commands) |
| **Safe move** | "go forward carefully" | Moves with continuous obstacle checking, stops permanently on detection |
| **Keep moving** | "keep going", "don't stop" | Continuous movement in background thread, auto-pauses on obstacle, resumes when clear |
| **Patrol** | "patrol", "explore" | Forward 3s → pause → backward 3s → pause → repeat, with obstacle awareness |
| **Follow** | "follow me" | Camera tracks largest face, variable-speed PWM steering (slow=45% for turns, fast=80% for approach), maintains ~80cm distance |
| **Spin** | "spin", "turn around" | Right turn for 2 seconds |
| **Emergency stop** | "stop", "halt", "freeze" | Immediately stops all motors and background movement modes |
| **Goodbye** | "goodbye", "shut down", "goodnight" | Graceful full shutdown — stops motors, speaks farewell, cleans up all subsystems |

---

## Run Modes

ECHO has three entry points depending on your needs:

### Standard Mode (main.py)
```bash
python3 main.py
```
The primary mode. Uses Faster-Whisper for local STT + Gemini for conversation + Gemini TTS for speech output. Best for reliable operation.

### Debug Mode (debug_main.py)
```bash
python3 debug_main.py
```
Same functionality as standard mode — `ECHODebug` inherits from the `ECHO` class and adds verbose, color-coded terminal output. Shows every sensor reading, command interpretation, API call, streaming sentence, gaze tracking, and timing breakdown in real-time. Includes a **live camera preview window** (OpenCV) showing the camera feed with face detection bounding boxes, current emotion label, and a per-emotion confidence bar chart — useful for monitoring the TFLite model's performance in real time. Also tracks session statistics (loop count, speech I/O, command breakdown including goodbye). Best for development and troubleshooting. Overrides handler methods to add debug logging while delegating to the base class via `super()` — including `_handle_chat()`, which uses the inherited producer-consumer TTS pipeline rather than a separate implementation. Includes the full streaming think→TTS pipeline, hybrid NLP fallback, and gaze tracking — all with debug output.

### Live API Mode (live_main.py)
```bash
python3 live_main.py
```
Full-featured mode using Gemini Live API for bidirectional audio streaming with complete hardware integration. Replaces the separate Whisper STT → Gemini Chat → Gemini TTS pipeline with a single streaming connection. The Live engine (`gemini_live.py`) provides:
- **Function calling** — 10 declared tools (`move_robot`, `stop_robot`, `start_follow_mode`, `start_patrol_mode`, `safe_move_forward`, `set_face_emotion`, `get_sensor_data`, `get_camera_emotion`, `set_volume`, `shutdown_robot`) so Gemini can control all robot hardware via natural conversation
- **Thread-safe architecture** — `queue.Queue` bridges sync mic/camera threads to the async event loop (not `asyncio.Queue`)
- **Proper API usage** — `send_realtime_input(audio=types.Blob(...))` for audio, `send_realtime_input(video=types.Blob(...))` for camera frames
- **Interruption handling** — clears audio queue when `server_content.interrupted` is True
- **Reconnection with exponential backoff** — up to 5 attempts, base 2s delay doubling each time
- **Context window compression** — configured at 25600 trigger tokens, sliding to 12800 for long sessions
- **Periodic context** — camera frames sent every 5s, emotion context text every 10s (only on change)
- **Face gaze tracking** — updates face gaze from camera face detection in main loop
- No local STT model needed (saves RAM + startup time)
- Lower latency (streaming vs request/response)
- Natural conversation with interruption support
- Gemini handles voice activity detection

---

## Setup Instructions

### 1. Raspberry Pi OS Setup

- Flash **Raspberry Pi OS Bookworm 64-bit** to an SD card
- Boot, connect to WiFi, enable SSH
- Ensure PipeWire is the audio server (default on Bookworm):
  ```bash
  pw-cli info 0  # Should show PipeWire version
  ```

### 2. Clone & Install

```bash
# Clone the repository
git clone https://github.com/coldboxer007/ECHOtest.git
cd ECHOtest

# Run the system dependency installer
chmod +x setup.sh
sudo ./setup.sh

# Or manually:
sudo apt update && sudo apt install -y \
    build-essential python3-dev python3-pip python3-venv \
    libportaudio2 portaudio19-dev libasound2-dev \
    alsa-utils espeak ffmpeg \
    libsdl2-dev libsdl2-ttf-dev libsdl2-image-dev \
    libsdl2-mixer-dev python3-pygame \
    libatlas-base-dev libhdf5-dev \
    v4l-utils pipewire pipewire-audio-client-libraries

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt

# Install TFLite runtime (recommended: AI Edge LiteRT, Google's official successor)
pip install ai-edge-litert
# Falls back to tflite-runtime or tensorflow.lite if unavailable
```

### 3. Environment Variables

Create a `.env` file in the project root:

```bash
# .env
GEMINI_API_KEY=your_gemini_api_key_here
```

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

### 4. Audio Configuration

ECHO uses a USB speaker/microphone combo. On Raspberry Pi with PipeWire:

```bash
# List audio devices
pw-cli list-objects Node | grep -E "node.name|media.class"

# Set the USB speaker as default (if not automatic)
# Find your device:
pactl list short sinks

# Set default:
pactl set-default-sink <your_usb_sink_name>

# Test speaker
speaker-test -D plughw:3,0 -t pink -l 1

# Test microphone
arecord -D plughw:3,0 -f cd -d 3 test.wav
aplay test.wav
```

### 5. First Run

```bash
source venv/bin/activate
python3 main.py
```

On first run, the Faster-Whisper `tiny.en` model (~39MB) will be downloaded from Hugging Face. Subsequent runs use the cached model.

**Expected startup sequence:**
1. GPIO configured, motors stopped
2. Sensors initialized (TRIG/ECHO/IR)
3. Camera opened (USB webcam)
4. Whisper model loaded (~3-5s first time, ~2s cached)
5. PyAudio initialized (mic + speaker found)
6. Gemini client connected
7. Face display opened (fullscreen on HDMI)
8. Startup greeting: "Hello! I'm Echo, your companion robot."
9. ECHO is running! Listening for commands...

---

## Auto-Start on Boot (One Startup)

To make ECHO start automatically when the Raspberry Pi boots — so you just power it on and the robot is ready — use a **systemd service**.

### Step 1: Create the Service File

```bash
sudo nano /etc/systemd/system/echo-robot.service
```

Paste:

```ini
[Unit]
Description=ECHO Companion Robot
After=network-online.target sound.target graphical.target
Wants=network-online.target

[Service]
Type=simple
User=sahiltanna7
WorkingDirectory=/home/sahiltanna7/Desktop/echo-test2/ECHOtest
Environment=DISPLAY=:0
Environment=XDG_RUNTIME_DIR=/run/user/1000
Environment=PULSE_SERVER=unix:/run/user/1000/pulse/native
Environment=PIPEWIRE_RUNTIME_DIR=/run/user/1000
ExecStartPre=/bin/sleep 10
ExecStart=/home/sahiltanna7/Desktop/echo-test2/ECHOtest/venv/bin/python3 /home/sahiltanna7/Desktop/echo-test2/ECHOtest/main.py
Restart=on-failure
RestartSec=5
StandardOutput=append:/var/log/echo-robot.log
StandardError=append:/var/log/echo-robot.log

[Install]
WantedBy=graphical.target
```

> **Key details:**
> - `After=graphical.target` — waits for the desktop/display server
> - `ExecStartPre=/bin/sleep 10` — gives PipeWire and USB devices time to initialize
> - `Environment=DISPLAY=:0` — needed for pygame to find the HDMI display
> - `Restart=on-failure` — auto-restarts if ECHO crashes
> - Logs go to `/var/log/echo-robot.log`

### Step 2: Enable and Start

```bash
# Reload systemd to pick up the new service
sudo systemctl daemon-reload

# Enable auto-start on boot
sudo systemctl enable echo-robot.service

# Start it now (without rebooting)
sudo systemctl start echo-robot.service

# Check status
sudo systemctl status echo-robot.service

# View logs
sudo journalctl -u echo-robot.service -f
# or
tail -f /var/log/echo-robot.log
```

### Step 3: Manage the Service

```bash
# Stop the robot
sudo systemctl stop echo-robot.service

# Disable auto-start
sudo systemctl disable echo-robot.service

# Restart after code changes
sudo systemctl restart echo-robot.service
```

### Alternative: Desktop Autostart (LXDE)

If you are using the Raspberry Pi desktop and prefer a simpler approach:

```bash
mkdir -p ~/.config/autostart
nano ~/.config/autostart/echo-robot.desktop
```

```ini
[Desktop Entry]
Type=Application
Name=ECHO Robot
Exec=bash -c "sleep 10 && cd /home/sahiltanna7/Desktop/echo-test2/ECHOtest && source venv/bin/activate && python3 main.py >> /var/log/echo-robot.log 2>&1"
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
```

---

## Testing Individual Components

The `test_components.py` file lets you test each subsystem independently:

```bash
source venv/bin/activate

# Test motor directions (forward, backward, left, right — 2s each)
python3 test_components.py motors

# Test ultrasonic + IR sensors (10 seconds of readings)
python3 test_components.py sensors

# Test camera + face detection
python3 test_components.py camera

# Test TFLite sentiment model
python3 test_components.py sentiment

# Test microphone recording
python3 test_components.py mic

# Test speech-to-text (speak into mic)
python3 test_components.py stt

# Test text-to-speech (plays through speaker)
python3 test_components.py tts

# Test Gemini API conversation
python3 test_components.py gemini

# Test face display (cycles through emotions)
python3 test_components.py face

# Test specific emotion
python3 test_components.py face-happy

# Test audio output on 3.5mm jack
python3 test_components.py audio-out

# Test everything
python3 test_components.py all
```

---

## Debugging & Troubleshooting

### GPIO Busy Error
```
lgpio.error: 'GPIO busy'
```
**Cause:** Previous process did not release GPIO pins.  
**Fix:** Kill all Python processes, then release GPIO:
```bash
pkill -9 -f "python3 main"
python3 -c "import RPi.GPIO as GPIO; GPIO.setmode(GPIO.BOARD); GPIO.cleanup()"
```

### Motors Spin on Shutdown
**Cause:** `GPIO.cleanup()` resets pins to INPUT (floating), which L298N reads as HIGH.  
**Fix:** Already handled — `motor_controller.py` intentionally does NOT call `GPIO.cleanup()`. Pins are left as LOW outputs.

### No Audio Output
**Cause:** PyAudio playback is silent through PipeWire's ALSA compatibility layer.  
**Fix:** ECHO uses `pw-play` (PipeWire native) for audio output. Verify:
```bash
echo "test" | espeak --stdout | pw-play --target=0 -
```

### Whisper Too Slow
**Cause:** base.en model with beam_size > 1 is too heavy for RPi4 CPU.  
**Fix:** Config uses `tiny.en` with `beam_size=1` (greedy decode). If accuracy is poor, try `small.en` with `beam_size=1`.

### Face Display Crashes After a Few Minutes
**Cause:** Per-frame SRCALPHA surface allocations (1024x768 x 4 bytes x 30fps = memory exhaustion).  
**Fix:** Already fixed — all glow effects use direct dim-color drawing. Scan lines are pre-rendered once. FPS reduced to 20.

### Camera Fails to Open After Kill
**Cause:** `/dev/video0` locked by killed process.  
**Fix:** Wait 5 seconds and restart, or:
```bash
sudo fuser -k /dev/video0
```

### PipeWire Not Running
```bash
systemctl --user status pipewire pipewire-pulse wireplumber
systemctl --user restart pipewire pipewire-pulse wireplumber
```

### "stay" Triggers Stop Instead of Movement
**Cause:** Originally, "stay", "hold", "wait" were in the stop keywords list, causing phrases like "move forward until you can stay your obstacle" to match as stop.  
**Fix:** Already fixed — overly broad stop keywords removed, and complex movement commands are checked BEFORE stop commands with smart disambiguation.

---

## Key Design Decisions & Lessons Learned

### 1. pygame.display.init() instead of pygame.init()
`pygame.init()` initializes ALL subsystems including the mixer, which opens the audio device and steals it from PipeWire. By only initializing `display` and `font`, we leave audio routing entirely to PipeWire/PyAudio/pw-play.

### 2. pw-play instead of PyAudio for output
On Raspberry Pi OS Bookworm with PipeWire, PyAudio's output stream produces silence. PipeWire's ALSA compatibility layer does not properly route PyAudio's output. Direct `pw-play` subprocess calls communicate with PipeWire natively and work reliably.

### 3. No GPIO.cleanup() on exit
Calling `GPIO.cleanup()` resets pins from OUTPUT to INPUT (high-impedance). The L298N motor driver interprets floating inputs as HIGH and spins the motors at full speed. Instead, we leave pins configured as OUTPUT with LOW level — keeping motors safely stopped.

### 4. Inverted motor wiring compensation
Rather than rewiring the physical motors (which would require disassembling the robot), we swapped the HIGH/LOW logic in software. Forward = IN2+IN4 instead of IN1+IN3.

### 5. Priority-based command routing
Simple keyword matching (checking "stop" keywords first) fails because complex sentences like "move forward and stop when there's an obstacle" contain the word "stop". The priority system checks complex multi-word movement phrases first and only falls through to stop if no movement intent is detected.

### 6. ALSA error suppression via ctypes
PyAudio's initialization triggers ALSA to print errors about missing PCM plugins. On PipeWire systems, these are harmless but can number in the hundreds. We install a custom C-level error handler before importing PyAudio to suppress them.

### 7. Greedy decode (beam_size=1) for Whisper
On RPi4 CPU, beam search with beam_size=3 takes 24 seconds to transcribe 7 seconds of audio. Greedy decode (beam_size=1) reduces this to ~2-3 seconds with minimal accuracy loss for clear English speech commands.

### 8. Pre-rendered overlays for face display
Originally, every frame allocated new pygame Surfaces with SRCALPHA for glow effects, scan lines, blush marks, etc. At 1024x768, each surface was 3.1MB. With dozens per frame at 30fps, this caused memory exhaustion and crashes after a few minutes. All effects now use either pre-rendered overlays (created once) or direct dim-color drawing.

### 9. system_instruction for Gemini persona
The system prompt is now passed via Gemini's `system_instruction` parameter rather than being prepended as a fake user/model message pair. This is the proper API usage — it saves tokens on every request and provides stronger persona adherence since the model treats it as a top-level instruction rather than conversation context.

### 10. Debug mode via inheritance
`debug_main.py` was refactored from 461 lines of ~80% duplicated code to 385 lines by having `ECHODebug` inherit from `ECHO`. All subsystem imports come through the base class. Override methods add debug output and call `super()` for the actual logic. This ensures bug fixes and new features in `main.py` automatically propagate to debug mode.

### 11. Motor watchdog for safety
A background watchdog thread monitors continuous motor operation and forces a stop after 30 seconds. Combined with `atexit.register()`, this prevents runaway motors if the program crashes, hangs, or the user forgets to stop. Non-blocking timed moves use background threads so the robot remains responsive to "stop" commands during movement.

### 12. Emotion temporal smoothing
Raw TFLite emotion predictions flicker rapidly between frames (e.g., happy → neutral → happy). An exponential moving average (alpha=0.4) smooths predictions over time, preventing jarring face animation changes while still responding quickly to genuine emotion shifts.

### 13. Streaming think→TTS pipeline
Instead of waiting for Gemini to generate the entire response before speaking, `think_stream()` yields sentences as they complete. The first sentence determines the response emotion and begins playing while subsequent sentences are still generating. This reduces perceived latency by 1-3 seconds on a typical 2-3 sentence response.

### 14. Hybrid NLP command interpretation
Pure keyword matching misses natural language movement phrases like "come ahead" or "move closer". Rather than sending every utterance to Gemini for classification (slow, wasteful), the hybrid approach tries fast local keywords first and only calls Gemini NLP when the result is `chat` — keeping the common case fast while catching ambiguous phrases.

### 15. PWM made optional
Some RPi setups have issues with software PWM (timing jitter, lgpio conflicts). The `MOTOR_PWM_ENABLED` flag allows falling back to simple GPIO on/off mode. All motor methods branch internally — `navigation.py` and callers don't need changes. `set_speed()` is a no-op in GPIO mode.

### 16. Consistent TTS voice across all speech
Movement acknowledgments ("Moving forward!", "Stopped!") originally used `force_fallback=True` to skip Gemini TTS for speed. However, this produced a jarring voice switch — the robot would chat in Kore (Gemini TTS) then bark orders in espeak's robotic voice. Round 4 removed all `force_fallback=True` calls so every utterance uses the same Gemini TTS voice, maintaining personality consistency.

### 17. Goodbye as a separate command type
"Shut down" was originally in the stop keyword list, which only stopped motors. Round 4 elevated goodbye/shutdown phrases to their own command type at priority 2a (before stop), triggering a full graceful shutdown: motors stop, `_running = False` breaks the main loop, `shutdown()` speaks farewell and cleans up all subsystems in reverse order.

### 18. Three-level TFLite import fallback
Google deprecated `tflite-runtime` in favor of `ai-edge-litert`. Rather than hard-coding one package, the import chain tries all three options: `ai_edge_litert.interpreter.Interpreter` → `tflite_runtime.interpreter.Interpreter` → `tensorflow.lite.experimental.Interpreter`. If all fail, TFLite inference is disabled and the system falls back to Gemini API emotion analysis. This ensures ECHO works regardless of which TFLite package the user has installed.

### 19. Whisper hallucination filtering
Whisper's tiny.en model is fast but prone to "hallucinating" text from silence or background noise — producing phrases like "thank you for watching", "subscribe to my channel", or repeating a single word. These phantom transcriptions trigger false commands. A static set of known hallucination patterns is checked post-transcription; matches return empty text.

### 20. Rate-limited Gemini emotion fallback
When TFLite is unavailable, ECHO falls back to `analyze_emotion_from_image()` (Gemini vision API) for facial emotion detection. Without rate-limiting, this fires every listen cycle (~3-8s), wasting API quota and adding latency. A 10-second cooldown ensures the fallback only fires periodically, reusing the last detected emotion in between.

### 21. NLP classification caching
The hybrid NLP fallback (`interpret_command_nlp`) calls Gemini to classify ambiguous phrases. If the user repeats or the system re-processes the same text within 2 seconds, the cached result is returned instantly instead of making another API call. Reduces redundant Gemini calls during rapid interaction.

### 22. Producer-consumer TTS pipeline
Instead of generating TTS audio for a sentence and then playing it sequentially, Round 6 introduced a queue-based producer-consumer pattern. A background producer thread reads sentences from `think_stream()` and pre-generates TTS audio via `generate_tts_audio()`, placing results into a bounded queue (maxsize=4). The main thread consumes from the queue, playing each sentence. This overlaps generation of sentence N+1 with playback of sentence N, saving 1-3 seconds per sentence after the first. A sentinel `None` value signals stream exhaustion, and a 30-second timeout prevents deadlocks.

### 23. Mouth sync via callback
Previously, mouth animation (`set_talking(True/False)`) was called manually in the chat handler based on rough timing estimates. Round 6 replaced this with a callback system in `SpeechEngine`: `_notify_talking(True, duration)` fires when `_play_audio_bytes()` actually begins playback, and `_notify_talking(False, 0.0)` fires when it ends. The ECHO class registers `_on_talking_changed` which calls `face.set_talking()`. This ensures perfect synchronization between mouth movement and actual audio output.

### 24. NLP skip heuristic for obvious chat
The hybrid NLP fallback called Gemini to classify every `chat`-type phrase, adding 1-2 seconds of latency even for obviously conversational input like "How are you doing today?". Round 6 added a heuristic: if the phrase has >6 words, starts with a question word (who/what/where/when/why/how), or contains a question mark, or contains chat indicator words (feel/think/tell/know), the NLP API call is skipped entirely. This saves 1-2 seconds on the vast majority of conversational interactions.

### 25. Listening and thinking face states
The face display now has explicit state indicators beyond just emotions. `set_state("listening")` shows pulsing ear arc indicators and centers the pupils for an attentive look. `set_state("thinking")` drifts the eyes upward and shows animated processing dots below the mouth. These states provide clear visual feedback about what ECHO is doing, replacing the previous approach of abusing `set_talking(True)` as a "thinking" indicator.

### 26. Simplified message format for Gemini
The message format sent to Gemini was changed from `[EMOTION DETECTED: X (confidence: Y%)]\nUser says: Z` to simply `[EMOTION: X] Z`. This saves tokens on every API call, reduces prompt noise, and provides clearer context for the model without unnecessary verbosity.

### 27. Raw pixel preprocessing for TFLite (matched to emotion_test_perfect.py)
The TFLite emotion model (`fer_3stage_fp16.tflite`) expects 224x224 RGB input as **raw 0-255 float32 pixel values** — no normalization. This was discovered by using `emotion_test_perfect.py` as the reference implementation, which detects all 7 emotions reliably. Previous code incorrectly applied EfficientNet normalization (`pixel / 127.5 - 1.0`, range [-1, 1]), which destroyed the model's input distribution. Additionally, BGR→RGB conversion is applied before inference (OpenCV captures BGR natively), and softmax is only applied when needed (`raw.min() < 0 or abs(raw.sum() - 1.0) > 0.01`). All files doing TFLite face inference (`camera_sentiment.py`, `camera_test.py`) are now aligned to this exact preprocessing.

### 28. Smart softmax detection
Rather than hardcoding whether to apply softmax to TFLite model output, the code checks if the output already sums to approximately 1.0 (within tolerance of 0.1). If it does, the output is already softmax-normalized and is used directly. If not, softmax is applied. This makes the code model-agnostic — it works correctly with both raw-logit models and softmax-output models without configuration changes.

### 29. Thread-safe queue bridge for Gemini Live
The original `gemini_live.py` used `asyncio.Queue.put_nowait()` called from synchronous threads (mic capture, camera) — this is NOT thread-safe and can corrupt the queue's internal state. The rewrite uses `queue.Queue` (from Python's `queue` module, which is thread-safe) as a bridge between sync threads and the async event loop. The async tasks poll with `asyncio.sleep()` intervals rather than blocking on `queue.get()`.

### 30. Function calling in Gemini Live mode
The original `live_main.py` was a barebones audio-only demo — the robot could hear and speak but not move, sense obstacles, or show emotions. The rewrite declares 10 function tools to the Gemini Live session, allowing the model to control all hardware subsystems via natural conversation. Gemini Live's function calling is synchronous — the model pauses generation until it receives the tool response, ensuring actions complete before the conversation continues.

### 31. Motor `_timed_stop` race condition fix
The original `_timed_stop()` method slept for the move duration, then unconditionally stopped the motors. If a new move command was issued during the sleep, the old timer would wake up and stop the new move prematurely. The fix snapshots `_move_start_time` as a move identity before sleeping, and only stops the motors if that identity still matches when the timer wakes — meaning a newer move has not superseded it.

### 32. Debug mode `_handle_chat` delegation
`debug_main.py` originally contained a full copy of the chat logic using synchronous `self.speech.speak()` per sentence — no producer-consumer pipeline, making it 2-3x slower than main.py. Round 7 replaced this with `super()._handle_chat()` delegation wrapped with debug print statements, getting the inherited TTS pipeline for free and ensuring future improvements to main.py automatically propagate.

### 33. Numpy RMS calculation
The silence detection RMS calculation in `speech_engine.py` used `struct.unpack` to extract individual int16 samples, then computed RMS in pure Python. On RPi4, this is slow for the number of audio samples per chunk. Replacing it with `np.frombuffer(data, dtype=np.int16)` and `np.sqrt(np.mean(samples.astype(np.float32)**2))` is ~5x faster since numpy uses optimized C/BLAS operations.

---

## Future Directions

### Planned Improvements

1. ~~**PWM Speed Control**~~ — **DONE (Round 2), made optional (Round 3).** Software PWM (1000Hz) on all motor pins with configurable duty cycle. Follow mode uses variable speed based on face position. PWM can be disabled via `MOTOR_PWM_ENABLED = False` in config.py for setups that only need GPIO on/off control.

2. ~~**Gemini Live API as Primary Mode**~~ — **DONE (Round 7).** `live_main.py` is now a full-featured mode with complete hardware integration via function calling (10 tools: move, stop, follow, patrol, safe_move, set_emotion, get_sensors, get_emotion, set_volume, shutdown). Thread-safe queue bridge, reconnection with exponential backoff, context window compression, periodic camera/emotion context injection. Eliminates the separate Whisper → Gemini → TTS pipeline.

3. ~~**Wake Word Detection**~~ — **DONE.** Post-Whisper text filter checks for "Echo"/"Hey Echo" prefix. Configurable via `WAKE_WORD_ENABLED` in config.py (off by default for minimal-friction interaction).

4. **SLAM / Mapping** — Use ultrasonic + camera data to build a rudimentary room map for autonomous navigation beyond simple obstacle avoidance.

5. **Multi-Person Tracking** — Extend the follow mode to track specific people using face embeddings, rather than just following the largest face.

6. ~~**Camera Vision**~~ — **DONE.** "What do you see?" / "Look around" sends camera frame to Gemini vision for scene description. Also used for Gemini-based emotion fallback when TFLite is unavailable.

7. **Edge TPU / Coral Accelerator** — Adding a Google Coral USB accelerator would speed up TFLite inference from ~100ms to ~10ms and enable real-time emotion detection at camera framerate.

8. ~~**Battery Monitoring**~~ — **STUB READY.** `battery_monitor.py` module created with `BatteryMonitor` class, simulated voltage readings, low/critical thresholds, background check loop. Ready for ADS1115 ADC hardware integration.

9. ~~**Volume Control**~~ — **DONE.** Voice-adjustable TTS volume ("louder", "quieter", "volume up/down") with 0.25x–2.0x range in 25% steps.

10. **OTA Updates** — Implement a simple git-pull mechanism so the robot can update its own code over WiFi.

11. **Web Dashboard** — A Flask/FastAPI web interface showing real-time sensor data, camera feed, conversation history, and manual controls accessible from any device on the local network.

12. **Persistent Memory** — Save conversation highlights and user preferences to a local SQLite database so ECHO remembers across restarts.

13. **Sound Localization** — With a stereo mic array, ECHO could turn toward the person speaking before responding.

14. ~~**Streaming Think→TTS Pipeline**~~ — **DONE (Round 3).** `think_stream()` yields sentences as they complete from Gemini. Each sentence is spoken immediately while the next generates, reducing perceived latency by 1-3 seconds.

15. ~~**In-Memory Whisper Transcription**~~ — **DONE (Round 3).** PCM audio is converted to float32 numpy array and passed directly to Faster-Whisper, eliminating temp WAV file I/O.

16. ~~**Thinking Audio Cue**~~ — **DONE (Round 3).** Brief ascending two-tone beep (440Hz→660Hz, ~200ms) plays before Gemini starts generating, so the user knows ECHO heard them.

17. ~~**Natural Language Command Understanding**~~ — **DONE (Round 3).** Hybrid NLP: fast local keyword match first, Gemini API fallback for ambiguous natural language phrases like "come ahead" or "move closer".

18. ~~**Face Display Overhaul**~~ — **DONE (Round 3).** Shape-morphing transitions, emotion-specific eyebrows, camera-directed gaze tracking, all-emotion talking mouths, pupil behaviors, brighter blush, reaction animations (bounce/shake), and curved eyelid blinks.

19. ~~**Goodbye / Graceful Shutdown**~~ — **DONE (Round 4).** Voice-triggered shutdown: "goodbye", "shut down", "goodnight", etc. Stops motors, speaks farewell, cleans up all subsystems. Added at priority 2a (before stop) so "shut down" doesn't just stop motors.

20. ~~**Voice Consistency**~~ — **DONE (Round 4).** All speech output now uses Gemini TTS (Kore voice) consistently. Previously, movement acknowledgments used `force_fallback=True` which bypassed Gemini TTS and used espeak — producing a jarring voice change. Removed all `force_fallback=True` calls.

21. ~~**Face Polish — Happy Eyes, Tears, Gleams**~~ — **DONE (Round 4).** Happy eyes rewritten as full round eyes with squint (not half-moon arcs). Sad emotion gets tear drops. All emotions get gleam highlights. Fear idle mouth trembles correctly. Scan lines drawn on top of face elements. CPU-safe (simple circles/lines only).

22. ~~**Emotional Intelligence**~~ — **DONE (Round 4, upgraded Round 9, Gemini classify added Round 10).** ECHO reacts to the user's emotions like a real friend — with its OWN emotional response, not by mirroring. If the user is sad, ECHO shows warmth/comfort. If angry, ECHO stays calm. `determine_response_emotion()` now uses a 3-tier system: (1) Gemini AI classification via `_classify_emotion_gemini()` — asks `gemini-3.1-flash-lite-preview` (thinkingBudget=0, max_output_tokens=8, temperature=0.0) to return a single emotion word, catching nuance that keywords miss. (2) Expanded keyword fallback via `_classify_emotion_keywords()` — ~90+ keywords across all 7 emotions. (3) Neutral default. System prompt explicitly instructs emotional intelligence. Face shows neutral while thinking (not user's detected emotion).

23. ~~**Debug Camera Preview**~~ — **DONE (Round 4).** OpenCV window in debug_main.py showing live camera feed with face detection boxes, emotion label overlay, and per-emotion EMA confidence bar chart for monitoring TFLite model performance.

24. **Multi-Language Support** — **INVESTIGATED (Round 4).** Pipeline is compatible: switch Whisper from `tiny.en` to `tiny` (multilingual), set `WHISPER_LANGUAGE` to target language or `None` for auto-detect, update system prompt. Gemini chat and TTS handle multiple languages natively. Main friction: command keyword lists are English-only, but the NLP fallback via Gemini handles other languages. Config-level switch when needed.

25. ~~**AI Edge LiteRT Migration**~~ — **DONE (Round 5).** Migrated TFLite inference from deprecated `tflite-runtime` to Google's official successor `ai-edge-litert`. Uses a 3-level import fallback chain: `ai_edge_litert` → `tflite_runtime` → `tensorflow.lite` → disabled. Drop-in replacement with identical `Interpreter` API. ARM64 wheels confirmed available for RPi4.

26. ~~**Whisper Hallucination Filtering**~~ — **DONE (Round 5).** Added post-transcription filter that catches known Whisper tiny.en artifacts ("thank you for watching", "subscribe", single repeated words, etc.) and discards them. Prevents false command triggers from silence/noise.

27. ~~**Whisper VAD Tuning & Resampling**~~ — **DONE (Round 5).** Tuned Silero VAD parameters for RPi: `min_silence_duration_ms=300`, `speech_pad_ms=200`, `threshold=0.35`. Upgraded audio resampling from numpy linear interpolation to scipy sinc interpolation (with numpy fallback).

28. ~~**Latency Optimizations**~~ — **DONE (Round 5).** Four improvements: (1) Thinking cue now uses direct PyAudio playback instead of temp WAV + subprocess. (2) Gemini conversation history trimmed from 40 to 20 entries. (3) NLP classification results cached for 2 seconds. (4) Sentence splitting regex now also splits on newlines.

29. ~~**Emotion Fallback Rate-Limiting**~~ — **DONE (Round 5).** Gemini API emotion fallback (used when TFLite is unavailable) now rate-limited to once per 10 seconds instead of every listen cycle. Prevents unnecessary API calls and reduces latency. Applied to both main.py and debug_main.py.

30. ~~**Audio Chunk Duration Increase**~~ — **DONE (Round 5).** `AUDIO_CHUNK_DURATION` increased from 5 to 8 seconds. The previous 5-second limit was truncating longer sentences mid-speech.

31. ~~**Producer-Consumer TTS Pipeline**~~ — **DONE (Round 6).** Queue-based producer-consumer pattern in `_handle_chat`. Producer thread pre-generates TTS audio for upcoming sentences while the current sentence plays. Overlaps generation with playback, saving 1-3 seconds per sentence after the first. Bounded queue (maxsize=4) with sentinel-based termination.

32. ~~**Mouth Sync Callback System**~~ — **DONE (Round 6).** Replaced manual `set_talking(True/False)` calls with a callback system in `SpeechEngine`. `_notify_talking()` fires when audio actually starts/stops playing in `_play_audio_bytes()`, ensuring mouth animation is perfectly synchronized with real audio output.

33. ~~**Listening & Thinking Face States**~~ — **DONE (Round 6).** `set_state("listening")` shows pulsing ear arcs and attentive centered pupils. `set_state("thinking")` drifts eyes upward with animated processing dots below the mouth. Provides clear visual feedback during the listen→think→speak pipeline.

34. ~~**NLP Skip Heuristic**~~ — **DONE (Round 6).** Added fast heuristic to skip the Gemini NLP classification API call for obviously conversational input (>6 words, question words, '?' present, chat indicator words). Saves 1-2 seconds on most chat interactions.

35. ~~**Conversation Quality Improvements**~~ — **DONE (Round 6, trimmed Round 9).** History reduced from 20→30→14 entries (7 exchanges) for optimal balance between context and API latency. Message format simplified from verbose `[EMOTION DETECTED: X (confidence: Y%)]\nUser says: Z` to `[EMOTION: X] Z`. System prompt rewritten with clearer persona, emotional intelligence guidelines, and conversation guidelines.

36. ~~**Silence Duration Tuning**~~ — **DONE (Round 6, reverted Round 8b).** `AUDIO_SILENCE_DURATION` tuned to 0.7s — 0.5s was cutting off speech too early, reverted closer to the Round 2 default for reliability.

37. ~~**TFLite Preprocessing Fix**~~ — **DONE (Round 8b).** Model expects raw 0-255 float32 pixel values with BGR→RGB conversion — no normalization. Preprocessing across all files (`camera_sentiment.py`, `camera_test.py`) now matched exactly to `emotion_test_perfect.py`, which detects all 7 emotions reliably.

38. ~~**Code Audit — Latency & Safety Fixes**~~ — **DONE (Round 7).** Removed 4 unnecessary `time.sleep(0.5)` calls from emotion change handlers. Moved inline imports (`subprocess`, `json`) to module level. Replaced `struct.unpack` RMS with numpy (~5x faster). Extracted `_EMOTION_DIRECTIONS` dict to class constant (was duplicated 3x). Fixed motor `_timed_stop` race condition. Fixed `slight_left/right` missing safety flags.

39. ~~**Full Gemini Live Mode with Function Calling**~~ — **DONE (Round 7).** Complete rewrite of `gemini_live.py` (~530 lines) and `live_main.py` (~310 lines). 10 function tools for full hardware control. Thread-safe queue bridge. Proper `types.Blob` API usage. Reconnection with exponential backoff. Context window compression. Periodic camera frame and emotion context injection.

40. ~~**Debug Mode Inheritance Cleanup**~~ — **DONE (Round 7).** `debug_main.py` `_handle_chat` replaced: was a full copy of chat logic with synchronous TTS (no pipeline). Now delegates to `super()._handle_chat()` with debug wrappers, inheriting the producer-consumer TTS pipeline. `_CHAT_INDICATORS` uses inherited class constant instead of duplicate dict.

44. ~~**Gemini-Based Emotion Classification for Face Display**~~ — **DONE (Round 10).** `determine_response_emotion()` rewritten as a 3-tier system. Primary: `_classify_emotion_gemini()` asks Gemini to classify the emotion of ECHO's response text with a single-word prompt (thinkingBudget=0, max_output_tokens=8, temperature=0.0). Tested 10/10 correct across all 7 emotions. Fallback: `_classify_emotion_keywords()` expanded from ~30 to ~90+ keywords covering all 7 emotions. Default: neutral. Added `_VALID_EMOTIONS` frozenset for validation. Fixes the bug where ECHO's face stayed neutral on responses like "That means a lot to me!" because the old keyword set didn't cover enough happy phrases.

41. ~~**Gemini Cloud STT**~~ — **DONE (Round 9).** Added `_transcribe_gemini()` method to `speech_engine.py`. Sends recorded audio as WAV to `gemini-3.1-flash-lite-preview` with `thinkingBudget=0` for fast transcription (~1-2s). Falls back to local Faster-Whisper on failure. Configurable via `GEMINI_STT_ENABLED` in config.py. More accurate than local tiny.en model and comparable speed over network.

42. ~~**Model Upgrade to Gemini 3.1 Flash Lite**~~ — **DONE (Round 9).** Upgraded chat and STT models from deprecated `gemini-2.0-flash`/`gemini-2.5-flash` to `gemini-3.1-flash-lite-preview` (newest, fastest, cost-efficient). Verified audio+image input support via live API tests. All non-chat API calls (STT, NLP classify, emotion fallback, scene analysis fallback) use `thinkingBudget=0` to eliminate thinking overhead. TTS stays on `gemini-2.5-flash-preview-tts` (no 3.x TTS model available). Robotics stays on `gemini-robotics-er-1.5-preview` (specialized, no equivalent).

43. ~~**Latency Improvements — Round 9**~~ — **DONE (Round 9).** Six improvements from latency analysis: (1) Non-blocking Gemini emotion fallback (runs in background thread). (2) Async movement handler speech (keep_moving, safe_move, patrol, follow use background TTS). (3) Scene analysis timeout+fallback (15s timeout on robotics model, falls back to chat model). (4) Chat history reduced from 30→14 entries. (5) Face shows neutral while thinking (not user's emotion). (6) System prompt updated with explicit emotional intelligence guidelines.

### Known Limitations

- **Single-language by default** — English only (Whisper tiny.en), but pipeline supports switching to multilingual mode via config (see Future Directions #24)
- **TFLite runtime optional** — sentiment defaults to "neutral" without it; uses 3-level import fallback (ai-edge-litert → tflite-runtime → tensorflow.lite); model expects 224x224 RGB with raw 0-255 float32 pixel values (no normalization); softmax applied only when output is raw logits
- **WiFi dependent** — Gemini API requires internet connectivity
- **5V/3.3V mismatch** — HC-SR04 ECHO pin needs a voltage divider
- **No rear sensors** — backward movement logs a warning but cannot detect obstacles behind the robot
- **Battery monitoring hardware** — `battery_monitor.py` exists as a stub; requires ADS1115 ADC for actual voltage readings

---

## Credits

- **Built by:** Sahil Tanna
- **AI:** Google Gemini 3.1 Flash Lite (conversation, STT), Gemini 2.5 Flash Preview (TTS), Gemini Robotics-ER 1.5 (vision)
- **STT:** [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) by SYSTRAN
- **Face Display:** [Pygame](https://www.pygame.org/)
- **Hardware:** Raspberry Pi Foundation
- **Motor Driver:** L298N H-Bridge

---

<p align="center">
  <em>ECHO — Because every robot deserves a personality.</em>
</p>
