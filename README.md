# 🤖 ECHO — Emotionally Connected Humanoid Observer

<p align="center">
  <strong>A Raspberry Pi 4B companion robot that sees, hears, speaks, moves, and feels.</strong>
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

ECHO is a **fully autonomous companion robot** built on a Raspberry Pi 4B. It combines:

- **Voice conversation** — powered by Google Gemini 2.5 Flash for intelligent, emotionally-aware dialogue
- **Speech recognition** — Faster-Whisper (tiny.en) running locally on the RPi CPU
- **Expressive speech** — Gemini TTS with emotional voice styling (happy, sad, surprised, etc.)
- **Animated robot face** — full-screen pygame display with smooth eye animations, blinking, pupil wandering, breathing, and emotion transitions
- **Motor control** — 6 DC motors (3 per side) via L298N driver for tank-steer movement
- **Obstacle avoidance** — HC-SR04 ultrasonic + IR sensors for real-time safety
- **Camera sentiment analysis** — USB webcam with Haar cascade face detection + optional TFLite emotion model
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
│  │(Zeb USB)│    │ (Whisper STT)│    │ (Gemini 2.5 Flash│    │
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
│              Raspberry Pi 4B (4GB RAM)                       │
│              Raspberry Pi OS (Bookworm 64-bit)               │
│              PipeWire 1.4.2 audio server                     │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User speaks → Mic captures audio (PyAudio @ 44100Hz)
  → Silence detection stops recording
  → Faster-Whisper tiny.en transcribes (beam_size=1, ~2-3s on RPi4)
  → GeminiBrain.interpret_command() routes to handler:
      ├─ Movement → NavigationController → MotorController → GPIO → L298N → Motors
      ├─ Safe move → Continuous obstacle checking while moving
      ├─ Keep moving → Background thread, auto-pause on obstacle
      ├─ Patrol → Back-and-forth loop with obstacle awareness
      ├─ Stop → Emergency stop all motors
      ├─ Follow → Camera face tracking + motor steering
      └─ Chat → Gemini 2.5 Flash generates response
               → determine_response_emotion() picks face emotion
               → FaceDisplay shows emotion
               → Gemini TTS generates speech audio
               → pw-play outputs through USB speaker
```

---

## Hardware Bill of Materials

| Component | Specification | Purpose |
|---|---|---|
| **Raspberry Pi 4B** | 4GB RAM, 64-bit Bookworm OS | Main computer |
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
| **STT** | Faster-Whisper (tiny.en, int8) | Local speech-to-text (~2-3s on RPi4) |
| **AI Brain** | Google Gemini 2.5 Flash | Conversation + command understanding |
| **TTS** | Gemini 2.5 Flash Preview TTS | Emotional text-to-speech (voice: Kore) |
| **Live API** | Gemini 2.5 Flash Native Audio Preview | Alternative: bidirectional audio streaming |
| **Face Display** | Pygame 2.6.1 (SDL2) | Animated robot face rendering |
| **Vision** | OpenCV 4.x + Haar cascades | Face detection |
| **Sentiment** | TFLite (FER 3-stage FP16) | Facial emotion recognition (optional) |
| **GPIO** | RPi.GPIO (lgpio backend) | Motor and sensor control |
| **Audio I/O** | PyAudio (capture) + pw-play (output) | Mic input + speaker output |
| **Fallback TTS** | espeak | Offline TTS when Gemini is unavailable |

---

## Project Structure

```
ECHOtest/
├── main.py                  # Main entry point — the listen→think→speak→act loop
├── debug_main.py            # Verbose debug version with color-coded terminal output
├── live_main.py             # Alternative: Gemini Live API bidirectional audio mode
├── config.py                # ALL settings, pin assignments, model paths, prompts
├── motor_controller.py      # L298N motor driver — forward/backward/turn/stop
├── sensor_controller.py     # HC-SR04 ultrasonic + IR obstacle detection
├── camera_sentiment.py      # USB webcam + TFLite facial emotion analysis
├── speech_engine.py         # Faster-Whisper STT + Gemini TTS + audio I/O
├── gemini_brain.py          # Gemini AI conversation + command interpretation
├── gemini_live.py           # Gemini Live API engine (bidirectional audio stream)
├── face_display.py          # Pygame animated robot face (eyes, mouth, emotions)
├── navigation.py            # High-level movement: manual, follow, patrol, safe move
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
    3. Interpret command (GeminiBrain.interpret_command())
    4. Route to handler based on command type
    5. Execute action (move, speak, or both)
    6. Update face display
    7. Loop back to listening
```

The loop skips listening while the robot is speaking (to avoid hearing its own voice).

### Speech-to-Text Pipeline

**File:** `speech_engine.py` → `_record_audio()` + `_transcribe()`

1. **Microphone capture:** PyAudio opens the USB mic at its native sample rate (44100Hz for Zeb SoundMX through PipeWire). Records in 1024-sample chunks.

2. **Silence detection:** Each chunk's RMS energy is computed. If RMS > 150 (configurable threshold), it's considered speech. Recording stops after 1.0 seconds of silence following speech, or after 5 seconds maximum.

3. **Whisper transcription:** The recorded PCM audio is saved to a temporary WAV file and fed to Faster-Whisper with:
   - Model: `tiny.en` (39MB — chosen for speed on RPi4 CPU, ~2-3s per transcription)
   - Compute type: `int8` (quantized for ARM CPU)
   - Beam size: `1` (greedy decode — 10x faster than beam_size=3)
   - VAD filter: enabled (filters out non-speech segments)
   - Language: English only

4. **ALSA error suppression:** Before importing PyAudio, we install a custom ALSA error handler using ctypes to suppress harmless "PCM plugin" warnings that interfere with PipeWire's ALSA compatibility layer.

**Performance evolution:**

| Model | Beam Size | Transcription Time | Notes |
|---|---|---|---|
| base.en | 3 | ~24 seconds | Original — unusable |
| base.en | 1 | ~15 seconds | Still too slow |
| **tiny.en** | **1** | **~2-3 seconds** | Final choice |

### AI Brain & Command Routing

**File:** `gemini_brain.py` → `interpret_command()` + `think()`

The brain has two functions:

1. **Command interpretation** (`interpret_command`): Parses user text to determine if it's a movement command or conversation. Uses a **priority-based keyword matching system** (see [Command Routing](#command-routing--priority-system) below).

2. **Conversation** (`think`): Sends the user's text + detected emotion to Gemini 2.5 Flash with:
   - System prompt defining ECHO's personality
   - Conversation history (last 20 messages)
   - Emotion tag from camera (e.g., `[EMOTION DETECTED: happy (75%)]`)
   - Temperature: 0.8, max_output_tokens: 350, top_p: 0.9

3. **Emotion determination** (`determine_response_emotion`): Analyzes Gemini's response text for emotional keywords to choose the appropriate face animation.

### Text-to-Speech Pipeline

**File:** `speech_engine.py` → `speak()` + `_speak_gemini()` + `_play_audio_bytes()`

1. **Gemini TTS:** The response text is sent to `gemini-2.5-flash-preview-tts` with:
   - Voice: "Kore" (firm, clear voice)
   - Emotion direction prepended (e.g., "Say this warmly and cheerfully with a smile in your voice:")
   - Output: raw PCM audio at 24kHz

2. **Volume boost:** The returned audio gets a +6dB boost (x2.0 amplitude) because Gemini TTS output tends to be quiet.

3. **Playback:** Audio is written to a temp WAV file and played via:
   - **Primary:** `pw-play` (PipeWire native — most reliable on RPi with PipeWire)
   - **Fallback 1:** `aplay` (ALSA)
   - **Fallback 2:** PyAudio stream (last resort)

4. **Espeak fallback:** If Gemini TTS fails entirely, `espeak` generates offline speech, piped through `pw-play`.

> **Why not PyAudio for playback?** On RPi with PipeWire, PyAudio playback was silent — PipeWire's ALSA compatibility layer didn't route PyAudio's output to the USB speaker. Subprocess-based `pw-play` communicates directly with PipeWire's native protocol and works reliably.

### Motor Control & Navigation

**Files:** `motor_controller.py` + `navigation.py`

**Motor Controller** (low-level):
- Controls 4 GPIO pins connected to L298N motor driver
- Each pin is set HIGH or LOW (no PWM yet — full speed only)
- All 6 motors are in 2 groups: left (IN1/IN2) and right (IN3/IN4)
- Differential drive: same direction = straight, opposite = turn in place
- GPIO pins initialized with `initial=GPIO.LOW` to prevent motor spin during setup
- `GPIO.cleanup()` is **intentionally NOT called** — it resets pins to INPUT (floating/high-impedance), which the L298N reads as HIGH and spins the motors

**Navigation Controller** (high-level):
- Wraps motor controller with sensor feedback
- Checks obstacles before forward movement
- Provides multiple movement modes (see [Movement Modes](#movement-modes))

### Sensor Feedback & Obstacle Avoidance

**File:** `sensor_controller.py`

- **HC-SR04 ultrasonic:** Sends 10us trigger pulse, measures echo return time, calculates distance. Readings below 2cm are ignored as sensor noise.
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
- **Smooth transitions:** Emotion changes blend over ~0.33 seconds using linear interpolation
- **Random blinking:** Every 2-5.5 seconds, eyes smoothly close and open (triangle wave)
- **Pupil wandering:** Pupils drift randomly with lerp smoothing during idle
- **Breathing animation:** Subtle vertical oscillation of eye positions
- **Talk animation:** Multi-frequency mouth movement (3 layered sine waves) when speaking
- **Idle micro-movements:** Gentle sway when neutral

**Performance optimizations:**
- Scan line overlay pre-rendered once at startup (was creating new surface every frame)
- All glow effects use direct dim-color drawing instead of per-frame SRCALPHA surface allocations
- Reduced from 30fps to 20fps (saves CPU for Whisper)
- `pygame.display.init()` + `pygame.font.init()` instead of `pygame.init()` (avoids pygame stealing the audio device from PipeWire)
- `pygame.QUIT` events are ignored (RPi window manager sends spurious QUIT events that would crash the display)

### Camera & Sentiment Analysis

**File:** `camera_sentiment.py`

1. **USB webcam** captures at 640x480 @ 15fps via OpenCV
2. **Haar cascade** (`haarcascade_frontalface_default.xml`) detects faces in grayscale
3. **TFLite model** (`fer_3stage_fp16.tflite`) classifies facial emotion:
   - Input: 48x48 grayscale face crop
   - Output: 7-class probability vector (angry, disgust, fear, happy, sad, surprise, neutral)
   - Confidence threshold: 40%
4. **Background loop** runs at 1Hz, updating `current_emotion` and `current_confidence`
5. **Gemini API fallback** available if TFLite runtime is not installed

---

## Command Routing — Priority System

The `interpret_command()` function uses a **strict priority ordering** to prevent false matches. This was redesigned after discovering that phrases like "move forward until you can stop at your obstacle" were being incorrectly classified as "stop" commands.

```
Priority 1: Complex / continuous movement
    ├── keep_moving  → "keep going", "continue moving", "don't stop"
    ├── safe_move    → "carefully", "stop when obstacle", "watch out"
    └── patrol       → "patrol", "back and forth", "explore"

Priority 2: Explicit stop (with disambiguation)
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
    └── follow → "follow me", "come here"

Priority 6: Chat (default)
    └── Everything else → sent to Gemini for conversation
```

---

## Movement Modes

| Mode | Trigger Phrase | Behavior |
|---|---|---|
| **Single move** | "move forward", "turn right" | Moves for default duration (1.0s move, 0.5s turn) |
| **Timed move** | "go forward for 3 seconds" | Moves for specified duration |
| **Safe move** | "go forward carefully" | Moves with continuous obstacle checking, stops permanently on detection |
| **Keep moving** | "keep going", "don't stop" | Continuous movement in background thread, auto-pauses on obstacle, resumes when clear |
| **Patrol** | "patrol", "explore" | Forward 3s → pause → backward 3s → pause → repeat, with obstacle awareness |
| **Follow** | "follow me" | Camera tracks largest face, steers toward it, maintains ~80cm distance |
| **Spin** | "spin", "turn around" | Right turn for 2 seconds |
| **Emergency stop** | "stop", "halt", "freeze" | Immediately stops all motors and background movement modes |

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
Same functionality but with verbose color-coded terminal output. Shows every sensor reading, command interpretation, and API call in real-time. Best for development.

### Live API Mode (live_main.py)
```bash
python3 live_main.py
```
Experimental mode using Gemini Live API for bidirectional audio streaming. Replaces the separate Whisper STT + Gemini Chat + Gemini TTS pipeline with a single streaming connection. Benefits:
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
    v4l-utils pipewire pipewire-alsa

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt
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

---

## Future Directions

### Planned Improvements

1. **PWM Speed Control** — Currently motors are full-speed only (GPIO HIGH/LOW). Adding software PWM via `RPi.GPIO.PWM` or hardware PWM would enable variable speed, smoother turns, and gentler movements.

2. **Gemini Live API as Primary Mode** — The `live_main.py` mode eliminates the separate Whisper → Gemini → TTS pipeline. With further testing, it could become the default, cutting latency from ~5-8s to under 2s for a full conversation turn.

3. **Wake Word Detection** — Add a lightweight wake word detector (e.g., Porcupine, openWakeWord) so ECHO only activates on "Hey Echo" instead of listening continuously.

4. **SLAM / Mapping** — Use ultrasonic + camera data to build a rudimentary room map for autonomous navigation beyond simple obstacle avoidance.

5. **Multi-Person Tracking** — Extend the follow mode to track specific people using face embeddings, rather than just following the largest face.

6. **Gemini Robotics-ER Integration** — The codebase already has `analyze_scene()` and `detect_person_position()` methods using Gemini Robotics-ER for richer scene understanding. These could power smarter navigation.

7. **Edge TPU / Coral Accelerator** — Adding a Google Coral USB accelerator would speed up TFLite inference from ~100ms to ~10ms and enable real-time emotion detection at camera framerate.

8. **Battery Monitoring** — Add an ADC (e.g., ADS1115) to monitor motor battery voltage and have ECHO warn when batteries are low.

9. **OTA Updates** — Implement a simple git-pull mechanism so the robot can update its own code over WiFi.

10. **Web Dashboard** — A Flask/FastAPI web interface showing real-time sensor data, camera feed, conversation history, and manual controls accessible from any device on the local network.

11. **Persistent Memory** — Save conversation highlights and user preferences to a local SQLite database so ECHO remembers across restarts.

12. **Sound Localization** — With a stereo mic array, ECHO could turn toward the person speaking before responding.

### Known Limitations

- **No PWM speed control** — motors are either full speed or stopped
- **Single-language** — English only (Whisper tiny.en)
- **No wake word** — listens continuously, uses silence detection
- **TFLite runtime optional** — sentiment defaults to "neutral" without it
- **WiFi dependent** — Gemini API requires internet connectivity
- **5V/3.3V mismatch** — HC-SR04 ECHO pin needs a voltage divider

---

## Credits

- **Built by:** Sahil Tanna
- **AI:** Google Gemini 2.5 Flash (conversation, TTS, vision)
- **STT:** [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) by SYSTRAN
- **Face Display:** [Pygame](https://www.pygame.org/)
- **Hardware:** Raspberry Pi Foundation
- **Motor Driver:** L298N H-Bridge

---

<p align="center">
  <em>ECHO — Because every robot deserves a personality.</em>
</p>
