# 🤖 ECHO — Emotionally Connected Humanoid Observer

A Raspberry Pi 4B-powered companion robot that sees your emotions, talks to you, and moves on your command.

## Features

- **Sentiment Analysis** — USB webcam + TFLite model detects facial emotions locally in real-time
- **Voice Interaction** — Faster-Whisper (STT) + Gemini TTS for natural conversation
- **AI Brain** — Gemini 2.5 Flash generates emotionally-aware responses
- **Expressive Face** — Small display shows animated faces matching detected/response emotions
- **Voice-Controlled Movement** — Tell ECHO to move forward, turn, stop, or follow you
- **Obstacle Avoidance** — Ultrasonic + IR sensors for safe navigation
- **Person Following** — Webcam-based person tracking for follow mode

## Hardware

| Component | Connection |
|---|---|
| Raspberry Pi 4B | Main board |
| L298N Motor Driver (6 motors, 2 sides) | GPIO 31, 33, 35, 37, GND 39 |
| Ultrasonic Sensor (HC-SR04) | Trigger: GPIO 16, Echo: GPIO 18 |
| IR Sensor | GPIO 11 |
| USB Microphone | USB port |
| USB Camera | USB port |
| Small Display (SPI/I2C/HDMI) | Display port |

## Pin Mapping (BOARD numbering)

```
L298N Motor Driver:
  IN1 → Pin 31 (GPIO 6)   — Left motors forward
  IN2 → Pin 33 (GPIO 13)  — Left motors backward
  IN3 → Pin 35 (GPIO 19)  — Right motors forward
  IN4 → Pin 37 (GPIO 26)  — Right motors backward
  GND → Pin 39 (GND)

Ultrasonic Sensor:
  TRIG → Pin 16 (GPIO 23)
  ECHO → Pin 18 (GPIO 24)

IR Sensor:
  OUT  → Pin 11 (GPIO 17)
```

## Installation

### 1. Clone & Setup
```bash
git clone <your-repo-url> ECHO
cd ECHO
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. System Dependencies (on Raspberry Pi)
```bash
sudo apt update
sudo apt install -y python3-dev python3-pip libatlas-base-dev \
  libportaudio2 portaudio19-dev ffmpeg \
  libsdl2-dev libsdl2-ttf-dev libsdl2-image-dev \
  python3-pygame espeak
```

### 3. Configure API Key
```bash
cp .env.example .env
# Edit .env and add your Gemini API key
nano .env
```

### 4. Place your TFLite model
```bash
# Copy your sentiment analysis .tflite file into the models/ directory
cp /path/to/your/model.tflite models/sentiment_model.tflite
```

### 5. Run
```bash
python3 main.py
```

## Voice Commands

| Command | Action |
|---|---|
| "move forward" / "go forward" | Drive forward |
| "move backward" / "go back" | Drive backward |
| "turn left" | Turn left |
| "turn right" | Turn right |
| "stop" | Stop all motors |
| "follow me" | Enter person-following mode |
| "stop following" | Exit follow mode |
| "how are you" / any question | Chat with Gemini AI |

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ USB Camera  │────▶│  Sentiment   │────▶│  Gemini AI  │
│             │     │  (TFLite)    │     │  Brain      │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
┌─────────────┐     ┌──────────────┐            │
│ USB Mic     │────▶│  Faster-     │────────────┘
│             │     │  Whisper STT │            │
└─────────────┘     └──────────────┘            │
                                                ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Ultrasonic  │────▶│  Navigation  │◀───▶│  Motor      │
│ + IR Sensor │     │  Controller  │     │  Controller │
└─────────────┘     └──────────────┘     └─────────────┘
                                                │
                    ┌──────────────┐            │
                    │  Gemini TTS  │◀───────────┘
                    │  Speaker     │
                    └──────────────┘
                    ┌──────────────┐
                    │  Face        │
                    │  Display     │
                    └──────────────┘
```

## Project Structure

```
ECHO/
├── main.py                 # Main orchestrator
├── config.py               # Pin mappings, settings, API config
├── motor_controller.py     # L298N motor driver control
├── sensor_controller.py    # Ultrasonic + IR sensors
├── camera_sentiment.py     # Webcam capture + TFLite inference
├── speech_engine.py        # STT (Faster-Whisper) + TTS (Gemini)
├── gemini_brain.py         # Gemini API conversation engine
├── face_display.py         # Pygame animated face display
├── navigation.py           # Movement logic + obstacle avoidance
├── requirements.txt        # Python dependencies
├── setup.sh                # System dependency installer
├── .env.example            # Environment variable template
├── models/                 # TFLite model files
│   └── sentiment_model.tflite
└── assets/
    └── sounds/             # Audio assets
```

## License

MIT
