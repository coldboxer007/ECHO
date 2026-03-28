#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# ECHO Robot — Raspberry Pi System Dependencies Installer
# Run: chmod +x setup.sh && sudo ./setup.sh
# ─────────────────────────────────────────────────────────────

set -e

echo "🤖 ECHO Robot — Installing system dependencies..."

sudo apt update && sudo apt upgrade -y

# Core build tools
sudo apt install -y build-essential python3-dev python3-pip python3-venv

# Audio (PortAudio for PyAudio, ALSA utils, espeak for fallback TTS)
sudo apt install -y libportaudio2 portaudio19-dev libasound2-dev \
    alsa-utils pulseaudio espeak ffmpeg

# SDL2 for Pygame display
sudo apt install -y libsdl2-dev libsdl2-ttf-dev libsdl2-image-dev \
    libsdl2-mixer-dev python3-pygame

# OpenCV dependencies
sudo apt install -y libatlas-base-dev libjasper-dev libqtgui4 \
    libqt4-test libhdf5-dev

# Camera tools
sudo apt install -y v4l-utils

# Python virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo ""
echo "✅ System dependencies installed!"
echo ""
echo "Next steps:"
echo "  1. Copy .env.example to .env and add your Gemini API key"
echo "  2. Place your .tflite model in models/sentiment_model.tflite"
echo "  3. Run: source venv/bin/activate && python3 main.py"
