"""
ECHO Robot — Speech Engine
============================
Speech-to-Text:  Faster-Whisper (lightweight, runs locally on RPi CPU)
Text-to-Speech:  Gemini TTS API (high-quality, emotional voice output)
                 Falls back to espeak if Gemini TTS is unavailable.

Audio I/O uses PyAudio for microphone capture and playback.
"""

import os
import io
import time
import wave
import struct
import logging
import tempfile
import threading
import numpy as np

logger = logging.getLogger("echo.speech")

# Audio I/O
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logger.warning("PyAudio not available — audio I/O disabled")

# Faster Whisper STT
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("faster-whisper not available — STT disabled")

# Gemini TTS
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("google-genai not available — TTS will use fallback")

from config import (
    GEMINI_API_KEY, GEMINI_TTS_MODEL,
    WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
    WHISPER_LANGUAGE, WHISPER_BEAM_SIZE,
    AUDIO_SAMPLE_RATE, AUDIO_CHANNELS, AUDIO_CHUNK_DURATION,
    AUDIO_SILENCE_THRESHOLD, AUDIO_SILENCE_DURATION,
    TTS_VOICE, TTS_SAMPLE_RATE, TTS_FALLBACK_ENGINE,
)


class SpeechEngine:
    """Handles speech-to-text (Faster Whisper) and text-to-speech (Gemini TTS)."""

    def __init__(self):
        self._whisper_model = None
        self._genai_client = None
        self._pyaudio = None
        self._is_speaking = False
        self._lock = threading.Lock()

        self._init_stt()
        self._init_tts()
        self._init_audio()
        logger.info("SpeechEngine initialized")

    # ═══════════════════════════════════════════
    # Initialization
    # ═══════════════════════════════════════════

    def _init_stt(self):
        """Load Faster Whisper model for speech-to-text."""
        if not WHISPER_AVAILABLE:
            logger.warning("Faster Whisper not loaded")
            return

        try:
            logger.info(f"Loading Whisper model: {WHISPER_MODEL_SIZE} ({WHISPER_COMPUTE_TYPE})...")
            self._whisper_model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
            )
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")

    def _init_tts(self):
        """Initialize Gemini client for TTS."""
        if not GENAI_AVAILABLE or not GEMINI_API_KEY:
            logger.warning("Gemini TTS not configured")
            return

        try:
            self._genai_client = genai.Client(api_key=GEMINI_API_KEY)
            logger.info("Gemini TTS client initialized")
        except Exception as e:
            logger.error(f"Failed to init Gemini client: {e}")

    def _init_audio(self):
        """Initialize PyAudio for mic input and speaker output."""
        if not PYAUDIO_AVAILABLE:
            return

        try:
            self._pyaudio = pyaudio.PyAudio()
            logger.info("PyAudio initialized")
        except Exception as e:
            logger.error(f"Failed to init PyAudio: {e}")

    # ═══════════════════════════════════════════
    # Speech-to-Text (Microphone → Text)
    # ═══════════════════════════════════════════

    def listen(self) -> str:
        """
        Record audio from microphone until silence is detected,
        then transcribe with Faster Whisper.
        Returns the transcribed text, or empty string on failure.
        """
        if not PYAUDIO_AVAILABLE or self._pyaudio is None:
            logger.error("Cannot listen — PyAudio not available")
            return ""

        logger.info("🎤 Listening...")
        audio_data = self._record_audio()

        if not audio_data:
            logger.info("No audio captured")
            return ""

        return self._transcribe(audio_data)

    def _record_audio(self) -> bytes:
        """
        Record from microphone with silence detection.
        Returns raw PCM audio bytes (16-bit, mono, 16kHz).
        """
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        frames = []
        silence_frames = 0
        silence_limit = int(AUDIO_SILENCE_DURATION * AUDIO_SAMPLE_RATE / CHUNK)
        max_frames = int(AUDIO_CHUNK_DURATION * AUDIO_SAMPLE_RATE / CHUNK)
        has_speech = False

        try:
            stream = self._pyaudio.open(
                format=FORMAT,
                channels=AUDIO_CHANNELS,
                rate=AUDIO_SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )

            for _ in range(max_frames):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)

                # Calculate RMS for silence detection
                samples = struct.unpack(f'{CHUNK}h', data)
                rms = (sum(s * s for s in samples) / CHUNK) ** 0.5

                if rms > AUDIO_SILENCE_THRESHOLD:
                    has_speech = True
                    silence_frames = 0
                else:
                    silence_frames += 1

                # Stop after enough silence (but only if we heard speech first)
                if has_speech and silence_frames >= silence_limit:
                    break

            stream.stop_stream()
            stream.close()

            if not has_speech:
                return b""

            return b"".join(frames)

        except Exception as e:
            logger.error(f"Recording error: {e}")
            return b""

    def _transcribe(self, audio_data: bytes) -> str:
        """Transcribe raw PCM audio bytes using Faster Whisper."""
        if self._whisper_model is None:
            logger.error("Whisper model not loaded")
            return ""

        try:
            # Save to temp WAV file (Faster Whisper reads files)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
                wf = wave.open(f, 'wb')
                wf.setnchannels(AUDIO_CHANNELS)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(AUDIO_SAMPLE_RATE)
                wf.writeframes(audio_data)
                wf.close()

            # Transcribe
            segments, info = self._whisper_model.transcribe(
                tmp_path,
                beam_size=WHISPER_BEAM_SIZE,
                language=WHISPER_LANGUAGE,
                vad_filter=True,
            )

            text = " ".join(segment.text.strip() for segment in segments).strip()
            logger.info(f"📝 Transcribed: '{text}'")

            # Clean up temp file
            os.unlink(tmp_path)

            return text

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

    # ═══════════════════════════════════════════
    # Text-to-Speech (Text → Speaker)
    # ═══════════════════════════════════════════

    def speak(self, text: str, emotion: str = "neutral"):
        """
        Convert text to speech and play through speaker.
        Uses Gemini TTS API with emotional voice styling.
        Falls back to espeak if Gemini is unavailable.
        """
        if not text:
            return

        with self._lock:
            self._is_speaking = True

        logger.info(f"🔊 Speaking: '{text[:60]}...' (emotion={emotion})")

        try:
            if self._genai_client is not None:
                self._speak_gemini(text, emotion)
            else:
                self._speak_fallback(text)
        except Exception as e:
            logger.error(f"TTS error: {e}")
            self._speak_fallback(text)
        finally:
            with self._lock:
                self._is_speaking = False

    def _speak_gemini(self, text: str, emotion: str = "neutral"):
        """Use Gemini TTS API for high-quality emotional speech."""
        # Build an expressive prompt with emotion cues
        emotion_directions = {
            "happy":    "Say this warmly and cheerfully with a smile in your voice:",
            "sad":      "Say this gently and softly with empathy:",
            "angry":    "Say this in a calm, reassuring tone:",
            "surprise": "Say this with gentle excitement and wonder:",
            "fear":     "Say this in a warm, comforting and reassuring way:",
            "disgust":  "Say this calmly and matter-of-factly:",
            "neutral":  "Say this in a friendly, conversational tone:",
        }
        direction = emotion_directions.get(emotion, emotion_directions["neutral"])
        prompt = f"{direction}\n\"{text}\""

        try:
            response = self._genai_client.models.generate_content(
                model=GEMINI_TTS_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=TTS_VOICE,
                            )
                        )
                    ),
                ),
            )

            # Extract audio data from response
            audio_data = None
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.mime_type.startswith("audio/"):
                    audio_data = part.inline_data.data
                    break

            if audio_data:
                self._play_audio_bytes(audio_data, TTS_SAMPLE_RATE)
            else:
                logger.warning("No audio in Gemini TTS response, using fallback")
                self._speak_fallback(text)

        except Exception as e:
            logger.error(f"Gemini TTS error: {e}")
            self._speak_fallback(text)

    def _speak_fallback(self, text: str):
        """Fallback TTS using espeak (works offline on RPi)."""
        try:
            import subprocess
            # espeak: -s = speed (words per minute), -v = voice
            cmd = ["espeak", "-s", "150", "-v", "en", text]
            subprocess.run(cmd, timeout=30, capture_output=True)
        except FileNotFoundError:
            logger.error("espeak not installed! Run: sudo apt install espeak")
        except Exception as e:
            logger.error(f"Fallback TTS error: {e}")

    def _play_audio_bytes(self, audio_bytes: bytes, sample_rate: int):
        """Play raw PCM audio bytes through speakers."""
        if not PYAUDIO_AVAILABLE or self._pyaudio is None:
            logger.error("Cannot play audio — PyAudio not available")
            return

        try:
            stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                output=True,
            )

            # Gemini TTS returns raw PCM 16-bit audio
            CHUNK = 4096
            for i in range(0, len(audio_bytes), CHUNK):
                stream.write(audio_bytes[i:i + CHUNK])

            stream.stop_stream()
            stream.close()

        except Exception as e:
            logger.error(f"Audio playback error: {e}")

    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return self._is_speaking

    # ═══════════════════════════════════════════
    # Cleanup
    # ═══════════════════════════════════════════

    def cleanup(self):
        """Release audio resources."""
        if self._pyaudio:
            self._pyaudio.terminate()
        logger.info("SpeechEngine cleaned up")
