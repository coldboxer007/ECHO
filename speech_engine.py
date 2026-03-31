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
import ctypes
import numpy as np

logger = logging.getLogger("echo.speech")

# ── Suppress ALSA error spam ──────────────────────────────
# ALSA prints harmless errors about missing PCM plugins during PyAudio init.
# On systems with PipeWire, these errors can prevent proper device enumeration.
# Suppressing them BEFORE importing pyaudio fixes the issue.
try:
    _ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(
        None, ctypes.c_char_p, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p
    )
    def _alsa_error_handler(filename, line, function, err, fmt):
        pass  # Swallow ALSA errors silently
    _c_alsa_handler = _ERROR_HANDLER_FUNC(_alsa_error_handler)
    _asound = ctypes.cdll.LoadLibrary('libasound.so.2')
    _asound.snd_lib_error_set_handler(_c_alsa_handler)
    logger.debug("ALSA error handler installed")
except Exception:
    pass  # Not on Linux / no ALSA — that's fine

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
        self._input_device_index = None
        self._input_sample_rate = AUDIO_SAMPLE_RATE
        self._output_device_index = None
        self._output_device_name = "default"
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

            # ── Find input device (microphone) ──
            input_name = ""
            try:
                info = self._pyaudio.get_default_input_device_info()
                self._input_device_index = int(info.get("index"))
                default_rate = int(info.get("defaultSampleRate", AUDIO_SAMPLE_RATE))
                input_name = str(info.get("name", ""))
                if default_rate > 0:
                    self._input_sample_rate = default_rate
                logger.info(f"Input device (default): idx={self._input_device_index} '{input_name}' @ {self._input_sample_rate} Hz")
            except Exception:
                # Default failed — scan all devices for a USB/external mic
                logger.warning("Default input device not found, scanning all devices...")
                usb_mic = None
                any_mic = None
                for i in range(self._pyaudio.get_device_count()):
                    try:
                        d = self._pyaudio.get_device_info_by_index(i)
                        if int(d.get("maxInputChannels", 0)) <= 0:
                            continue
                        name = str(d.get("name", ""))
                        logger.debug(f"  Input candidate: idx={i} '{name}'")
                        if any_mic is None:
                            any_mic = d
                        # Prefer USB/external mic devices
                        if any(kw in name.lower() for kw in ["usb", "zeb", "external", "webcam", "mic"]):
                            usb_mic = d
                            break
                    except Exception:
                        continue

                chosen = usb_mic or any_mic
                if chosen:
                    self._input_device_index = int(chosen.get("index"))
                    input_name = str(chosen.get("name", ""))
                    rate = int(chosen.get("defaultSampleRate", AUDIO_SAMPLE_RATE))
                    if rate > 0:
                        self._input_sample_rate = rate
                    logger.info(f"Input device (scanned): idx={self._input_device_index} '{input_name}' @ {self._input_sample_rate} Hz")
                else:
                    logger.error("No input device found! Mic will not work.")

            # ── Find output device (speaker) ──
            try:
                selected_output = None
                for i in range(self._pyaudio.get_device_count()):
                    d = self._pyaudio.get_device_info_by_index(i)
                    if int(d.get("maxOutputChannels", 0)) <= 0:
                        continue
                    if input_name and input_name in str(d.get("name", "")):
                        selected_output = d
                        break

                if selected_output is None:
                    try:
                        selected_output = self._pyaudio.get_default_output_device_info()
                    except Exception:
                        # Default output also failed — find any output device
                        logger.warning("Default output device not found, scanning...")
                        for i in range(self._pyaudio.get_device_count()):
                            d = self._pyaudio.get_device_info_by_index(i)
                            if int(d.get("maxOutputChannels", 0)) > 0:
                                selected_output = d
                                break

                if selected_output:
                    self._output_device_index = int(selected_output.get("index"))
                    self._output_device_name = str(selected_output.get("name", "default"))
                    logger.info(
                        f"Output device: idx={self._output_device_index} name='{self._output_device_name}'"
                    )
                else:
                    logger.error("No output device found! Speaker will not work.")
            except Exception as e:
                logger.warning(f"Could not select output device, using system default: {e}")

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
        audio_data, sample_rate = self._record_audio()

        if not audio_data:
            logger.info("No audio captured")
            return ""

        return self._transcribe(audio_data, sample_rate)

    def _record_audio(self) -> tuple[bytes, int]:
        """
        Record from microphone with silence detection.
        Returns raw PCM audio bytes (16-bit, mono, 16kHz).
        """
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        frames = []
        silence_frames = 0
        has_speech = False
        selected_rate = self._input_sample_rate or AUDIO_SAMPLE_RATE

        candidate_rates = []
        for rate in [selected_rate, AUDIO_SAMPLE_RATE, 48000, 44100, 32000, 22050, 16000]:
            if rate and rate not in candidate_rates:
                candidate_rates.append(rate)

        stream = None
        for rate in candidate_rates:
            try:
                stream = self._pyaudio.open(
                    format=FORMAT,
                    channels=AUDIO_CHANNELS,
                    rate=rate,
                    input=True,
                    input_device_index=self._input_device_index,
                    frames_per_buffer=CHUNK,
                )
                selected_rate = rate
                break
            except Exception as e:
                logger.warning(f"Mic open failed at {rate} Hz: {e}")

        if stream is None:
            logger.error("Recording error: could not open microphone at any supported sample rate")
            return b"", AUDIO_SAMPLE_RATE

        silence_limit = int(AUDIO_SILENCE_DURATION * selected_rate / CHUNK)
        max_frames = int(AUDIO_CHUNK_DURATION * selected_rate / CHUNK)

        try:
            logger.info(f"Recording at {selected_rate} Hz")

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
                return b"", selected_rate

            return b"".join(frames), selected_rate

        except Exception as e:
            logger.error(f"Recording error: {e}")
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
            return b"", selected_rate

    def _transcribe(self, audio_data: bytes, sample_rate: int) -> str:
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
                wf.setframerate(sample_rate)
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
            # Write to WAV first, then play via pw-play for proper routing
            tmp_wav = os.path.join(tempfile.gettempdir(), "echo_espeak.wav")
            subprocess.run(
                ["espeak", "-s", "150", "-v", "en", "-w", tmp_wav, text],
                timeout=30, capture_output=True,
            )
            # Play the WAV through PipeWire
            for cmd in [["pw-play", tmp_wav], ["aplay", tmp_wav]]:
                try:
                    r = subprocess.run(cmd, timeout=30, capture_output=True)
                    if r.returncode == 0:
                        break
                except FileNotFoundError:
                    continue
            try:
                os.unlink(tmp_wav)
            except Exception:
                pass
        except FileNotFoundError:
            logger.error("espeak not installed! Run: sudo apt install espeak")
        except Exception as e:
            logger.error(f"Fallback TTS error: {e}")

    def _resample_pcm16_mono(self, audio_bytes: bytes, src_rate: int, dst_rate: int) -> bytes:
        """Resample 16-bit mono PCM from src_rate to dst_rate."""
        if src_rate == dst_rate or not audio_bytes:
            return audio_bytes

        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        if len(samples) < 2:
            return audio_bytes

        old_x = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
        new_len = max(1, int(len(samples) * (dst_rate / src_rate)))
        new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        resampled = np.interp(new_x, old_x, samples.astype(np.float32))
        return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()

    def _play_audio_bytes(self, audio_bytes: bytes, sample_rate: int):
        """
        Play raw PCM audio bytes through speakers.
        Uses subprocess-based playback (pw-play / aplay) for reliability
        on RPi with PipeWire. Falls back to PyAudio if neither is available.
        """
        if not audio_bytes:
            logger.warning("No audio data to play")
            return

        import subprocess

        # Boost volume — Gemini TTS output can be quiet
        try:
            samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            samples *= 2.0  # +6dB boost
            samples = np.clip(samples, -32768, 32767)
            audio_bytes = samples.astype(np.int16).tobytes()
        except Exception:
            pass  # Play original if boost fails

        # Write PCM data to a temporary WAV file
        tmp_wav = os.path.join(tempfile.gettempdir(), "echo_tts_out.wav")
        try:
            with wave.open(tmp_wav, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_bytes)
        except Exception as e:
            logger.error(f"Failed to write temp WAV: {e}")
            return

        file_size = os.path.getsize(tmp_wav)
        duration = len(audio_bytes) / (sample_rate * 2)  # 2 bytes per sample, mono
        logger.info(f"Playing {duration:.1f}s audio ({file_size} bytes) @ {sample_rate} Hz")

        # Try pw-play first (PipeWire native, most reliable)
        played = False
        for player_cmd in [
            ["pw-play", tmp_wav],
            ["aplay", tmp_wav],
        ]:
            try:
                result = subprocess.run(
                    player_cmd,
                    capture_output=True, text=True,
                    timeout=max(30, duration + 10),
                )
                if result.returncode == 0:
                    logger.info(f"Audio playback OK via {player_cmd[0]}")
                    played = True
                    break
                else:
                    logger.warning(f"{player_cmd[0]} failed: {result.stderr.strip()[:100]}")
            except FileNotFoundError:
                continue
            except subprocess.TimeoutExpired:
                logger.warning(f"{player_cmd[0]} timed out")
            except Exception as e:
                logger.warning(f"{player_cmd[0]} error: {e}")

        # Last resort: PyAudio playback
        if not played and PYAUDIO_AVAILABLE and self._pyaudio is not None:
            logger.info("Falling back to PyAudio playback...")
            stream = None
            try:
                stream = self._pyaudio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    output=True,
                    output_device_index=self._output_device_index,
                )
                CHUNK = 4096
                for i in range(0, len(audio_bytes), CHUNK):
                    stream.write(audio_bytes[i:i + CHUNK])
                stream.stop_stream()
                stream.close()
                logger.info("Audio playback OK via PyAudio")
                played = True
            except Exception as e:
                logger.error(f"PyAudio playback error: {e}")
                if stream:
                    try:
                        stream.stop_stream()
                        stream.close()
                    except Exception:
                        pass

        if not played:
            logger.error("All audio playback methods failed!")

        # Clean up temp file
        try:
            os.unlink(tmp_wav)
        except Exception:
            pass

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
