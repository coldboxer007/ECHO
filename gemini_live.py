"""
ECHO Robot — Gemini Live API Voice Engine
============================================
Replaces the separate Whisper STT + Gemini TTS pipeline with a single
bidirectional audio stream through the Gemini Live API.

How it works:
  1. Mic audio (PCM 16kHz) is streamed to Gemini in real-time
  2. Gemini processes speech, understands it, and responds with audio
  3. Response audio (PCM 24kHz) is played through pw-play / aplay
  4. Camera frames can be sent for visual context

Requires:  pip install google-genai pyaudio
"""

import os
import sys
import io
import wave
import time
import asyncio
import logging
import tempfile
import threading
import traceback
import ctypes
import numpy as np

logger = logging.getLogger("echo.live")

# ── Suppress ALSA error spam ──
try:
    _ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(
        None, ctypes.c_char_p, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p
    )
    def _alsa_error_handler(filename, line, function, err, fmt):
        pass
    _c_alsa_handler = _ERROR_HANDLER_FUNC(_alsa_error_handler)
    _asound = ctypes.cdll.LoadLibrary('libasound.so.2')
    _asound.snd_lib_error_set_handler(_c_alsa_handler)
except Exception:
    pass

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logger.error("pyaudio not available! Install with: pip install pyaudio")

try:
    from google import genai
    from google.genai import types
except ImportError:
    logger.error("google-genai not available! pip install google-genai")
    sys.exit(1)

if sys.version_info < (3, 11, 0):
    try:
        import taskgroup, exceptiongroup
        asyncio.TaskGroup = taskgroup.TaskGroup
        asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup
    except ImportError:
        logger.warning("Python < 3.11 — install taskgroup and exceptiongroup packages")

from config import (
    GEMINI_API_KEY, SYSTEM_PROMPT, TTS_VOICE,
)

# ─────────────────────────────────────────────
# Audio Configuration
# ─────────────────────────────────────────────
FORMAT = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None
CHANNELS = 1
SEND_SAMPLE_RATE = 16000     # Mic → Gemini
RECEIVE_SAMPLE_RATE = 24000  # Gemini → Speaker
CHUNK_SIZE = 1024

# ─────────────────────────────────────────────
# Gemini Live API Configuration
# ─────────────────────────────────────────────
MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"


class GeminiLiveEngine:
    """
    Bidirectional voice engine using Gemini Live API.
    Handles both speech input and audio output through a single
    persistent streaming connection.
    """

    def __init__(self, on_turn_start=None, on_turn_end=None, on_text=None):
        """
        Callbacks:
          on_turn_start()  — called when Gemini starts responding
          on_turn_end()    — called when Gemini finishes a turn
          on_text(str)     — called with transcribed/response text
        """
        self._on_turn_start = on_turn_start
        self._on_turn_end = on_turn_end
        self._on_text = on_text

        self._running = False
        self._session = None
        self._audio_in_queue = None
        self._out_queue = None
        self._audio_stream = None
        self._pya = None
        self._loop = None
        self._thread = None

        # Build the client
        api_key = GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set! Set it in .env or environment.")

        self._client = genai.Client(
            api_key=api_key,
            http_options={"api_version": "v1beta"},
        )

        # Live session config
        self._config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=TTS_VOICE or "Zephyr"
                    )
                )
            ),
            system_instruction=SYSTEM_PROMPT,
            context_window_compression=types.ContextWindowCompressionConfig(
                trigger_tokens=25600,
                sliding_window=types.SlidingWindow(target_tokens=12800),
            ),
        )

        logger.info("GeminiLiveEngine initialized")

    # ═══════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════

    def start(self):
        """Start the live audio session in a background thread."""
        if not PYAUDIO_AVAILABLE:
            logger.error("Cannot start — pyaudio is not installed")
            return

        if self._running:
            logger.warning("Already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("🎙️ Gemini Live session starting...")

    def stop(self):
        """Stop the live session."""
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Gemini Live session stopped")

    def send_text(self, text: str):
        """Send a text message to the session (non-blocking)."""
        if self._session and self._loop:
            asyncio.run_coroutine_threadsafe(
                self._send_text_async(text), self._loop
            )

    def send_image(self, jpeg_bytes: bytes):
        """Send a camera frame to give Gemini visual context."""
        if self._session and self._loop and self._out_queue:
            import base64
            payload = {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(jpeg_bytes).decode()
            }
            try:
                self._out_queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass  # Skip frame if queue is full

    @property
    def is_running(self):
        return self._running

    # ═══════════════════════════════════════════
    # Internal: Async Event Loop
    # ═══════════════════════════════════════════

    def _run_loop(self):
        """Run the asyncio event loop in a background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run())
        except Exception as e:
            logger.error(f"Live session error: {e}")
            traceback.print_exc()
        finally:
            self._running = False

    async def _run(self):
        """Main async loop — connect and manage all audio tasks."""
        self._pya = pyaudio.PyAudio()
        try:
            async with (
                self._client.aio.live.connect(
                    model=MODEL, config=self._config
                ) as session,
                asyncio.TaskGroup() as tg,
            ):
                self._session = session
                self._audio_in_queue = asyncio.Queue()
                self._out_queue = asyncio.Queue(maxsize=5)

                logger.info("✅ Connected to Gemini Live API")

                tg.create_task(self._listen_audio())
                tg.create_task(self._send_realtime())
                tg.create_task(self._receive_audio())
                tg.create_task(self._play_audio())

                # Keep running until stopped
                while self._running:
                    await asyncio.sleep(0.1)

                raise asyncio.CancelledError("Shutdown requested")

        except asyncio.CancelledError:
            logger.info("Live session cancelled")
        except Exception as e:
            logger.error(f"Live session error: {e}")
            traceback.print_exc()
        finally:
            if self._audio_stream:
                try:
                    self._audio_stream.stop_stream()
                    self._audio_stream.close()
                except Exception:
                    pass
            if self._pya:
                self._pya.terminate()
            self._session = None

    # ═══════════════════════════════════════════
    # Audio Input: Microphone → Gemini
    # ═══════════════════════════════════════════

    async def _listen_audio(self):
        """Capture mic audio and push to output queue."""
        try:
            mic_info = self._pya.get_default_input_device_info()
            self._audio_stream = await asyncio.to_thread(
                self._pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=SEND_SAMPLE_RATE,
                input=True,
                input_device_index=mic_info["index"],
                frames_per_buffer=CHUNK_SIZE,
            )
            logger.info(f"🎤 Mic open: {mic_info['name']} @ {SEND_SAMPLE_RATE} Hz")

            kwargs = {"exception_on_overflow": False} if __debug__ else {}

            while self._running:
                data = await asyncio.to_thread(
                    self._audio_stream.read, CHUNK_SIZE, **kwargs
                )
                payload = {"data": data, "mime_type": "audio/pcm"}
                try:
                    self._out_queue.put_nowait(payload)
                except asyncio.QueueFull:
                    # Drop oldest to keep stream real-time
                    _ = self._out_queue.get_nowait()
                    self._out_queue.put_nowait(payload)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Mic error: {e}")
        finally:
            if self._audio_stream:
                try:
                    self._audio_stream.stop_stream()
                    self._audio_stream.close()
                except Exception:
                    pass

    async def _send_realtime(self):
        """Send queued audio/media chunks to Gemini."""
        try:
            while self._running:
                msg = await self._out_queue.get()
                if msg.get("mime_type", "").startswith("audio/"):
                    await self._session.send_realtime_input(audio=msg)
                else:
                    await self._session.send_realtime_input(media=msg)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Send error: {e}")

    async def _send_text_async(self, text: str):
        """Send a text prompt to the live session."""
        try:
            await self._session.send_client_content(
                turns=types.Content(parts=[types.Part(text=text)]),
                turn_complete=True,
            )
        except Exception as e:
            logger.error(f"Send text error: {e}")

    # ═══════════════════════════════════════════
    # Audio Output: Gemini → Speaker
    # ═══════════════════════════════════════════

    async def _receive_audio(self):
        """Read response audio chunks from Gemini and queue for playback."""
        try:
            while self._running:
                turn = self._session.receive()
                first_chunk = True
                async for response in turn:
                    if first_chunk:
                        first_chunk = False
                        if self._on_turn_start:
                            self._on_turn_start()

                    if data := response.data:
                        self._audio_in_queue.put_nowait(data)
                        continue
                    if text := response.text:
                        logger.info(f"📝 Gemini text: {text}")
                        if self._on_text:
                            self._on_text(text)

                # Turn complete — flush the queue to stop old audio
                if self._on_turn_end:
                    self._on_turn_end()
                while not self._audio_in_queue.empty():
                    self._audio_in_queue.get_nowait()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Receive error: {e}")

    async def _play_audio(self):
        """
        Play received audio chunks through speakers.
        Uses subprocess pw-play for reliability on RPi + PipeWire.
        Collects chunks into a buffer and plays periodically.
        """
        BUFFER_TIMEOUT = 0.3  # Wait up to 300ms between chunks before flushing
        buffer = bytearray()

        try:
            while self._running:
                try:
                    chunk = await asyncio.wait_for(
                        self._audio_in_queue.get(), timeout=BUFFER_TIMEOUT
                    )
                    buffer.extend(chunk)
                except asyncio.TimeoutError:
                    # No more chunks — flush buffer
                    if buffer:
                        await self._flush_audio(bytes(buffer))
                        buffer.clear()

        except asyncio.CancelledError:
            if buffer:
                await self._flush_audio(bytes(buffer))
        except Exception as e:
            logger.error(f"Play error: {e}")

    async def _flush_audio(self, pcm_data: bytes):
        """Write PCM data to temp WAV and play via pw-play."""
        if not pcm_data:
            return

        # Volume boost
        try:
            samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
            samples *= 2.0  # +6dB
            samples = np.clip(samples, -32768, 32767)
            pcm_data = samples.astype(np.int16).tobytes()
        except Exception:
            pass

        tmp_wav = os.path.join(tempfile.gettempdir(), "echo_live_out.wav")
        try:
            with wave.open(tmp_wav, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(RECEIVE_SAMPLE_RATE)
                wf.writeframes(pcm_data)
        except Exception as e:
            logger.error(f"WAV write error: {e}")
            return

        duration = len(pcm_data) / (RECEIVE_SAMPLE_RATE * 2)

        # Play with pw-play (PipeWire native) or aplay fallback
        import subprocess
        for cmd in [["pw-play", tmp_wav], ["aplay", tmp_wav]]:
            try:
                result = await asyncio.to_thread(
                    subprocess.run, cmd,
                    capture_output=True, text=True,
                    timeout=max(15, duration + 5),
                )
                if result.returncode == 0:
                    break
            except FileNotFoundError:
                continue
            except Exception as e:
                logger.warning(f"{cmd[0]} error: {e}")

        try:
            os.unlink(tmp_wav)
        except Exception:
            pass

    # ═══════════════════════════════════════════
    # Cleanup
    # ═══════════════════════════════════════════

    def cleanup(self):
        """Stop and release all resources."""
        self.stop()
        logger.info("GeminiLiveEngine cleaned up")


# ═══════════════════════════════════════════════════
# Standalone Test
# ═══════════════════════════════════════════════════

def main():
    """Run the Live API engine standalone for testing."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 50)
    print("  🎙️ ECHO — Gemini Live Voice Test")
    print("  Speak into your microphone.")
    print("  Gemini will respond with audio.")
    print("  Press Ctrl+C to stop.")
    print("=" * 50)

    def on_start():
        print("\n🤖 [Gemini is speaking...]")

    def on_end():
        print("🤖 [Turn complete]\n")
        print("🎤 Listening...")

    def on_text(text):
        print(f"📝 {text}")

    engine = GeminiLiveEngine(
        on_turn_start=on_start,
        on_turn_end=on_end,
        on_text=on_text,
    )

    try:
        engine.start()
        print("🎤 Listening...")
        # Keep main thread alive
        while engine.is_running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\n👋 Stopping...")
    finally:
        engine.cleanup()


if __name__ == "__main__":
    main()
