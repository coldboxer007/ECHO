"""
ECHO Robot — Gemini Live API Voice Engine (Full Integration)
==============================================================
Bidirectional audio stream through the Gemini Live API with full
robot control via function calling.

How it works:
  1. Mic audio (PCM 16kHz) is streamed to Gemini in real-time
  2. Gemini processes speech, understands it, and responds with audio
  3. Response audio (PCM 24kHz) is played through pw-play / aplay
  4. Camera frames sent periodically for visual context
  5. Function calling lets Gemini control motors, navigation, face, sensors
  6. Reconnection with exponential backoff for reliability

Requires:  pip install google-genai pyaudio
"""

import os
import sys
import json
import wave
import time
import asyncio
import logging
import tempfile
import threading
import traceback
import subprocess
import ctypes
import queue as thread_queue  # Thread-safe queue for cross-thread communication
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

# ─────────────────────────────────────────────
# Function Calling Tool Declarations
# ─────────────────────────────────────────────
# These are the tools Gemini can call to control the robot hardware.
# Function calling in Live API is synchronous — the model waits for
# the tool response before continuing.

ROBOT_TOOLS = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="move_robot",
            description="Move the robot in a direction. Use for forward, backward, left, right movements.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "direction": types.Schema(
                        type="STRING",
                        enum=["forward", "backward", "left", "right"],
                        description="Direction to move",
                    ),
                    "duration": types.Schema(
                        type="NUMBER",
                        description="Duration in seconds (0.5-10). Default 1.0 if not specified.",
                    ),
                },
                required=["direction"],
            ),
        ),
        types.FunctionDeclaration(
            name="stop_robot",
            description="Stop all robot movement immediately. Use when asked to stop, halt, or freeze.",
            parameters=types.Schema(type="OBJECT", properties={}),
        ),
        types.FunctionDeclaration(
            name="start_follow_mode",
            description="Start following the person using camera tracking. The robot will follow the detected person.",
            parameters=types.Schema(type="OBJECT", properties={}),
        ),
        types.FunctionDeclaration(
            name="start_patrol_mode",
            description="Start autonomous patrol mode. Robot moves back and forth until stopped.",
            parameters=types.Schema(type="OBJECT", properties={}),
        ),
        types.FunctionDeclaration(
            name="safe_move_forward",
            description="Move forward carefully with obstacle detection. Stops if an obstacle is detected.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "duration": types.Schema(
                        type="NUMBER",
                        description="Duration in seconds (1-15). Default 8.0.",
                    ),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="set_face_emotion",
            description="Change the robot's face display to show an emotion.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "emotion": types.Schema(
                        type="STRING",
                        enum=["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"],
                        description="The emotion to display on the robot's face",
                    ),
                },
                required=["emotion"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_sensor_data",
            description="Read current sensor data (distance to obstacle, IR sensor state). Use when asked about surroundings or before moving.",
            parameters=types.Schema(type="OBJECT", properties={}),
        ),
        types.FunctionDeclaration(
            name="get_camera_emotion",
            description="Get the currently detected facial emotion from the camera. Use when asked about how someone looks or feels.",
            parameters=types.Schema(type="OBJECT", properties={}),
        ),
        types.FunctionDeclaration(
            name="set_volume",
            description="Adjust the robot's speech volume.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "direction": types.Schema(
                        type="STRING",
                        enum=["up", "down"],
                        description="Whether to increase or decrease volume",
                    ),
                },
                required=["direction"],
            ),
        ),
        types.FunctionDeclaration(
            name="shutdown_robot",
            description="Initiate a graceful shutdown of the robot. Use when the user says goodbye, goodnight, or asks to shut down.",
            parameters=types.Schema(type="OBJECT", properties={}),
        ),
    ]),
]

# Extended system prompt for Live mode with function calling context
LIVE_SYSTEM_PROMPT = SYSTEM_PROMPT + """

You have direct control of your robot body through function calls. Use them naturally:
- When asked to move, call move_robot with the direction and optional duration.
- When asked to stop, call stop_robot.
- When asked to follow someone, call start_follow_mode.
- When asked to patrol or explore, call start_patrol_mode.
- Call get_sensor_data before moving if you want to check for obstacles.
- Call get_camera_emotion to check how someone is feeling.
- Call set_face_emotion to change your facial expression to match the conversation mood.
- When someone says goodbye, call shutdown_robot.
- You can see through your camera — images are sent to you periodically.
- Keep your spoken responses concise (2-3 sentences) since you're speaking in real-time.
- You receive the user's detected emotion with each camera frame — adapt your tone accordingly.
"""


class GeminiLiveEngine:
    """
    Bidirectional voice engine using Gemini Live API with full robot control.
    Handles speech I/O, function calling for hardware, and reconnection.
    """

    # Reconnection settings
    _MAX_RECONNECT_ATTEMPTS = 5
    _RECONNECT_BASE_DELAY = 2.0   # seconds, doubles each attempt

    def __init__(self, on_turn_start=None, on_turn_end=None, on_text=None,
                 on_function_call=None, on_input_transcript=None):
        """
        Callbacks:
          on_turn_start()           — called when Gemini starts responding
          on_turn_end()             — called when Gemini finishes a turn
          on_text(str)              — called with response text
          on_function_call(name, args) → result_dict  — called to execute robot functions
          on_input_transcript(str)  — called with user's transcribed speech
        """
        self._on_turn_start = on_turn_start
        self._on_turn_end = on_turn_end
        self._on_text = on_text
        self._on_function_call = on_function_call
        self._on_input_transcript = on_input_transcript

        self._running = False
        self._session = None
        self._audio_play_queue = None    # asyncio.Queue for audio playback
        self._mic_queue = None           # thread_queue.Queue for mic→async bridge
        self._image_queue = None         # thread_queue.Queue for image→async bridge
        self._audio_stream = None
        self._pya = None
        self._loop = None
        self._thread = None
        self._volume = 1.0
        self._volume_lock = threading.Lock()
        self._reconnect_count = 0

        # Build the client
        api_key = GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set! Set it in .env or environment.")

        self._client = genai.Client(
            api_key=api_key,
            http_options={"api_version": "v1beta"},
        )

        # Live session config with function calling tools
        self._config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=TTS_VOICE or "Kore"
                    )
                )
            ),
            system_instruction=LIVE_SYSTEM_PROMPT,
            tools=ROBOT_TOOLS,
            context_window_compression=types.ContextWindowCompressionConfig(
                trigger_tokens=25600,
                sliding_window=types.SlidingWindow(target_tokens=12800),
            ),
        )

        logger.info("GeminiLiveEngine initialized (with function calling)")

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
        self._mic_queue = thread_queue.Queue(maxsize=50)
        self._image_queue = thread_queue.Queue(maxsize=3)
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Gemini Live session starting...")

    def stop(self):
        """Stop the live session."""
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Gemini Live session stopped")

    def send_text(self, text: str):
        """Send a text message to the session (non-blocking, thread-safe)."""
        if self._session and self._loop:
            asyncio.run_coroutine_threadsafe(
                self._send_text_async(text), self._loop
            )

    def send_image(self, jpeg_bytes: bytes):
        """Send a camera frame to give Gemini visual context (thread-safe)."""
        try:
            # Drop oldest if full — keep real-time
            if self._image_queue and self._image_queue.full():
                try:
                    self._image_queue.get_nowait()
                except thread_queue.Empty:
                    pass
            if self._image_queue:
                self._image_queue.put_nowait(jpeg_bytes)
        except Exception:
            pass  # Non-critical — skip frame

    def set_volume(self, level: float):
        """Set volume multiplier (0.25–2.0)."""
        with self._volume_lock:
            self._volume = max(0.25, min(2.0, level))
        logger.info(f"Live volume set to {self._volume:.2f}x")

    def adjust_volume(self, delta: float) -> float:
        """Adjust volume by delta. Returns new level."""
        with self._volume_lock:
            self._volume = max(0.25, min(2.0, self._volume + delta))
            return self._volume

    @property
    def volume(self) -> float:
        with self._volume_lock:
            return self._volume

    @property
    def is_running(self):
        return self._running

    # ═══════════════════════════════════════════
    # Internal: Async Event Loop
    # ═══════════════════════════════════════════

    def _run_loop(self):
        """Run the asyncio event loop in a background thread with reconnection."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_with_reconnect())
        except Exception as e:
            logger.error(f"Live session fatal error: {e}")
            traceback.print_exc()
        finally:
            self._running = False

    async def _run_with_reconnect(self):
        """Connect to Gemini Live API with automatic reconnection on failure."""
        while self._running:
            try:
                await self._run()
                # Clean exit — don't reconnect
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._reconnect_count += 1
                if self._reconnect_count > self._MAX_RECONNECT_ATTEMPTS:
                    logger.error(f"Max reconnection attempts ({self._MAX_RECONNECT_ATTEMPTS}) exceeded. Giving up.")
                    break

                delay = self._RECONNECT_BASE_DELAY * (2 ** (self._reconnect_count - 1))
                delay = min(delay, 30.0)  # Cap at 30s
                logger.warning(
                    f"Live session disconnected: {e}. "
                    f"Reconnecting in {delay:.0f}s (attempt {self._reconnect_count}/{self._MAX_RECONNECT_ATTEMPTS})..."
                )
                await asyncio.sleep(delay)

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
                self._audio_play_queue = asyncio.Queue()
                self._reconnect_count = 0  # Reset on successful connect

                logger.info("Connected to Gemini Live API (with function calling)")

                tg.create_task(self._capture_mic())
                tg.create_task(self._send_audio())
                tg.create_task(self._send_images())
                tg.create_task(self._receive_responses())
                tg.create_task(self._play_audio())

                # Keep running until stopped
                while self._running:
                    await asyncio.sleep(0.1)

                raise asyncio.CancelledError("Shutdown requested")

        except asyncio.CancelledError:
            logger.info("Live session cancelled")
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

    async def _capture_mic(self):
        """Capture mic audio and push to thread-safe queue."""
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
            logger.info(f"Mic open: {mic_info['name']} @ {SEND_SAMPLE_RATE} Hz")

            while self._running:
                data = await asyncio.to_thread(
                    self._audio_stream.read, CHUNK_SIZE,
                    exception_on_overflow=False,
                )
                # Thread-safe put into mic queue
                try:
                    if self._mic_queue.full():
                        try:
                            self._mic_queue.get_nowait()
                        except thread_queue.Empty:
                            pass
                    self._mic_queue.put_nowait(data)
                except thread_queue.Full:
                    pass  # Drop frame — keep real-time

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
                self._audio_stream = None

    async def _send_audio(self):
        """Send mic audio from thread-safe queue to Gemini."""
        try:
            while self._running:
                try:
                    data = self._mic_queue.get(timeout=0.1)
                except thread_queue.Empty:
                    continue
                await self._session.send_realtime_input(
                    audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000")
                )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Send audio error: {e}")

    async def _send_images(self):
        """Send camera frames from thread-safe queue to Gemini."""
        try:
            while self._running:
                try:
                    jpeg_bytes = self._image_queue.get(timeout=0.5)
                except thread_queue.Empty:
                    continue
                await self._session.send_realtime_input(
                    video=types.Blob(data=jpeg_bytes, mime_type="image/jpeg")
                )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Send image error: {e}")

    async def _send_text_async(self, text: str):
        """Send a text prompt to the live session."""
        try:
            await self._session.send_realtime_input(text=text)
        except Exception as e:
            logger.error(f"Send text error: {e}")

    # ═══════════════════════════════════════════
    # Response Processing: Gemini → Robot
    # ═══════════════════════════════════════════

    async def _receive_responses(self):
        """Read responses from Gemini — audio, text, function calls, interruptions."""
        try:
            while self._running:
                turn = self._session.receive()
                first_chunk = True
                async for response in turn:
                    server_content = response.server_content

                    if server_content:
                        # ── Audio + text from model turn ──
                        if server_content.model_turn:
                            if first_chunk:
                                first_chunk = False
                                if self._on_turn_start:
                                    self._on_turn_start()

                            for part in server_content.model_turn.parts:
                                if part.inline_data:
                                    self._audio_play_queue.put_nowait(part.inline_data.data)
                                if part.text:
                                    logger.info(f"Gemini text: {part.text}")
                                    if self._on_text:
                                        self._on_text(part.text)

                        # ── Input transcription (what the user said) ──
                        if server_content.input_transcription:
                            text = server_content.input_transcription.text
                            if text and self._on_input_transcript:
                                self._on_input_transcript(text)

                        # ── Output transcription (what Gemini said) ──
                        if server_content.output_transcription:
                            text = server_content.output_transcription.text
                            if text:
                                logger.debug(f"Output transcript: {text}")

                        # ── Interruption — clear audio queue ──
                        if server_content.interrupted:
                            logger.info("User interrupted — clearing audio queue")
                            while not self._audio_play_queue.empty():
                                try:
                                    self._audio_play_queue.get_nowait()
                                except asyncio.QueueEmpty:
                                    break

                        # ── Turn complete ──
                        if server_content.turn_complete:
                            if self._on_turn_end:
                                self._on_turn_end()
                            first_chunk = True

                    # ── Function calls from Gemini ──
                    if response.tool_call:
                        await self._handle_tool_call(response.tool_call)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Receive error: {e}")
            traceback.print_exc()

    async def _handle_tool_call(self, tool_call):
        """Execute function calls from Gemini and send results back."""
        function_responses = []

        for fc in tool_call.function_calls:
            name = fc.name
            args = dict(fc.args) if fc.args else {}
            logger.info(f"Function call: {name}({args})")

            # Execute via callback (runs on the event loop thread)
            result = {"status": "error", "message": "No handler registered"}
            if self._on_function_call:
                try:
                    result = await asyncio.to_thread(
                        self._on_function_call, name, args
                    )
                except Exception as e:
                    logger.error(f"Function call '{name}' failed: {e}")
                    result = {"status": "error", "message": str(e)}

            function_responses.append(
                types.FunctionResponse(
                    name=name,
                    id=fc.id,
                    response=result,
                )
            )

        # Send all function responses back to Gemini
        if function_responses:
            try:
                await self._session.send_tool_response(
                    function_responses=function_responses
                )
            except Exception as e:
                logger.error(f"Error sending tool response: {e}")

    # ═══════════════════════════════════════════
    # Audio Output: Gemini → Speaker
    # ═══════════════════════════════════════════

    async def _play_audio(self):
        """
        Play received audio chunks through speakers.
        Collects chunks into a buffer and plays periodically to avoid
        per-chunk subprocess overhead.
        """
        BUFFER_TIMEOUT = 0.3  # Wait up to 300ms between chunks before flushing
        buffer = bytearray()

        try:
            while self._running:
                try:
                    chunk = await asyncio.wait_for(
                        self._audio_play_queue.get(), timeout=BUFFER_TIMEOUT
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

        # Volume boost — base 2x gain + user volume multiplier
        try:
            with self._volume_lock:
                total_gain = 2.0 * self._volume
            samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
            np.multiply(samples, total_gain, out=samples)
            np.clip(samples, -32768, 32767, out=samples)
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
    print("  ECHO — Gemini Live Voice Test")
    print("  Speak into your microphone.")
    print("  Gemini will respond with audio.")
    print("  Press Ctrl+C to stop.")
    print("=" * 50)

    def on_start():
        print("\n[Gemini is speaking...]")

    def on_end():
        print("[Turn complete]\n")
        print("Listening...")

    def on_text(text):
        print(f"Gemini: {text}")

    def on_input(text):
        print(f"You: {text}")

    def on_func(name, args):
        print(f"Function call: {name}({args})")
        return {"status": "ok", "message": f"Simulated {name}"}

    engine = GeminiLiveEngine(
        on_turn_start=on_start,
        on_turn_end=on_end,
        on_text=on_text,
        on_function_call=on_func,
        on_input_transcript=on_input,
    )

    try:
        engine.start()
        print("Listening...")
        while engine.is_running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        engine.cleanup()


if __name__ == "__main__":
    main()
