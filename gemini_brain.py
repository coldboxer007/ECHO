"""
ECHO Robot — Gemini AI Brain
==============================
The conversational AI core. Takes user text + detected emotion,
sends to Gemini 2.5 Flash, and returns an emotionally-aware response.

Also provides vision capability via Gemini Robotics-ER for person detection
and scene understanding when needed.
"""

import re
import time
import logging
import threading
from typing import Generator

logger = logging.getLogger("echo.brain")

# Pre-compile regex used on every command parse
_DURATION_RE = re.compile(r'for\s+(\d+)\s*(?:second|sec|s\b)')
# Sentence boundary for streaming: split on . ! ? followed by space/end, OR on newlines
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+|(?<=\n)')

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("google-genai not available — Gemini Brain disabled")

from config import (
    GEMINI_API_KEY, GEMINI_CHAT_MODEL, GEMINI_ROBOTICS_MODEL,
    SYSTEM_PROMPT,
)


class GeminiBrain:
    """Gemini-powered conversational AI with emotion awareness."""

    # Maximum chat history entries to keep in memory (pairs of user+model)
    # 20 entries = 10 exchanges — enough context for conversation continuity
    # while keeping API request size manageable on RPi (fewer tokens = faster response)
    MAX_HISTORY_ENTRIES = 20  # 10 exchanges

    # NLP command classification cooldown: avoid calling Gemini NLP for every
    # utterance classified as 'chat'. Cache the last result briefly.
    _NLP_COOLDOWN_SECS = 2.0

    def __init__(self):
        self._client = None
        self._chat_history = []
        self._lock = threading.Lock()

        # NLP classification cache
        self._last_nlp_text = ""
        self._last_nlp_result = None
        self._last_nlp_time = 0.0

        self._init_client()
        logger.info("GeminiBrain initialized")

    def _init_client(self):
        """Initialize Gemini API client."""
        if not GENAI_AVAILABLE or not GEMINI_API_KEY:
            logger.error("Gemini API not configured — set GEMINI_API_KEY in .env")
            return

        try:
            self._client = genai.Client(api_key=GEMINI_API_KEY)
            logger.info(f"Gemini client initialized (model={GEMINI_CHAT_MODEL})")
        except Exception as e:
            logger.error(f"Failed to init Gemini client: {e}")

    def think(self, user_text: str, emotion: str = "neutral", confidence: float = 0.0) -> str:
        """
        Generate a response to user input, informed by detected emotion.

        Args:
            user_text: What the user said (transcribed speech)
            emotion: Detected facial emotion from camera
            confidence: Confidence of the emotion detection

        Returns:
            Response text from Gemini
        """
        if self._client is None:
            return "I'm having trouble connecting to my brain right now. Let me try again."

        # Build the message with emotion context
        emotion_tag = ""
        if emotion != "neutral" and confidence > 0.3:
            emotion_tag = f"[EMOTION DETECTED: {emotion} (confidence: {confidence:.0%})]\n"

        full_message = f"{emotion_tag}User says: {user_text}"

        logger.info(f"🧠 Thinking... input='{user_text}' emotion={emotion}")

        try:
            # Build conversation with history (system prompt sent via config)
            messages = []

            # Add conversation history (keep last 10 exchanges = 20 entries)
            with self._lock:
                for entry in self._chat_history[-self.MAX_HISTORY_ENTRIES:]:
                    messages.append(entry)

            # Add current user message
            user_content = types.Content(
                role="user",
                parts=[types.Part.from_text(text=full_message)]
            )
            messages.append(user_content)

            # Call Gemini with system_instruction (proper API usage, saves tokens)
            response = self._client.models.generate_content(
                model=GEMINI_CHAT_MODEL,
                contents=messages,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.8,
                    max_output_tokens=350,  # Rich conversational responses
                    top_p=0.9,
                ),
            )

            reply = response.text.strip()
            logger.info(f"💬 Gemini says: '{reply[:80]}...'")

            # Update history
            model_content = types.Content(
                role="model",
                parts=[types.Part.from_text(text=reply)]
            )

            with self._lock:
                self._chat_history.append(user_content)
                self._chat_history.append(model_content)
                # Trim to prevent unbounded memory growth
                if len(self._chat_history) > self.MAX_HISTORY_ENTRIES:
                    self._chat_history = self._chat_history[-self.MAX_HISTORY_ENTRIES:]

            return reply

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return "Sorry, I had a little hiccup thinking about that. Can you say that again?"

    def think_stream(self, user_text: str, emotion: str = "neutral", confidence: float = 0.0) -> Generator[str, None, None]:
        """
        Streaming version of think() — yields sentences as they become available.
        This allows TTS to start speaking the first sentence while the rest
        is still being generated, saving 1-3 seconds of perceived latency.

        Yields:
            Individual sentences as they are completed.
            The full response is also saved to chat history after all chunks.
        """
        if self._client is None:
            yield "I'm having trouble connecting to my brain right now. Let me try again."
            return

        # Build the message with emotion context
        emotion_tag = ""
        if emotion != "neutral" and confidence > 0.3:
            emotion_tag = f"[EMOTION DETECTED: {emotion} (confidence: {confidence:.0%})]\n"

        full_message = f"{emotion_tag}User says: {user_text}"

        logger.info(f"🧠 Thinking (stream)... input='{user_text}' emotion={emotion}")

        try:
            # Build conversation with history
            messages = []
            with self._lock:
                for entry in self._chat_history[-self.MAX_HISTORY_ENTRIES:]:
                    messages.append(entry)

            user_content = types.Content(
                role="user",
                parts=[types.Part.from_text(text=full_message)]
            )
            messages.append(user_content)

            # Stream the response from Gemini
            response_stream = self._client.models.generate_content_stream(
                model=GEMINI_CHAT_MODEL,
                contents=messages,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.8,
                    max_output_tokens=350,
                    top_p=0.9,
                ),
            )

            # Accumulate text and yield complete sentences
            buffer = ""
            full_reply = ""

            for chunk in response_stream:
                if chunk.text:
                    buffer += chunk.text

                    # Split buffer on sentence boundaries
                    sentences = _SENTENCE_RE.split(buffer)
                    # Keep the last fragment (may be incomplete sentence)
                    if len(sentences) > 1:
                        # Yield all complete sentences
                        for sentence in sentences[:-1]:
                            sentence = sentence.strip()
                            if sentence:
                                full_reply += sentence + " "
                                yield sentence
                        buffer = sentences[-1]

            # Yield any remaining text in buffer
            remaining = buffer.strip()
            if remaining:
                full_reply += remaining
                yield remaining

            full_reply = full_reply.strip()
            logger.info(f"💬 Gemini says (streamed): '{full_reply[:80]}...'")

            # Update history with the complete response
            model_content = types.Content(
                role="model",
                parts=[types.Part.from_text(text=full_reply)]
            )
            with self._lock:
                self._chat_history.append(user_content)
                self._chat_history.append(model_content)
                if len(self._chat_history) > self.MAX_HISTORY_ENTRIES:
                    self._chat_history = self._chat_history[-self.MAX_HISTORY_ENTRIES:]

        except Exception as e:
            logger.error(f"Gemini streaming API error: {e}")
            yield "Sorry, I had a little hiccup thinking about that. Can you say that again?"

    def interpret_command(self, text: str) -> dict:
        """
        Parse user text to determine if it's a movement command or conversation.

        Returns dict with:
            - 'type': 'move' | 'keep_moving' | 'safe_move' | 'patrol' |
                      'follow' | 'stop' | 'goodbye' | 'clear_history' |
                      'look' | 'volume' | 'chat'
            - 'direction': 'forward' | 'backward' | 'left' | 'right' (for move)
            - 'duration': optional float seconds
            - 'volume_direction': 'up' | 'down' (for volume)
            - 'text': original text
        """
        text_lower = text.lower().strip()
        words = text_lower.split()

        # ── PRIORITY 1: Complex / continuous movement commands ──
        # Check these FIRST — they contain words like "stop when" and "obstacle"
        # that would otherwise false-match the stop command.

        # "keep moving", "keep going", "go until I say stop", "don't stop"
        keep_kw = ['keep moving', 'keep going', 'until i say stop', 'don\'t stop',
                   'keep driving', 'continue moving', 'continue going',
                   'keep walking', 'go go go']
        if any(kw in text_lower for kw in keep_kw):
            direction = 'forward'
            if any(w in text_lower for w in ['back', 'backward', 'backwards', 'reverse']):
                direction = 'backward'
            return {'type': 'keep_moving', 'direction': direction, 'text': text}

        # "move carefully", "stop when there's an obstacle", "watch for obstacles"
        safe_kw = ['carefully', 'watch for obstacle', 'stop when', 'obstacle',
                   'slowly', 'be careful', 'watch out', 'stay your obstacle']
        if any(kw in text_lower for kw in safe_kw) and \
           any(w in text_lower for w in ['move', 'go', 'forward', 'drive', 'walk', 'until', 'stay']):
            direction = 'forward'
            if any(w in text_lower for w in ['back', 'backward', 'reverse']):
                direction = 'backward'
            return {'type': 'safe_move', 'direction': direction, 'text': text}

        # "patrol", "go back and forth", "pace around"
        patrol_kw = ['patrol', 'back and forth', 'pace', 'wander', 'explore', 'roam']
        if any(kw in text_lower for kw in patrol_kw):
            return {'type': 'patrol', 'text': text}

        # ── PRIORITY 2a: Goodbye / shutdown commands ──
        # Graceful shutdown — must come before 'stop' so "shut down" and
        # "goodbye" trigger a full shutdown, not just a motor stop.
        goodbye_phrases = ['goodbye', 'good bye', 'bye bye', 'bye echo',
                           'goodnight', 'good night', 'shut down', 'power off',
                           'turn off', 'go to sleep', 'see you later',
                           'see you soon', 'i\'m leaving', 'time to sleep']
        if any(kw in text_lower for kw in goodbye_phrases):
            return {'type': 'goodbye', 'text': text}

        # ── PRIORITY 2b: Explicit stop commands ──
        # Only match unambiguous stop phrases — NOT "stay", "hold", "wait" which
        # appear in movement sentences like "stop when there's an obstacle"
        stop_phrases = ['stop', 'halt', 'freeze', 'stand still', 'don\'t move',
                        'stop moving', 'stop following', 'cancel']
        # Must be an explicit stop — not part of a movement sentence
        if any(kw in text_lower for kw in stop_phrases):
            # But don't trigger stop if there's ALSO a movement intent
            has_move_word = any(w in text_lower for w in
                               ['forward', 'go', 'move forward', 'backward', 'left', 'right',
                                'keep', 'carefully', 'obstacle', 'patrol', 'until'])
            if not has_move_word:
                return {'type': 'stop', 'text': text}

        # ── PRIORITY 3: Duration extraction: "go forward for 3 seconds" ──
        duration = None
        dur_match = _DURATION_RE.search(text_lower)
        if dur_match:
            duration = float(dur_match.group(1))

        # ── PRIORITY 4: Standard movement commands ──
        move_keywords = {
            'forward':  ['move forward', 'go forward', 'go ahead', 'drive forward', 'move ahead',
                         'forward', 'straight', 'go straight', 'advance'],
            'backward': ['move backward', 'go backward', 'go back', 'reverse', 'move back',
                         'back up', 'backward', 'backwards', 'back'],
            'left':     ['turn left', 'go left', 'rotate left', 'spin left', 'left'],
            'right':    ['turn right', 'go right', 'rotate right', 'spin right', 'right'],
        }

        for direction, keywords in move_keywords.items():
            for kw in keywords:
                if kw in text_lower:
                    cmd = {'type': 'move', 'direction': direction, 'text': text}
                    if duration:
                        cmd['duration'] = duration
                    return cmd

        # "spin", "turn around", "do a circle"
        if any(kw in text_lower for kw in ['spin', 'turn around', 'circle', '360', 'rotate']):
            return {'type': 'move', 'direction': 'right', 'duration': 2.0, 'text': text}

        # Follow commands — require explicit "follow me" or "come with/here"
        # "follow" alone is too broad (matches "can you follow instructions?")
        follow_phrases = ['follow me', 'come with me', 'come here', 'come to me']
        if any(kw in text_lower for kw in follow_phrases):
            return {'type': 'follow', 'text': text}

        # Clear history / forget command
        if any(kw in text_lower for kw in ['clear history', 'forget everything',
                                            'start over', 'reset conversation',
                                            'new conversation', 'forget what']):
            return {'type': 'clear_history', 'text': text}

        # "What do you see?" / "look around" / "describe" — visual query
        look_phrases = ['what do you see', 'look around', 'describe what',
                        'what\'s around', 'what is around', 'what\'s in front',
                        'who do you see', 'can you see', 'what are you looking at']
        if any(kw in text_lower for kw in look_phrases):
            return {'type': 'look', 'text': text}

        # Volume control — louder / quieter / volume up / volume down
        if any(kw in text_lower for kw in ['louder', 'volume up', 'speak up',
                                            'turn up', 'raise volume', 'too quiet']):
            return {'type': 'volume', 'direction': 'up', 'text': text}
        if any(kw in text_lower for kw in ['quieter', 'volume down', 'lower volume',
                                            'turn down', 'too loud', 'not so loud',
                                            'speak softer', 'softer']):
            return {'type': 'volume', 'direction': 'down', 'text': text}

        # Everything else is conversation
        return {'type': 'chat', 'text': text}

    def interpret_command_nlp(self, text: str) -> dict:
        """
        Use Gemini to interpret ambiguous natural language as a robot command.
        Called as a fallback when local keyword matching returns 'chat' but the
        text might be a movement/action command phrased in natural language
        (e.g., "come ahead", "move closer", "go that way").

        Includes a short cooldown cache to avoid redundant API calls for
        repeated/similar utterances in quick succession.

        Returns:
            Same dict format as interpret_command(). Returns {'type': 'chat'}
            if Gemini determines it's just conversation.
        """
        if self._client is None:
            return {'type': 'chat', 'text': text}

        # ── Cache check: return cached result if same text within cooldown ──
        now = time.monotonic()
        if (text == self._last_nlp_text
                and self._last_nlp_result is not None
                and (now - self._last_nlp_time) < self._NLP_COOLDOWN_SECS):
            logger.debug(f"NLP cache hit for: '{text}'")
            return self._last_nlp_result

        try:
            import json as _json
            prompt = (
                "You are a robot command classifier. Given the user's speech, determine if it's "
                "a physical movement command or just conversation.\n\n"
                "Valid command types:\n"
                '- {"type":"move","direction":"forward"} — move forward (includes: come ahead, move closer, approach, come to me, walk forward, go)\n'
                '- {"type":"move","direction":"backward"} — move backward (includes: back away, retreat, move away)\n'
                '- {"type":"move","direction":"left"} — turn left\n'
                '- {"type":"move","direction":"right"} — turn right\n'
                '- {"type":"stop"} — stop moving (includes: wait, hold on, stay)\n'
                '- {"type":"goodbye"} — user is saying goodbye/leaving (includes: goodbye, bye, see you later, goodnight, shut down, power off, go to sleep)\n'
                '- {"type":"follow"} — follow the person\n'
                '- {"type":"patrol"} — patrol/explore\n'
                '- {"type":"chat"} — not a movement command, just conversation\n\n'
                f'User said: "{text}"\n\n'
                "Respond with ONLY a JSON object. No markdown, no explanation."
            )

            response = self._client.models.generate_content(
                model=GEMINI_CHAT_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=64,
                ),
            )

            result_text = response.text.strip()
            # Clean markdown code fences if present
            if result_text.startswith("```"):
                result_text = result_text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            data = _json.loads(result_text)
            cmd_type = data.get("type", "chat")

            # Validate and build result
            result = {'type': 'chat', 'text': text}

            if cmd_type == "move":
                direction = data.get("direction", "forward")
                if direction in ("forward", "backward", "left", "right"):
                    logger.info(f"NLP classified as move:{direction}")
                    result = {'type': 'move', 'direction': direction, 'text': text}
            elif cmd_type in ("stop", "follow", "patrol", "goodbye"):
                logger.info(f"NLP classified as {cmd_type}")
                result = {'type': cmd_type, 'text': text}

            # Cache the result
            self._last_nlp_text = text
            self._last_nlp_result = result
            self._last_nlp_time = time.monotonic()
            return result

        except Exception as e:
            logger.warning(f"NLP command interpretation failed: {e}")
            return {'type': 'chat', 'text': text}

    def analyze_scene(self, image_bytes: bytes) -> str:
        """
        Use Gemini Robotics-ER to analyze a camera frame.
        Returns a text description of the scene.
        """
        if self._client is None:
            return "Cannot analyze scene — no Gemini client"

        try:
            response = self._client.models.generate_content(
                model=GEMINI_ROBOTICS_MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
                    "Briefly describe what you see. If there are people, describe their posture and approximate position."
                ],
                config=types.GenerateContentConfig(
                    temperature=0.5,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Scene analysis error: {e}")
            return ""

    def detect_person_position(self, image_bytes: bytes) -> dict:
        """
        Use Gemini Robotics-ER to find a person's position in the frame.
        Returns dict with 'found': bool, 'point': [y, x] normalized to 0-1000.
        """
        if self._client is None:
            return {'found': False}

        try:
            response = self._client.models.generate_content(
                model=GEMINI_ROBOTICS_MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
                    'Point to the person closest to the camera. '
                    'The answer should follow the json format: [{"point": [y, x], "label": "person"}]. '
                    'The points are in [y, x] format normalized to 0-1000. '
                    'If no person is found, return an empty list [].'
                ],
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )

            import json
            text = response.text.strip()
            # Clean markdown code fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            data = json.loads(text)

            if data and len(data) > 0:
                point = data[0].get('point', [500, 500])
                return {'found': True, 'point': point}
            return {'found': False}

        except Exception as e:
            logger.error(f"Person detection error: {e}")
            return {'found': False}

    def analyze_emotion_from_image(self, image_bytes: bytes) -> tuple:
        """
        Fallback sentiment analysis using Gemini API.
        Analyzes a camera frame to detect the primary facial emotion.

        Args:
            image_bytes: JPEG-encoded image bytes

        Returns:
            (emotion: str, confidence: float)
        """
        if self._client is None:
            return "neutral", 0.0

        try:
            response = self._client.models.generate_content(
                model=GEMINI_CHAT_MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
                    (
                        "Analyze the person's facial expression in this image. "
                        "Respond with ONLY a single JSON object: "
                        '{"emotion": "<one of: happy, sad, angry, surprise, fear, disgust, neutral>", '
                        '"confidence": <0.0-1.0>}. '
                        'If no person is visible, return: {"emotion": "neutral", "confidence": 0.0}'
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=64,
                ),
            )

            import json
            text = response.text.strip()
            # Clean markdown code fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            data = json.loads(text)

            emotion = data.get("emotion", "neutral").lower()
            confidence = float(data.get("confidence", 0.0))

            valid_emotions = ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]
            if emotion not in valid_emotions:
                emotion = "neutral"
                confidence = 0.0

            logger.info(f"Gemini emotion fallback: {emotion} ({confidence:.0%})")
            return emotion, confidence

        except Exception as e:
            logger.error(f"Gemini emotion analysis error: {e}")
            return "neutral", 0.0

    def determine_response_emotion(self, response_text: str, user_emotion: str) -> str:
        """
        Pick an appropriate display emotion for the robot's spoken response.
        Analyzes keywords in the response to choose the best facial expression.
        Prioritizes matching the response content; mirrors user emotion for empathy;
        defaults to neutral so the face doesn't appear stuck in one expression.
        """
        text = response_text.lower()

        if any(w in text for w in ['sorry', 'sad', 'unfortunate', 'tough', 'difficult', 'condolence']):
            return 'sad'
        if any(w in text for w in ['happy', 'glad', 'great', 'wonderful', 'love', 'fantastic', 'awesome', 'exciting']):
            return 'happy'
        if any(w in text for w in ['wow', 'amazing', 'incredible', 'surprising', 'no way']):
            return 'surprise'
        if any(w in text for w in ['careful', 'danger', 'worried', 'scary', 'afraid']):
            return 'fear'

        # Mirror user's emotion for empathy; default to neutral (not happy)
        # so the face naturally reflects context rather than always smiling.
        return user_emotion

    def clear_history(self):
        """Clear conversation history."""
        with self._lock:
            self._chat_history.clear()
        logger.info("Conversation history cleared")

    def cleanup(self):
        """Release resources."""
        self.clear_history()
        logger.info("GeminiBrain cleaned up")
