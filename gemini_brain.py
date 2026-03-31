"""
ECHO Robot — Gemini AI Brain
==============================
The conversational AI core. Takes user text + detected emotion,
sends to Gemini 2.5 Flash, and returns an emotionally-aware response.

Also provides vision capability via Gemini Robotics-ER for person detection
and scene understanding when needed.
"""

import logging
import threading

logger = logging.getLogger("echo.brain")

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

    def __init__(self):
        self._client = None
        self._chat_history = []
        self._lock = threading.Lock()

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
            # Build conversation with system prompt + history
            messages = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=SYSTEM_PROMPT)]
                ),
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(
                        text=(
                            "Understood! I'm ECHO, ready to be a warm and empathetic companion. "
                            "I'll keep my responses concise and emotionally aware."
                        )
                    )]
                ),
            ]

            # Add conversation history (keep last 10 exchanges)
            with self._lock:
                for entry in self._chat_history[-20:]:
                    messages.append(entry)

            # Add current user message
            user_content = types.Content(
                role="user",
                parts=[types.Part.from_text(text=full_message)]
            )
            messages.append(user_content)

            # Call Gemini
            response = self._client.models.generate_content(
                model=GEMINI_CHAT_MODEL,
                contents=messages,
                config=types.GenerateContentConfig(
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

            return reply

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return "Sorry, I had a little hiccup thinking about that. Can you say that again?"

    def interpret_command(self, text: str) -> dict:
        """
        Parse user text to determine if it's a movement command or conversation.

        Returns dict with:
            - 'type': 'move' | 'keep_moving' | 'safe_move' | 'patrol' | 'follow' | 'stop' | 'chat'
            - 'direction': 'forward' | 'backward' | 'left' | 'right' (for move)
            - 'duration': optional float seconds
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

        # ── PRIORITY 2: Explicit stop commands ──
        # Only match unambiguous stop phrases — NOT "stay", "hold", "wait" which
        # appear in movement sentences like "stop when there's an obstacle"
        stop_phrases = ['stop', 'halt', 'freeze', 'stand still', 'don\'t move',
                        'stop moving', 'stop following', 'shut down', 'cancel']
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
        import re
        dur_match = re.search(r'for\s+(\d+)\s*(?:second|sec|s\b)', text_lower)
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

        # Follow commands
        if any(kw in text_lower for kw in ['follow me', 'follow', 'come with me', 'come here']):
            return {'type': 'follow', 'text': text}

        # Everything else is conversation
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

        # Mirror user's emotion for empathy; default to friendly happy for neutral
        return user_emotion if user_emotion != 'neutral' else 'happy'

    def clear_history(self):
        """Clear conversation history."""
        with self._lock:
            self._chat_history.clear()
        logger.info("Conversation history cleared")

    def cleanup(self):
        """Release resources."""
        self.clear_history()
        logger.info("GeminiBrain cleaned up")
