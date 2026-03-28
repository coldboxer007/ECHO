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
                    parts=[types.Part.from_text(SYSTEM_PROMPT)]
                ),
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(
                        "Understood! I'm ECHO, ready to be a warm and empathetic companion. "
                        "I'll keep my responses concise and emotionally aware."
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
                parts=[types.Part.from_text(full_message)]
            )
            messages.append(user_content)

            # Call Gemini
            response = self._client.models.generate_content(
                model=GEMINI_CHAT_MODEL,
                contents=messages,
                config=types.GenerateContentConfig(
                    temperature=0.8,
                    max_output_tokens=256,
                    top_p=0.9,
                ),
            )

            reply = response.text.strip()
            logger.info(f"💬 Gemini says: '{reply[:80]}...'")

            # Update history
            model_content = types.Content(
                role="model",
                parts=[types.Part.from_text(reply)]
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
            - 'type': 'move' | 'follow' | 'stop' | 'chat'
            - 'direction': 'forward' | 'backward' | 'left' | 'right' (for move)
            - 'text': original text
        """
        text_lower = text.lower().strip()

        # Movement commands
        move_keywords = {
            'forward':  ['move forward', 'go forward', 'go ahead', 'drive forward', 'move ahead'],
            'backward': ['move backward', 'go backward', 'go back', 'reverse', 'move back', 'back up'],
            'left':     ['turn left', 'go left', 'rotate left'],
            'right':    ['turn right', 'go right', 'rotate right'],
        }

        for direction, keywords in move_keywords.items():
            for kw in keywords:
                if kw in text_lower:
                    return {'type': 'move', 'direction': direction, 'text': text}

        # Follow commands
        if any(kw in text_lower for kw in ['follow me', 'follow', 'come with me', 'come here']):
            return {'type': 'follow', 'text': text}

        # Stop commands
        if any(kw in text_lower for kw in ['stop', 'halt', 'freeze', 'stay', 'stop following']):
            return {'type': 'stop', 'text': text}

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

    def clear_history(self):
        """Clear conversation history."""
        with self._lock:
            self._chat_history.clear()
        logger.info("Conversation history cleared")

    def cleanup(self):
        """Release resources."""
        self.clear_history()
        logger.info("GeminiBrain cleaned up")
