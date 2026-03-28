"""
ECHO Robot — Face Display
===========================
Renders an animated face on the small display using Pygame.
The face changes expression based on the current detected/response emotion.

Expressions:
  - happy:    Upturned eyes, wide smile
  - sad:      Droopy eyes, downturned mouth
  - angry:    Angled eyebrows, frown
  - surprise: Wide round eyes, O mouth
  - fear:     Wide eyes, wavy mouth
  - disgust:  Squinted eyes, tongue out
  - neutral:  Default relaxed face
"""

import math
import time
import logging
import threading

logger = logging.getLogger("echo.face")

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logger.warning("Pygame not available — face display disabled")

from config import (
    DISPLAY_WIDTH, DISPLAY_HEIGHT, DISPLAY_FPS,
    DISPLAY_FULLSCREEN, EMOTION_COLORS,
)


class FaceDisplay:
    """Animated face display that reflects detected emotions."""

    def __init__(self):
        self._screen = None
        self._clock = None
        self._running = False
        self._emotion = "neutral"
        self._target_emotion = "neutral"
        self._blink_timer = 0.0
        self._is_blinking = False
        self._talk_phase = 0.0
        self._is_talking = False
        self._breath_phase = 0.0
        self._lock = threading.Lock()

        self._init_display()
        logger.info("FaceDisplay initialized")

    def _init_display(self):
        """Initialize Pygame display."""
        if not PYGAME_AVAILABLE:
            return

        try:
            pygame.init()

            flags = 0
            if DISPLAY_FULLSCREEN:
                flags = pygame.FULLSCREEN

            self._screen = pygame.display.set_mode(
                (DISPLAY_WIDTH, DISPLAY_HEIGHT), flags
            )
            pygame.display.set_caption("ECHO Face")
            self._clock = pygame.time.Clock()
            pygame.mouse.set_visible(False)
            logger.info(f"Display initialized: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")

        except Exception as e:
            logger.error(f"Failed to init display: {e}")
            self._screen = None

    def set_emotion(self, emotion: str):
        """Set the target emotion for the face to transition to."""
        with self._lock:
            self._target_emotion = emotion

    def set_talking(self, talking: bool):
        """Set whether the mouth should animate as talking."""
        with self._lock:
            self._is_talking = talking

    # ═══════════════════════════════════════════
    # Drawing
    # ═══════════════════════════════════════════

    def _draw_face(self):
        """Render the current face expression."""
        if self._screen is None:
            return

        with self._lock:
            emotion = self._target_emotion
            is_talking = self._is_talking

        # Colors
        bg_color = (20, 20, 30)  # Dark background
        face_color = EMOTION_COLORS.get(emotion, (200, 200, 200))
        eye_color = (255, 255, 255)
        pupil_color = (30, 30, 40)
        mouth_color = (30, 30, 40)

        cx = DISPLAY_WIDTH // 2   # Center X
        cy = DISPLAY_HEIGHT // 2  # Center Y

        # Breathing animation (subtle scale)
        self._breath_phase += 0.02
        breath = math.sin(self._breath_phase) * 2

        # Blink timer
        self._blink_timer += 1 / DISPLAY_FPS
        if self._blink_timer > 3.5:  # Blink every ~3.5 seconds
            self._is_blinking = True
            if self._blink_timer > 3.7:
                self._is_blinking = False
                self._blink_timer = 0

        # Talk animation
        if is_talking:
            self._talk_phase += 0.3
        talk_offset = math.sin(self._talk_phase) * 5 if is_talking else 0

        # Clear screen
        self._screen.fill(bg_color)

        # ── Face background (rounded rect / circle) ──
        face_rect = pygame.Rect(
            cx - 90, cy - 80 + breath, 180, 160
        )
        pygame.draw.ellipse(self._screen, face_color, face_rect)

        # ── Eyes ──
        eye_y = cy - 25 + breath
        left_eye_x = cx - 35
        right_eye_x = cx + 35
        eye_w, eye_h = 28, 28

        if self._is_blinking:
            # Closed eyes — horizontal lines
            pygame.draw.line(self._screen, pupil_color,
                             (left_eye_x - 14, eye_y), (left_eye_x + 14, eye_y), 3)
            pygame.draw.line(self._screen, pupil_color,
                             (right_eye_x - 14, eye_y), (right_eye_x + 14, eye_y), 3)
        elif emotion == "happy":
            # Happy eyes — upturned arcs (^  ^)
            pygame.draw.arc(self._screen, pupil_color,
                            (left_eye_x - 14, eye_y - 10, 28, 20),
                            0, math.pi, 4)
            pygame.draw.arc(self._screen, pupil_color,
                            (right_eye_x - 14, eye_y - 10, 28, 20),
                            0, math.pi, 4)
        elif emotion == "sad":
            # Sad eyes — droopy
            pygame.draw.ellipse(self._screen, eye_color,
                                (left_eye_x - 12, eye_y - 8, 24, 20))
            pygame.draw.ellipse(self._screen, eye_color,
                                (right_eye_x - 12, eye_y - 8, 24, 20))
            pygame.draw.circle(self._screen, pupil_color, (left_eye_x, eye_y + 2), 6)
            pygame.draw.circle(self._screen, pupil_color, (right_eye_x, eye_y + 2), 6)
            # Droopy eyebrows
            pygame.draw.line(self._screen, pupil_color,
                             (left_eye_x - 16, eye_y - 18), (left_eye_x + 12, eye_y - 14), 3)
            pygame.draw.line(self._screen, pupil_color,
                             (right_eye_x - 12, eye_y - 14), (right_eye_x + 16, eye_y - 18), 3)
        elif emotion == "angry":
            # Angry eyes — angled brows, narrow
            pygame.draw.ellipse(self._screen, eye_color,
                                (left_eye_x - 12, eye_y - 6, 24, 16))
            pygame.draw.ellipse(self._screen, eye_color,
                                (right_eye_x - 12, eye_y - 6, 24, 16))
            pygame.draw.circle(self._screen, pupil_color, (left_eye_x, eye_y), 5)
            pygame.draw.circle(self._screen, pupil_color, (right_eye_x, eye_y), 5)
            # Angry eyebrows \  /
            pygame.draw.line(self._screen, pupil_color,
                             (left_eye_x - 16, eye_y - 12), (left_eye_x + 10, eye_y - 20), 4)
            pygame.draw.line(self._screen, pupil_color,
                             (right_eye_x - 10, eye_y - 20), (right_eye_x + 16, eye_y - 12), 4)
        elif emotion == "surprise":
            # Surprise — big round eyes
            pygame.draw.circle(self._screen, eye_color, (left_eye_x, eye_y), 16)
            pygame.draw.circle(self._screen, eye_color, (right_eye_x, eye_y), 16)
            pygame.draw.circle(self._screen, pupil_color, (left_eye_x, eye_y), 7)
            pygame.draw.circle(self._screen, pupil_color, (right_eye_x, eye_y), 7)
            # Raised eyebrows
            pygame.draw.arc(self._screen, pupil_color,
                            (left_eye_x - 18, eye_y - 30, 36, 20),
                            0, math.pi, 3)
            pygame.draw.arc(self._screen, pupil_color,
                            (right_eye_x - 18, eye_y - 30, 36, 20),
                            0, math.pi, 3)
        elif emotion == "fear":
            # Fear — wide eyes with small pupils
            pygame.draw.circle(self._screen, eye_color, (left_eye_x, eye_y), 15)
            pygame.draw.circle(self._screen, eye_color, (right_eye_x, eye_y), 15)
            pygame.draw.circle(self._screen, pupil_color, (left_eye_x - 2, eye_y - 2), 4)
            pygame.draw.circle(self._screen, pupil_color, (right_eye_x + 2, eye_y - 2), 4)
        elif emotion == "disgust":
            # Disgust — squinted eyes
            pygame.draw.ellipse(self._screen, eye_color,
                                (left_eye_x - 14, eye_y - 4, 28, 10))
            pygame.draw.ellipse(self._screen, eye_color,
                                (right_eye_x - 14, eye_y - 4, 28, 10))
            pygame.draw.circle(self._screen, pupil_color, (left_eye_x, eye_y), 4)
            pygame.draw.circle(self._screen, pupil_color, (right_eye_x, eye_y), 4)
        else:
            # Neutral — normal round eyes
            pygame.draw.circle(self._screen, eye_color, (left_eye_x, eye_y), 13)
            pygame.draw.circle(self._screen, eye_color, (right_eye_x, eye_y), 13)
            pygame.draw.circle(self._screen, pupil_color, (left_eye_x, eye_y), 6)
            pygame.draw.circle(self._screen, pupil_color, (right_eye_x, eye_y), 6)

        # ── Mouth ──
        mouth_y = cy + 30 + breath

        if emotion == "happy":
            # Wide smile arc
            pygame.draw.arc(self._screen, mouth_color,
                            (cx - 30, mouth_y - 15 + talk_offset, 60, 30 + abs(talk_offset)),
                            math.pi, 2 * math.pi, 4)
        elif emotion == "sad":
            # Downturned frown
            pygame.draw.arc(self._screen, mouth_color,
                            (cx - 25, mouth_y + talk_offset, 50, 20),
                            0, math.pi, 4)
        elif emotion == "angry":
            # Tight frown line
            pygame.draw.line(self._screen, mouth_color,
                             (cx - 25, mouth_y + 2), (cx + 25, mouth_y + 2), 4)
            pygame.draw.line(self._screen, mouth_color,
                             (cx - 25, mouth_y + 2), (cx - 18, mouth_y + 8), 3)
            pygame.draw.line(self._screen, mouth_color,
                             (cx + 25, mouth_y + 2), (cx + 18, mouth_y + 8), 3)
        elif emotion == "surprise":
            # O-shaped mouth
            o_height = int(18 + abs(talk_offset))
            pygame.draw.ellipse(self._screen, mouth_color,
                                (cx - 12, mouth_y - 8, 24, o_height), 0)
        elif emotion == "fear":
            # Wavy worried mouth
            points = []
            for i in range(20):
                px = cx - 20 + i * 2
                py = mouth_y + math.sin(i * 0.8 + self._talk_phase) * 4
                points.append((px, py))
            if len(points) > 1:
                pygame.draw.lines(self._screen, mouth_color, False, points, 3)
        elif emotion == "disgust":
            # Tongue out
            pygame.draw.line(self._screen, mouth_color,
                             (cx - 20, mouth_y), (cx + 20, mouth_y), 3)
            pygame.draw.ellipse(self._screen, (200, 100, 100),
                                (cx - 8, mouth_y, 16, 12))
        else:
            # Neutral — small line or gentle curve
            if is_talking:
                h = int(8 + abs(talk_offset))
                pygame.draw.ellipse(self._screen, mouth_color,
                                    (cx - 15, mouth_y - 4, 30, h), 0)
            else:
                pygame.draw.line(self._screen, mouth_color,
                                 (cx - 20, mouth_y), (cx + 20, mouth_y), 3)

        # ── Blush for happy/surprise ──
        if emotion in ("happy", "surprise"):
            blush_alpha = 80
            blush_surface = pygame.Surface((30, 16), pygame.SRCALPHA)
            pygame.draw.ellipse(blush_surface, (255, 150, 150, blush_alpha),
                                (0, 0, 30, 16))
            self._screen.blit(blush_surface, (left_eye_x - 15, eye_y + 18))
            self._screen.blit(blush_surface, (right_eye_x - 15, eye_y + 18))

        pygame.display.flip()

    # ═══════════════════════════════════════════
    # Main Loop
    # ═══════════════════════════════════════════

    def start(self):
        """Start the face display in a background thread."""
        if not PYGAME_AVAILABLE or self._screen is None:
            logger.warning("Cannot start face display — Pygame not available")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Face display started")

    def stop(self):
        """Stop the face display."""
        self._running = False
        logger.info("Face display stopped")

    def _run_loop(self):
        """Main render loop."""
        while self._running:
            # Handle Pygame events (required to prevent freezing)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self._running = False
                        return

            self._draw_face()
            self._clock.tick(DISPLAY_FPS)

    def cleanup(self):
        """Shut down display."""
        self.stop()
        if PYGAME_AVAILABLE:
            try:
                pygame.quit()
            except Exception:
                pass
        logger.info("FaceDisplay cleaned up")
