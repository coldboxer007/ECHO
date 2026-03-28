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
        """Render the current face expression scaled for display size."""
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

        # Scale factor relative to original 320x240 design
        sx = DISPLAY_WIDTH / 320.0
        sy = DISPLAY_HEIGHT / 240.0
        s = min(sx, sy)  # Uniform scale

        # Breathing animation (subtle scale)
        self._breath_phase += 0.02
        breath = math.sin(self._breath_phase) * (3 * s)

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
        talk_offset = math.sin(self._talk_phase) * (8 * s) if is_talking else 0

        # Clear screen
        self._screen.fill(bg_color)

        # ── Face background (rounded ellipse) ──
        fw = int(180 * s)
        fh = int(160 * s)
        face_rect = pygame.Rect(
            cx - fw // 2, cy - fh // 2 + breath, fw, fh
        )
        pygame.draw.ellipse(self._screen, face_color, face_rect)

        # ── Eyes ──
        eye_y = int(cy - 25 * s + breath)
        left_eye_x = int(cx - 35 * s)
        right_eye_x = int(cx + 35 * s)

        # Scaled sizes
        eye_r = int(13 * s)       # neutral eye radius
        pupil_r = int(6 * s)      # neutral pupil radius
        line_w = max(2, int(3 * s))
        thick_w = max(3, int(4 * s))

        if self._is_blinking:
            # Closed eyes — horizontal lines
            hw = int(14 * s)
            pygame.draw.line(self._screen, pupil_color,
                             (left_eye_x - hw, eye_y), (left_eye_x + hw, eye_y), line_w)
            pygame.draw.line(self._screen, pupil_color,
                             (right_eye_x - hw, eye_y), (right_eye_x + hw, eye_y), line_w)
        elif emotion == "happy":
            # Happy eyes — upturned arcs (^  ^)
            aw = int(28 * s)
            ah = int(20 * s)
            pygame.draw.arc(self._screen, pupil_color,
                            (left_eye_x - aw // 2, eye_y - ah // 2, aw, ah),
                            0, math.pi, thick_w)
            pygame.draw.arc(self._screen, pupil_color,
                            (right_eye_x - aw // 2, eye_y - ah // 2, aw, ah),
                            0, math.pi, thick_w)
        elif emotion == "sad":
            # Sad eyes — droopy
            ew = int(24 * s)
            eh = int(20 * s)
            pygame.draw.ellipse(self._screen, eye_color,
                                (left_eye_x - ew // 2, eye_y - eh // 2, ew, eh))
            pygame.draw.ellipse(self._screen, eye_color,
                                (right_eye_x - ew // 2, eye_y - eh // 2, ew, eh))
            pr = int(6 * s)
            pygame.draw.circle(self._screen, pupil_color, (left_eye_x, eye_y + int(2 * s)), pr)
            pygame.draw.circle(self._screen, pupil_color, (right_eye_x, eye_y + int(2 * s)), pr)
            # Droopy eyebrows
            brow_off = int(16 * s)
            brow_h = int(18 * s)
            brow_tilt = int(14 * s)
            pygame.draw.line(self._screen, pupil_color,
                             (left_eye_x - brow_off, eye_y - brow_h),
                             (left_eye_x + int(12 * s), eye_y - brow_tilt), line_w)
            pygame.draw.line(self._screen, pupil_color,
                             (right_eye_x - int(12 * s), eye_y - brow_tilt),
                             (right_eye_x + brow_off, eye_y - brow_h), line_w)
        elif emotion == "angry":
            # Angry eyes — angled brows, narrow
            ew = int(24 * s)
            eh = int(16 * s)
            pygame.draw.ellipse(self._screen, eye_color,
                                (left_eye_x - ew // 2, eye_y - eh // 2, ew, eh))
            pygame.draw.ellipse(self._screen, eye_color,
                                (right_eye_x - ew // 2, eye_y - eh // 2, ew, eh))
            pr = int(5 * s)
            pygame.draw.circle(self._screen, pupil_color, (left_eye_x, eye_y), pr)
            pygame.draw.circle(self._screen, pupil_color, (right_eye_x, eye_y), pr)
            # Angry eyebrows \  /
            b1 = int(16 * s)
            b2 = int(10 * s)
            b3 = int(12 * s)
            b4 = int(20 * s)
            pygame.draw.line(self._screen, pupil_color,
                             (left_eye_x - b1, eye_y - b3), (left_eye_x + b2, eye_y - b4), thick_w)
            pygame.draw.line(self._screen, pupil_color,
                             (right_eye_x - b2, eye_y - b4), (right_eye_x + b1, eye_y - b3), thick_w)
        elif emotion == "surprise":
            # Surprise — big round eyes
            big_r = int(16 * s)
            pr = int(7 * s)
            pygame.draw.circle(self._screen, eye_color, (left_eye_x, eye_y), big_r)
            pygame.draw.circle(self._screen, eye_color, (right_eye_x, eye_y), big_r)
            pygame.draw.circle(self._screen, pupil_color, (left_eye_x, eye_y), pr)
            pygame.draw.circle(self._screen, pupil_color, (right_eye_x, eye_y), pr)
            # Raised eyebrows
            bw = int(36 * s)
            bh = int(20 * s)
            by = int(30 * s)
            pygame.draw.arc(self._screen, pupil_color,
                            (left_eye_x - bw // 2, eye_y - by, bw, bh),
                            0, math.pi, line_w)
            pygame.draw.arc(self._screen, pupil_color,
                            (right_eye_x - bw // 2, eye_y - by, bw, bh),
                            0, math.pi, line_w)
        elif emotion == "fear":
            # Fear — wide eyes with small pupils
            big_r = int(15 * s)
            pr = int(4 * s)
            off = int(2 * s)
            pygame.draw.circle(self._screen, eye_color, (left_eye_x, eye_y), big_r)
            pygame.draw.circle(self._screen, eye_color, (right_eye_x, eye_y), big_r)
            pygame.draw.circle(self._screen, pupil_color, (left_eye_x - off, eye_y - off), pr)
            pygame.draw.circle(self._screen, pupil_color, (right_eye_x + off, eye_y - off), pr)
        elif emotion == "disgust":
            # Disgust — squinted eyes
            ew = int(28 * s)
            eh = int(10 * s)
            pr = int(4 * s)
            pygame.draw.ellipse(self._screen, eye_color,
                                (left_eye_x - ew // 2, eye_y - eh // 2, ew, eh))
            pygame.draw.ellipse(self._screen, eye_color,
                                (right_eye_x - ew // 2, eye_y - eh // 2, ew, eh))
            pygame.draw.circle(self._screen, pupil_color, (left_eye_x, eye_y), pr)
            pygame.draw.circle(self._screen, pupil_color, (right_eye_x, eye_y), pr)
        else:
            # Neutral — normal round eyes
            pygame.draw.circle(self._screen, eye_color, (left_eye_x, eye_y), eye_r)
            pygame.draw.circle(self._screen, eye_color, (right_eye_x, eye_y), eye_r)
            pygame.draw.circle(self._screen, pupil_color, (left_eye_x, eye_y), pupil_r)
            pygame.draw.circle(self._screen, pupil_color, (right_eye_x, eye_y), pupil_r)

        # ── Mouth ──
        mouth_y = int(cy + 30 * s + breath)

        if emotion == "happy":
            # Wide smile arc
            mw = int(60 * s)
            mh = int(30 * s)
            pygame.draw.arc(self._screen, mouth_color,
                            (cx - mw // 2, mouth_y - mh // 2 + int(talk_offset),
                             mw, mh + int(abs(talk_offset))),
                            math.pi, 2 * math.pi, thick_w)
        elif emotion == "sad":
            # Downturned frown
            mw = int(50 * s)
            mh = int(20 * s)
            pygame.draw.arc(self._screen, mouth_color,
                            (cx - mw // 2, mouth_y + int(talk_offset), mw, mh),
                            0, math.pi, thick_w)
        elif emotion == "angry":
            # Tight frown line with downturned corners
            hw = int(25 * s)
            corner = int(8 * s)
            pygame.draw.line(self._screen, mouth_color,
                             (cx - hw, mouth_y + 2), (cx + hw, mouth_y + 2), thick_w)
            pygame.draw.line(self._screen, mouth_color,
                             (cx - hw, mouth_y + 2), (cx - hw + int(7 * s), mouth_y + corner), line_w)
            pygame.draw.line(self._screen, mouth_color,
                             (cx + hw, mouth_y + 2), (cx + hw - int(7 * s), mouth_y + corner), line_w)
        elif emotion == "surprise":
            # O-shaped mouth
            ow = int(24 * s)
            oh = int(18 * s) + int(abs(talk_offset))
            pygame.draw.ellipse(self._screen, mouth_color,
                                (cx - ow // 2, mouth_y - oh // 2, ow, oh), 0)
        elif emotion == "fear":
            # Wavy worried mouth
            points = []
            mw = int(20 * s)
            for i in range(20):
                px = cx - mw + i * int(2 * s)
                py = mouth_y + math.sin(i * 0.8 + self._talk_phase) * (4 * s)
                points.append((int(px), int(py)))
            if len(points) > 1:
                pygame.draw.lines(self._screen, mouth_color, False, points, line_w)
        elif emotion == "disgust":
            # Tongue out
            hw = int(20 * s)
            pygame.draw.line(self._screen, mouth_color,
                             (cx - hw, mouth_y), (cx + hw, mouth_y), line_w)
            tw = int(8 * s)
            th = int(12 * s)
            pygame.draw.ellipse(self._screen, (200, 100, 100),
                                (cx - tw, mouth_y, tw * 2, th))
        else:
            # Neutral — small line or gentle curve
            hw = int(20 * s)
            if is_talking:
                h = int(8 * s + abs(talk_offset))
                pygame.draw.ellipse(self._screen, mouth_color,
                                    (cx - int(15 * s), mouth_y - int(4 * s),
                                     int(30 * s), h), 0)
            else:
                pygame.draw.line(self._screen, mouth_color,
                                 (cx - hw, mouth_y), (cx + hw, mouth_y), line_w)

        # ── Blush for happy/surprise ──
        if emotion in ("happy", "surprise"):
            blush_alpha = 80
            bw = int(30 * s)
            bh = int(16 * s)
            blush_surface = pygame.Surface((bw, bh), pygame.SRCALPHA)
            pygame.draw.ellipse(blush_surface, (255, 150, 150, blush_alpha),
                                (0, 0, bw, bh))
            self._screen.blit(blush_surface, (left_eye_x - bw // 2, eye_y + int(18 * s)))
            self._screen.blit(blush_surface, (right_eye_x - bw // 2, eye_y + int(18 * s)))

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
