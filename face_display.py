"""
ECHO Robot — Face Display (Full-Screen Robot Face)
====================================================
Renders an animated robot face on the 5" HDMI display (800x480).
The face fills the entire screen with:
  - Left eye on the left side of the display
  - Right eye on the right side of the display
  - Mouth centered at the bottom

Round 3 enhancements:
  - Shape-morphing transitions (smooth eye/mouth geometry blending)
  - Eyebrows for ALL emotions (neutral + happy now have them)
  - Camera-directed gaze via set_gaze(x, y) API
  - All emotion mouths animate during speech
  - Emotion-specific pupil behavior (fear darts, sad droops, angry constricts)
  - Brighter blush effect (visible on dark background)
  - Reaction animations (bounce/shake on emotion change)
  - Curved eyelid blinks (arcs instead of rectangles)

Emotions: happy, sad, angry, surprise, fear, disgust, neutral
"""

import math
import time
import random
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

# ── Emotion-specific eye geometry parameters ──
# (eye_width_mult, eye_height_mult, iris_size_mult, pupil_size_mult)
# These are interpolated during transitions for smooth shape morphing.
_EYE_PARAMS = {
    "neutral":  {"ow": 0.90, "oh": 0.90, "iris": 0.58, "pupil": 0.50},
    "happy":    {"ow": 1.30, "oh": 1.00, "iris": 0.45, "pupil": 0.50},  # arc-style, wider
    "sad":      {"ow": 1.30, "oh": 1.00, "iris": 0.42, "pupil": 0.50},
    "angry":    {"ow": 1.40, "oh": 0.65, "iris": 0.35, "pupil": 0.50},
    "surprise": {"ow": 1.15, "oh": 1.15, "iris": 0.55, "pupil": 0.50},
    "fear":     {"ow": 1.05, "oh": 1.05, "iris": 0.30, "pupil": 0.45},
    "disgust":  {"ow": 1.40, "oh": 0.40, "iris": 0.45, "pupil": 0.50},
}

# ── Emotion-specific mouth geometry ──
# (width_mult, height_mult) of base MOUTH_W / MOUTH_H
_MOUTH_PARAMS = {
    "neutral":  {"w": 0.90, "h": 0.20},
    "happy":    {"w": 2.40, "h": 2.20},
    "sad":      {"w": 2.00, "h": 1.80},
    "angry":    {"w": 1.30, "h": 0.10},
    "surprise": {"w": 1.00, "h": 1.60},
    "fear":     {"w": 1.40, "h": 0.30},
    "disgust":  {"w": 1.10, "h": 0.20},
}


class FaceDisplay:
    """Full-screen robot face with left eye, right eye, and centered mouth."""

    def __init__(self):
        self._screen = None
        self._clock = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        # ── Emotion state with smooth transitions ──
        self._emotion = "neutral"
        self._target_emotion = "neutral"
        self._emotion_blend = 1.0        # 0.0 = old emotion, 1.0 = target emotion
        self._prev_emotion = "neutral"

        # ── Shape morphing: smoothly interpolated geometry params ──
        self._morph_eye_ow = 0.90
        self._morph_eye_oh = 0.90
        self._morph_iris = 0.58
        self._morph_pupil = 0.50
        self._morph_mouth_w = 0.90
        self._morph_mouth_h = 0.20

        # ── Blink state — random intervals ──
        self._blink_timer = 0.0
        self._next_blink = random.uniform(2.0, 5.0)  # Random interval
        self._is_blinking = False
        self._blink_progress = 0.0       # 0.0 -> 1.0 -> 0.0 for smooth blink

        # ── Talk animation ──
        self._talk_phase = 0.0
        self._is_talking = False
        self._talk_amplitude = 0.0       # Smoothed amplitude for natural talking

        # ── Breathing ──
        self._breath_phase = 0.0

        # ── Pupil wander / gaze tracking ──
        self._pupil_offset_x = 0.0
        self._pupil_offset_y = 0.0
        self._pupil_target_x = 0.0
        self._pupil_target_y = 0.0
        self._pupil_wander_timer = 0.0
        self._pupil_wander_interval = random.uniform(1.5, 4.0)
        # Camera-directed gaze: when set, overrides random wander
        self._gaze_x = None              # None = use random wander
        self._gaze_y = None
        self._gaze_timeout = 0.0         # Auto-expire gaze after N seconds

        # ── Color transition ──
        self._current_color = (200, 200, 200)  # Start neutral gray
        self._target_color = (200, 200, 200)

        # ── Idle micro-movements ──
        self._idle_phase = 0.0

        # ── Reaction animation (bounce/shake on emotion change) ──
        self._reaction_timer = 0.0       # Counts down from ~0.4s on emotion change
        self._reaction_type = None       # 'bounce' or 'shake'
        self._reaction_intensity = 0.0   # Decays over time

        # ── Pre-rendered overlays (created in _init_display) ──
        self._scan_overlay = None

        # ── Layout (800x480) ──
        self.W = DISPLAY_WIDTH
        self.H = DISPLAY_HEIGHT

        # Eye positions — spread wide across the display
        self.LEFT_EYE_X = int(self.W * 0.25)
        self.RIGHT_EYE_X = int(self.W * 0.75)
        self.EYE_Y = int(self.H * 0.38)
        self.EYE_RADIUS = int(min(self.W, self.H) * 0.15)

        # Mouth position — centered lower
        self.MOUTH_X = self.W // 2
        self.MOUTH_Y = int(self.H * 0.78)
        self.MOUTH_W = int(self.W * 0.18)
        self.MOUTH_H = int(self.H * 0.08)

        logger.info("FaceDisplay initialized (robot face mode)")

    def _init_display(self):
        """Initialize Pygame display (display + font ONLY — not audio).
        Using pygame.display.init() instead of pygame.init() prevents
        the pygame mixer from stealing the audio device from PyAudio."""
        if not PYGAME_AVAILABLE:
            return

        try:
            # CRITICAL: Only init display & font — NOT audio/mixer
            # pygame.init() would grab the audio device and block PyAudio
            pygame.display.init()
            pygame.font.init()

            if DISPLAY_FULLSCREEN:
                # Use the ACTUAL screen resolution for true fullscreen
                info = pygame.display.Info()
                self.W = info.current_w
                self.H = info.current_h
                self._screen = pygame.display.set_mode(
                    (self.W, self.H), pygame.FULLSCREEN
                )
            else:
                self.W = DISPLAY_WIDTH
                self.H = DISPLAY_HEIGHT
                self._screen = pygame.display.set_mode(
                    (self.W, self.H)
                )

            # Recalculate face layout for actual screen size
            self._recalculate_layout()

            pygame.display.set_caption("ECHO Face")
            self._clock = pygame.time.Clock()
            pygame.mouse.set_visible(False)

            # Cache status font (avoids re-creating every frame)
            self._status_font = pygame.font.Font(None, 22)

            # Pre-render scan lines overlay once (avoids 3MB SRCALPHA alloc per frame)
            self._scan_overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            for y_line in range(0, self.H, 4):
                pygame.draw.line(self._scan_overlay, (255, 255, 255, 5),
                               (0, y_line), (self.W, y_line), 1)

            logger.info(f"Display initialized: {self.W}x{self.H}")

        except Exception as e:
            logger.error(f"Failed to init display: {e}")
            self._screen = None

    def set_emotion(self, emotion: str):
        """Set the target emotion for the face to transition to."""
        with self._lock:
            if emotion != self._target_emotion:
                self._prev_emotion = self._target_emotion
                self._target_emotion = emotion
                self._emotion_blend = 0.0  # Start transition
                self._target_color = EMOTION_COLORS.get(emotion, (200, 200, 200))
                # Trigger reaction animation
                self._reaction_timer = 0.4  # 400ms reaction
                # Choose reaction type based on emotion
                if emotion in ("surprise", "fear"):
                    self._reaction_type = "bounce"
                elif emotion in ("angry", "disgust"):
                    self._reaction_type = "shake"
                else:
                    self._reaction_type = "bounce"
                self._reaction_intensity = 1.0

    def set_talking(self, talking: bool):
        """Set whether the mouth should animate as talking."""
        with self._lock:
            self._is_talking = talking

    def set_gaze(self, x: float, y: float):
        """Direct the pupils toward a point.
        x: -1.0 (far left) to 1.0 (far right)
        y: -1.0 (up) to 1.0 (down)
        Overrides random pupil wander. Call with None to resume wander."""
        with self._lock:
            if x is None or y is None:
                self._gaze_x = None
                self._gaze_y = None
            else:
                self._gaze_x = max(-1.0, min(1.0, x))
                self._gaze_y = max(-1.0, min(1.0, y))
                self._gaze_timeout = 3.0  # Auto-expire after 3 seconds

    def _recalculate_layout(self):
        """Recalculate all face element positions for current resolution.
        Called after display init when we know the actual screen size."""
        self.LEFT_EYE_X = int(self.W * 0.25)
        self.RIGHT_EYE_X = int(self.W * 0.75)
        self.EYE_Y = int(self.H * 0.38)
        self.EYE_RADIUS = int(min(self.W, self.H) * 0.15)
        self.MOUTH_X = self.W // 2
        self.MOUTH_Y = int(self.H * 0.78)
        self.MOUTH_W = int(self.W * 0.18)
        self.MOUTH_H = int(self.H * 0.08)
        logger.info(f"Layout: eyes at ({self.LEFT_EYE_X},{self.EYE_Y}) & ({self.RIGHT_EYE_X},{self.EYE_Y}), R={self.EYE_RADIUS}, mouth at ({self.MOUTH_X},{self.MOUTH_Y})")

    # =============================================
    # Drawing Helpers
    # =============================================

    def _draw_glow_circle(self, center, radius, color, layers=2):
        """Draw a soft glow around a circle for the robot LED look.
        Uses small correctly-sized surfaces instead of full-screen SRCALPHA."""
        for i in range(layers, 0, -1):
            alpha = max(8, int(35 / i))
            glow_r = radius + i * 8
            size = glow_r * 2 + 2
            surf = pygame.Surface((size, size), pygame.SRCALPHA)
            glow_col = (color[0], color[1], color[2], alpha)
            pygame.draw.circle(surf, glow_col, (size // 2, size // 2), glow_r)
            self._screen.blit(surf, (center[0] - size // 2, center[1] - size // 2))

    def _draw_glow_line(self, start, end, color, width=4, layers=2):
        """Draw a glowing line. Uses dimmed direct drawing to avoid SRCALPHA allocs."""
        for i in range(layers, 0, -1):
            frac = 0.25 / i  # Dimming factor instead of alpha
            w = width + i * 3
            dim_col = (max(0, int(color[0] * frac)),
                       max(0, int(color[1] * frac)),
                       max(0, int(color[2] * frac)))
            pygame.draw.line(self._screen, dim_col, start, end, w)
        pygame.draw.line(self._screen, color, start, end, width)

    def _draw_glow_arc(self, rect, start_angle, stop_angle, color, width=4, layers=2):
        """Draw a glowing arc. Uses dimmed direct drawing to avoid SRCALPHA allocs."""
        for i in range(layers, 0, -1):
            frac = 0.25 / i
            w = width + i * 2
            dim_col = (max(0, int(color[0] * frac)),
                       max(0, int(color[1] * frac)),
                       max(0, int(color[2] * frac)))
            pygame.draw.arc(self._screen, dim_col, rect, start_angle, stop_angle, w)
        pygame.draw.arc(self._screen, color, rect, start_angle, stop_angle, width)

    # =============================================
    # Animation Helpers
    # =============================================

    def _lerp(self, a, b, t):
        """Linear interpolation between a and b."""
        return a + (b - a) * max(0.0, min(1.0, t))

    def _lerp_color(self, c1, c2, t):
        """Interpolate between two RGB colors."""
        t = max(0.0, min(1.0, t))
        return (
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t),
        )

    def _update_animations(self, dt):
        """Update all animation states each frame."""
        with self._lock:
            target_emotion = self._target_emotion
            is_talking = self._is_talking

        # ── Smooth emotion transition ──
        if self._emotion_blend < 1.0:
            self._emotion_blend = min(1.0, self._emotion_blend + dt * 3.0)  # ~0.33s transition
            if self._emotion_blend >= 1.0:
                self._emotion = target_emotion
        self._current_color = self._lerp_color(
            EMOTION_COLORS.get(self._prev_emotion, (200, 200, 200)),
            self._target_color,
            self._emotion_blend
        )

        # ── Shape morphing: interpolate eye/mouth geometry ──
        prev_ep = _EYE_PARAMS.get(self._prev_emotion, _EYE_PARAMS["neutral"])
        targ_ep = _EYE_PARAMS.get(target_emotion, _EYE_PARAMS["neutral"])
        t = self._emotion_blend
        self._morph_eye_ow = self._lerp(prev_ep["ow"], targ_ep["ow"], t)
        self._morph_eye_oh = self._lerp(prev_ep["oh"], targ_ep["oh"], t)
        self._morph_iris = self._lerp(prev_ep["iris"], targ_ep["iris"], t)
        self._morph_pupil = self._lerp(prev_ep["pupil"], targ_ep["pupil"], t)

        prev_mp = _MOUTH_PARAMS.get(self._prev_emotion, _MOUTH_PARAMS["neutral"])
        targ_mp = _MOUTH_PARAMS.get(target_emotion, _MOUTH_PARAMS["neutral"])
        self._morph_mouth_w = self._lerp(prev_mp["w"], targ_mp["w"], t)
        self._morph_mouth_h = self._lerp(prev_mp["h"], targ_mp["h"], t)

        # ── Breathing (slow, subtle) ──
        self._breath_phase += dt * 1.8
        breath = math.sin(self._breath_phase) * 3.5

        # ── Blink with random intervals and smooth open/close ──
        self._blink_timer += dt
        if not self._is_blinking and self._blink_timer >= self._next_blink:
            self._is_blinking = True
            self._blink_progress = 0.0
        if self._is_blinking:
            self._blink_progress += dt * 8.0  # Fast blink (~0.25s total)
            if self._blink_progress >= 2.0:
                self._is_blinking = False
                self._blink_timer = 0.0
                self._next_blink = random.uniform(2.0, 5.5)

        # ── Talk amplitude smoothing ──
        if is_talking:
            self._talk_phase += dt * 14.0
            # Multi-frequency for natural jaw movement
            raw = (math.sin(self._talk_phase) * 0.5 +
                   math.sin(self._talk_phase * 1.7) * 0.3 +
                   math.sin(self._talk_phase * 0.6) * 0.2)
            target_amp = abs(raw)
            self._talk_amplitude = self._lerp(self._talk_amplitude, target_amp, dt * 12.0)
        else:
            self._talk_amplitude = self._lerp(self._talk_amplitude, 0.0, dt * 8.0)

        # ── Pupil: gaze tracking or random wander ──
        with self._lock:
            gaze_x = self._gaze_x
            gaze_y = self._gaze_y

        if gaze_x is not None:
            # Camera-directed gaze: map [-1,1] to pixel offset within eye
            wander_range = self.EYE_RADIUS * 0.25
            self._pupil_target_x = gaze_x * wander_range
            self._pupil_target_y = gaze_y * wander_range * 0.6
            # Decay gaze timeout
            with self._lock:
                self._gaze_timeout -= dt
                if self._gaze_timeout <= 0:
                    self._gaze_x = None
                    self._gaze_y = None
        else:
            # Random pupil wander (idle look-around)
            self._pupil_wander_timer += dt
            if self._pupil_wander_timer >= self._pupil_wander_interval:
                self._pupil_wander_timer = 0.0
                self._pupil_wander_interval = random.uniform(1.5, 4.0)
                wander_range = self.EYE_RADIUS * 0.15
                self._pupil_target_x = random.uniform(-wander_range, wander_range)
                self._pupil_target_y = random.uniform(-wander_range * 0.5, wander_range * 0.5)

        # ── Emotion-specific pupil behavior ──
        emotion = target_emotion
        if emotion == "fear":
            # Fear: rapid darting pupils
            dart = math.sin(self._idle_phase * 12) * self.EYE_RADIUS * 0.08
            self._pupil_target_x += dart
            self._pupil_target_y += math.cos(self._idle_phase * 9) * self.EYE_RADIUS * 0.04
        elif emotion == "sad":
            # Sad: pupils droop downward
            self._pupil_target_y += self.EYE_RADIUS * 0.08
        elif emotion == "angry":
            # Angry: pupils constrict toward center (reduce wander range)
            self._pupil_target_x *= 0.3
            self._pupil_target_y *= 0.3

        # Smooth interpolation toward target
        gaze_speed = 4.0 if gaze_x is not None else 2.5
        self._pupil_offset_x = self._lerp(self._pupil_offset_x, self._pupil_target_x, dt * gaze_speed)
        self._pupil_offset_y = self._lerp(self._pupil_offset_y, self._pupil_target_y, dt * gaze_speed)

        # ── Idle micro-movement ──
        self._idle_phase += dt * 0.7

        # ── Reaction animation decay ──
        if self._reaction_timer > 0:
            self._reaction_timer -= dt
            self._reaction_intensity = max(0.0, self._reaction_timer / 0.4)
        else:
            self._reaction_intensity = 0.0
            self._reaction_type = None

        return breath

    # =============================================
    # Main Face Drawing
    # =============================================

    def _draw_face(self, dt):
        """Render the full robot face."""
        if self._screen is None:
            return

        # Update animations
        breath = self._update_animations(dt)

        with self._lock:
            emotion = self._target_emotion
            is_talking = self._is_talking

        # ── Colors ──
        bg_color = (10, 12, 18)
        accent = self._current_color  # Smoothly transitioned color
        eye_white = (215, 220, 230)
        pupil_color = (12, 14, 20)
        dim_accent = tuple(max(0, c // 4) for c in accent)

        # Blink amount: 0.0 = open, 1.0 = closed
        blink_amount = 0.0
        if self._is_blinking:
            # Triangle wave: 0->1->0 over blink_progress 0->2
            blink_amount = 1.0 - abs(self._blink_progress - 1.0)

        # Talk offset for mouth
        talk_offset = self._talk_amplitude * 18.0

        # Subtle idle sway
        idle_sway_x = math.sin(self._idle_phase) * 1.5
        idle_sway_y = math.cos(self._idle_phase * 0.8) * 1.0

        # ── Reaction offset (bounce or shake) ──
        reaction_dx = 0.0
        reaction_dy = 0.0
        if self._reaction_intensity > 0 and self._reaction_type:
            ri = self._reaction_intensity
            if self._reaction_type == "bounce":
                # Quick vertical bounce that decays
                reaction_dy = math.sin(ri * math.pi * 4) * ri * 8.0
            elif self._reaction_type == "shake":
                # Horizontal shake that decays
                reaction_dx = math.sin(ri * math.pi * 6) * ri * 6.0

        # ── Clear screen ──
        self._screen.fill(bg_color)

        # ── Pre-rendered scan lines overlay (created once in _init_display) ──
        if self._scan_overlay is not None:
            self._screen.blit(self._scan_overlay, (0, 0))

        # ── Eye sockets (subtle dark panels behind eyes) ──
        socket_r = self.EYE_RADIUS + 22
        for ex in [self.LEFT_EYE_X, self.RIGHT_EYE_X]:
            ey = int(self.EYE_Y + breath + idle_sway_y + reaction_dy)
            # Subtle pulsing glow behind socket (dim direct draw, no SRCALPHA)
            glow_frac = 0.06 + math.sin(self._breath_phase * 0.5) * 0.03
            dim_glow = (max(0, int(accent[0] * glow_frac)),
                        max(0, int(accent[1] * glow_frac)),
                        max(0, int(accent[2] * glow_frac)))
            pygame.draw.circle(self._screen, dim_glow, (int(ex + reaction_dx), ey), socket_r + 8)
            pygame.draw.circle(self._screen, (18, 20, 28), (int(ex + reaction_dx), ey), socket_r)

        # ── Draw both eyes ──
        for eye_x in [self.LEFT_EYE_X, self.RIGHT_EYE_X]:
            eye_y = int(self.EYE_Y + breath + idle_sway_y + reaction_dy)
            ex = int(eye_x + reaction_dx)
            is_left = (eye_x == self.LEFT_EYE_X)
            px = int(self._pupil_offset_x + idle_sway_x)
            py = int(self._pupil_offset_y)
            self._draw_eye(ex, eye_y, emotion, is_left, accent, eye_white,
                          pupil_color, blink_amount, px, py)

        # ── Draw mouth ──
        mouth_y = int(self.MOUTH_Y + breath + idle_sway_y + reaction_dy)
        mouth_x = int(self.MOUTH_X + reaction_dx)
        self._draw_mouth(mouth_x, mouth_y, emotion, is_talking,
                        talk_offset, accent, self._talk_amplitude)

        # ── Status indicator at bottom ──
        try:
            font = self._status_font
            status = f"ECHO  |  {emotion.upper()}"
            if is_talking:
                status += "  |  SPEAKING"
            label = font.render(status, True, dim_accent)
            self._screen.blit(label, (self.W // 2 - label.get_width() // 2, self.H - 24))
        except Exception:
            pass

        pygame.display.flip()

    # =============================================
    # Eye Drawing
    # =============================================

    def _draw_eye(self, cx, cy, emotion, is_left, accent, white, pupil_color,
                  blink_amount=0.0, pupil_dx=0, pupil_dy=0):
        """Draw one eye at position (cx, cy) with emotion-specific shape.
        blink_amount: 0.0 = fully open, 1.0 = fully closed.
        pupil_dx/dy: pupil wander offset.
        Uses morphed geometry parameters for smooth transitions."""
        R = self.EYE_RADIUS  # base radius

        # ── Full blink: glowing horizontal line ──
        if blink_amount > 0.85:
            hw = int(R * 1.3)
            self._draw_glow_line((cx - hw, cy), (cx + hw, cy), accent, width=4)
            return

        # Scale vertical openness based on blink (partial blink squishes the eye)
        v_scale = 1.0 - blink_amount * 0.7

        # ── Morphed geometry ──
        m_ow = self._morph_eye_ow
        m_oh = self._morph_eye_oh
        m_iris = self._morph_iris
        m_pupil = self._morph_pupil

        if emotion == "happy":
            # ── Happy: upward arc (^ ^) -- anime style ──
            arc_w = int(R * 2.6)
            arc_h = int(R * 2.0 * v_scale)
            rect = (cx - arc_w // 2, cy - arc_h // 2, arc_w, arc_h)
            self._draw_glow_arc(rect, 0.15, math.pi - 0.15, accent, width=6)
            # Add sparkle dots for extra happy
            if v_scale > 0.5:
                sparkle_r = int(R * 0.08)
                sx = cx + int(R * 0.5 * (-1 if is_left else 1))
                sy = cy - int(R * 0.5 * v_scale)
                pygame.draw.circle(self._screen, (255, 255, 255), (sx, sy), sparkle_r)
            # ── Happy eyebrows: gentle raised arcs ──
            brow_y = cy - int(R * 1.0 * v_scale) - 12
            bw = int(R * 0.8)
            self._draw_glow_arc(
                (cx - bw, brow_y - 6, bw * 2, 20),
                0.1, math.pi - 0.1, accent, width=3
            )

        elif emotion == "sad":
            # ── Sad: half-closed droopy eyes ──
            ow = int(R * m_ow)
            oh = int(R * m_oh * v_scale)
            pygame.draw.ellipse(self._screen, white, (cx - ow, cy - oh // 2, ow * 2, oh))
            # Heavy eyelid covering upper portion (direct draw, no SRCALPHA)
            lid_h = max(1, int(oh * (0.55 + blink_amount * 0.3)))
            pygame.draw.rect(self._screen, pupil_color,
                           (cx - ow - 2, cy - oh // 2 - 2, ow * 2 + 4, lid_h))
            # Iris -- positioned low with droop
            iris_r = int(R * m_iris)
            ix = cx + pupil_dx
            iy = cy + int(R * 0.12) + pupil_dy
            pygame.draw.circle(self._screen, accent, (ix, iy), iris_r)
            pupil_r = int(iris_r * m_pupil)
            pygame.draw.circle(self._screen, pupil_color, (ix, iy), pupil_r)
            # Droopy eyebrow
            brow_y = cy - oh // 2 - 14
            tilt = 18 if is_left else -18
            self._draw_glow_line(
                (cx - ow, brow_y - tilt), (cx + ow, brow_y + tilt),
                accent, width=3
            )

        elif emotion == "angry":
            # ── Angry: narrowed eyes with V-shaped brows ──
            ow = int(R * m_ow)
            oh = int(R * m_oh * v_scale)
            pygame.draw.ellipse(self._screen, white, (cx - ow, cy - oh // 2, ow * 2, oh))
            # Constricted iris with reduced wander
            iris_r = int(R * m_iris)
            ix = cx + pupil_dx
            iy = cy + pupil_dy
            pygame.draw.circle(self._screen, accent, (ix, iy), iris_r)
            pupil_r = int(iris_r * m_pupil)
            pygame.draw.circle(self._screen, pupil_color, (ix, iy), pupil_r)
            # Angry V-brow: \/ shape
            brow_y = cy - oh // 2 - 16
            inner_x = cx + (int(R * 0.5) if is_left else -int(R * 0.5))
            outer_x = cx - (int(R * 1.2) if is_left else -int(R * 1.2))
            self._draw_glow_line(
                (outer_x, brow_y + 22), (inner_x, brow_y - 8),
                accent, width=5
            )
            # Subtle vein lines (deterministic -- no per-frame random or SRCALPHA)
            dim_vein = (max(0, accent[0] // 8), max(0, accent[1] // 8), max(0, accent[2] // 8))
            for frac in [0.2, 0.5, 0.8]:
                vx = cx - ow + int(ow * 2 * frac)
                vy = cy - oh // 2 + int(oh * (0.3 if frac < 0.5 else 0.7))
                pygame.draw.line(self._screen, dim_vein, (cx, cy), (vx, vy), 1)

        elif emotion == "surprise":
            # ── Surprise: big wide circles ──
            big_r = int(R * m_ow * (1.0 + (1.0 - v_scale) * 0.1))
            pygame.draw.circle(self._screen, white, (cx, cy), big_r)
            # Large iris with wander
            iris_r = int(big_r * m_iris)
            ix = cx + pupil_dx
            iy = cy + pupil_dy
            pygame.draw.circle(self._screen, accent, (ix, iy), iris_r)
            pupil_r = int(iris_r * m_pupil)
            pygame.draw.circle(self._screen, pupil_color, (ix, iy), pupil_r)
            # Animated gleam highlight (shifts slightly)
            gleam_r = int(big_r * 0.14)
            gleam_shift = math.sin(self._idle_phase * 2) * 3
            gx = cx - int(big_r * 0.28) + int(gleam_shift)
            gy = cy - int(big_r * 0.28)
            pygame.draw.circle(self._screen, (255, 255, 255), (gx, gy), gleam_r)
            # Small secondary gleam
            pygame.draw.circle(self._screen, (255, 255, 255),
                             (cx + int(big_r * 0.15), cy + int(big_r * 0.15)),
                             max(1, gleam_r // 3))
            # Raised eyebrow arc
            brow_y = cy - big_r - 16
            bw = int(R * 1.0)
            self._draw_glow_arc(
                (cx - bw, brow_y - 8, bw * 2, 28),
                0, math.pi, accent, width=4
            )
            self._draw_glow_circle((cx, cy), big_r, accent, layers=3)

        elif emotion == "fear":
            # ── Fear: wide eyes, darting tiny pupils ──
            big_r = int(R * m_ow)
            pygame.draw.circle(self._screen, white, (cx, cy), big_r)
            # Tiny trembling iris (darting is applied via pupil behavior in _update_animations)
            iris_r = int(big_r * m_iris)
            ix = cx + pupil_dx
            iy = cy - 6 + pupil_dy
            pygame.draw.circle(self._screen, accent, (ix, iy), iris_r)
            pupil_r = int(iris_r * m_pupil)
            pygame.draw.circle(self._screen, pupil_color, (ix, iy), pupil_r)
            # Worried eyebrows (tilted)
            brow_y = cy - big_r - 12
            tilt = -14 if is_left else 14
            self._draw_glow_line(
                (cx - int(R * 0.7), brow_y + tilt),
                (cx + int(R * 0.7), brow_y - tilt),
                accent, width=3
            )

        elif emotion == "disgust":
            # ── Disgust: heavily squinted ──
            ow = int(R * m_ow)
            oh = int(R * m_oh * v_scale)
            pygame.draw.ellipse(self._screen, white, (cx - ow, cy - oh // 2, ow * 2, oh))
            iris_r = int(oh * m_iris) if oh > 0 else 4
            ix = cx + pupil_dx
            iy = cy + pupil_dy
            pygame.draw.circle(self._screen, accent, (ix, iy), max(2, iris_r))
            pupil_r = int(iris_r * m_pupil)
            pygame.draw.circle(self._screen, pupil_color, (ix, iy), max(1, pupil_r))
            if is_left:
                self._draw_glow_line(
                    (cx - ow, cy - oh // 2 - 14 + 12),
                    (cx + int(ow * 0.6), cy - oh // 2 - 14 - 16),
                    accent, width=4
                )
            else:
                self._draw_glow_line(
                    (cx - int(ow * 0.6), cy - oh // 2 - 14),
                    (cx + ow, cy - oh // 2 - 14),
                    accent, width=3
                )

        else:
            # ── Neutral: clean round eyes with iris + pupil wander ──
            outer_r = int(R * m_ow * (1.0 - blink_amount * 0.3))
            if outer_r < 5:
                # Almost closed -- just draw a line
                hw = int(R * 1.0)
                self._draw_glow_line((cx - hw, cy), (cx + hw, cy), accent, width=3)
                return
            pygame.draw.circle(self._screen, white, (cx, cy), outer_r)
            # Iris with wander
            iris_r = int(outer_r * m_iris)
            ix = cx + pupil_dx
            iy = cy + pupil_dy
            pygame.draw.circle(self._screen, accent, (ix, iy), iris_r)
            # Pupil
            pupil_r = int(iris_r * m_pupil)
            pygame.draw.circle(self._screen, pupil_color, (ix, iy), pupil_r)
            # Gleam (follows pupil slightly)
            gleam_r = int(outer_r * 0.12)
            gx = cx - int(outer_r * 0.22) + pupil_dx // 3
            gy = cy - int(outer_r * 0.22) + pupil_dy // 3
            pygame.draw.circle(self._screen, (255, 255, 255), (gx, gy), gleam_r)
            # Subtle glow ring
            self._draw_glow_circle((cx, cy), outer_r, accent, layers=2)
            # ── Neutral eyebrows: subtle flat arcs above each eye ──
            brow_y = cy - outer_r - 14
            bw = int(R * 0.7)
            self._draw_glow_line(
                (cx - bw, brow_y), (cx + bw, brow_y),
                accent, width=2
            )

        # ── Curved eyelid blink (arc instead of rectangle) ──
        if 0.1 < blink_amount <= 0.85 and emotion not in ("happy",):
            lid_progress = blink_amount  # 0.0 = open, ~0.85 = nearly closed
            lid_r = int(R * 1.5)
            # Draw a filled arc from the top that covers more of the eye as blink progresses
            # We approximate with an ellipse clipped to the upper portion
            lid_h = int(R * 2.4 * lid_progress)
            lid_w = int(R * 3.0)
            # Background-colored ellipse descending from above the eye
            lid_rect = (cx - lid_w // 2, cy - R - 10, lid_w, lid_h * 2)
            pygame.draw.ellipse(self._screen, (10, 12, 18), lid_rect)
            # Eyelid edge line (curved)
            if lid_h > 5:
                edge_rect = (cx - lid_w // 2, cy - R - 10 + lid_h - 8, lid_w, 16)
                self._draw_glow_arc(edge_rect, math.pi + 0.3, 2 * math.pi - 0.3,
                                   accent, width=2, layers=1)

    # =============================================
    # Mouth Drawing
    # =============================================

    def _draw_mouth(self, cx, cy, emotion, is_talking, talk_offset, color, talk_amp=0.0):
        """Draw the mouth at position (cx, cy) with emotion-specific shape.
        talk_amp: 0.0-1.0 smoothed amplitude for natural jaw movement.
        ALL emotions now animate during speech."""
        mw = self.MOUTH_W   # half-width
        mh = self.MOUTH_H   # half-height
        thick = 5

        if emotion == "happy":
            # ── Wide smile arc ──
            arc_w = int(mw * self._morph_mouth_w)
            arc_h = int(mh * self._morph_mouth_h) + int(talk_amp * 12)
            rect = (cx - arc_w // 2, cy - arc_h // 4, arc_w, arc_h)
            self._draw_glow_arc(rect, math.pi + 0.2, 2 * math.pi - 0.2, color, width=thick)
            if is_talking and talk_amp > 0.1:
                mouth_open = int(talk_amp * 18)
                if mouth_open > 3:
                    # Dark mouth interior
                    pygame.draw.ellipse(
                        self._screen, (25, 12, 18),
                        (cx - arc_w // 5, cy + 2, arc_w // 2 - 10, mouth_open + 8)
                    )
                    # Teeth hint
                    teeth_w = arc_w // 3 - 14
                    teeth_h = min(6, mouth_open // 3)
                    if teeth_h > 2:
                        pygame.draw.rect(
                            self._screen, (230, 235, 240),
                            (cx - teeth_w // 2, cy + 2, teeth_w, teeth_h)
                        )

        elif emotion == "sad":
            # ── Downturned frown arc — opens during speech ──
            arc_w = int(mw * self._morph_mouth_w)
            arc_h = int(mh * self._morph_mouth_h)
            if is_talking and talk_amp > 0.05:
                # Sad talk: frown opens slightly (trembling mouth)
                mouth_open = int(talk_amp * 12)
                # Draw the frown
                rect = (cx - arc_w // 2, cy - arc_h // 2, arc_w, arc_h)
                self._draw_glow_arc(rect, 0.2, math.pi - 0.2, color, width=thick)
                # Add opening beneath the frown
                if mouth_open > 2:
                    open_w = int(arc_w * 0.5)
                    pygame.draw.ellipse(
                        self._screen, (25, 15, 20),
                        (cx - open_w // 2, cy + 2, open_w, mouth_open + 4)
                    )
                    pygame.draw.ellipse(
                        self._screen, color,
                        (cx - open_w // 2, cy + 2, open_w, mouth_open + 4), 2
                    )
            else:
                rect = (cx - arc_w // 2, cy - arc_h // 2, arc_w, arc_h)
                self._draw_glow_arc(rect, 0.2, math.pi - 0.2, color, width=thick)

        elif emotion == "angry":
            # ── Tight line with sharp downturned corners — grits open during speech ──
            hw = int(mw * self._morph_mouth_w)
            if is_talking and talk_amp > 0.05:
                # Angry talk: tight-jawed open, teeth bared
                mouth_open = int(talk_amp * 10)
                self._draw_glow_line((cx - hw, cy), (cx + hw, cy), color, width=thick)
                if mouth_open > 2:
                    # Dark gap below line
                    open_w = int(hw * 1.2)
                    pygame.draw.rect(
                        self._screen, (25, 12, 18),
                        (cx - open_w // 2, cy + 2, open_w, mouth_open + 2)
                    )
                    # Teeth
                    teeth_w = int(open_w * 0.8)
                    teeth_h = min(4, mouth_open // 2)
                    if teeth_h > 1:
                        pygame.draw.rect(
                            self._screen, (230, 235, 240),
                            (cx - teeth_w // 2, cy + 2, teeth_w, teeth_h)
                        )
            else:
                self._draw_glow_line((cx - hw, cy), (cx + hw, cy), color, width=thick)
            # Downturned corners
            self._draw_glow_line(
                (cx - hw, cy), (cx - hw + 18, cy + 14), color, width=3
            )
            self._draw_glow_line(
                (cx + hw, cy), (cx + hw - 18, cy + 14), color, width=3
            )

        elif emotion == "surprise":
            # ── O-shaped mouth, pulses when talking ──
            ow = int(mh * self._morph_mouth_w) + int(talk_amp * 8)
            oh = int(mh * self._morph_mouth_h) + int(talk_amp * 10)
            rect = (cx - ow, cy - oh // 2, ow * 2, oh)
            # Filled dark inside
            pygame.draw.ellipse(self._screen, (20, 12, 18), rect)
            self._draw_glow_arc(rect, 0, 2 * math.pi, color, width=thick)

        elif emotion == "fear":
            # ── Wavy trembling line — opens during speech ──
            wave_w = int(mw * self._morph_mouth_w)
            if is_talking and talk_amp > 0.05:
                # Fear talk: trembling open mouth
                mouth_open = int(talk_amp * 14)
                # Draw wavy outline
                points_top = []
                points_bot = []
                segments = 20
                for i in range(segments):
                    px = cx - wave_w + i * (2 * wave_w // max(segments - 1, 1))
                    wave = math.sin(i * 0.85 + self._talk_phase * 2.5) * 4
                    points_top.append((int(px), int(cy + wave - mouth_open // 2)))
                    points_bot.append((int(px), int(cy + wave + mouth_open // 2)))
                if len(points_top) > 1:
                    dim_col = (max(0, color[0] // 5), max(0, color[1] // 5), max(0, color[2] // 5))
                    pygame.draw.lines(self._screen, color, False, points_top, thick)
                    pygame.draw.lines(self._screen, color, False, points_bot, thick - 1)
                    # Dark interior
                    if mouth_open > 3:
                        open_w = int(wave_w * 1.0)
                        pygame.draw.ellipse(
                            self._screen, (25, 15, 20),
                            (cx - open_w // 2, cy - mouth_open // 3, open_w, mouth_open)
                        )
            else:
                points = []
                segments = 28
                for i in range(segments):
                    px = cx - wave_w + i * (2 * wave_w // max(segments - 1, 1))
                    py = cy + math.sin(i * 0.85 + self._talk_phase * 2.5) * 7
                    points.append((int(px), int(py)))
                if len(points) > 1:
                    dim_col = (max(0, color[0] // 5), max(0, color[1] // 5), max(0, color[2] // 5))
                    pygame.draw.lines(self._screen, dim_col, False, points, thick + 4)
                    pygame.draw.lines(self._screen, color, False, points, thick)

        elif emotion == "disgust":
            # ── Asymmetric frown + tongue — tongue moves during speech ──
            hw = int(mw * self._morph_mouth_w)
            self._draw_glow_line((cx - hw, cy - 4), (cx + hw, cy + 4), color, width=thick)
            if is_talking and talk_amp > 0.05:
                # Disgust talk: tongue wiggles
                tongue_offset = int(math.sin(self._talk_phase * 3) * 4)
                tw = int(22 + talk_amp * 8)
                th = int(20 + talk_amp * 6)
                tongue_col = (200, 95, 100)
                pygame.draw.ellipse(
                    self._screen, tongue_col,
                    (cx - tw // 2 + tongue_offset, cy + 4, tw, th)
                )
                # Open mouth area
                open_h = int(talk_amp * 8)
                if open_h > 2:
                    pygame.draw.ellipse(
                        self._screen, (25, 15, 20),
                        (cx - hw // 2, cy, hw, open_h + 4)
                    )
            else:
                # Tongue (static)
                tw, th = 22, 20
                tongue_col = (200, 95, 100)
                pygame.draw.ellipse(
                    self._screen, tongue_col,
                    (cx - tw // 2, cy + 4, tw, th)
                )

        else:
            # ── Neutral: simple line or talking oval ──
            hw = int(mw * self._morph_mouth_w)
            if is_talking and talk_amp > 0.05:
                # Natural talking: jaw opens and closes with varying amplitude
                mouth_h = int(8 + talk_amp * 22)
                mouth_w = int(hw * (0.7 + talk_amp * 0.3))
                # Inner dark mouth
                pygame.draw.ellipse(
                    self._screen, (25, 15, 20),
                    (cx - mouth_w // 2, cy - mouth_h // 4, mouth_w, mouth_h), 0
                )
                # Lip outline
                pygame.draw.ellipse(
                    self._screen, color,
                    (cx - mouth_w // 2, cy - mouth_h // 4, mouth_w, mouth_h), 3
                )
                # Tongue hint when mouth is wide open
                if talk_amp > 0.6:
                    tongue_w = int(mouth_w * 0.4)
                    tongue_h = int(mouth_h * 0.3)
                    tongue_col = (180, 80, 90)
                    pygame.draw.ellipse(
                        self._screen, tongue_col,
                        (cx - tongue_w // 2, cy + mouth_h // 4, tongue_w, tongue_h)
                    )
            else:
                # Resting: slight smile curve instead of flat line
                smile = 3  # Slight upward curve
                pts = [
                    (cx - hw, cy),
                    (cx - hw // 2, cy + smile),
                    (cx, cy + smile + 1),
                    (cx + hw // 2, cy + smile),
                    (cx + hw, cy),
                ]
                if len(pts) > 1:
                    dim_col = (max(0, color[0] // 5), max(0, color[1] // 5), max(0, color[2] // 5))
                    pygame.draw.lines(self._screen, dim_col, False, pts, thick + 3)
                    pygame.draw.lines(self._screen, color, False, pts, thick)

        # ── Blush marks for happy/surprise (brighter, more visible) ──
        if emotion in ("happy", "surprise"):
            bw, bh = 44, 20
            # Much brighter blush: pinkish-red clearly visible on dark (10,12,18) bg
            blush_col = (90, 35, 40)
            blush_y = self.EYE_Y + int(self.EYE_RADIUS * 0.8)
            pygame.draw.ellipse(self._screen, blush_col,
                              (self.LEFT_EYE_X - bw // 2, blush_y, bw, bh))
            pygame.draw.ellipse(self._screen, blush_col,
                              (self.RIGHT_EYE_X - bw // 2, blush_y, bw, bh))

    # =============================================
    # Main Loop
    # =============================================

    def start(self):
        """Start the face display in a background thread."""
        if not PYGAME_AVAILABLE:
            logger.warning("Cannot start face display -- Pygame not available")
            return

        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Face display started")

    def stop(self):
        """Stop the face display."""
        self._running = False

        thread = self._thread
        if thread and thread.is_alive() and threading.current_thread() is not thread:
            thread.join(timeout=2.0)
        self._thread = None

        logger.info("Face display stopped")

    def _run_loop(self):
        """Main render loop."""
        self._init_display()
        if self._screen is None or self._clock is None:
            logger.error("Face display loop cannot start -- display init failed")
            self._running = False
            return

        last_time = time.time()

        try:
            while self._running:
                # Handle Pygame events (required to prevent freezing)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        # Ignore QUIT events -- face lifecycle is controlled by _running flag.
                        # On RPi, the window manager can send spurious QUIT events.
                        logger.debug("Ignoring pygame.QUIT event (RPi window manager)")
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            logger.info("ESC pressed -- stopping face display")
                            self._running = False
                            break

                if not self._running:
                    break

                # Calculate delta time for smooth animations
                now = time.time()
                dt = min(now - last_time, 0.1)  # Cap at 100ms to prevent jumps
                last_time = now

                self._draw_face(dt)
                self._clock.tick(DISPLAY_FPS)
        except Exception as e:
            logger.error(f"Face display loop crashed: {e}", exc_info=True)
            self._running = False
        finally:
            if PYGAME_AVAILABLE:
                try:
                    pygame.display.quit()
                    pygame.font.quit()
                except Exception:
                    pass
            self._screen = None
            self._clock = None
            # Note: do NOT set self._thread = None here -- we're running
            # inside the thread. The stop() method handles join().
            logger.info("Face display loop exited")

    def cleanup(self):
        """Shut down display."""
        self.stop()
        logger.info("FaceDisplay cleaned up")
