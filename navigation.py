"""
ECHO Robot — Navigation Controller
=====================================
High-level movement logic combining motor control and sensor data.

Modes:
  - Manual: Direct voice-command movement (forward, backward, turn)
  - Follow: Person-tracking using webcam face detection
  - Autonomous: (future) waypoint navigation with obstacle avoidance
"""

import time
import logging
import threading
import cv2

logger = logging.getLogger("echo.navigation")

from config import (
    MOTOR_MOVE_DURATION, MOTOR_TURN_DURATION,
    OBSTACLE_DISTANCE_CM,
    FOLLOW_TARGET_DISTANCE_CM, FOLLOW_TURN_THRESHOLD_PX,
    FOLLOW_LOST_TIMEOUT,
    CAMERA_WIDTH,
)


class NavigationController:
    """
    Orchestrates motor movements with sensor feedback.
    Provides manual control, obstacle avoidance, and person following.
    """

    def __init__(self, motors, sensors, camera):
        """
        Args:
            motors: MotorController instance
            sensors: SensorController instance
            camera: CameraSentiment instance
        """
        self.motors = motors
        self.sensors = sensors
        self.camera = camera

        self._follow_mode = False
        self._follow_thread = None
        self._running = False
        self._lock = threading.Lock()

        logger.info("NavigationController initialized")

    # ═══════════════════════════════════════════
    # Manual Movement (Voice Commands)
    # ═══════════════════════════════════════════

    def execute_move(self, direction: str, duration: float = None):
        """
        Execute a movement command with obstacle checking.

        Args:
            direction: 'forward', 'backward', 'left', 'right'
            duration: How long to move (seconds). None = default.
        """
        # Stop follow mode if active
        if self._follow_mode:
            self.stop_follow()

        # Check for obstacles before moving forward
        if direction == 'forward' and self.sensors.is_obstacle_ahead():
            logger.warning("Cannot move forward — obstacle detected!")
            return False

        if direction == 'forward':
            self.motors.forward(duration or MOTOR_MOVE_DURATION)
        elif direction == 'backward':
            self.motors.backward(duration or MOTOR_MOVE_DURATION)
        elif direction == 'left':
            self.motors.turn_left(duration or MOTOR_TURN_DURATION)
        elif direction == 'right':
            self.motors.turn_right(duration or MOTOR_TURN_DURATION)
        else:
            logger.warning(f"Unknown direction: {direction}")
            return False

        return True

    def emergency_stop(self):
        """Immediately stop everything."""
        self._follow_mode = False
        self.motors.stop()
        logger.warning("⛔ Emergency stop!")

    # ═══════════════════════════════════════════
    # Person Following Mode
    # ═══════════════════════════════════════════

    def start_follow(self):
        """Start person-following mode."""
        with self._lock:
            if self._follow_mode:
                logger.info("Already in follow mode")
                return
            self._follow_mode = True
            self._running = True

        self._follow_thread = threading.Thread(target=self._follow_loop, daemon=True)
        self._follow_thread.start()
        logger.info("🏃 Follow mode STARTED")

    def stop_follow(self):
        """Stop person-following mode."""
        with self._lock:
            self._follow_mode = False
            self._running = False
        self.motors.stop()
        logger.info("🛑 Follow mode STOPPED")

    @property
    def is_following(self) -> bool:
        with self._lock:
            return self._follow_mode

    def _follow_loop(self):
        """
        Main follow loop:
        1. Capture frame
        2. Detect face center
        3. Steer toward face (turn left/right)
        4. Move forward if distance allows
        5. Stop if person lost for too long
        """
        frame_center_x = CAMERA_WIDTH // 2
        lost_timer = 0.0
        last_time = time.time()

        while self._running and self._follow_mode:
            now = time.time()
            dt = now - last_time
            last_time = now

            # Check obstacles first
            if self.sensors.is_obstacle_ahead():
                self.motors.stop()
                time.sleep(0.2)
                continue

            # Detect face
            frame = self.camera.capture_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            face_center = self.camera.get_face_center(frame)

            if face_center is None:
                # Person lost
                lost_timer += dt
                if lost_timer > FOLLOW_LOST_TIMEOUT:
                    logger.info("Person lost — stopping follow")
                    self.motors.stop()
                else:
                    # Slow down gradually
                    self.motors.stop()
                time.sleep(0.1)
                continue

            # Person found — reset lost timer
            lost_timer = 0.0
            face_x, face_y = face_center

            # Calculate offset from center
            offset_x = face_x - frame_center_x

            # Steering
            if abs(offset_x) > FOLLOW_TURN_THRESHOLD_PX:
                if offset_x > 0:
                    # Person is to the right
                    self.motors.slight_right()
                else:
                    # Person is to the left
                    self.motors.slight_left()
            else:
                # Person is roughly centered — check distance
                distance = self.sensors.read_distance()

                if distance > FOLLOW_TARGET_DISTANCE_CM:
                    # Too far — move forward
                    self.motors.forward()
                elif distance < FOLLOW_TARGET_DISTANCE_CM * 0.6:
                    # Too close — back up slightly
                    self.motors.backward()
                else:
                    # Good distance — stop
                    self.motors.stop()

            time.sleep(0.1)  # ~10 Hz control loop

        self.motors.stop()

    # ═══════════════════════════════════════════
    # Obstacle-Aware Continuous Move
    # ═══════════════════════════════════════════

    def safe_forward(self, duration: float = 2.0, check_interval: float = 0.1):
        """
        Move forward while continuously checking for obstacles.
        Stops immediately if obstacle detected.
        """
        self.motors.forward()
        elapsed = 0.0

        while elapsed < duration:
            if self.sensors.is_obstacle_ahead():
                self.motors.stop()
                logger.warning("Obstacle! Stopped during safe_forward.")
                return False
            time.sleep(check_interval)
            elapsed += check_interval

        self.motors.stop()
        return True

    # ═══════════════════════════════════════════
    # Cleanup
    # ═══════════════════════════════════════════

    def cleanup(self):
        """Stop all movement and clean up."""
        self.stop_follow()
        self.motors.stop()
        logger.info("NavigationController cleaned up")
