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
        self._continuous_mode = False
        self._continuous_thread = None
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

    def safe_forward(self, duration: float = 5.0, check_interval: float = 0.1):
        """
        Move forward while continuously checking for obstacles.
        Stops immediately if obstacle detected.
        """
        if self._follow_mode:
            self.stop_follow()
        self._continuous_mode = True
        self.motors.forward()
        elapsed = 0.0

        while elapsed < duration and self._continuous_mode:
            if self.sensors.is_obstacle_ahead():
                self.motors.stop()
                logger.warning("⚠️ Obstacle detected! Stopped safely.")
                self._continuous_mode = False
                return False
            time.sleep(check_interval)
            elapsed += check_interval

        self.motors.stop()
        self._continuous_mode = False
        return True

    # ═══════════════════════════════════════════
    # Continuous Movement (keep going until stop)
    # ═══════════════════════════════════════════

    def start_continuous_move(self, direction: str = 'forward'):
        """
        Start moving continuously in a direction until stop is called.
        Checks obstacles every 100ms. Runs in background thread.
        """
        self.stop_continuous()  # Stop any existing continuous movement
        if self._follow_mode:
            self.stop_follow()

        self._continuous_mode = True
        self._continuous_thread = threading.Thread(
            target=self._continuous_loop, args=(direction,), daemon=True
        )
        self._continuous_thread.start()
        logger.info(f"🔄 Continuous {direction} movement STARTED")

    def stop_continuous(self):
        """Stop continuous movement."""
        if self._continuous_mode:
            self._continuous_mode = False
            self.motors.stop()
            logger.info("🛑 Continuous movement STOPPED")

    @property
    def is_continuous(self) -> bool:
        return self._continuous_mode

    def _continuous_loop(self, direction: str):
        """Background loop for continuous movement with obstacle checking."""
        move_fn = {
            'forward': self.motors.forward,
            'backward': self.motors.backward,
        }.get(direction, self.motors.forward)

        move_fn()  # Start moving (no duration = continuous)

        while self._continuous_mode:
            # Check obstacles for forward movement
            if direction == 'forward' and self.sensors.is_obstacle_ahead():
                self.motors.stop()
                logger.warning("⚠️ Obstacle! Pausing continuous movement...")
                # Wait until obstacle clears
                while self._continuous_mode and self.sensors.is_obstacle_ahead():
                    time.sleep(0.2)
                if self._continuous_mode:
                    logger.info("✅ Obstacle cleared, resuming movement")
                    move_fn()
            time.sleep(0.1)

        self.motors.stop()

    # ═══════════════════════════════════════════
    # Patrol Mode (back and forth)
    # ═══════════════════════════════════════════

    def start_patrol(self, forward_duration: float = 3.0, pause: float = 1.0):
        """
        Patrol back and forth: forward → pause → backward → pause → repeat.
        Stops on obstacle or when stop_continuous() is called.
        """
        self.stop_continuous()
        if self._follow_mode:
            self.stop_follow()

        self._continuous_mode = True
        self._continuous_thread = threading.Thread(
            target=self._patrol_loop, args=(forward_duration, pause), daemon=True
        )
        self._continuous_thread.start()
        logger.info("🔄 Patrol mode STARTED")

    def _patrol_loop(self, fwd_dur: float, pause: float):
        """Background loop for patrol movement."""
        while self._continuous_mode:
            # Forward leg
            logger.info("🔄 Patrol: moving forward")
            self.motors.forward()
            elapsed = 0.0
            while elapsed < fwd_dur and self._continuous_mode:
                if self.sensors.is_obstacle_ahead():
                    self.motors.stop()
                    logger.warning("⚠️ Patrol: obstacle during forward, reversing early")
                    break
                time.sleep(0.1)
                elapsed += 0.1
            self.motors.stop()

            if not self._continuous_mode:
                break

            # Pause
            time.sleep(pause)
            if not self._continuous_mode:
                break

            # Backward leg
            logger.info("🔄 Patrol: moving backward")
            self.motors.backward()
            elapsed = 0.0
            while elapsed < fwd_dur and self._continuous_mode:
                time.sleep(0.1)
                elapsed += 0.1
            self.motors.stop()

            if not self._continuous_mode:
                break

            # Pause
            time.sleep(pause)

        self.motors.stop()
        self._continuous_mode = False

    # ═══════════════════════════════════════════
    # Cleanup
    # ═══════════════════════════════════════════

    def cleanup(self):
        """Stop all movement and clean up."""
        self.stop_continuous()
        self.stop_follow()
        self.motors.stop()
        logger.info("NavigationController cleaned up")
