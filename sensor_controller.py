"""
ECHO Robot — Sensor Controller
================================
Manages ultrasonic (HC-SR04) and IR sensors for obstacle detection.

Ultrasonic: Measures distance in cm using TRIG/ECHO pulses.
IR: Digital obstacle detection (active-low or active-high configurable).
"""

import time
import logging
import threading

logger = logging.getLogger("echo.sensors")

try:
    import RPi.GPIO as GPIO
    RPI_AVAILABLE = True
except ImportError:
    RPI_AVAILABLE = False
    logger.warning("RPi.GPIO not available — sensors will return simulated data")

from config import (
    ULTRASONIC_TRIG, ULTRASONIC_ECHO, ULTRASONIC_TIMEOUT,
    IR_PIN, IR_OBSTACLE_ACTIVE_LOW,
    OBSTACLE_DISTANCE_CM,
)


class SensorController:
    """Reads ultrasonic distance and IR obstacle sensors."""

    def __init__(self):
        self._distance = 999.0  # cm — start assuming no obstacle
        self._ir_blocked = False
        self._running = False
        self._lock = threading.Lock()
        self._setup_gpio()
        logger.info("SensorController initialized")

    def _setup_gpio(self):
        """Configure GPIO pins for sensors."""
        if not RPI_AVAILABLE:
            return

        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)

        # Ultrasonic
        GPIO.setup(ULTRASONIC_TRIG, GPIO.OUT)
        GPIO.setup(ULTRASONIC_ECHO, GPIO.IN)
        GPIO.output(ULTRASONIC_TRIG, GPIO.LOW)

        # IR
        GPIO.setup(IR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        logger.info(
            f"Sensor pins: TRIG={ULTRASONIC_TRIG}, ECHO={ULTRASONIC_ECHO}, IR={IR_PIN}"
        )

    def read_distance(self) -> float:
        """
        Measure distance using HC-SR04 ultrasonic sensor.
        Returns distance in cm, or 999.0 on timeout/error.
        """
        if not RPI_AVAILABLE:
            return 999.0  # Simulated: no obstacle

        try:
            # Send 10µs trigger pulse
            GPIO.output(ULTRASONIC_TRIG, GPIO.HIGH)
            time.sleep(0.00001)  # 10 microseconds
            GPIO.output(ULTRASONIC_TRIG, GPIO.LOW)

            # Wait for echo to go HIGH (start of return pulse)
            start_time = time.time()
            timeout_start = start_time
            while GPIO.input(ULTRASONIC_ECHO) == GPIO.LOW:
                start_time = time.time()
                if start_time - timeout_start > ULTRASONIC_TIMEOUT:
                    return 999.0  # Timeout

            # Wait for echo to go LOW (end of return pulse)
            end_time = start_time
            while GPIO.input(ULTRASONIC_ECHO) == GPIO.HIGH:
                end_time = time.time()
                if end_time - start_time > ULTRASONIC_TIMEOUT:
                    return 999.0  # Timeout

            # Calculate distance: speed of sound = 34300 cm/s, divide by 2 for round trip
            elapsed = end_time - start_time
            distance = (elapsed * 34300) / 2.0

            return round(distance, 1)

        except Exception as e:
            logger.error(f"Ultrasonic read error: {e}")
            return 999.0

    def read_ir(self) -> bool:
        """
        Read IR obstacle sensor.
        Returns True if obstacle is detected.
        """
        if not RPI_AVAILABLE:
            return False  # Simulated: no obstacle

        try:
            value = GPIO.input(IR_PIN)
            if IR_OBSTACLE_ACTIVE_LOW:
                return value == GPIO.LOW  # LOW = obstacle detected
            else:
                return value == GPIO.HIGH  # HIGH = obstacle detected
        except Exception as e:
            logger.error(f"IR read error: {e}")
            return False

    def is_obstacle_ahead(self) -> bool:
        """Check if there's an obstacle using both sensors."""
        distance = self.read_distance()
        ir_blocked = self.read_ir()

        with self._lock:
            self._distance = distance
            self._ir_blocked = ir_blocked

        # Readings below 2cm are likely sensor noise — ignore them
        if distance < 2.0:
            distance = 999.0

        obstacle = distance < OBSTACLE_DISTANCE_CM or ir_blocked

        if obstacle:
            logger.debug(
                f"Obstacle detected! Distance: {distance}cm, IR: {ir_blocked}"
            )

        return obstacle

    @property
    def last_distance(self) -> float:
        """Get the most recent distance reading."""
        with self._lock:
            return self._distance

    @property
    def last_ir_blocked(self) -> bool:
        """Get the most recent IR reading."""
        with self._lock:
            return self._ir_blocked

    # ── Background Monitoring ──

    def start_monitoring(self, interval: float = 0.2):
        """Start a background thread that continuously reads sensors."""
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop, args=(interval,), daemon=True
        )
        self._thread.start()
        logger.info(f"Sensor monitoring started (interval={interval}s)")

    def stop_monitoring(self):
        """Stop the background sensor monitoring thread."""
        self._running = False
        logger.info("Sensor monitoring stopped")

    def _monitor_loop(self, interval: float):
        """Continuously poll sensors."""
        while self._running:
            self.is_obstacle_ahead()
            time.sleep(interval)

    # ── Cleanup ──

    def cleanup(self):
        """Stop monitoring and release resources."""
        self.stop_monitoring()
        logger.info("SensorController cleaned up")
