"""
ECHO Robot — Motor Controller
==============================
Controls the L298N motor driver connected to 6 DC motors (3 per side).
Left side motors: IN1 (forward), IN2 (backward)
Right side motors: IN3 (forward), IN4 (backward)

Uses BOARD pin numbering.
"""

import time
import logging

logger = logging.getLogger("echo.motors")

try:
    import RPi.GPIO as GPIO
    RPI_AVAILABLE = True
except ImportError:
    RPI_AVAILABLE = False
    logger.warning("RPi.GPIO not available — motor commands will be simulated")

from config import (
    MOTOR_IN1, MOTOR_IN2, MOTOR_IN3, MOTOR_IN4,
    MOTOR_DEFAULT_SPEED, MOTOR_TURN_DURATION, MOTOR_MOVE_DURATION,
)


class MotorController:
    """Drives the L298N motor driver for differential-drive (tank-steer) robot."""

    def __init__(self):
        self.running = False
        self._setup_gpio()
        self.stop()  # Ensure all motors are stopped on startup
        logger.info("MotorController initialized")

    def _setup_gpio(self):
        """Configure GPIO pins for motor control."""
        if not RPI_AVAILABLE:
            logger.info("GPIO simulation mode — no physical motors")
            return

        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)

        # Use initial=GPIO.LOW so each pin is driven LOW the instant it
        # becomes an output — no floating gap that could spin the motors.
        for pin in [MOTOR_IN1, MOTOR_IN2, MOTOR_IN3, MOTOR_IN4]:
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

        logger.info(
            f"Motor pins configured: IN1={MOTOR_IN1}, IN2={MOTOR_IN2}, "
            f"IN3={MOTOR_IN3}, IN4={MOTOR_IN4}"
        )

    def _set_motors(self, in1: bool, in2: bool, in3: bool, in4: bool):
        """Set all four motor input pins."""
        if not RPI_AVAILABLE:
            logger.debug(f"SIM Motors: IN1={in1} IN2={in2} IN3={in3} IN4={in4}")
            return

        GPIO.output(MOTOR_IN1, GPIO.HIGH if in1 else GPIO.LOW)
        GPIO.output(MOTOR_IN2, GPIO.HIGH if in2 else GPIO.LOW)
        GPIO.output(MOTOR_IN3, GPIO.HIGH if in3 else GPIO.LOW)
        GPIO.output(MOTOR_IN4, GPIO.HIGH if in4 else GPIO.LOW)

    # ── Movement Commands ──

    def forward(self, duration: float = None):
        """Drive both sides forward."""
        logger.info(f"Moving FORWARD{f' for {duration}s' if duration else ''}")
        self.running = True
        # Wiring is swapped: IN2=LEFT FWD, IN4=RIGHT FWD
        self._set_motors(False, True, False, True)
        if duration:
            time.sleep(duration)
            self.stop()

    def backward(self, duration: float = None):
        """Drive both sides backward."""
        logger.info(f"Moving BACKWARD{f' for {duration}s' if duration else ''}")
        self.running = True
        # Wiring is swapped: IN1=LEFT BWD, IN3=RIGHT BWD
        self._set_motors(True, False, True, False)
        if duration:
            time.sleep(duration)
            self.stop()

    def turn_left(self, duration: float = None):
        """Turn left: right side forward, left side backward."""
        duration = duration or MOTOR_TURN_DURATION
        logger.info(f"Turning LEFT for {duration}s")
        self.running = True
        # Wiring swapped: IN1=LEFT BWD, IN4=RIGHT FWD
        self._set_motors(True, False, False, True)
        if duration:
            time.sleep(duration)
            self.stop()

    def turn_right(self, duration: float = None):
        """Turn right: left side forward, right side backward."""
        duration = duration or MOTOR_TURN_DURATION
        logger.info(f"Turning RIGHT for {duration}s")
        self.running = True
        # Wiring swapped: IN2=LEFT FWD, IN3=RIGHT BWD
        self._set_motors(False, True, True, False)
        if duration:
            time.sleep(duration)
            self.stop()

    def stop(self):
        """Stop all motors."""
        logger.info("Motors STOPPED")
        self.running = False
        self._set_motors(False, False, False, False)

    def slight_left(self):
        """Gentle left correction for follow mode — right forward only."""
        self._set_motors(False, False, False, True)

    def slight_right(self):
        """Gentle right correction for follow mode — left forward only."""
        self._set_motors(False, True, False, False)

    # ── Cleanup ──

    def cleanup(self):
        """Stop motors and leave pins as LOW outputs.

        We intentionally do NOT call GPIO.cleanup() because that resets
        the pins to INPUT (high-impedance / floating), which the L298N
        reads as HIGH and spins the motors.  Leaving them as LOW outputs
        keeps the motors safely off even after the program exits.
        """
        self.stop()
        # GPIO.cleanup() is deliberately omitted.
        logger.info("MotorController cleaned up (pins left LOW)")
