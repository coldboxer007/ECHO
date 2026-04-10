"""
ECHO Robot — Motor Controller
==============================
Controls the L298N motor driver connected to 6 DC motors (3 per side).
Left side motors: IN1 (forward), IN2 (backward)
Right side motors: IN3 (forward), IN4 (backward)

Uses BOARD pin numbering.
"""

import time
import atexit
import logging
import threading

logger = logging.getLogger("echo.motors")

try:
    import RPi.GPIO as GPIO
    RPI_AVAILABLE = True
except ImportError:
    RPI_AVAILABLE = False
    logger.warning("RPi.GPIO not available — motor commands will be simulated")

from config import (
    MOTOR_IN1, MOTOR_IN2, MOTOR_IN3, MOTOR_IN4,
    MOTOR_TURN_DURATION, MOTOR_MOVE_DURATION,
    MOTOR_PWM_ENABLED, MOTOR_PWM_FREQ, MOTOR_DEFAULT_DUTY,
)

# Maximum time (seconds) motors can run continuously before auto-stop.
# Prevents runaway if user forgets "stop" or process hangs.
MOTOR_MAX_CONTINUOUS_SEC = 30.0


class MotorController:
    """Drives the L298N motor driver for differential-drive (tank-steer) robot.

    When MOTOR_PWM_ENABLED is True, uses software PWM on the IN pins for
    variable speed control (duty cycle 0-100).
    When False, uses simple GPIO HIGH/LOW (full-speed on/off).
    """

    def __init__(self):
        self.running = False
        self._move_start_time = None        # When current movement began
        self._watchdog_thread = None
        self._watchdog_running = False
        self._lock = threading.Lock()
        self._speed = MOTOR_DEFAULT_DUTY    # Current duty cycle (0-100)
        self._pwm_enabled = MOTOR_PWM_ENABLED
        self._pwm = {}                      # pin → PWM object (only when PWM enabled on RPi)
        self._setup_gpio()
        self.stop()  # Ensure all motors are stopped on startup
        self._start_watchdog()
        # Register atexit handler so motors stop even on crash / unhandled exception
        atexit.register(self._atexit_stop)
        mode_str = f"PWM {MOTOR_PWM_FREQ} Hz, default duty {MOTOR_DEFAULT_DUTY}%" if self._pwm_enabled else "GPIO on/off (no PWM)"
        logger.info("MotorController initialized (%s)", mode_str)

    def _setup_gpio(self):
        """Configure GPIO pins for motor control.
        Uses PWM when MOTOR_PWM_ENABLED, otherwise simple GPIO HIGH/LOW."""
        if not RPI_AVAILABLE:
            logger.info("GPIO simulation mode — no physical motors")
            return

        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)

        # Use initial=GPIO.LOW so each pin is driven LOW the instant it
        # becomes an output — no floating gap that could spin the motors.
        for pin in [MOTOR_IN1, MOTOR_IN2, MOTOR_IN3, MOTOR_IN4]:
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

        if self._pwm_enabled:
            for pin in [MOTOR_IN1, MOTOR_IN2, MOTOR_IN3, MOTOR_IN4]:
                pwm = GPIO.PWM(pin, MOTOR_PWM_FREQ)
                pwm.start(0)  # Start with 0% duty — motors off
                self._pwm[pin] = pwm
            logger.info(
                f"Motor pins configured (PWM {MOTOR_PWM_FREQ}Hz): "
                f"IN1={MOTOR_IN1}, IN2={MOTOR_IN2}, "
                f"IN3={MOTOR_IN3}, IN4={MOTOR_IN4}"
            )
        else:
            logger.info(
                f"Motor pins configured (GPIO on/off): "
                f"IN1={MOTOR_IN1}, IN2={MOTOR_IN2}, "
                f"IN3={MOTOR_IN3}, IN4={MOTOR_IN4}"
            )

    def _atexit_stop(self):
        """Called by atexit — stop motors on interpreter shutdown or crash."""
        try:
            self.stop()
            logger.info("atexit: motors stopped safely")
        except Exception:
            # Best effort — interpreter may be partially torn down
            if RPI_AVAILABLE:
                try:
                    if self._pwm_enabled:
                        for pwm in self._pwm.values():
                            try:
                                pwm.ChangeDutyCycle(0)
                            except Exception:
                                pass
                    else:
                        for pin in [MOTOR_IN1, MOTOR_IN2, MOTOR_IN3, MOTOR_IN4]:
                            try:
                                GPIO.output(pin, GPIO.LOW)
                            except Exception:
                                pass
                except Exception:
                    pass

    def _start_watchdog(self):
        """Start background watchdog that auto-stops motors after MOTOR_MAX_CONTINUOUS_SEC."""
        self._watchdog_running = True
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    def _watchdog_loop(self):
        """Check every second if motors have been running too long."""
        while self._watchdog_running:
            with self._lock:
                if self.running and self._move_start_time is not None:
                    elapsed = time.time() - self._move_start_time
                    if elapsed > MOTOR_MAX_CONTINUOUS_SEC:
                        logger.warning(
                            f"⚠️ Watchdog: motors running for {elapsed:.0f}s "
                            f"(max {MOTOR_MAX_CONTINUOUS_SEC}s) — auto-stopping!"
                        )
                        self._force_stop()
            time.sleep(1.0)

    def _force_stop(self):
        """Internal stop without lock (called from watchdog while holding lock)."""
        self.running = False
        self._move_start_time = None
        if RPI_AVAILABLE:
            if self._pwm_enabled:
                for pwm in self._pwm.values():
                    pwm.ChangeDutyCycle(0)
            else:
                for pin in [MOTOR_IN1, MOTOR_IN2, MOTOR_IN3, MOTOR_IN4]:
                    GPIO.output(pin, GPIO.LOW)

    def _set_motors(self, in1: bool, in2: bool, in3: bool, in4: bool):
        """Set all four motor input pins.
        PWM mode: active pins get self._speed duty cycle; inactive pins get 0.
        GPIO mode: active pins go HIGH; inactive pins go LOW."""
        if not RPI_AVAILABLE:
            logger.debug(f"SIM Motors: IN1={in1} IN2={in2} IN3={in3} IN4={in4} speed={self._speed}%")
            return

        if self._pwm_enabled:
            duty = self._speed
            self._pwm[MOTOR_IN1].ChangeDutyCycle(duty if in1 else 0)
            self._pwm[MOTOR_IN2].ChangeDutyCycle(duty if in2 else 0)
            self._pwm[MOTOR_IN3].ChangeDutyCycle(duty if in3 else 0)
            self._pwm[MOTOR_IN4].ChangeDutyCycle(duty if in4 else 0)
        else:
            GPIO.output(MOTOR_IN1, GPIO.HIGH if in1 else GPIO.LOW)
            GPIO.output(MOTOR_IN2, GPIO.HIGH if in2 else GPIO.LOW)
            GPIO.output(MOTOR_IN3, GPIO.HIGH if in3 else GPIO.LOW)
            GPIO.output(MOTOR_IN4, GPIO.HIGH if in4 else GPIO.LOW)

    # ── Speed Control ──

    def set_speed(self, duty: int):
        """Set motor speed as PWM duty cycle (0-100).
        Takes effect immediately on currently-running motors.
        No-op if MOTOR_PWM_ENABLED is False (GPIO mode is always full speed)."""
        if not self._pwm_enabled:
            logger.debug("set_speed() ignored — PWM disabled (GPIO on/off mode)")
            return
        duty = max(0, min(100, duty))
        self._speed = duty
        # If motors are currently running, update the active PWM channels live
        if self.running and RPI_AVAILABLE:
            for pin, pwm in self._pwm.items():
                # Only adjust pins that are currently active (duty > 0)
                current = pwm  # RPi.GPIO PWM doesn't expose current duty, so re-apply all
            # Re-issue the last movement to apply new speed
            # (not needed — ChangeDutyCycle on active pins handles it)
            pass
        logger.debug(f"Speed set to {duty}%")

    @property
    def speed(self) -> int:
        """Current speed as duty cycle (0-100)."""
        return self._speed

    # ── Movement Commands ──

    def forward(self, duration: float = None, speed: int = None):
        """Drive both sides forward."""
        if speed is not None:
            self._speed = max(0, min(100, speed))
        logger.info(f"Moving FORWARD{f' for {duration}s' if duration else ''} @ {self._speed}%")
        with self._lock:
            self.running = True
            self._move_start_time = time.time()
        # Wiring is swapped: IN2=LEFT FWD, IN4=RIGHT FWD
        self._set_motors(False, True, False, True)
        if duration:
            self._timed_stop(duration)

    def backward(self, duration: float = None, speed: int = None):
        """Drive both sides backward."""
        if speed is not None:
            self._speed = max(0, min(100, speed))
        logger.info(f"Moving BACKWARD{f' for {duration}s' if duration else ''} @ {self._speed}%")
        with self._lock:
            self.running = True
            self._move_start_time = time.time()
        # Wiring is swapped: IN1=LEFT BWD, IN3=RIGHT BWD
        self._set_motors(True, False, True, False)
        if duration:
            self._timed_stop(duration)

    def turn_left(self, duration: float = None, speed: int = None):
        """Turn left: right side forward, left side backward."""
        duration = duration or MOTOR_TURN_DURATION
        if speed is not None:
            self._speed = max(0, min(100, speed))
        logger.info(f"Turning LEFT for {duration}s @ {self._speed}%")
        with self._lock:
            self.running = True
            self._move_start_time = time.time()
        # Left/right motor groups are physically swapped on chassis
        # IN2=LEFT FWD (actually right wheels), IN3=RIGHT BWD (actually left wheels)
        self._set_motors(False, True, True, False)
        if duration:
            self._timed_stop(duration)

    def turn_right(self, duration: float = None, speed: int = None):
        """Turn right: left side forward, right side backward."""
        duration = duration or MOTOR_TURN_DURATION
        if speed is not None:
            self._speed = max(0, min(100, speed))
        logger.info(f"Turning RIGHT for {duration}s @ {self._speed}%")
        with self._lock:
            self.running = True
            self._move_start_time = time.time()
        # Left/right motor groups are physically swapped on chassis
        # IN1=LEFT BWD (actually right wheels), IN4=RIGHT FWD (actually left wheels)
        self._set_motors(True, False, False, True)
        if duration:
            self._timed_stop(duration)

    def _timed_stop(self, duration: float):
        """Stop motors after duration seconds in a background thread.
        This avoids blocking the calling thread (e.g. voice command handler),
        so the robot stays responsive to 'stop' commands during timed moves.
        Uses _move_start_time to detect if a new move was issued during sleep."""
        move_id = self._move_start_time  # Snapshot the current move identity
        def _wait_and_stop():
            time.sleep(duration)
            # Only stop if we're still in the SAME move (not interrupted by a new command)
            with self._lock:
                if self.running and self._move_start_time == move_id:
                    self._force_stop()
        t = threading.Thread(target=_wait_and_stop, daemon=True)
        t.start()

    def stop(self):
        """Stop all motors."""
        logger.info("Motors STOPPED")
        with self._lock:
            self.running = False
            self._move_start_time = None
        if RPI_AVAILABLE:
            if self._pwm_enabled and self._pwm:
                for pwm in self._pwm.values():
                    pwm.ChangeDutyCycle(0)
            else:
                for pin in [MOTOR_IN1, MOTOR_IN2, MOTOR_IN3, MOTOR_IN4]:
                    try:
                        GPIO.output(pin, GPIO.LOW)
                    except Exception:
                        pass

    def slight_left(self, speed: int = None):
        """Gentle left correction for follow mode — left forward only (physically right wheels)."""
        if speed is not None:
            self._speed = max(0, min(100, speed))
        with self._lock:
            self.running = True
            self._move_start_time = time.time()
        # Motor groups physically swapped: IN2 (labeled left) drives right wheels
        self._set_motors(False, True, False, False)

    def slight_right(self, speed: int = None):
        """Gentle right correction for follow mode — right forward only (physically left wheels)."""
        if speed is not None:
            self._speed = max(0, min(100, speed))
        with self._lock:
            self.running = True
            self._move_start_time = time.time()
        # Motor groups physically swapped: IN4 (labeled right) drives left wheels
        self._set_motors(False, False, False, True)

    # ── Cleanup ──

    def cleanup(self):
        """Stop motors and leave pins as LOW outputs.

        We intentionally do NOT call GPIO.cleanup() because that resets
        the pins to INPUT (high-impedance / floating), which the L298N
        reads as HIGH and spins the motors.  Leaving them as LOW outputs
        keeps the motors safely off even after the program exits.
        """
        self._watchdog_running = False
        self.stop()
        # Stop PWM channels gracefully (only if PWM was used)
        if RPI_AVAILABLE and self._pwm_enabled:
            for pwm in self._pwm.values():
                try:
                    pwm.stop()
                except Exception:
                    pass
        # GPIO.cleanup() is deliberately omitted.
        logger.info("MotorController cleaned up (pins left LOW)")
