"""
ECHO Robot — Battery Monitor (Stub)
======================================
Placeholder module for future battery voltage monitoring via ADC
(e.g., ADS1115 on I2C reading a voltage divider on the battery pack).

Currently provides:
  - Simulated voltage reading (always returns full charge)
  - Low-battery warning threshold
  - Periodic check loop that can be started as a background thread

To enable real monitoring:
  1. Wire a voltage divider from battery pack to ADS1115 channel 0
  2. pip install adafruit-circuitpython-ads1x15
  3. Uncomment the ADC code below and adjust DIVIDER_RATIO
"""

import logging
import threading
import time

logger = logging.getLogger("echo.battery")

# ─── Configuration ───
BATTERY_FULL_VOLTAGE = 12.6     # 3S LiPo fully charged (4.2V * 3)
BATTERY_LOW_VOLTAGE = 10.5      # ~3.5V per cell — time to stop moving
BATTERY_CRITICAL_VOLTAGE = 9.9  # ~3.3V per cell — shutdown recommended
BATTERY_CHECK_INTERVAL = 30.0   # Seconds between checks

# Voltage divider ratio: Vbattery = Vadc * DIVIDER_RATIO
# Example: 47kΩ + 10kΩ divider → ratio ≈ 5.7
DIVIDER_RATIO = 5.7

# ADC reference (ADS1115 default gain=1 → ±4.096V range)
ADC_VOLTAGE_REF = 4.096
ADC_MAX_VALUE = 32767  # 15-bit signed


class BatteryMonitor:
    """
    Monitors battery voltage and provides low-battery warnings.

    Currently a stub — returns simulated full-charge readings.
    Replace _read_voltage() with real ADC code when hardware is wired.
    """

    def __init__(self, on_low_battery=None, on_critical_battery=None):
        """
        Args:
            on_low_battery: Callback(voltage) when battery drops below LOW threshold
            on_critical_battery: Callback(voltage) when battery drops below CRITICAL threshold
        """
        self._on_low = on_low_battery
        self._on_critical = on_critical_battery
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._last_voltage = BATTERY_FULL_VOLTAGE

        # --- Uncomment for real ADS1115 ADC ---
        # try:
        #     import board
        #     import busio
        #     import adafruit_ads1x15.ads1115 as ADS
        #     from adafruit_ads1x15.analog_in import AnalogIn
        #     i2c = busio.I2C(board.SCL, board.SDA)
        #     ads = ADS.ADS1115(i2c)
        #     self._adc_channel = AnalogIn(ads, ADS.P0)
        #     self._adc_available = True
        #     logger.info("ADS1115 ADC initialized for battery monitoring")
        # except Exception as e:
        #     self._adc_channel = None
        #     self._adc_available = False
        #     logger.info(f"No ADC hardware — battery monitor in stub mode: {e}")
        self._adc_available = False

        logger.info("BatteryMonitor initialized (stub mode)")

    def _read_voltage(self) -> float:
        """
        Read battery voltage from ADC.
        Returns simulated full voltage when no ADC is connected.
        """
        if not self._adc_available:
            return BATTERY_FULL_VOLTAGE  # Stub: always full

        # --- Uncomment for real ADC reading ---
        # try:
        #     adc_voltage = self._adc_channel.voltage
        #     battery_voltage = adc_voltage * DIVIDER_RATIO
        #     return round(battery_voltage, 2)
        # except Exception as e:
        #     logger.error(f"ADC read error: {e}")
        #     return self._last_voltage
        return BATTERY_FULL_VOLTAGE

    @property
    def voltage(self) -> float:
        """Current battery voltage."""
        with self._lock:
            return self._last_voltage

    @property
    def percentage(self) -> float:
        """Estimated battery percentage (linear approximation)."""
        v = self.voltage
        if v >= BATTERY_FULL_VOLTAGE:
            return 100.0
        if v <= BATTERY_CRITICAL_VOLTAGE:
            return 0.0
        return ((v - BATTERY_CRITICAL_VOLTAGE) /
                (BATTERY_FULL_VOLTAGE - BATTERY_CRITICAL_VOLTAGE)) * 100.0

    @property
    def is_low(self) -> bool:
        return self.voltage < BATTERY_LOW_VOLTAGE

    @property
    def is_critical(self) -> bool:
        return self.voltage < BATTERY_CRITICAL_VOLTAGE

    def start_monitoring(self):
        """Start periodic battery checks in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info(f"Battery monitoring started (interval={BATTERY_CHECK_INTERVAL}s)")

    def _monitor_loop(self):
        """Background loop that checks battery voltage periodically."""
        while self._running:
            try:
                voltage = self._read_voltage()
                with self._lock:
                    self._last_voltage = voltage

                if voltage < BATTERY_CRITICAL_VOLTAGE:
                    logger.warning(f"CRITICAL battery: {voltage:.1f}V")
                    if self._on_critical:
                        self._on_critical(voltage)
                elif voltage < BATTERY_LOW_VOLTAGE:
                    logger.warning(f"Low battery: {voltage:.1f}V")
                    if self._on_low:
                        self._on_low(voltage)
                else:
                    logger.debug(f"Battery OK: {voltage:.1f}V ({self.percentage:.0f}%)")

            except Exception as e:
                logger.error(f"Battery monitor error: {e}")

            time.sleep(BATTERY_CHECK_INTERVAL)

    def cleanup(self):
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("BatteryMonitor cleaned up")
