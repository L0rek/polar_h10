import asyncio
import logging
import time
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

from bleak import BleakClient, BleakError

from .enums import GattUUID, PMDCommand, PMDMeasurement, PMDStatus
from .utils import (
    deserialize_pmd_parameters,
    get_timestamps_list,
    process_acc_samples,
    process_ecg_samples,
    process_hr_samples,
    serialize_pmd_parameters,
)


class NotificationType(Enum):
    """Notification types for measurement data."""

    ECG = 0
    ACC = 2
    BATTERY_LVL = 101
    DISCONNECT = 102
    HEAR_RATE = 201


class PolarH10Error(Exception):
    """Base exception for Polar H10-related errors."""


class PolarH10:
    """
    Polar H10 BLE Heart Rate Monitor interface.

    This class handles BLE communication with a Polar H10 device, including
    device connection, notifications, and processing of various measurements.
    """

    def __init__(self, address: str) -> None:
        self._client = BleakClient(address, self._handle_disconnect)
        self._is_connected = False
        self._battery_level: Optional[int] = None
        self._device_info: Dict[str, str] = {}
        self._callbacks: Dict[Enum, Callable[[Dict], None]] = {}
        self._p_timestamp: Dict[Enum, int] = {}
        self._config: Dict[Enum, Dict[str, List[int]]] = {}
        self._current_config: Dict[Enum, Dict[str, int]] = {}
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(self.__class__.__name__)
        self._device_time_delta = 0

        self._config[NotificationType.HEAR_RATE] = {}
        self._config[NotificationType.DISCONNECT] = {}

    def _handle_disconnect(self, _client: BleakClient) -> None:
        """Handle device disconnection event."""
        self._is_connected = False

        self._logger.warning("Device unexpectedly disconnected")
        callback = self._callbacks.get(NotificationType.DISCONNECT)
        if callback:
            callback({"disconnect": None})

    def _handle_battery_level(self, _sender: int, data: bytearray) -> None:
        """Process battery level notification."""
        self._battery_level = data[0]
        callback = self._callbacks.get(NotificationType.BATTERY_LVL)
        if callback:
            callback({"battery_level": data[0]})

    def _handle_heart_rate_measurement(self, _sender: int, data: bytearray) -> None:
        """Process heart rate measurement notification."""
        hr_callback = self._callbacks.get(NotificationType.HEAR_RATE)
        if not hr_callback:
            return

        measurement = process_hr_samples(data)
        self._logger.debug("Heart rate update %s", str(measurement))
        if measurement:
            hr_callback(measurement)

    def _handle_pmd_data(self, _sender: int, data: bytearray) -> None:
        """Process incoming PMD data notification."""
        now = time.time_ns()
        now += int(time.localtime().tm_gmtoff * 1e9)

        measurement_type = PMDMeasurement(data[0])
        timestamp = int.from_bytes(data[1:9], "little")
        frame_type = data[9]
        samples = data[10:]

        if abs(now - timestamp - self._device_time_delta) > 1e9:
            dif = now - timestamp - self._device_time_delta
            self._device_time_delta += dif
            self._logger.info(f"Synch device time dt:{dif // 1e6} s")

        notify_type = NotificationType(measurement_type.value)
        callback = self._callbacks.get(notify_type)
        if not callback:
            return

        try:
            if measurement_type == PMDMeasurement.ECG:
                n_samples, processed = process_ecg_samples(frame_type, samples)
            elif measurement_type == PMDMeasurement.ACC:
                n_samples, processed = process_acc_samples(frame_type, samples)
            else:
                self._logger.warning("Unprocessed measurement type: %s", measurement_type)
                return
        except Exception as e:  # pylint: disable=broad-except
            self._logger.error("Error processing PMD data: %s", str(e))

        sample_rate = self._current_config[notify_type]["SAMPLE_RATE"]
        time_list = get_timestamps_list(
            n_samples,
            sample_rate,
            timestamp,
            self._p_timestamp[notify_type],
            self._device_time_delta,
        )
        self._p_timestamp[notify_type] = timestamp

        result = {"time": time_list, **processed}

        callback(result)

    async def _get_device_information(self) -> None:
        """Retrieve and store device information."""
        info_mapping = [
            (GattUUID.MANUFACTURER_NAME, "manufacturer"),
            (GattUUID.MODEL_NUMBER, "model"),
            (GattUUID.SERIAL_NUMBER, "serial_number"),
            (GattUUID.HARDWARE_REVISION, "hardware_version"),
            (GattUUID.FIRMWARE_REVISION, "firmware_version"),
            (GattUUID.SOFTWARE_REVISION, "software_version"),
            (GattUUID.SYSTEM_ID, "system_id"),
        ]
        self._device_info = {}
        for uuid, key in info_mapping:
            try:
                value = await self._client.read_gatt_char(uuid)
                if uuid == GattUUID.SYSTEM_ID:
                    self._device_info[key] = value.hex()
                else:
                    self._device_info[key] = value.decode().strip()
            except BleakError as e:
                self._logger.error("Failed to read %s: %s", key, e)
                self._device_info[key] = "Unknown"
        self._logger.debug("Device information retrieved")

    async def _pmd_control_point_request(
        self,
        command: PMDCommand,
        measurement_type: PMDMeasurement,
        parameters: bytes = b"",
    ) -> Tuple[PMDStatus, Dict[str, List[int]]]:
        """
        Execute a PMD Control Point command.

        Returns a tuple of the response status and any parameters retrieved.
        """
        response_event = asyncio.Event()
        response_status = PMDStatus.ERROR_INVALID_STATE
        response_params = {}

        def handle_response(_sender: int, data: bytearray) -> None:
            nonlocal response_status, response_params, response_event
            if data[1] != command.value or data[2] != measurement_type.value:
                return
            response_status = PMDStatus(data[3])
            if response_status == PMDStatus.SUCCESS:
                self._logger.debug("Received PMD response parameters.")
                response_params = deserialize_pmd_parameters(data[5:])
            response_event.set()

        try:
            async with self._lock:
                await self._client.start_notify(GattUUID.PMD_CONTROL_POINT, handle_response)
                payload = (
                    command.value.to_bytes(1, "little") + measurement_type.value.to_bytes(1, "little") + parameters
                )
                self._logger.debug("Sending payload: %s", payload)
                await self._client.write_gatt_char(GattUUID.PMD_CONTROL_POINT, payload, response=True)
                await asyncio.wait_for(response_event.wait(), timeout=5)
            return response_status, response_params
        except TimeoutError:
            self._logger.error("PMD Control Point response timeout")
            return PMDStatus.ERROR_INVALID_STATE, {}
        finally:
            await self._client.stop_notify(GattUUID.PMD_CONTROL_POINT)

    async def _enable_battery_notifications(self) -> None:
        """Enable battery level notifications."""
        try:
            data = await self._client.read_gatt_char(GattUUID.BATTERY_LEVEL)
            self._battery_level = data[0]
            await self._client.start_notify(GattUUID.BATTERY_LEVEL, self._handle_battery_level)
            self._config[NotificationType.BATTERY_LVL] = {}
        except BleakError as e:
            self._logger.error("Failed to read battery level: %s", e)
            raise PolarH10Error("Battery read failed") from e
        self._logger.debug("Battery notifications enabled")

    async def _enable_hr_notification(self) -> None:
        """Enable heart rate notifications."""
        await self._client.start_notify(
            GattUUID.HEART_RATE_MEASUREMENT,
            self._handle_heart_rate_measurement,
        )
        self._logger.info("Heart rate notifications enabled")

    async def _disable_hr_notification(self) -> None:
        """Disable heart rate notifications."""
        await self._client.stop_notify(GattUUID.HEART_RATE_MEASUREMENT)
        self._logger.info("Heart rate notifications enabled")

    async def _initialize_pmd_service(self) -> None:
        """Initialize the PMD Service and query supported measurements."""
        await self._client.start_notify(GattUUID.PMD_DATA, self._handle_pmd_data)
        self._logger.debug("PMD notifications enabled")
        for measurement in PMDMeasurement:
            status, config = await self._pmd_control_point_request(PMDCommand.GET_MEASUREMENT_SETTINGS, measurement)
            if status == PMDStatus.SUCCESS:
                self._config[NotificationType(measurement.value)] = config
                self._logger.debug("PMD %s config: %s", measurement.name, config)

    @staticmethod
    def _validate_pmd_parameters(
        notification_type: NotificationType,
        pmd_config: Dict[Enum, Dict[str, List[int]]],
        parameters: Dict[str, int],
    ) -> Optional[Dict[str, int]]:
        """
        Validate PMD parameters against the supported configuration.

        Raises:
            ValueError: If a parameter is not supported or invalid.

        """
        params = {}
        for k, v in pmd_config[notification_type].items():
            if isinstance(v, list):
                params[k] = v[-1]
            else:
                params[k] = v

        if not params:
            return None

        if not parameters:
            return params
        params.update(parameters)

        valid_params = {}
        for setting, value in params.items():
            if setting not in pmd_config[notification_type]:
                raise ValueError(f"Unsupported setting {setting} for {notification_type.name}")
            if value not in pmd_config[notification_type][setting]:
                raise ValueError(f"Invalid value {value} for {setting}")
            valid_params[setting] = value
        return valid_params

    @property
    def available_configs(self) -> Dict[Enum, Dict[str, List[int]]]:
        """Available configurable parameters."""
        return self._config

    @property
    def current_config(self) -> Dict[Enum, Dict[str, int]]:
        """Current config parameters."""
        return self._current_config

    @property
    def battery_level(self) -> Optional[int]:
        """Current battery level percentage."""
        return self._battery_level

    @property
    def device_information(self) -> Dict[str, str]:
        """Device information dictionary."""
        return self._device_info

    async def connect(self) -> None:
        """
        Connect to the Polar H10 device and initialize services.

        Raises:
            PolarH10Error: If connection fails or the device is unsupported.

        """
        if self._client.is_connected:
            self._logger.warning("Device is already connected")
            return
        self._logger.info("Connecting to device at %s", self._client.address)
        try:
            await self._client.connect()
            name = await self._client.read_gatt_char(GattUUID.DEVICE_NAME)
            device_name = name.decode().strip()
            if not device_name.startswith("Polar H10"):
                raise PolarH10Error(f"Unsupported device: {device_name}")
            self._logger.info("Connected to %s", device_name)
            self._is_connected = True
            await self._get_device_information()
            await self._enable_battery_notifications()
            await self._initialize_pmd_service()
        except BleakError as e:
            self._logger.error("Connection failed: %s", str(e))
            raise PolarH10Error("Connection failed") from e

    async def disconnect(self) -> None:
        """Disconnect from the device."""
        if self._client.is_connected:
            await self._client.disconnect()
            self._is_connected = False
            self._logger.info("Device disconnected")

    async def register_notification(
        self,
        notification_type: NotificationType,
        callback: Callable[[Dict], None],
        parameters: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Register a callback for PMD measurement data.

        Args:
            notification_type: The type of measurement to register.
            callback: The function to be called with measurement data.
            parameters: Optional parameters to adjust the measurement settings.

        Raises:
            ValueError: If the measurement type is unsupported or already registered.
            PolarH10Error: If the PMD command fails.

        """
        if notification_type not in self._config:
            raise ValueError(f"Unsupported measurement type {notification_type.name}")
        if notification_type in self._callbacks:
            raise ValueError("Notification already registered for this type")

        valid_params = self._validate_pmd_parameters(notification_type, self._config, parameters)

        self._callbacks[notification_type] = callback
        self._current_config[notification_type] = valid_params

        if notification_type == NotificationType.HEAR_RATE:
            await self._enable_hr_notification()
            return

        if notification_type.value >= 100:
            return

        try:
            self._p_timestamp[notification_type] = 0
            status, _ = await self._pmd_control_point_request(
                PMDCommand.START_MEASUREMENT,
                PMDMeasurement(notification_type.value),
                serialize_pmd_parameters(valid_params),
            )
            if status != PMDStatus.SUCCESS:
                raise PolarH10Error(f"PMD command failed: {status.name}")
            self._logger.info("Notifications enabled %s", notification_type.name)
        except Exception as e:
            del self._callbacks[notification_type]
            del self._current_config[notification_type]
            raise e

    async def remove_notification(self, notification_type: NotificationType) -> None:
        """
        Register a callback for PMD measurement data.

        Args:
            notification_type: The type of measurement to register.

        Raises:
            ValueError: If the measurement type is unsupported or not registered.
            PolarH10Error: If the PMD command fails.

        """
        if notification_type not in self._config:
            raise ValueError(f"Unsupported measurement type {notification_type.name}")
        if notification_type not in self._callbacks:
            raise ValueError("Notification not registered for this type")

        if notification_type == NotificationType.HEAR_RATE:
            await self._enable_hr_notification()
            return

        if notification_type.value >= 100:
            return

        try:
            status, _ = await self._pmd_control_point_request(
                PMDCommand.STOP_MEASUREMENT,
                PMDMeasurement(notification_type.value),
            )
            if status != PMDStatus.SUCCESS:
                raise PolarH10Error(f"PMD command failed: {status.name}")
            del self._callbacks[notification_type]
            del self._current_config[notification_type]
            self._logger.info("Notifications disable %s", notification_type.name)
        except Exception as e:
            raise e
