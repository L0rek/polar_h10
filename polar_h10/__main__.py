import argparse
import asyncio
import contextlib
import logging
import traceback
from datetime import datetime
from typing import Dict

from bleak import BleakScanner

try:
    from .core import NotificationType, PolarH10
except ImportError:
    from core import NotificationType, PolarH10


async def find_device_by_name(name: str = "Polar H10", scan_time: float = 10.0) -> str:
    """
    Scan continuously for ``scan_time`` seconds and return discovered devices with specific name.

    Args:
        name: Device name.
        scan_time: Bluetooth scan timeout.

    Returns:
        Device address if was found

    """
    print(f"Scanning for BLE devices for {scan_time} seconds...")
    devices = await BleakScanner.discover(timeout=scan_time)
    for d in devices:
        if d.name and name in d.name:
            print(f"Found device: {d.name} ({d.address})")
            return d.address
    return None


class PolarH10Demo:
    """Polar H10 demo class."""

    def __init__(self, hr: bool, ecg_file: str, acc_file: str) -> None:
        self._hr = hr
        self._ecg_file_name = ecg_file
        self._acc_file_name = acc_file
        self._device = None
        self._ecg_file = None
        self._acc_file = None

    async def _register_notification(self) -> None:
        if self._hr:
            await self._device.register_notification(NotificationType.HEAR_RATE, self._print_hr)

        if self._ecg_file_name:
            await self._device.register_notification(NotificationType.ECG, self._save_ecg_data)

        if self._acc_file_name:
            await self._device.register_notification(NotificationType.ACC, self._save_acc_data)

    async def connect(self, addr: str) -> None:
        """Connect to device with specific address."""
        print(f"Connecting to the Polar H10 ({addr})...")
        self._device = PolarH10(addr)
        await self._device.connect()

        print(f"Battery Level: {self._device.battery_level}%")

        print("Device Information:")
        for key, value in self._device.device_information.items():
            print(f"    {key}: {value}")

        print("Device available configuration:")
        for notification, config in self._device.available_configs.items():
            print(f"    {notification.name}:")
            if not config:
                print("        ---")
                continue
            for name, value in config.items():
                if isinstance(value, list):
                    print(f"        {name}: {', '.join([str(i) for i in value])}")
                else:
                    print(f"        {name}: {value}")

        await self._register_notification()

        print("Current configuration:")
        for notification, config in self._device.current_config.items():
            print(f"    {notification.name}:")
            if not config:
                print("        ---")
                continue
            for name, value in config.items():
                print(f"        {name}: {value}")

    def _save_ecg_data(self, data: Dict) -> None:
        if not self._ecg_file:
            self._ecg_file = open(f"{self._ecg_file_name}.csv", "w")  # noqa: SIM115
            self._ecg_file.write("Time [ns], Lead 1 [uV]\n")
        for t, lead in zip(*data.values()):
            self._ecg_file.write(f"{t}, {lead}\n")

    def _save_acc_data(self, data: Dict) -> None:
        if not self._acc_file:
            self._acc_file = open(f"{self._acc_file_name}.csv", "w")  # noqa: SIM115
            self._acc_file.write("Time [ns], X Axis [mG], Y Axis [mG], Z Axis [mG]\n")
        for t, *axis in zip(*data.values()):
            self._acc_file.write(f"{t}, {', '.join([str(i) for i in axis])}\n")

    def _print_hr(self, data: Dict) -> None:
        line = datetime.now().astimezone().strftime("%H:%M:%S.%f")[:-3]
        line += ",  "
        for key, val in data.items():
            if key == "rr_interval":
                line += f"{key}: [{', '.join([str(i) for i in val])}]"
            else:
                line += f"{key}: {val},  "
        print(line)

    async def disconnect(self) -> None:
        """Disconnect and close files."""
        if self._device:
            await self._device.disconnect()
            print("Disconnected.")
            if self._ecg_file:
                self._ecg_file.close()
            if self._acc_file:
                self._acc_file.close()


async def run(args: argparse.Namespace) -> None:
    """Start asyncio loop."""
    demo = PolarH10Demo(args.hr, args.ecg_file, args.acc_file)
    try:
        if not args.device:
            addr = await find_device_by_name(args.name)
        else:
            addr = args.device

        if not addr:
            print("Device not found")
            return

        await demo.connect(addr)

        print("Recording started. Press Ctrl+C to exit.")
        while True:
            await asyncio.sleep(1)

    except Exception as e:
        print(f"Error {e!s}")
        print(traceback.format_exc())
    finally:
        if demo:
            await demo.disconnect()


def main() -> None:
    """Entry demo point."""
    logging.basicConfig(level=logging.ERROR)

    parser = argparse.ArgumentParser(
        prog="Polar H10 demo.",
        description="Example program using the Polar H10 module",
    )

    parser.add_argument(
        "--hr",
        action="store_true",
        help="Enables printing of the heart rate to the console.",
    )
    parser.add_argument("-e", "--ecg-file", help="Save ecg data to file.")
    parser.add_argument("-a", "--acc-file", help="Save accelerometer data to file.")
    parser.add_argument("-n", "--name", default="Polar H10", help="Set device name")
    parser.add_argument("-d", "--device", help="Set device address")

    try:
        asyncio.run(run(parser.parse_args()))
    except KeyboardInterrupt:
        contextlib.suppress(KeyboardInterrupt)


if __name__ == "__main__":
    main()
