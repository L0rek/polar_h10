from typing import Dict, Iterable, List, Optional, Tuple, Union

from .enums import PMDSetting


def iter_batched(iterable: Iterable, n: int, strict: bool = False) -> Iterable:
    """Batch data from the iterable into iterable of length n. The last batch may be shorter than n."""
    if n < 1:
        raise ValueError("n must be at least one")
    iterable_len = len(iterable)
    for idx in range(0, iterable_len, n):
        batch = iterable[idx : min(idx + n, iterable_len)]
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


def serialize_pmd_parameters(parameters: Dict[str, Union[List[int], int]]) -> bytes:
    """
    Serialize PMD parameters to bytes.

    Args:
        parameters: A dictionary of setting to integer values.

    Returns:
        The serialized parameters as a bytes object.

    """
    data = bytearray()
    for setting, value in parameters.items():
        data += PMDSetting[setting].value.to_bytes(1, "little")
        if isinstance(value, list):
            data += len(value).to_bytes(1, "little")
            for x in value:
                data += x.to_bytes(get_setting_size(setting), "little")
        else:
            data += b"\x01"
            data += value.to_bytes(get_setting_size(setting), "little")

    return bytes(data)


def deserialize_pmd_parameters(data: bytes) -> Union[List[int], int]:
    """
    Convert PMD parameter bytes into a configuration dictionary.

    Args:
        data: Bytes representing the PMD settings.

    Returns:
        A dictionary mapping each setting to a list of its values.

    """
    params = {}
    offset = 0
    while offset + 2 < len(data):
        setting_type = PMDSetting(data[offset])
        offset += 1
        count = data[offset]
        offset += 1
        value_size = get_setting_size(setting_type)
        value = []
        if count == 1:
            value = int.from_bytes(data[offset : offset + value_size], "little")
            offset += value_size
        else:
            for _ in range(count):
                if offset + value_size > len(data):
                    break
                tmp = int.from_bytes(data[offset : offset + value_size], "little")
                value.append(tmp)
                offset += value_size

        params[setting_type.name] = value
    return params


def get_timestamps_list(n_samples: int, sample_rate: int, timestamp: int, p_timestamp: int = 0, delta: int = 0) -> list:
    """
    Generate timestamps list.

    Args:
        n_samples: Number of samples
        sample_rate: Sample rate
        timestamp: Current frame timestamp
        p_timestamp: Previous frame timestamp (optional)
        delta: difference between device and host time (optional)

    Returns:
       A list of timestamps for samples.

    """
    if n_samples <= 0:
        raise ValueError("Number of samples must be grater than zero")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be grater than zero")

    if timestamp < p_timestamp:
        raise ValueError("Current timestamp must be grater than previous")

    timestamp_delta = int(1e9 // sample_rate)
    calculate_timestamp_delta = (timestamp - p_timestamp) // n_samples

    # checking if we have lost any samples
    if abs((calculate_timestamp_delta - timestamp_delta) / timestamp_delta) > 0.1:
        p_timestamp = timestamp - (n_samples * timestamp_delta)
    else:
        timestamp_delta = calculate_timestamp_delta

    p_timestamp += timestamp_delta
    return [p_timestamp + i * timestamp_delta + delta for i in range(n_samples)]


def process_ecg_samples(frame_type: int, data: bytes) -> Tuple[int, Dict[str, list]]:
    """
    Process ECG samples (3 bytes per sample).

    Args:
        frame_type: The frame type indicator.
        data: The raw bytes of ECG data.

    Returns:
        A tuple number of samples and dictionary with the key "lead_1"
        mapping to a list of processed sample values.

    """
    if frame_type != 0:
        raise ValueError("Unsupported ecg frame type")

    result = {"lead_1": []}
    sample_size = 3
    n_samples = 0

    for sample in iter_batched(data, sample_size, True):
        n_samples += 1
        result["lead_1"].append(int.from_bytes(sample, "little", signed=True))

    return n_samples, result


def process_acc_samples(frame_type: int, data: bytes) -> Tuple[int, Dict[str, list]]:
    """
    Process accelerometer samples.

    Args:
        frame_type: The frame type which determines sample size.
        data: The raw bytes of accelerometer data.

    Returns:
        A tuple number of samples and dictionary with keys 'x', 'y', and 'z'
        mapping to their sample lists.

    """
    if frame_type < 0 or frame_type > 2:
        raise ValueError("Unsupported acc frame type")

    result = {"x": [], "y": [], "z": []}
    sample_size = frame_type + 1
    n_samples = 0

    for frame in iter_batched(data, sample_size * 3, True):
        n_samples += 1
        for axis, sample in zip(result.keys(), iter_batched(frame, sample_size, True)):
            result[axis].append(int.from_bytes(sample, "little", signed=True))

    return n_samples, result


def process_hr_samples(data: bytes) -> Optional[Dict]:
    """
    Process heart rate samples.

    Args:
        data: The raw bytes of heart rate data.

    Returns:
        A dictionary containing heart rate and, if available, additional metrics.

    """
    if len(data) < 2:
        return None
    idx = 0
    flags = data[0]
    idx += 1
    sample = {}
    if flags & 0x01:
        sample["heart_rate"] = int.from_bytes(data[idx : idx + 2], "little")
        idx += 2
    else:
        sample["heart_rate"] = data[idx]
        idx += 1
    if flags & 0x04:
        sample["sensor_contact"] = bool(flags & 0x02)
    if flags & 0x08:
        sample["energy_expended"] = int.from_bytes(data[idx : idx + 2], "little")
        idx += 2
    if flags & 0x10:
        rr = []
        for val in iter_batched(data[idx:], 2, True):
            rr.append(int.from_bytes(val, "little") * 1000 // 1024)
        sample["rr_interval"] = rr
    return sample


def get_setting_size(setting: PMDSetting) -> int:
    """Return the byte size for the given PMD setting value."""
    return {
        PMDSetting.CHANNELS: 1,
        PMDSetting.FACTOR: 4,
        PMDSetting.RANGE_MILLIUNIT: 8,
    }.get(setting, 2)
