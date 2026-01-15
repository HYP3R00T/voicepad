import io
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import sounddevice as sd
import soundfile as sf


@dataclass
class AudioDevice:
    """Represents a physical audio input device found on the system."""

    index: int
    name: str
    channels: int
    sample_rate: int

    def __str__(self):
        return f"[{self.index}] {self.name} ({self.channels} in, {self.sample_rate}Hz)"


def get_input_devices() -> list[AudioDevice]:
    """
    Queries the OS for available audio devices.
    Returns a list of devices that support at least 1 input channel.
    """
    devices: list[AudioDevice] = []

    class _DeviceInfo(TypedDict, total=False):
        """Typing for the dict-like objects returned by sounddevice.query_devices()."""

        name: str
        max_input_channels: int
        default_samplerate: float

    # sd.query_devices() returns a DeviceList, which behaves like a sequence of dicts.
    all_devices = cast(Sequence[_DeviceInfo], sd.query_devices())

    for idx, dev in enumerate(all_devices):
        # We only care about devices that can RECORD (input channels > 0)
        max_inputs = dev.get("max_input_channels") or 0
        if max_inputs > 0:
            # Clean up the name if needed (Windows sometimes adds API prefixes)
            clean_name = dev.get("name", f"Device {idx}")

            # sounddevice returns samplerate as float, cast to int for cleanliness
            default_sr = dev.get("default_samplerate", 0.0)
            try:
                rate = int(default_sr)  # type: ignore[arg-type]
                if rate <= 0:
                    raise ValueError("invalid sample rate")
            except (TypeError, ValueError):
                # Fallback to a sensible default if the device does not report a valid rate
                rate = 44100

            device = AudioDevice(index=idx, name=clean_name, channels=int(max_inputs), sample_rate=rate)
            devices.append(device)

    return devices


def get_device_by_index(index: int) -> AudioDevice:
    """Helper to fetch a specific device by its OS index."""
    devices = get_input_devices()
    for dev in devices:
        if dev.index == index:
            return dev
    raise ValueError(f"No input device found with index {index}")


def print_devices():
    devices = get_input_devices()
    for dev in devices:
        print(dev)


def record_voice(device_index: int, duration: float) -> bytes:
    """
    Record audio from the given device for `duration` seconds.
    Saves to project root as recording.wav and returns audio bytes.

    Args:
        device_index: OS device index.
        duration: Recording duration in seconds.

    Returns:
        Bytes containing the encoded audio.
    """
    if duration <= 0:
        raise ValueError("duration must be positive")

    dev = get_device_by_index(device_index)

    frames = int(duration * dev.sample_rate)
    arr = sd.rec(frames, samplerate=dev.sample_rate, channels=dev.channels, dtype="float32", device=device_index)
    sd.wait()

    buf = io.BytesIO()
    sf.write(buf, arr, samplerate=dev.sample_rate, format="WAV")
    data = buf.getvalue()

    project_root = Path(__file__).parents[3]
    out_path = project_root / "recording.wav"
    out_path.write_bytes(data)

    return data
