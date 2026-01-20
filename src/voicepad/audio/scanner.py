import logging
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import sounddevice as sd
import soundfile as sf

from voicepad.audio.utils import get_recording_path
from voicepad.config import get_config

logger = logging.getLogger(__name__)


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


def _capture_audio(
    stream: sd.InputStream,
    dev: AudioDevice,
    is_recording: list[bool],
    writer: sf.SoundFile,
) -> None:
    """Read audio in background thread and write directly to file."""
    try:
        with stream:
            logger.info(f"Recording from: {dev.name} ({dev.sample_rate}Hz)")

            while is_recording[0]:
                chunk, overflow = stream.read(int(dev.sample_rate))
                if overflow:
                    logger.warning("Audio overflow detected")

                # Write chunk immediately so transcriber can read growing file
                writer.write(chunk)

    except Exception as e:
        logger.error(f"Stream error: {e}")


def record_voice(
    device_index: int,
    output_file: Path | str | None = None,
) -> tuple[bytes, Path | None]:
    """
    Record audio continuously until user presses Enter.

    Writes audio directly to disk as it records so transcription can poll
    the growing file. Uses configured recordings path by default.

    Args:
        device_index: OS device index.
        output_file: Optional path to write audio incrementally. If None, uses configured recordings path.

    Returns:
        Tuple of (WAV bytes, output file path)
    """
    dev = get_device_by_index(device_index)
    is_recording = [True]

    # Decide output path
    if output_file:
        output_path = Path(output_file)
    else:
        config = get_config()
        config.recordings_path.mkdir(parents=True, exist_ok=True)
        output_path = get_recording_path(config.recordings_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    stream = sd.InputStream(
        device=device_index,
        samplerate=dev.sample_rate,
        channels=dev.channels,
        dtype="float32",
        blocksize=int(dev.sample_rate),
    )

    # Open writer for incremental writes
    writer = sf.SoundFile(
        file=str(output_path),
        mode="w",
        samplerate=dev.sample_rate,
        channels=dev.channels,
        format="WAV",
    )

    thread = threading.Thread(
        target=_capture_audio,
        args=(stream, dev, is_recording, writer),
        daemon=True,
    )
    thread.start()

    # Use sys.stdin.readline() instead of input() to avoid terminal clearing
    sys.stdin.readline()

    is_recording[0] = False
    thread.join(timeout=2)

    writer.close()

    # Read bytes to return
    data = output_path.read_bytes()

    return data, output_path
