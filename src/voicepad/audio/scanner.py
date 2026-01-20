import io
import logging
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict, cast

import numpy as np
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


def record_voice(device_index: int, duration: float) -> bytes:
    """
    Record audio from the given device for `duration` seconds.
    Saves to configured recordings path with timestamp.

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

    # Get config and create output path with timestamp
    config = get_config()
    config.recordings_path.mkdir(parents=True, exist_ok=True)

    out_path = get_recording_path(config.recordings_path)
    out_path.write_bytes(data)

    return data


def _capture_audio(
    stream: sd.InputStream,
    dev: AudioDevice,
    audio_buffer: deque,
    is_recording: list[bool],
    frames_received: list[int],
    last_stats: list[float],
) -> None:
    """Read audio in background thread."""
    try:
        with stream:
            logger.info(f"Recording from: {dev.name} ({dev.sample_rate}Hz)")
            print(f"Recording from: {dev.name} (Press Enter to stop)")

            while is_recording[0]:
                chunk, overflow = stream.read(int(dev.sample_rate))
                if overflow:
                    logger.warning("Audio overflow detected")

                for frame in chunk:
                    audio_buffer.append(frame)
                frames_received[0] += len(chunk)

                # Print stats every 5 seconds
                now = time.time()
                if now - last_stats[0] > 5:
                    buffer_sec = len(audio_buffer) / dev.sample_rate
                    total_sec = frames_received[0] / dev.sample_rate
                    mem_mb = (len(audio_buffer) * dev.channels * 4) / (1024 * 1024)
                    print(f"  Total: {total_sec:.0f}s | Buffer: {buffer_sec:.0f}s | Memory: {mem_mb:.1f}MB")
                    last_stats[0] = now

    except Exception as e:
        logger.error(f"Stream error: {e}")


def record_voice_continuous(device_index: int, keep_duration_sec: float = 300) -> bytes:
    """
    Continuously capture audio using a rolling buffer with bounded memory.

    Args:
        device_index: OS device index.
        keep_duration_sec: How many seconds of audio to keep (default: 5 minutes).

    Returns:
        WAV bytes of the last `keep_duration_sec` seconds of audio.
    """
    dev = get_device_by_index(device_index)
    max_frames = int(keep_duration_sec * dev.sample_rate)
    audio_buffer = deque(maxlen=max_frames)
    is_recording = [True]
    frames_received = [0]
    last_stats = [time.time()]

    stream = sd.InputStream(
        device=device_index,
        samplerate=dev.sample_rate,
        channels=dev.channels,
        dtype="float32",
        blocksize=int(dev.sample_rate),
    )

    thread = threading.Thread(
        target=_capture_audio,
        args=(stream, dev, audio_buffer, is_recording, frames_received, last_stats),
        daemon=True,
    )
    thread.start()
    input()
    is_recording[0] = False
    thread.join(timeout=2)

    if not audio_buffer:
        return b""

    audio = np.array(list(audio_buffer), dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, audio, samplerate=dev.sample_rate, format="WAV")
    data = buf.getvalue()

    # Get config and create output path with timestamp
    config = get_config()
    config.recordings_path.mkdir(parents=True, exist_ok=True)

    out_path = get_recording_path(config.recordings_path)
    out_path.write_bytes(data)
    print(f"✓ Saved: {out_path} ({len(audio) / dev.sample_rate:.1f}s)")

    return data
