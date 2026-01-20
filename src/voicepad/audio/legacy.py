"""Evolution of recording functionality - documenting our development progress.

This module tracks the historical development of Voicepad's audio recording system,
showing how we progressed from simple device detection to sophisticated continuous
recording with VAD-based chunk processing.

Recording Evolution:
    1. Device Detection (Step 1)
       - Detect available audio input devices
       - Query device capabilities (channels, sample rate)

    2. Fixed Duration Recording (Step 2)
       - Record audio for a specified number of seconds
       - Simple blocking approach with sounddevice.rec()
       - Single file output

    3. Continuous Recording with Manual Stop (Step 3)
       - Record indefinitely until user presses Enter
       - Uses background thread for non-blocking capture
       - Rolling buffer to manage memory with bounded size

    4. Continuous Recording with VAD Chunking (Step 4) - CURRENT
       - Record entire session to single file
       - Process audio chunks in real-time using Voice Activity Detection
       - Temporary chunks for transcription, discarded after processing
       - Original recording always preserved intact
"""

import io
import logging
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
import soundfile as sf

from voicepad.audio.scanner import AudioDevice, get_device_by_index
from voicepad.audio.utils import get_recording_path
from voicepad.config import get_config

logger = logging.getLogger(__name__)


# ============================================================================
# STEP 1: DEVICE DETECTION
# ============================================================================


def legacy_detect_devices() -> list[AudioDevice]:
    """
    Step 1 - Basic device detection.

    Queries the OS for available audio input devices and their capabilities.
    This was the first step in the recording journey - knowing what hardware
    we have available.

    Returns:
        List of available AudioDevice objects with input capability.
    """
    from voicepad.audio.scanner import get_input_devices

    return get_input_devices()


# ============================================================================
# STEP 2: FIXED DURATION RECORDING
# ============================================================================


def legacy_record_voice_fixed_duration(device_index: int, duration: float) -> bytes:
    """
    Step 2 - Fixed duration recording (DEPRECATED).

    The first recording implementation. Records audio from a device for
    exactly N seconds using a blocking call. Simple but inflexible.

    Evolution Notes:
        - ✓ Works for fixed-length recordings
        - ✗ Cannot stop early
        - ✗ User must know duration beforehand
        - ✗ No flexibility for natural conversation pauses

    Args:
        device_index: OS device index to record from.
        duration: Recording duration in seconds.

    Returns:
        Bytes containing the encoded WAV audio.

    Raises:
        ValueError: If duration is not positive.
    """
    if duration <= 0:
        raise ValueError("duration must be positive")

    dev = get_device_by_index(device_index)

    frames = int(duration * dev.sample_rate)
    arr = sd.rec(
        frames,
        samplerate=dev.sample_rate,
        channels=dev.channels,
        dtype="float32",
        device=device_index,
    )
    sd.wait()

    buf = io.BytesIO()
    sf.write(buf, arr, samplerate=dev.sample_rate, format="WAV")
    data = buf.getvalue()

    # Save to disk with timestamp
    config = get_config()
    config.recordings_path.mkdir(parents=True, exist_ok=True)

    out_path = get_recording_path(config.recordings_path)
    out_path.write_bytes(data)

    return data


# ============================================================================
# STEP 3: CONTINUOUS RECORDING WITH MANUAL STOP
# ============================================================================


def legacy_capture_audio_background(
    stream: sd.InputStream,
    dev: AudioDevice,
    audio_buffer: deque,
    is_recording: list[bool],
    frames_received: list[int],
    last_stats: list[float],
) -> None:
    """
    Step 3a - Background audio capture thread (HELPER).

    Reads audio from a stream in a background thread without blocking
    the main thread. This pattern allowed the recording to be stopped
    by the user at any time.

    Evolution Notes:
        - Introduced threading to allow manual stop
        - Buffering for memory management
        - Real-time statistics reporting

    Args:
        stream: Open sounddevice input stream.
        dev: AudioDevice metadata.
        audio_buffer: Deque to accumulate audio samples.
        is_recording: List[bool] flag to signal stop (mutable for thread sync).
        frames_received: List[int] counter for total frames captured.
        last_stats: List[float] for tracking stats report timing.
    """
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


def legacy_record_voice_continuous(device_index: int, keep_duration_sec: float = 300) -> bytes:
    """
    Step 3b - Continuous recording with manual stop (PREDECESSOR OF CURRENT).

    Records audio indefinitely until user presses Enter. Uses a rolling buffer
    to keep memory usage bounded. This was a major improvement over fixed
    duration as it allowed natural stop points.

    Evolution Notes:
        - ✓ User can stop at any time
        - ✓ Memory management with rolling buffer
        - ✓ Background thread allows UI interaction
        - ✗ Data loss if recording exceeds buffer (rolling buffer overwrites)
        - ✗ No processing during recording

    Args:
        device_index: OS device index to record from.
        keep_duration_sec: Max buffer duration in seconds (default: 5 minutes).
                          Data older than this is discarded.

    Returns:
        Bytes containing the encoded WAV audio (last N seconds).
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
        target=legacy_capture_audio_background,
        args=(stream, dev, audio_buffer, is_recording, frames_received, last_stats),
        daemon=True,
    )
    thread.start()
    input()  # Wait for user to press Enter
    is_recording[0] = False
    thread.join(timeout=2)

    if not audio_buffer:
        return b""

    audio = np.array(list(audio_buffer), dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, audio, samplerate=dev.sample_rate, format="WAV")
    data = buf.getvalue()

    config = get_config()
    config.recordings_path.mkdir(parents=True, exist_ok=True)

    out_path = get_recording_path(config.recordings_path)
    out_path.write_bytes(data)
    print(f"✓ Saved: {out_path} ({len(audio) / dev.sample_rate:.1f}s)")

    return data


# ============================================================================
# STEP 4: CONTINUOUS RECORDING WITH VAD CHUNKING (CURRENT)
# ============================================================================
# See: voicepad/audio/vad_processor.py and voicepad/audio/session_manager.py
#
# Current Implementation:
#   - Record entire session to one single file (preserves full audio)
#   - Process chunks in real-time using VAD (Voice Activity Detection)
#   - Configurable wait duration (e.g., 2 minutes) before VAD processes
#   - Detect silence to create chunk boundaries
#   - Process and discard chunks (for transcription)
#   - Original recording always kept intact
#
# Evolution Notes:
#   ✓ Full session recording - no data loss
#   ✓ Real-time chunk processing for transcription
#   ✓ Configurable chunk timing
#   ✓ Preserves context across chunks
#   ✓ Parallel processes: recording + processing
#   ✓ Transcription-ready chunks with audio context
# ============================================================================
