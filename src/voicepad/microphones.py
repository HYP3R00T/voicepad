import logging
import subprocess

logger = logging.getLogger(__name__)


def _detect_sounddevice() -> list[dict]:
    """Detect microphones using sounddevice library."""
    try:
        import sounddevice as sd

        microphones: list[dict] = []
        devices = sd.query_devices()

        if isinstance(devices, dict):
            devices = [devices]

        for idx, device in enumerate(devices):
            if device.get("max_input_channels", 0) > 0:
                microphones.append({
                    "name": device.get("name", f"Microphone {idx}"),
                    "index": idx,
                    "channels": device.get("max_input_channels", 1),
                    "sample_rate": int(device.get("default_samplerate", 48000)),
                })

        return microphones
    except Exception as e:
        logger.debug(f"sounddevice detection failed: {e}")
        return []


def _detect_pulseaudio() -> list[dict]:
    """Detect microphones using PulseAudio."""
    try:
        result = subprocess.run(
            ["pactl", "list", "sources"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            return []

        microphones: list[dict] = []
        lines = result.stdout.split("\n")
        current_device: dict = {}

        for line in lines:
            line = line.strip()
            if line.startswith("Source #"):
                if current_device:
                    microphones.append(current_device)
                current_device = {"index": len(microphones)}
            elif line.startswith("device.description"):
                name = line.split("=", 1)[-1].strip().strip('"')
                current_device["name"] = name
            elif line.startswith("device.class"):
                device_class = line.split("=", 1)[-1].strip().strip('"')
                current_device["class"] = device_class

        if current_device:
            microphones.append(current_device)

        return microphones
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.debug(f"PulseAudio detection failed: {e}")
        return []


def _detect_alsa() -> list[dict]:
    """Detect microphones using ALSA."""
    try:
        result = subprocess.run(
            ["arecord", "-l"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            return []

        microphones: list[dict] = []
        lines = result.stdout.split("\n")

        for line in lines:
            line = line.strip()
            if line.startswith("card"):
                # Format: card 0: CARD_NAME [DESCRIPTION], device 0: DEVICE_NAME [DESCRIPTION]
                parts = line.split(":", 1)
                if len(parts) == 2:
                    card_name = parts[1].strip()
                    microphones.append({"name": card_name})

        return microphones
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.debug(f"ALSA detection failed: {e}")
        return []


def _detect_windows_wsl() -> list[dict]:
    """Detect microphones from Windows via WSL2 using WMI."""
    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                "Get-WmiObject -Class Win32_SoundDevice | Select-Object -ExpandProperty Name",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            microphones: list[dict] = []
            lines = result.stdout.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line:
                    microphones.append({"name": line})
            if microphones:
                return microphones
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.debug(f"Windows WMI detection failed: {e}")

    return []


def get_microphones() -> list[dict]:
    """Detect and return list of available microphones in the system.

    Tries multiple detection methods:
    1. sounddevice library
    2. PulseAudio (pactl)
    3. ALSA (arecord)
    4. /proc/asound/cards (Linux kernel info)
    5. Windows via WSL2 (powershell)

    Returns:
        List of dictionaries containing microphone information with keys:
        - name: Display name of the microphone
        - index: Device index (if available)
        - channels: Number of input channels (if available)
        - sample_rate: Default sample rate (if available)
    """
    microphones: list[dict] = []

    # Try detection methods in order
    microphones = _detect_sounddevice()
    if microphones:
        return microphones

    microphones = _detect_pulseaudio()
    if microphones:
        return microphones

    microphones = _detect_alsa()
    if microphones:
        return microphones

    microphones = _detect_windows_wsl()
    if microphones:
        return microphones

    # Fallback: return empty list with no microphones found
    return microphones
