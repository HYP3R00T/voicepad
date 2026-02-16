"""Configuration management commands."""

import logging
from dataclasses import dataclass
from typing import TypedDict, cast

import sounddevice as sd
import typer
from voicepad_core import get_config, get_config_with_metadata

logger = logging.getLogger(__name__)

config_app = typer.Typer(help="Configuration management commands")


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    channels: int
    sample_rate: int

    def __str__(self) -> str:
        return f"[{self.index}] {self.name} ({self.channels} in, {self.sample_rate}Hz)"


def _get_input_devices() -> list[AudioDevice]:
    class _DeviceInfo(TypedDict, total=False):
        name: str
        max_input_channels: int
        default_samplerate: float

    devices: list[AudioDevice] = []
    all_devices = cast(list[_DeviceInfo], sd.query_devices())

    for idx, dev in enumerate(all_devices):
        max_inputs = dev.get("max_input_channels") or 0
        if max_inputs <= 0:
            continue

        name = dev.get("name", f"Device {idx}")
        default_rate = dev.get("default_samplerate", 0.0)
        rate = int(default_rate) if isinstance(default_rate, (int, float)) and default_rate > 0 else 44100
        devices.append(AudioDevice(index=idx, name=name, channels=int(max_inputs), sample_rate=rate))

    return devices


@config_app.command("input")
def list_input_devices() -> None:
    """List available audio input devices for configuration."""
    devices = _get_input_devices()
    if not devices:
        typer.secho("No audio input devices detected.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    config = get_config()
    configured_index = config.input_device_index

    typer.echo("Configured default input device:")
    if configured_index is None:
        typer.echo("  not set")
    else:
        try:
            configured_device = next(dev for dev in devices if dev.index == configured_index)
            typer.echo(f"  {configured_index}: {configured_device.name}")
        except StopIteration:
            typer.echo(f"  {configured_index}: not found")

    typer.echo("\nAvailable audio input devices:")
    typer.echo("-" * 60)
    for dev in devices:
        selected_marker = " (configured)" if dev.index == configured_index else ""
        typer.echo(f"{dev}{selected_marker}")
    typer.echo("-" * 60)

    _show_config_hint()


def _show_config_hint() -> None:
    config_file = None
    config_source = None

    _config, metadata = get_config_with_metadata()
    if metadata.per_field:
        first_field_source = next(iter(metadata.per_field.values()))
        config_file = first_field_source.source_path
        config_source = first_field_source.source

    if config_file and config_source != "defaults":
        typer.echo(f"Config file: {config_file}")
        typer.echo("Set input_device_index in that file to persist the default.")
    else:
        typer.echo("Config file: using defaults (no external config found)")
        typer.echo("Set input_device_index in voicepad.yaml to persist the default.")


@config_app.command("system")
def system_info() -> None:
    """Display comprehensive system information for transcription."""
    from voicepad_core import (
        get_cpu_info,
        get_ram_info,
        gpu_diagnostics,
    )

    typer.echo("System Information")
    typer.echo("=" * 60)

    # RAM info
    ram = get_ram_info()
    typer.echo("\nRAM:")
    typer.echo(f"   Total: {ram.total_gb:.1f} GB")
    typer.echo(f"   Available: {ram.available_gb:.1f} GB")

    if ram.available_gb >= 8:
        typer.secho("   [OK] Sufficient RAM for transcription", fg=typer.colors.GREEN)
    elif ram.available_gb >= 4:
        typer.secho("   [WARNING] Moderate RAM - use smaller models", fg=typer.colors.YELLOW)
    else:
        typer.secho("   [WARNING] Limited RAM - use tiny model only", fg=typer.colors.YELLOW)

    # CPU info
    cpu = get_cpu_info()
    typer.echo("\nCPU:")
    typer.echo(f"   Cores: {cpu.count}")
    if cpu.model_name:
        typer.echo(f"   Model: {cpu.model_name}")

    # GPU diagnostics
    gpu = gpu_diagnostics()
    typer.echo("\nGPU:")

    if gpu.nvidia_smi.success:
        typer.secho("   NVIDIA Driver: [OK] Detected", fg=typer.colors.GREEN)
    else:
        typer.secho("   NVIDIA Driver: [NOT FOUND]", fg=typer.colors.YELLOW)

    if gpu.ctranslate2_cuda.success:
        count = gpu.ctranslate2_cuda.cuda_device_count or 0
        typer.secho(f"   CUDA Devices: {count} device(s) available", fg=typer.colors.GREEN)
    else:
        typer.echo("   CUDA Devices: Not available")

    if gpu.faster_whisper_gpu.success:
        typer.secho("   faster-whisper GPU: [OK] Compatible", fg=typer.colors.GREEN)
    else:
        typer.echo("   faster-whisper GPU: Not compatible (CPU mode only)")

    typer.echo("=" * 60)


@config_app.command("recommend")
def recommend_model() -> None:
    """Get model recommendations based on system capabilities."""
    from voicepad_core import (
        get_available_models,
        get_cpu_info,
        get_model_recommendation,
        get_ram_info,
        gpu_diagnostics,
    )
    from voicepad_core.diagnostics.models import SystemInfo

    # Gather system info
    ram = get_ram_info()
    cpu = get_cpu_info()
    gpu = gpu_diagnostics()
    available_models = get_available_models()

    system_info = SystemInfo(ram=ram, cpu=cpu, gpu_diagnostics=gpu)

    # Get recommendation
    recommendation = get_model_recommendation(system_info, available_models)

    # Display recommendation
    typer.echo("Model Recommendation Based on System Capabilities")
    typer.echo("=" * 60)
    typer.echo("\nRecommended Configuration:")
    typer.secho(f"   Model: {recommendation.recommended_model}", fg=typer.colors.GREEN, bold=True)
    typer.secho(f"   Device: {recommendation.recommended_device}", fg=typer.colors.GREEN, bold=True)
    typer.secho(f"   Compute Type: {recommendation.recommended_compute_type}", fg=typer.colors.GREEN, bold=True)

    typer.echo("\nReason:")
    typer.echo(f"   {recommendation.reason}")

    if recommendation.alternative_models:
        typer.echo("\nAlternative Models:")
        for alt in recommendation.alternative_models[:5]:  # Show top 5
            typer.echo(f"   - {alt}")

    typer.echo("\nTo apply this configuration, add to voicepad.yaml:")
    typer.echo(f"   transcription_model: {recommendation.recommended_model}")
    typer.echo(f"   transcription_device: {recommendation.recommended_device}")
    typer.echo(f"   transcription_compute_type: {recommendation.recommended_compute_type}")
    typer.echo("=" * 60)


@config_app.command("transcription")
def transcription_config() -> None:
    """View current transcription configuration."""
    from voicepad_core import (
        get_available_models,
        get_config,
        get_cpu_info,
        get_model_recommendation,
        get_ram_info,
        gpu_diagnostics,
    )
    from voicepad_core.diagnostics.models import SystemInfo

    config = get_config()

    typer.echo("Transcription Configuration")
    typer.echo("=" * 60)
    typer.echo("\nCurrent Settings:")
    typer.echo(f"   Model: {config.transcription_model}")
    typer.echo(f"   Device: {config.transcription_device}")
    typer.echo(f"   Compute Type: {config.transcription_compute_type}")

    # Get recommendation for comparison
    ram = get_ram_info()
    cpu = get_cpu_info()
    gpu = gpu_diagnostics()
    available_models = get_available_models()
    system_info = SystemInfo(ram=ram, cpu=cpu, gpu_diagnostics=gpu)
    recommendation = get_model_recommendation(system_info, available_models)

    typer.echo("\nRecommended Settings:")
    if (
        config.transcription_model == recommendation.recommended_model
        and config.transcription_device == recommendation.recommended_device
        and config.transcription_compute_type == recommendation.recommended_compute_type
    ):
        typer.secho("   [OK] Your configuration matches the recommendation!", fg=typer.colors.GREEN)
    else:
        typer.secho("   [INFO] Consider updating your configuration:", fg=typer.colors.YELLOW)
        typer.echo(f"   Model: {recommendation.recommended_model}")
        typer.echo(f"   Device: {recommendation.recommended_device}")
        typer.echo(f"   Compute Type: {recommendation.recommended_compute_type}")
        typer.echo(f"\n   Reason: {recommendation.reason}")

    typer.echo("=" * 60)
    _show_config_hint()


@config_app.command("models")
def list_models() -> None:
    """List all available Whisper models."""
    from voicepad_core import get_available_models

    models = get_available_models()

    typer.echo("Available Whisper Models")
    typer.echo("=" * 60)

    # Categorize models
    categories = {
        "Tiny": [],
        "Base": [],
        "Small": [],
        "Medium": [],
        "Large": [],
        "Turbo": [],
        "Distil": [],
    }

    for model in models:
        model_lower = model.lower()
        if "turbo" in model_lower:
            categories["Turbo"].append(model)
        elif "distil" in model_lower:
            categories["Distil"].append(model)
        elif "large" in model_lower:
            categories["Large"].append(model)
        elif "medium" in model_lower:
            categories["Medium"].append(model)
        elif "small" in model_lower:
            categories["Small"].append(model)
        elif "base" in model_lower:
            categories["Base"].append(model)
        elif "tiny" in model_lower:
            categories["Tiny"].append(model)

    for category, category_models in categories.items():
        if category_models:
            typer.echo(f"\n{category}:")
            for model in sorted(category_models):
                en_only = " (English-only)" if ".en" in model else ""
                typer.echo(f"   - {model}{en_only}")

    typer.echo("\n" + "=" * 60)
    typer.echo(f"Total: {len(models)} models available")
    typer.echo("\nTip: Use 'voicepad config recommend' to get a model recommendation")
    typer.echo("     based on your system capabilities.")
