"""Interactive initialization wizard for VoicePad configuration setup."""

import logging
from pathlib import Path

import beaupy
from rich.console import Console
from rich.table import Table

from voicepad.config.settings import (
    MODEL_CATEGORIES,
    VRAM_ESTIMATES,
    Config,
    TranscriptionConfig,
    get_config,
    save_config,
)
from voicepad.system_utils import check_gpu_capabilities, recommend_faster_whisper_model

logger = logging.getLogger(__name__)
console = Console()


def display_system_info() -> tuple[bool, str | None, float | None]:
    """Display system capabilities and return GPU info.

    Returns:
        Tuple of (has_cuda, device_name, memory_gb)
    """
    gpu_info = check_gpu_capabilities()

    # Create system info table
    table = Table(title="🔍 System Capabilities Detection", show_header=False, box=None)
    table.add_row("PyTorch Available", "✓ Yes" if gpu_info.torch_available else "✗ No")
    table.add_row("Faster-Whisper Available", "✓ Yes" if gpu_info.faster_whisper_available else "✗ No")
    table.add_row()
    table.add_row("Device Type", gpu_info.device_type.upper(), style="bold cyan")

    if gpu_info.device_type == "cuda":
        table.add_row("Device Name", gpu_info.device_name or "N/A", style="bold green")
        if gpu_info.total_memory_gb:
            table.add_row("VRAM", f"{gpu_info.total_memory_gb:.1f} GB", style="bold green")
        if gpu_info.cuda_version:
            table.add_row("CUDA Version", gpu_info.cuda_version)
    else:
        table.add_row("Device", gpu_info.device_name or "CPU Only")

    console.print(table)
    console.print()

    return gpu_info.device_type == "cuda", gpu_info.device_name, gpu_info.total_memory_gb


def recommend_model_interactive(has_cuda: bool, device_name: str | None, memory_gb: float | None) -> str:
    """Interactively guide user through model selection.

    Returns:
        model_size
    """
    # Get recommendation
    rec = recommend_faster_whisper_model("cuda" if has_cuda else "cpu", memory_gb)

    console.print()
    console.print("[bold cyan]🤖 Model Selection[/bold cyan]")
    console.print()
    console.print("📋 [bold]Model Recommendation:[/bold]")
    console.print(f"  Recommended Model: [bold]{rec.model_size}[/bold]")
    console.print(f"  Note: {rec.notes}")
    console.print()

    # Ask user choice
    choice = beaupy.select(
        [
            "✓ Use recommended model",
            "🎯 Manually select model",
            "⏭️  Skip model selection (will be asked on first use)",
        ],
        cursor_index=0,
    )

    if choice is None:
        return rec.model_size

    choice_text = str(choice).lower()

    if "recommended" in choice_text:
        return rec.model_size

    if "skip" in choice_text:
        return "auto"

    # Manual selection
    return select_model_manually()


def select_model_manually() -> str:
    """Guide user through manual model selection.

    Returns:
        model_size
    """
    console.print()
    console.print("[bold cyan]📂 Available Models by Category[/bold cyan]")
    console.print()

    # Display categories
    categories = list(MODEL_CATEGORIES.keys())
    category_choice = beaupy.select(categories, cursor_index=1)

    if category_choice is None:
        return "auto"

    category_key = str(category_choice)

    # Display models in category
    models = MODEL_CATEGORIES.get(category_key, [])
    model_display = [f"{m} ({VRAM_ESTIMATES.get(m, 'N/A')})" for m in models]

    console.print()
    model_with_vram = beaupy.select(model_display, cursor_index=0)

    if model_with_vram is None:
        return "auto"

    # Extract model name (before the parentheses)
    selected_model = str(model_with_vram).split(" (")[0]

    console.print()
    console.print(f"✓ Selected: [bold green]{selected_model}[/bold green]")

    return selected_model


def prompt_path(label: str, default: Path) -> Path:
    """Prompt user for a file path.

    Args:
        label: Label for the path (e.g., "Audio recordings")
        default: Default path

    Returns:
        Selected path
    """
    console.print()
    console.print(f"📁 {label}")
    console.print(f"   Default: {default}")

    user_input = beaupy.prompt("Path", initial_value="")

    if not user_input or user_input.strip() == "":
        return default

    path = Path(user_input).expanduser()
    return path


def show_summary(config: Config) -> None:
    """Display configuration summary.

    Args:
        config: Configuration to display
    """
    console.print()
    console.print("[bold cyan]📝 Configuration Summary[/bold cyan]")
    console.print()

    # Create summary table
    table = Table(show_header=False, box=None)
    table.add_row()
    table.add_row("[bold cyan]🎵 Audio & Output:[/bold cyan]")
    table.add_row("  Recordings Path", str(config.recordings_path))
    table.add_row("  Markdown Path", str(config.markdown_path))
    table.add_row()
    table.add_row("[bold cyan]🤖 Transcription:[/bold cyan]")
    table.add_row("  Model", config.transcription.model)

    console.print(table)


def verify_and_create_directories(config: Config) -> bool:
    """Verify paths and create directories if needed.

    Args:
        config: Configuration with paths to verify

    Returns:
        True if successful, False if there were errors
    """
    console.print()
    console.print("[bold cyan]🔧 Finalizing Setup[/bold cyan]")
    console.print()

    success = True

    # Verify/create recordings path
    try:
        config.recordings_path.mkdir(parents=True, exist_ok=True)
        console.print(f"[bold green]✓ Recordings path ready:[/bold green] {config.recordings_path}")
    except Exception as e:
        console.print(f"[bold red]✗ Failed to create recordings path:[/bold red] {e}")
        success = False

    # Verify/create markdown path
    try:
        config.markdown_path.mkdir(parents=True, exist_ok=True)
        console.print(f"[bold green]✓ Markdown path ready:[/bold green] {config.markdown_path}")
    except Exception as e:
        console.print(f"[bold red]✗ Failed to create markdown path:[/bold red] {e}")
        success = False

    return success


def run_interactive_init() -> None:
    """Run the complete interactive initialization wizard."""
    console.print()
    console.print("[bold blue]🎙️  Welcome to VoicePad Setup![/bold blue]")
    console.print("This wizard will help you configure VoicePad for your system.")
    console.print()

    try:
        # Load current config to check if already configured
        current_config = get_config()
        is_reconfiguring = current_config.transcription.model != "auto"

        if is_reconfiguring and not beaupy.confirm(
            "VoicePad appears to be already configured. Re-run setup?", default_is_yes=False
        ):
            console.print("[yellow]✓ Setup cancelled. Current configuration remains unchanged.[/yellow]")
            return

    except Exception as e:
        console.print(f"[yellow]Note: Could not load existing config: {e}[/yellow]")

    # Step 1: Display system capabilities
    has_cuda, device_name, memory_gb = display_system_info()

    # Step 2: Model selection
    model_size = recommend_model_interactive(has_cuda, device_name, memory_gb)

    # Step 3: Path configuration
    console.print("[bold cyan]📁 Path Configuration[/bold cyan]")

    recordings_path = prompt_path("Audio Recordings Directory", Path("data/recordings"))
    markdown_path = prompt_path("Transcripts Directory", Path("data/markdown"))

    # Step 4: Create new config
    new_config = Config(
        recordings_path=recordings_path,
        markdown_path=markdown_path,
        transcription=TranscriptionConfig(
            model=model_size,  # type: ignore[arg-type]
        ),
    )

    # Step 6: Show summary
    show_summary(new_config)

    # Step 7: Confirm and save
    console.print()
    if not beaupy.confirm("Save this configuration?", default_is_yes=True):
        console.print("[yellow]✓ Setup cancelled. Configuration not saved.[/yellow]")
        return

    # Step 8: Create directories and save
    if verify_and_create_directories(new_config):
        save_config(new_config)
        console.print()
        console.print("[bold green]✅ Setup Complete![/bold green]")
        console.print()
        console.print("[bold]📖 Next steps:[/bold]")
        console.print("  • Record audio: [cyan]voicepad audio record[/cyan]")
        console.print("  • Transcribe: [cyan]voicepad transcribe file.wav[/cyan]")
        console.print("  • Open UI: [cyan]voicepad[/cyan]")
        console.print("  • View config: [cyan]voicepad config show[/cyan]")
    else:
        console.print()
        console.print("[yellow]⚠️  Configuration saved, but some directories could not be created.[/yellow]")
        console.print("Please check file permissions and create directories manually if needed.")
