"""Helper functions for settings screen."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.widgets import Input, Label, Select, Static
from voicepad_core import VALID_TRANSCRIPTION_MODELS

from voicepad.tui.components.checkbox import VoiceCheckbox

if TYPE_CHECKING:
    from voicepad_core.config import Config


def populate_settings_form(
    container: Static,
    config: Config,
    config_path_str: str,
) -> None:
    """Populate the settings form with current config values.

    Extracted from app.py _populate_settings method.
    """
    from voicepad.cli.config import _get_input_devices

    # Config path label
    exists_hint = "" if Path(config_path_str).exists() else "  [dim red](not yet created)[/]"
    path_label = Label(
        f"[dim]config file:[/]  {config_path_str}{exists_hint}",
        id="settings-config-path",
    )
    container.mount(path_label)

    # User-facing fields
    user_fields = {
        "recordings_path": "Where your WAV recordings are saved",
        "markdown_path": "Where your transcription files are saved",
        "transcription_model": "Whisper model to use for transcription",
        "input_device_index": "Microphone to record from",
        "theme": "UI color theme",
    }

    # Build device options
    audio_devices = _get_input_devices()
    device_options: list[tuple[str, int]] = [("system default", -1)]
    device_options += [(f"[{d.index}]  {d.name}", d.index) for d in audio_devices]

    # Create field widgets
    for field_name, hint in user_fields.items():
        # Use the Config *class* model_fields to avoid accessing Pydantic
        # metadata on an instance (deprecated in Pydantic v3).
        field_info = type(config).model_fields.get(field_name)
        if field_info is None:
            continue
        current_val = getattr(config, field_name)

        key_label = Label(
            f"[bold]{field_name}[/]  [dim]—  {hint}[/]",
            classes="settings-key",
        )

        if field_name == "transcription_model":
            options = [(m, m) for m in VALID_TRANSCRIPTION_MODELS]
            current_str = str(current_val) if current_val is not None else "turbo"
            widget = Select(
                options=options,
                value=current_str if current_str in VALID_TRANSCRIPTION_MODELS else "turbo",
                id="setting-transcription_model",
                classes="settings-input",
                allow_blank=False,
            )
        elif field_name == "theme":
            # Get available themes from the app
            from voicepad.tui.theme import get_available_themes

            available_themes = get_available_themes()
            options = [(t, t) for t in available_themes]
            current_str = str(current_val) if current_val is not None else "voicepad-dark"
            widget = Select(
                options=options,
                value=current_str if current_str in available_themes else "voicepad-dark",
                id="setting-theme",
                classes="settings-input",
                allow_blank=False,
            )
        elif field_name == "input_device_index":
            current_idx = current_val if current_val is not None else -1
            valid_values = {v for _, v in device_options}
            widget = Select(
                options=device_options,
                value=current_idx if current_idx in valid_values else -1,
                id="setting-input_device_index",
                classes="settings-input",
                allow_blank=False,
            )
        else:
            widget = Input(
                value=str(current_val) if current_val is not None else "",
                placeholder=str(field_info.default) if field_info.default is not None else "",
                id=f"setting-{field_name}",
                classes="settings-input",
            )

        row = Static(classes="settings-row")
        container.mount(row)
        row.mount(key_label, widget)

    # Hotkey picker
    _mount_hotkey_picker(container, config)


def _mount_hotkey_picker(container: Static, config: Config) -> None:
    """Mount the hotkey picker UI."""
    from voicepad.tui.utils.hotkey_utils import (
        HOTKEY_KEYS,
        build_hotkey_str,
        parse_hotkey_str,
    )

    hotkey_label = Label(
        "[bold]global_hotkey[/]  [dim]—  System-wide record/stop shortcut[/]",
        classes="settings-key",
    )
    hotkey_row = Static(classes="settings-row", id="hotkey-row")
    container.mount(hotkey_row)
    hotkey_row.mount(hotkey_label)

    mods, key_char = parse_hotkey_str(config.global_hotkey)

    mod_row = Static(classes="hotkey-mod-row", id="hotkey-mod-row")
    hotkey_row.mount(mod_row)

    _modifiers = [("Ctrl", "ctrl"), ("Alt", "alt"), ("Shift", "shift"), ("Win", "cmd")]
    for label_text, mod_id in _modifiers:
        cb = VoiceCheckbox(
            label_text,
            value=(mod_id in mods),
            id=f"hotkey-mod-{mod_id}",
            classes="hotkey-checkbox",
        )
        mod_row.mount(cb)

    key_options = [(k, k) for k in HOTKEY_KEYS]
    current_key = key_char if key_char in HOTKEY_KEYS else "v"
    key_select = Select(
        options=key_options,
        value=current_key,
        id="hotkey-key-select",
        classes="hotkey-key-select",
        allow_blank=False,
    )
    hotkey_row.mount(key_select)

    preview = build_hotkey_str(mods, current_key)
    hotkey_row.mount(
        Label(
            f"[dim]{preview or 'disabled'}[/]",
            id="hotkey-preview",
            classes="hotkey-preview",
        )
    )
