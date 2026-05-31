"""Settings handler for VoicePad TUI."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from textual import on
from textual.widgets import Button, Input, Label, Select, Static
from voicepad_core import VALID_TRANSCRIPTION_MODELS
from voicepad_core.config import Config as _Config

from voicepad.tui.components.checkbox import VoiceCheckbox
from voicepad.tui.utils.hotkey_utils import HOTKEY_KEYS as _HOTKEY_KEYS
from voicepad.tui.utils.hotkey_utils import build_hotkey_str as _build_hotkey_str
from voicepad.tui.utils.hotkey_utils import parse_hotkey_str as _parse_hotkey_str

if TYPE_CHECKING:
    from voicepad.tui.app import VoicePadApp


class SettingsHandler:
    """Handles settings tab functionality."""

    def __init__(self, app: VoicePadApp) -> None:
        self.app = app

    def refresh_settings_values(self) -> None:
        """Update settings form widget values to match the current config in-place."""
        with contextlib.suppress(Exception):
            from voicepad.cli.config import _get_input_devices

            # Update device dropdown
            devices = _get_input_devices()
            device_options: list[tuple[str, int]] = [("System default", -1)]
            device_options += [(d.name, d.index) for d in devices]
            valid = {v for _, v in device_options}
            current_idx = self.app.config.input_device_index if self.app.config.input_device_index is not None else -1
            sel_device = self.app.query_one("#setting-input_device_index", Select)
            sel_device.set_options(device_options)
            sel_device.value = current_idx if current_idx in valid else -1

        with contextlib.suppress(Exception):
            # Update model dropdown
            sel_model = self.app.query_one("#setting-transcription_model", Select)
            sel_model.value = self.app.config.transcription_model

        with contextlib.suppress(Exception):
            # Update theme dropdown from TUI config
            sel_theme = self.app.query_one("#setting-theme", Select)
            sel_theme.value = self.app.tui_config.theme

        with contextlib.suppress(Exception):
            # Update path inputs
            from textual.widgets import Input as _Input

            self.app.query_one("#setting-recordings_path", _Input).value = str(self.app.config.recordings_path)
            self.app.query_one("#setting-markdown_path", _Input).value = str(self.app.config.markdown_path)

        with contextlib.suppress(Exception):
            # Sync hotkey picker from config
            mods, key = _parse_hotkey_str(self.app.config.global_hotkey)
            for mod_id in ("ctrl", "alt", "shift", "cmd"):
                self.app.query_one(f"#hotkey-mod-{mod_id}", VoiceCheckbox).value = mod_id in mods
            sel = self.app.query_one("#hotkey-key-select", Select)
            if key in _HOTKEY_KEYS:
                sel.value = key
            self._update_hotkey_preview()

    def refresh_config_path_label(self) -> None:
        """Update the settings config path label to reflect current file state."""
        from utilityhub_config import get_config_path

        with contextlib.suppress(Exception):
            config_path = get_config_path("voicepad", format="yaml")
            exists_hint = "" if config_path.exists() else "  [dim red](not yet created)[/]"
            self.app.query_one("#settings-config-path", Label).update(
                f"[dim]config file:[/]  {config_path}{exists_hint}"
            )

    def populate_settings(self) -> None:
        """Build the settings form — only user-facing fields shown."""
        from utilityhub_config import get_config_path
        from voicepad_core.config.settings import get_config_with_metadata

        from voicepad.cli.config import _get_input_devices

        user_fields = {
            "recordings_path": "Where your WAV recordings are saved",
            "markdown_path": "Where your transcription files are saved",
            "transcription_model": "Whisper model to use for transcription",
            "input_device_index": "Microphone to record from",
        }

        # Build device options once — reused for the Select widget
        audio_devices = _get_input_devices()
        device_options: list[tuple[str, int]] = [("System default", -1)]
        device_options += [(d.name, d.index) for d in audio_devices]

        container = self.app.query_one("#settings-fields", Static)
        _, meta = get_config_with_metadata()

        # Show the config file path at the top — indicate if it exists or not
        config_path = get_config_path("voicepad", format="yaml")
        exists_hint = "" if config_path.exists() else "  [dim red](not yet created)[/]"
        path_label = Label(
            f"[dim]config file:[/]  {config_path}{exists_hint}",
            id="settings-config-path",
        )
        container.mount(path_label)

        for field_name, hint in user_fields.items():
            field_info = _Config.model_fields.get(field_name)
            if field_info is None:
                continue
            current_val = getattr(self.app.config, field_name)

            # Single-line label: "field_name  —  hint description"
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
            elif field_name == "input_device_index":
                # current_val is int | None; use -1 as the sentinel for "system default"
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

        # ── Theme picker (TUI-only setting) ───────────────────────────
        from voicepad.tui.theme import get_available_themes

        available_themes = get_available_themes()
        current_theme = self.app.tui_config.theme
        theme_label = Label(
            "[bold]theme[/]  [dim]—  UI color theme[/]",
            classes="settings-key",
        )
        theme_widget = Select(
            options=[(t, t) for t in available_themes],
            value=current_theme if current_theme in available_themes else "tokyo-night",
            id="setting-theme",
            classes="settings-input",
            allow_blank=False,
        )
        theme_row = Static(classes="settings-row")
        container.mount(theme_row)
        theme_row.mount(theme_label, theme_widget)

        # ── Hotkey picker ──────────────────────────────────────────────
        hotkey_label = Label(
            "[bold]global_hotkey[/]  [dim]—  System-wide record/stop shortcut[/]",
            classes="settings-key",
        )
        hotkey_row = Static(classes="settings-row", id="hotkey-row")
        container.mount(hotkey_row)
        hotkey_row.mount(hotkey_label)

        # Parse current hotkey into modifiers + key
        mods, key_char = _parse_hotkey_str(self.app.config.global_hotkey)

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

        key_options = [(k, k) for k in _HOTKEY_KEYS]
        current_key = key_char if key_char in _HOTKEY_KEYS else "v"
        key_select = Select(
            options=key_options,
            value=current_key,
            id="hotkey-key-select",
            classes="hotkey-key-select",
            allow_blank=False,
        )
        hotkey_row.mount(key_select)

        preview = _build_hotkey_str(mods, current_key)
        hotkey_row.mount(
            Label(
                f"[dim]{preview or 'disabled'}[/]",
                id="hotkey-preview",
                classes="hotkey-preview",
            )
        )

    def get_hotkey_from_picker(self) -> str:
        """Read modifier checkboxes + key dropdown and return pynput hotkey string."""
        mods: list[str] = []
        for mod_id in ("ctrl", "alt", "shift", "cmd"):
            with contextlib.suppress(Exception):
                if self.app.query_one(f"#hotkey-mod-{mod_id}", VoiceCheckbox).value:
                    mods.append(mod_id)
        key = "v"
        with contextlib.suppress(Exception):
            sel = self.app.query_one("#hotkey-key-select", Select)
            if sel.value is not Select.BLANK:
                key = str(sel.value)
        return _build_hotkey_str(mods, key)

    def _update_hotkey_preview(self) -> None:
        """Refresh the preview label from current picker state."""
        with contextlib.suppress(Exception):
            preview = self.get_hotkey_from_picker()
            self.app.query_one("#hotkey-preview", Label).update(f"[dim]{preview or 'disabled'}[/]")

    @on(VoiceCheckbox.Changed, ".hotkey-checkbox")
    def on_hotkey_checkbox_changed(self) -> None:
        self._update_hotkey_preview()

    @on(Select.Changed, "#hotkey-key-select")
    def on_hotkey_key_changed(self) -> None:
        self._update_hotkey_preview()

    @on(Button.Pressed, "#settings-save-btn")
    def on_settings_save(self) -> None:
        """Read user-facing inputs, merge with existing config, write to voicepad.yaml."""
        from utilityhub_config import get_config_path, write_config

        user_fields = ["recordings_path", "markdown_path", "transcription_model", "input_device_index"]

        status = self.app.query_one("#settings-status", Label)
        errors: list[str] = []

        # Start from current config values (preserves hidden fields)
        raw = self.app.config.model_dump(mode="json")

        # Read global hotkey from the picker widgets
        raw["global_hotkey"] = self.get_hotkey_from_picker()

        for field_name in user_fields:
            field_info = _Config.model_fields.get(field_name)
            if field_info is None:
                continue
            with contextlib.suppress(Exception):
                if field_name == "transcription_model":
                    sel = self.app.query_one("#setting-transcription_model", Select)
                    raw[field_name] = str(sel.value) if sel.value is not Select.BLANK else raw[field_name]
                elif field_name == "input_device_index":
                    sel = self.app.query_one("#setting-input_device_index", Select)
                    if sel.value is not Select.BLANK:
                        v = int(str(sel.value))
                        raw[field_name] = None if v == -1 else v
                else:
                    inp = self.app.query_one(f"#setting-{field_name}", Input)
                    val_str = inp.value.strip()
                    annotation = field_info.annotation
                    if val_str == "" or val_str.lower() == "none":
                        raw[field_name] = None
                    elif annotation in (int, "int") or "int" in str(annotation):
                        try:
                            raw[field_name] = int(val_str)
                        except ValueError:
                            errors.append(f"{field_name}: expected a number")
                    else:
                        raw[field_name] = val_str

        if errors:
            status.update(f"[red]\U000f0156  {'; '.join(errors)}[/]")
            return

        try:
            new_config = _Config(**raw)

            for path_field in (new_config.recordings_path, new_config.markdown_path):
                path_field.mkdir(parents=True, exist_ok=True)

            # Always write to the global config — never a project-local file
            global_path = get_config_path("voicepad", format="yaml")
            write_config(new_config, "voicepad", path=global_path, format="yaml")

            hotkey_changed = new_config.global_hotkey != self.app.config.global_hotkey
            model_changed = (
                new_config.transcription_model != self.app.config.transcription_model
                or new_config.transcription_device != self.app.config.transcription_device
                or new_config.transcription_compute_type != self.app.config.transcription_compute_type
            )
            object.__setattr__(self.app, "config", new_config)

            if hotkey_changed:
                if self.app._hotkey_listener is not None:
                    with contextlib.suppress(Exception):
                        self.app._hotkey_listener.stop()  # type: ignore[union-attr]
                if self.app._overlay is not None:
                    with contextlib.suppress(Exception):
                        self.app._overlay.stop()  # type: ignore[union-attr]
                self.app._hotkey_listener = None
                self.app._overlay = None
                self.app._start_hotkey_listener()

            if model_changed and not self.app._recording and not self.app._transcribing:
                from voicepad_core import _model_cache

                _model_cache.clear()
                self.app._model_ready = False
                self.app._set_status("transcribing", "loading model…")
                self.app.query_one("#header-model", Label).update("[dim]M:[/] loading…")
                self.app._warm_model_worker()
                status.update("[green]\U000f012c  saved — reloading model[/]")
            else:
                status.update("[green]\U000f012c  saved[/]")

            # Apply theme from the theme picker — watch_theme handles persistence
            with contextlib.suppress(Exception):
                sel = self.app.query_one("#setting-theme", Select)
                if sel.value is not Select.BLANK:
                    self.app.theme = str(sel.value)

            self.app.set_timer(3.0, lambda: status.update(""))
        except Exception as e:
            status.update(f"[red]\U000f0156  {e}[/]")
