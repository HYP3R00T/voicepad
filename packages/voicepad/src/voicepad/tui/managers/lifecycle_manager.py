"""Lifecycle management for VoicePad TUI application."""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

from voicepad.tui.modals import SetupModal

if TYPE_CHECKING:
    from voicepad.tui.app import VoicePadApp

logger = logging.getLogger(__name__)


class LifecycleManager:
    """Manages app lifecycle events: mount, unmount, and first-run setup."""

    def __init__(self, app: VoicePadApp) -> None:
        self.app = app

    def on_mount(self) -> None:
        """Initialize the app after mounting."""
        # Apply theme from TUI config
        self.app.theme = self.app.tui_config.theme

        self.app._load_history_from_disk()
        self.app._populate_settings()
        self.check_first_run()

    def on_unmount(self) -> None:
        """Clean up resources when the app exits."""
        self._stop_hotkey_listener()
        self._stop_overlay()

    def _stop_hotkey_listener(self) -> None:
        """Stop the global hotkey listener if running."""
        if self.app._hotkey_listener is not None:
            with contextlib.suppress(Exception):
                self.app._hotkey_listener.stop()

    def _stop_overlay(self) -> None:
        """Stop the status overlay if running."""
        if self.app._overlay is not None:
            with contextlib.suppress(Exception):
                self.app._overlay.stop()

    def check_first_run(self) -> None:
        """Show setup modal if config is missing or model not downloaded."""
        from utilityhub_config import get_config_path
        from voicepad_core import model_downloaded

        config_path = get_config_path("voicepad", format="yaml")
        config_missing = not config_path.exists()
        model_ready = model_downloaded(self.app.config.transcription_model, self.app.config)

        if config_missing or not model_ready:

            def _setup_callback(result: tuple[str, int | None] | None) -> None:
                if result is not None:
                    self.on_setup_done(result)

            self.app.push_screen(SetupModal(self.app.config), callback=_setup_callback)
        else:
            self.app._warm_model_worker()

    def on_setup_done(self, result: tuple[str, int | None]) -> None:
        """Handle setup wizard completion and write config."""
        from utilityhub_config import get_config_path, write_config
        from voicepad_core.config import Config as _Config

        chosen_model, chosen_device = result
        raw = self.app.config.model_dump(mode="json")
        raw["transcription_model"] = chosen_model
        raw["input_device_index"] = chosen_device

        try:
            new_config = _Config(**raw)
            global_path = get_config_path("voicepad", format="yaml")
            global_path.parent.mkdir(parents=True, exist_ok=True)
            write_config(new_config, "voicepad", path=global_path, format="yaml")
            object.__setattr__(self.app, "config", new_config)
            logger.info(f"Config written to {global_path}")
        except Exception as e:
            logger.warning(f"Could not write config: {e}")

        self.app._refresh_config_path_label()
        self.app._refresh_settings_values()
        self.app._warm_model_worker()
