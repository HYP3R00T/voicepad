"""Event handlers for VoicePad TUI."""

from voicepad.tui.handlers.history_handler import HistoryHandler
from voicepad.tui.handlers.hotkey_handler import HotkeyHandler
from voicepad.tui.handlers.recording_handler import RecordingHandler
from voicepad.tui.handlers.settings_handler import SettingsHandler

__all__ = [
    "HistoryHandler",
    "HotkeyHandler",
    "RecordingHandler",
    "SettingsHandler",
]
