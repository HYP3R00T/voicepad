"""CLI commands for running the Voicepad UI application."""

import logging

from voicepad.ui.voicepad_ui import VoicepadUI

logger = logging.getLogger(__name__)


def start_ui() -> None:
    """Start the Voicepad interactive UI application."""
    voicepad_app = VoicepadUI()
    voicepad_app.run()
