"""Tests for clipboard utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from voicepad.tui.utils.clipboard import copy_to_clipboard


class TestCopyToClipboard:
    """Test suite for copy_to_clipboard function."""

    @patch("pyperclip.copy")
    def test_copy_to_clipboard_calls_pyperclip(self, mock_copy: MagicMock) -> None:
        """copy_to_clipboard calls pyperclip.copy with text."""
        text = "Hello, World!"

        copy_to_clipboard(text)

        mock_copy.assert_called_once_with(text)

    @patch("pyperclip.copy")
    def test_copy_to_clipboard_handles_empty_string(self, mock_copy: MagicMock) -> None:
        """copy_to_clipboard handles empty string."""
        copy_to_clipboard("")

        mock_copy.assert_called_once_with("")

    @patch("pyperclip.copy")
    def test_copy_to_clipboard_handles_multiline_text(self, mock_copy: MagicMock) -> None:
        """copy_to_clipboard handles multiline text."""
        text = "Line 1\nLine 2\nLine 3"

        copy_to_clipboard(text)

        mock_copy.assert_called_once_with(text)

    @patch("pyperclip.copy")
    def test_copy_to_clipboard_handles_unicode(self, mock_copy: MagicMock) -> None:
        """copy_to_clipboard handles unicode characters."""
        text = "Hello 世界 🌍"

        copy_to_clipboard(text)

        mock_copy.assert_called_once_with(text)

    @patch("voicepad.tui.utils.clipboard.logger")
    @patch("pyperclip.copy")
    def test_copy_to_clipboard_logs_warning_on_import_error(self, mock_copy: MagicMock, mock_logger: MagicMock) -> None:
        """copy_to_clipboard logs warning when pyperclip import fails."""
        mock_copy.side_effect = ImportError("pyperclip not installed")

        copy_to_clipboard("test")

        mock_logger.warning.assert_called_once()
        assert "Clipboard copy failed" in str(mock_logger.warning.call_args)

    @patch("voicepad.tui.utils.clipboard.logger")
    @patch("pyperclip.copy")
    def test_copy_to_clipboard_logs_warning_on_runtime_error(
        self, mock_copy: MagicMock, mock_logger: MagicMock
    ) -> None:
        """copy_to_clipboard logs warning on runtime errors."""
        mock_copy.side_effect = RuntimeError("Clipboard unavailable")

        copy_to_clipboard("test")

        mock_logger.warning.assert_called_once()
        assert "Clipboard copy failed" in str(mock_logger.warning.call_args)

    @patch("voicepad.tui.utils.clipboard.logger")
    @patch("pyperclip.copy")
    def test_copy_to_clipboard_does_not_raise_on_error(self, mock_copy: MagicMock, mock_logger: MagicMock) -> None:
        """copy_to_clipboard does not raise exceptions."""
        mock_copy.side_effect = RuntimeError("Unexpected error")

        # Should not raise
        copy_to_clipboard("test")

        mock_logger.warning.assert_called_once()

    @patch("pyperclip.copy")
    def test_copy_to_clipboard_handles_large_text(self, mock_copy: MagicMock) -> None:
        """copy_to_clipboard handles large text blocks."""
        text = "A" * 10000  # 10KB of text

        copy_to_clipboard(text)

        mock_copy.assert_called_once_with(text)
