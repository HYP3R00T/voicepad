"""Tests for voicepad.tui.managers.layout_builder."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from voicepad.tui.managers.layout_builder import LayoutBuilder


class TestLayoutBuilder:
    @pytest.fixture
    def mock_app(self):
        """Create a mock VoicePadApp."""
        app = MagicMock()
        app.query_one = MagicMock()
        return app

    @pytest.fixture
    def builder(self, mock_app):
        """Create a LayoutBuilder instance."""
        return LayoutBuilder(app=mock_app, version="1.0.0")

    def test_init_stores_app_and_version(self, mock_app) -> None:
        builder = LayoutBuilder(app=mock_app, version="2.0.0")
        assert builder.app is mock_app
        assert builder.version == "2.0.0"

    def test_mount_widgets_calls_mount_methods(self, builder) -> None:
        with (
            patch.object(builder, "_mount_record_tab_widgets") as mock_record,
            patch.object(builder, "_mount_history_tab_widgets") as mock_history,
        ):
            builder.mount_widgets()
        mock_record.assert_called_once()
        mock_history.assert_called_once()

    def test_mount_record_tab_widgets_mounts_labels_and_button(self, builder) -> None:
        mock_tx = MagicMock()
        builder.app.query_one.return_value = mock_tx
        builder._mount_record_tab_widgets()
        # Should call mount 3 times (tx-text, tx-meta, tx-copy-btn)
        assert mock_tx.mount.call_count == 3

    def test_mount_history_tab_widgets_mounts_option_list(self, builder) -> None:
        mock_hist_list = MagicMock()
        builder.app.query_one.return_value = mock_hist_list
        builder._mount_history_tab_widgets()
        # Should call mount once for OptionList
        mock_hist_list.mount.assert_called_once()
