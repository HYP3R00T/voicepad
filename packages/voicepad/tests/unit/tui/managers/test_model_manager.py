"""Tests for ModelManager."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest


class TestModelManagerInit:
    """Tests for ModelManager initialization."""

    def test_init_stores_app_reference(self) -> None:
        """Test that __init__ stores the app reference."""
        from voicepad.tui.managers.model_manager import ModelManager

        mock_app = Mock()
        manager = ModelManager(mock_app)

        assert manager.app is mock_app


class TestOnModelReady:
    """Tests for on_model_ready method."""

    @pytest.fixture
    def mock_app(self) -> Mock:
        """Create a mock app with necessary attributes."""
        app = Mock()
        app.config = Mock()
        app.config.transcription_model = "base"
        app._model_ready = False
        app._hotkey_listener = None
        app._start_hotkey_listener = Mock()

        # Mock query_one for labels
        mock_model_label = Mock()
        mock_status_label = Mock()

        def query_one_side_effect(selector: str, widget_type: type) -> Mock:
            if selector == "#header-model":
                return mock_model_label
            elif selector == "#status":
                return mock_status_label
            return Mock()

        app.query_one = Mock(side_effect=query_one_side_effect)
        return app

    def test_stores_warm_result(self, mock_app: Mock) -> None:
        """Test that on_model_ready stores the warm result."""
        from voicepad.tui.managers.model_manager import ModelManager

        manager = ModelManager(mock_app)
        result = Mock()
        result.error = None
        result.fallback = False
        result.device = "cuda"

        manager.on_model_ready(result)

        assert mock_app._warm_result is result

    def test_handles_model_error(self, mock_app: Mock) -> None:
        """Test that on_model_ready handles model errors."""
        from voicepad.tui.managers.model_manager import ModelManager

        manager = ModelManager(mock_app)
        result = Mock()
        result.error = "Model loading failed"

        with patch.object(manager, "set_status") as mock_set_status:
            manager.on_model_ready(result)

            mock_set_status.assert_called_once_with("error", "model error: Model loading failed")
            # Should not set model_ready to True
            assert mock_app._model_ready is False

    def test_updates_model_label_on_success(self, mock_app: Mock) -> None:
        """Test that on_model_ready updates model label on success."""
        from textual.widgets import Label
        from voicepad.tui.managers.model_manager import ModelManager

        manager = ModelManager(mock_app)
        result = Mock()
        result.error = None
        result.fallback = False
        result.device = "cuda"

        manager.on_model_ready(result)

        mock_model_label = mock_app.query_one("#header-model", Label)
        mock_model_label.update.assert_called_once_with("[dim]model:[/] base  [dim]device:[/] cuda")

    def test_shows_fallback_indicator(self, mock_app: Mock) -> None:
        """Test that on_model_ready shows CPU fallback indicator."""
        from textual.widgets import Label
        from voicepad.tui.managers.model_manager import ModelManager

        manager = ModelManager(mock_app)
        result = Mock()
        result.error = None
        result.fallback = True
        result.device = "cpu"

        manager.on_model_ready(result)

        mock_model_label = mock_app.query_one("#header-model", Label)
        mock_model_label.update.assert_called_once_with("[dim]model:[/] base  [dim]device:[/] cpu  cpu fallback")

    def test_sets_status_to_ready(self, mock_app: Mock) -> None:
        """Test that on_model_ready sets status to ready."""
        from voicepad.tui.managers.model_manager import ModelManager

        manager = ModelManager(mock_app)
        result = Mock()
        result.error = None
        result.fallback = False
        result.device = "cuda"

        with patch.object(manager, "set_status") as mock_set_status:
            manager.on_model_ready(result)

            mock_set_status.assert_called_once_with("ready", "ready")

    def test_sets_model_ready_flag(self, mock_app: Mock) -> None:
        """Test that on_model_ready sets the model_ready flag."""
        from voicepad.tui.managers.model_manager import ModelManager

        manager = ModelManager(mock_app)
        result = Mock()
        result.error = None
        result.fallback = False
        result.device = "cuda"

        manager.on_model_ready(result)

        assert mock_app._model_ready is True

    def test_starts_hotkey_listener_if_not_running(self, mock_app: Mock) -> None:
        """Test that on_model_ready starts hotkey listener if not already running."""
        from voicepad.tui.managers.model_manager import ModelManager

        manager = ModelManager(mock_app)
        result = Mock()
        result.error = None
        result.fallback = False
        result.device = "cuda"

        mock_app._hotkey_listener = None

        manager.on_model_ready(result)

        mock_app._start_hotkey_listener.assert_called_once()

    def test_does_not_start_hotkey_listener_if_already_running(self, mock_app: Mock) -> None:
        """Test that on_model_ready does not start hotkey listener if already running."""
        from voicepad.tui.managers.model_manager import ModelManager

        manager = ModelManager(mock_app)
        result = Mock()
        result.error = None
        result.fallback = False
        result.device = "cuda"

        mock_app._hotkey_listener = Mock()  # Already running

        manager.on_model_ready(result)

        mock_app._start_hotkey_listener.assert_not_called()


class TestReloadModel:
    """Tests for reload_model method."""

    @pytest.fixture
    def mock_app(self) -> Mock:
        """Create a mock app with necessary attributes."""
        app = Mock()
        app._recording = False
        app._transcribing = False
        app._model_ready = True

        # Mock query_one for labels
        mock_model_label = Mock()
        mock_status_label = Mock()

        def query_one_side_effect(selector: str, widget_type: type) -> Mock:
            if selector == "#header-model":
                return mock_model_label
            elif selector == "#status":
                return mock_status_label
            return Mock()

        app.query_one = Mock(side_effect=query_one_side_effect)
        return app

    def test_does_not_reload_while_recording(self, mock_app: Mock) -> None:
        """Test that reload_model does nothing while recording."""
        from voicepad.tui.managers.model_manager import ModelManager

        mock_app._recording = True
        manager = ModelManager(mock_app)

        with patch("voicepad_core._model_cache") as mock_cache:
            manager.reload_model()

            mock_cache.clear.assert_not_called()

    def test_does_not_reload_while_transcribing(self, mock_app: Mock) -> None:
        """Test that reload_model does nothing while transcribing."""
        from voicepad.tui.managers.model_manager import ModelManager

        mock_app._transcribing = True
        manager = ModelManager(mock_app)

        with patch("voicepad_core._model_cache") as mock_cache:
            manager.reload_model()

            mock_cache.clear.assert_not_called()

    def test_clears_model_cache(self, mock_app: Mock) -> None:
        """Test that reload_model clears the model cache."""
        from voicepad.tui.managers.model_manager import ModelManager

        manager = ModelManager(mock_app)

        with patch("voicepad_core._model_cache") as mock_cache:
            manager.reload_model()

            mock_cache.clear.assert_called_once()

    def test_sets_model_ready_to_false(self, mock_app: Mock) -> None:
        """Test that reload_model sets model_ready to False."""
        from voicepad.tui.managers.model_manager import ModelManager

        manager = ModelManager(mock_app)

        with patch("voicepad_core._model_cache"):
            manager.reload_model()

            assert mock_app._model_ready is False

    def test_updates_status_to_reloading(self, mock_app: Mock) -> None:
        """Test that reload_model updates status to reloading."""
        from voicepad.tui.managers.model_manager import ModelManager

        manager = ModelManager(mock_app)

        with (
            patch("voicepad_core._model_cache"),
            patch.object(manager, "set_status") as mock_set_status,
        ):
            manager.reload_model()

            mock_set_status.assert_called_once_with("transcribing", "reloading model…")

    def test_updates_model_label_to_loading(self, mock_app: Mock) -> None:
        """Test that reload_model updates model label to loading."""
        from textual.widgets import Label
        from voicepad.tui.managers.model_manager import ModelManager

        manager = ModelManager(mock_app)

        with patch("voicepad_core._model_cache"):
            manager.reload_model()

            mock_model_label = mock_app.query_one("#header-model", Label)
            mock_model_label.update.assert_called_once_with("[dim]model:[/] loading…")

    def test_calls_warm_model_worker(self, mock_app: Mock) -> None:
        """Test that reload_model calls warm_model_worker."""
        from voicepad.tui.managers.model_manager import ModelManager

        manager = ModelManager(mock_app)

        with patch("voicepad_core._model_cache"):
            manager.reload_model()

            mock_app._warm_model_worker.assert_called_once()


class TestSetStatus:
    """Tests for set_status method."""

    @pytest.fixture
    def mock_app(self) -> Mock:
        """Create a mock app with necessary attributes."""
        app = Mock()
        mock_label = Mock()
        app.query_one = Mock(return_value=mock_label)
        return app

    def test_updates_label_with_ready_state(self, mock_app: Mock) -> None:
        """Test that set_status updates label with ready state."""
        from textual.widgets import Label
        from voicepad.tui.managers.model_manager import ModelManager

        manager = ModelManager(mock_app)
        manager.set_status("ready", "ready")

        mock_label = mock_app.query_one("#status", Label)
        mock_label.remove_class.assert_called_once_with("ready", "recording", "transcribing", "error")
        mock_label.add_class.assert_called_once_with("ready")
        mock_label.update.assert_called_once_with("  ready")

    def test_updates_label_with_recording_state(self, mock_app: Mock) -> None:
        """Test that set_status updates label with recording state."""
        from textual.widgets import Label
        from voicepad.tui.managers.model_manager import ModelManager

        manager = ModelManager(mock_app)
        manager.set_status("recording", "recording…")

        mock_label = mock_app.query_one("#status", Label)
        mock_label.update.assert_called_once_with("\U000f044a  recording…")

    def test_updates_label_with_transcribing_state(self, mock_app: Mock) -> None:
        """Test that set_status updates label with transcribing state."""
        from textual.widgets import Label
        from voicepad.tui.managers.model_manager import ModelManager

        manager = ModelManager(mock_app)
        manager.set_status("transcribing", "transcribing…")

        mock_label = mock_app.query_one("#status", Label)
        mock_label.update.assert_called_once_with("\U000f051f  transcribing…")

    def test_updates_label_with_error_state(self, mock_app: Mock) -> None:
        """Test that set_status updates label with error state."""
        from textual.widgets import Label
        from voicepad.tui.managers.model_manager import ModelManager

        manager = ModelManager(mock_app)
        manager.set_status("error", "error occurred")

        mock_label = mock_app.query_one("#status", Label)
        mock_label.update.assert_called_once_with("\U000f0159  error occurred")

    def test_uses_default_icon_for_unknown_state(self, mock_app: Mock) -> None:
        """Test that set_status uses default icon for unknown state."""
        from textual.widgets import Label
        from voicepad.tui.managers.model_manager import ModelManager

        manager = ModelManager(mock_app)
        manager.set_status("unknown", "unknown state")

        mock_label = mock_app.query_one("#status", Label)
        mock_label.update.assert_called_once_with("\U000f051f  unknown state")

    def test_removes_all_state_classes(self, mock_app: Mock) -> None:
        """Test that set_status removes all state classes before adding new one."""
        from textual.widgets import Label
        from voicepad.tui.managers.model_manager import ModelManager

        manager = ModelManager(mock_app)
        manager.set_status("recording", "recording…")

        mock_label = mock_app.query_one("#status", Label)
        mock_label.remove_class.assert_called_once_with("ready", "recording", "transcribing", "error")
