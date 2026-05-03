"""Tests for header component."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Label
from voicepad.tui.components.header import HeaderWidget


class HeaderTestApp(App):
    """Test app for HeaderWidget."""

    def __init__(self, version: str = "dev"):
        super().__init__()
        self.version = version

    def compose(self) -> ComposeResult:
        yield HeaderWidget(version=self.version)


class TestHeaderWidget:
    """Test HeaderWidget component."""

    def test_init_default_version(self):
        """Test HeaderWidget initialization with default version."""
        widget = HeaderWidget()
        assert widget.id == "header"
        assert widget._version == "dev"

    def test_init_custom_version(self):
        """Test HeaderWidget initialization with custom version."""
        widget = HeaderWidget(version="v1.2.3")
        assert widget.id == "header"
        assert widget._version == "v1.2.3"

    def test_compose_yields_labels(self):
        """Test that compose yields the correct labels."""
        widget = HeaderWidget(version="v1.0.0")
        result = list(widget.compose())

        assert len(result) == 4
        assert all(isinstance(label, Label) for label in result)

        # Check IDs
        assert result[0].id == "header-title"
        assert result[1].id == "header-version"
        assert result[2].id == "status"
        assert result[3].id == "header-model"

    def test_compose_label_ids(self):
        """Test that compose creates labels with correct IDs."""
        widget = HeaderWidget(version="v2.0.0")
        result = list(widget.compose())

        ids = [label.id for label in result]
        assert ids == ["header-title", "header-version", "status", "header-model"]

    @pytest.mark.asyncio
    async def test_set_status_ready(self):
        """Test set_status with ready state."""
        app = HeaderTestApp()
        async with app.run_test():
            widget = app.query_one(HeaderWidget)
            widget.set_status("ready", "ready to record")

            status_label = widget.query_one("#status", Label)
            assert status_label.has_class("ready")
            assert not status_label.has_class("recording")
            assert not status_label.has_class("transcribing")
            assert not status_label.has_class("error")

    @pytest.mark.asyncio
    async def test_set_status_recording(self):
        """Test set_status with recording state."""
        app = HeaderTestApp()
        async with app.run_test():
            widget = app.query_one(HeaderWidget)
            widget.set_status("recording", "recording audio")

            status_label = widget.query_one("#status", Label)
            assert status_label.has_class("recording")
            assert not status_label.has_class("ready")

    @pytest.mark.asyncio
    async def test_set_status_transcribing(self):
        """Test set_status with transcribing state."""
        app = HeaderTestApp()
        async with app.run_test():
            widget = app.query_one(HeaderWidget)
            widget.set_status("transcribing", "processing audio")

            status_label = widget.query_one("#status", Label)
            assert status_label.has_class("transcribing")

    @pytest.mark.asyncio
    async def test_set_status_error(self):
        """Test set_status with error state."""
        app = HeaderTestApp()
        async with app.run_test():
            widget = app.query_one(HeaderWidget)
            widget.set_status("error", "something went wrong")

            status_label = widget.query_one("#status", Label)
            assert status_label.has_class("error")

    @pytest.mark.asyncio
    async def test_set_status_unknown_state(self):
        """Test set_status with unknown state uses default icon."""
        app = HeaderTestApp()
        async with app.run_test():
            widget = app.query_one(HeaderWidget)
            widget.set_status("unknown", "unknown state")

            status_label = widget.query_one("#status", Label)
            # Should not crash and should update the label
            assert status_label is not None

    @pytest.mark.asyncio
    async def test_set_status_removes_previous_classes(self):
        """Test that set_status removes previous state classes."""
        app = HeaderTestApp()
        async with app.run_test():
            widget = app.query_one(HeaderWidget)

            # Set to recording first
            widget.set_status("recording", "recording")
            status_label = widget.query_one("#status", Label)
            assert status_label.has_class("recording")

            # Change to ready
            widget.set_status("ready", "ready")
            assert status_label.has_class("ready")
            assert not status_label.has_class("recording")
            assert not status_label.has_class("transcribing")
            assert not status_label.has_class("error")

    @pytest.mark.asyncio
    async def test_set_model_info_basic(self):
        """Test set_model_info with basic parameters."""
        app = HeaderTestApp()
        async with app.run_test():
            widget = app.query_one(HeaderWidget)
            widget.set_model_info("turbo", "cuda")

            model_label = widget.query_one("#header-model", Label)
            assert model_label is not None

    @pytest.mark.asyncio
    async def test_set_model_info_with_fallback(self):
        """Test set_model_info with CPU fallback."""
        app = HeaderTestApp()
        async with app.run_test():
            widget = app.query_one(HeaderWidget)
            widget.set_model_info("base", "cpu", fallback=True)

            model_label = widget.query_one("#header-model", Label)
            assert model_label is not None

    @pytest.mark.asyncio
    async def test_set_model_info_without_fallback(self):
        """Test set_model_info without CPU fallback."""
        app = HeaderTestApp()
        async with app.run_test():
            widget = app.query_one(HeaderWidget)
            widget.set_model_info("large", "cuda", fallback=False)

            model_label = widget.query_one("#header-model", Label)
            assert model_label is not None

    @pytest.mark.asyncio
    async def test_header_displays_version(self):
        """Test that header displays the correct version."""
        app = HeaderTestApp(version="v3.1.4")
        async with app.run_test():
            widget = app.query_one(HeaderWidget)
            version_label = widget.query_one("#header-version", Label)
            assert version_label is not None

    @pytest.mark.asyncio
    async def test_status_icon_changes(self):
        """Test that status icon changes with different states."""
        app = HeaderTestApp()
        async with app.run_test():
            widget = app.query_one(HeaderWidget)

            # Test each state
            for state in ["ready", "recording", "transcribing", "error"]:
                widget.set_status(state, f"test {state}")
                status_label = widget.query_one("#status", Label)
                assert status_label.has_class(state)

    @pytest.mark.asyncio
    async def test_multiple_status_updates(self):
        """Test multiple consecutive status updates."""
        app = HeaderTestApp()
        async with app.run_test():
            widget = app.query_one(HeaderWidget)

            # Rapid status changes
            widget.set_status("ready", "ready")
            widget.set_status("recording", "recording")
            widget.set_status("transcribing", "transcribing")
            widget.set_status("ready", "ready again")

            status_label = widget.query_one("#status", Label)
            assert status_label.has_class("ready")
            assert not status_label.has_class("recording")
            assert not status_label.has_class("transcribing")

    @pytest.mark.asyncio
    async def test_model_info_updates(self):
        """Test multiple model info updates."""
        app = HeaderTestApp()
        async with app.run_test():
            widget = app.query_one(HeaderWidget)

            # Update model info multiple times
            widget.set_model_info("tiny", "cpu")
            widget.set_model_info("base", "cuda")
            widget.set_model_info("turbo", "cuda", fallback=False)

            model_label = widget.query_one("#header-model", Label)
            assert model_label is not None
