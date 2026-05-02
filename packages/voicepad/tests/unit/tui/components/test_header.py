"""Tests for HeaderWidget component."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Label
from voicepad.tui.components.header import HeaderWidget


class HeaderTestApp(App[None]):
    """Test app for HeaderWidget."""

    def __init__(self, version: str = "dev") -> None:
        super().__init__()
        self.version = version

    def compose(self) -> ComposeResult:
        yield HeaderWidget(version=self.version)


class TestHeaderWidget:
    """Test suite for HeaderWidget component."""

    def test_init_with_default_version(self) -> None:
        """HeaderWidget initializes with default version 'dev'."""
        widget = HeaderWidget()
        assert widget._version == "dev"
        assert widget.id == "header"

    def test_init_with_custom_version(self) -> None:
        """HeaderWidget initializes with custom version string."""
        widget = HeaderWidget(version="1.2.3")
        assert widget._version == "1.2.3"
        assert widget.id == "header"

    def test_compose_yields_four_labels(self) -> None:
        """compose() yields exactly four Label widgets."""
        widget = HeaderWidget(version="1.0.0")
        children = list(widget.compose())
        assert len(children) == 4
        assert all(isinstance(child, Label) for child in children)

    def test_compose_label_ids(self) -> None:
        """compose() yields labels with correct IDs."""
        widget = HeaderWidget(version="1.0.0")
        children = list(widget.compose())
        ids = [child.id for child in children]
        assert ids == ["header-title", "header-version", "status", "header-model"]

    async def test_header_displays_correct_version(self) -> None:
        """Header displays the correct version string."""
        app = HeaderTestApp(version="2.5.0")
        async with app.run_test():
            header = app.query_one(HeaderWidget)
            version_label = header.query_one("#header-version", Label)
            # Access the label's render method to get text
            rendered = version_label.render()
            assert "2.5.0" in str(rendered)

    async def test_header_displays_title(self) -> None:
        """Header displays VoicePad title."""
        app = HeaderTestApp()
        async with app.run_test():
            header = app.query_one(HeaderWidget)
            title_label = header.query_one("#header-title", Label)
            rendered = title_label.render()
            assert "VoicePad" in str(rendered)

    async def test_set_status_ready(self) -> None:
        """set_status() updates status label for 'ready' state."""
        app = HeaderTestApp()
        async with app.run_test():
            header = app.query_one(HeaderWidget)
            header.set_status("ready", "Ready to record")

            status_label = header.query_one("#status", Label)
            assert status_label.has_class("ready")
            assert not status_label.has_class("recording")
            assert not status_label.has_class("transcribing")
            assert not status_label.has_class("error")

            rendered = status_label.render()
            assert "Ready to record" in str(rendered)

    async def test_set_status_recording(self) -> None:
        """set_status() updates status label for 'recording' state with correct icon."""
        app = HeaderTestApp()
        async with app.run_test():
            header = app.query_one(HeaderWidget)
            header.set_status("recording", "Recording...")

            status_label = header.query_one("#status", Label)
            assert status_label.has_class("recording")

            rendered = status_label.render()
            rendered_str = str(rendered)
            assert "\U000f044a" in rendered_str  # Recording icon
            assert "Recording..." in rendered_str

    async def test_set_status_transcribing(self) -> None:
        """set_status() updates status label for 'transcribing' state with correct icon."""
        app = HeaderTestApp()
        async with app.run_test():
            header = app.query_one(HeaderWidget)
            header.set_status("transcribing", "Processing audio")

            status_label = header.query_one("#status", Label)
            assert status_label.has_class("transcribing")

            rendered = status_label.render()
            rendered_str = str(rendered)
            assert "\U000f051f" in rendered_str  # Transcribing icon
            assert "Processing audio" in rendered_str

    async def test_set_status_error(self) -> None:
        """set_status() updates status label for 'error' state with correct icon."""
        app = HeaderTestApp()
        async with app.run_test():
            header = app.query_one(HeaderWidget)
            header.set_status("error", "Failed to record")

            status_label = header.query_one("#status", Label)
            assert status_label.has_class("error")

            rendered = status_label.render()
            rendered_str = str(rendered)
            assert "\U000f0159" in rendered_str  # Error icon
            assert "Failed to record" in rendered_str

    async def test_set_status_unknown_state_uses_default_icon(self) -> None:
        """set_status() uses default icon for unknown state."""
        app = HeaderTestApp()
        async with app.run_test():
            header = app.query_one(HeaderWidget)
            header.set_status("unknown", "Unknown state")

            status_label = header.query_one("#status", Label)
            assert status_label.has_class("unknown")

            rendered = status_label.render()
            rendered_str = str(rendered)
            assert "\U000f051f" in rendered_str  # Default icon
            assert "Unknown state" in rendered_str

    async def test_set_status_removes_previous_classes(self) -> None:
        """set_status() removes all state classes before adding new one."""
        app = HeaderTestApp()
        async with app.run_test():
            header = app.query_one(HeaderWidget)

            # Set initial state
            header.set_status("ready", "Ready")
            status_label = header.query_one("#status", Label)
            assert status_label.has_class("ready")

            # Change to different state
            header.set_status("error", "Error occurred")
            assert not status_label.has_class("ready")
            assert status_label.has_class("error")

    async def test_set_status_empty_state_no_class_added(self) -> None:
        """set_status() with empty state string doesn't add a class."""
        app = HeaderTestApp()
        async with app.run_test():
            header = app.query_one(HeaderWidget)
            header.set_status("", "No state")

            status_label = header.query_one("#status", Label)
            # Should not have any state classes
            assert not status_label.has_class("ready")
            assert not status_label.has_class("recording")
            assert not status_label.has_class("transcribing")
            assert not status_label.has_class("error")

            rendered = status_label.render()
            assert "No state" in str(rendered)

    async def test_set_model_info_basic(self) -> None:
        """set_model_info() updates model label with model and device info."""
        app = HeaderTestApp()
        async with app.run_test():
            header = app.query_one(HeaderWidget)
            header.set_model_info("base.en", "cuda")

            model_label = header.query_one("#header-model", Label)
            rendered = model_label.render()
            rendered_str = str(rendered)

            assert "base.en" in rendered_str
            assert "cuda" in rendered_str
            assert "model:" in rendered_str
            assert "device:" in rendered_str

    async def test_set_model_info_with_fallback(self) -> None:
        """set_model_info() includes fallback text when fallback=True."""
        app = HeaderTestApp()
        async with app.run_test():
            header = app.query_one(HeaderWidget)
            header.set_model_info("tiny", "cpu", fallback=True)

            model_label = header.query_one("#header-model", Label)
            rendered = model_label.render()
            rendered_str = str(rendered)

            assert "tiny" in rendered_str
            assert "cpu" in rendered_str
            assert "cpu fallback" in rendered_str

    async def test_set_model_info_without_fallback(self) -> None:
        """set_model_info() excludes fallback text when fallback=False."""
        app = HeaderTestApp()
        async with app.run_test():
            header = app.query_one(HeaderWidget)
            header.set_model_info("large-v3", "cuda", fallback=False)

            model_label = header.query_one("#header-model", Label)
            rendered = model_label.render()
            rendered_str = str(rendered)

            assert "large-v3" in rendered_str
            assert "cuda" in rendered_str
            assert "fallback" not in rendered_str

    async def test_set_model_info_default_fallback_is_false(self) -> None:
        """set_model_info() defaults to fallback=False."""
        app = HeaderTestApp()
        async with app.run_test():
            header = app.query_one(HeaderWidget)
            header.set_model_info("medium", "cuda")

            model_label = header.query_one("#header-model", Label)
            rendered = model_label.render()
            rendered_str = str(rendered)

            assert "medium" in rendered_str
            assert "cuda" in rendered_str
            assert "fallback" not in rendered_str

    async def test_set_model_info_formatting(self) -> None:
        """set_model_info() uses correct formatting with dim tags."""
        app = HeaderTestApp()
        async with app.run_test():
            header = app.query_one(HeaderWidget)
            header.set_model_info("base", "cuda")

            model_label = header.query_one("#header-model", Label)
            rendered = model_label.render()
            rendered_str = str(rendered)

            # Check that dim tags are present in the formatted string
            assert "model:" in rendered_str
            assert "device:" in rendered_str

    async def test_status_icon_mapping(self) -> None:
        """Verify all status states have correct icon mappings."""
        app = HeaderTestApp()
        async with app.run_test():
            header = app.query_one(HeaderWidget)

            # Test each state and verify icon is present
            states_and_icons = [
                ("ready", ""),  # Empty icon for ready
                ("recording", "\U000f044a"),
                ("transcribing", "\U000f051f"),
                ("error", "\U000f0159"),
            ]

            for state, expected_icon in states_and_icons:
                header.set_status(state, f"Test {state}")
                status_label = header.query_one("#status", Label)
                rendered = status_label.render()
                rendered_str = str(rendered)

                if expected_icon:  # Only check if icon is not empty
                    assert expected_icon in rendered_str, f"Icon for {state} not found"
                assert f"Test {state}" in rendered_str

    async def test_multiple_status_changes(self) -> None:
        """Test multiple consecutive status changes work correctly."""
        app = HeaderTestApp()
        async with app.run_test():
            header = app.query_one(HeaderWidget)

            # Cycle through different states
            header.set_status("ready", "Ready")
            status_label = header.query_one("#status", Label)
            assert status_label.has_class("ready")

            header.set_status("recording", "Recording")
            assert not status_label.has_class("ready")
            assert status_label.has_class("recording")

            header.set_status("transcribing", "Transcribing")
            assert not status_label.has_class("recording")
            assert status_label.has_class("transcribing")

            header.set_status("error", "Error")
            assert not status_label.has_class("transcribing")
            assert status_label.has_class("error")

    async def test_multiple_model_info_updates(self) -> None:
        """Test multiple consecutive model info updates work correctly."""
        app = HeaderTestApp()
        async with app.run_test():
            header = app.query_one(HeaderWidget)

            # First update
            header.set_model_info("tiny", "cpu")
            model_label = header.query_one("#header-model", Label)
            rendered = model_label.render()
            assert "tiny" in str(rendered)
            assert "cpu" in str(rendered)

            # Second update
            header.set_model_info("base", "cuda")
            rendered = model_label.render()
            rendered_str = str(rendered)
            assert "base" in rendered_str
            assert "cuda" in rendered_str
            assert "tiny" not in rendered_str  # Old value should be gone
