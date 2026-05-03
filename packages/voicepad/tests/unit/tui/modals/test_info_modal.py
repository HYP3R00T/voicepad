"""Tests for InfoModal."""

from __future__ import annotations

from textual.app import App
from textual.widgets import Button, Link, Static
from voicepad.tui.modals.info_modal import InfoModal


class InfoModalTestApp(App[None]):
    """Test app for InfoModal."""

    def __init__(self) -> None:
        super().__init__()
        self.dismissed = False

    def on_mount(self) -> None:
        def callback(result: None) -> None:
            self.dismissed = True

        self.push_screen(InfoModal(), callback)


class TestInfoModal:
    """Test suite for InfoModal."""

    async def test_modal_displays_title(self) -> None:
        """Modal displays VoicePad title."""
        app = InfoModalTestApp()
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, InfoModal)

            title = modal.query_one("#info-title", Static)
            rendered = title.render()
            assert "VoicePad" in str(rendered)

    async def test_modal_displays_subtitle(self) -> None:
        """Modal displays subtitle about privacy."""
        app = InfoModalTestApp()
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, InfoModal)

            subtitle = modal.query_one("#info-subtitle", Static)
            rendered = subtitle.render()
            rendered_str = str(rendered)
            assert "Your voice, your data" in rendered_str
            assert "never leaves your machine" in rendered_str

    async def test_modal_displays_guarantees(self) -> None:
        """Modal displays privacy guarantees."""
        app = InfoModalTestApp()
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, InfoModal)

            guarantees = modal.query("#info-guarantees .guarantee-line")
            assert len(guarantees) == 3

            texts = [str(g.render()) for g in guarantees]
            combined = " ".join(texts)
            assert "local processing" in combined
            assert "GPU-accelerated" in combined
            assert "No cloud" in combined

    async def test_modal_displays_philosophy(self) -> None:
        """Modal displays build philosophy."""
        app = InfoModalTestApp()
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, InfoModal)

            philosophy = modal.query_one("#info-philosophy", Static)
            rendered = philosophy.render()
            rendered_str = str(rendered)
            assert "Python" in rendered_str
            assert "Textual" in rendered_str
            assert "Whisper" in rendered_str

    async def test_modal_has_github_link(self) -> None:
        """Modal has GitHub link."""
        app = InfoModalTestApp()
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, InfoModal)

            github_link = modal.query_one("#github-link", Link)
            assert "github.com/HYP3R00T/voicepad" in github_link.url

    async def test_modal_has_sponsor_link(self) -> None:
        """Modal has sponsor link."""
        app = InfoModalTestApp()
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, InfoModal)

            sponsor_link = modal.query_one("#sponsor-link", Link)
            assert "github.com/sponsors/HYP3R00T" in sponsor_link.url

    async def test_modal_displays_sponsor_title(self) -> None:
        """Modal displays sponsor section title."""
        app = InfoModalTestApp()
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, InfoModal)

            sponsor_title = modal.query_one("#info-sponsor-title", Static)
            rendered = sponsor_title.render()
            assert "Support the project" in str(rendered)

    async def test_modal_displays_microcopy(self) -> None:
        """Modal displays microcopy about community support."""
        app = InfoModalTestApp()
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, InfoModal)

            microcopy = modal.query_one("#info-microcopy", Static)
            rendered = microcopy.render()
            rendered_str = str(rendered)
            assert "Privacy-first" in rendered_str
            assert "community support" in rendered_str

    async def test_modal_displays_meta_info(self) -> None:
        """Modal displays version and author metadata."""
        app = InfoModalTestApp()
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, InfoModal)

            meta = modal.query_one("#info-meta", Static)
            rendered = meta.render()
            rendered_str = str(rendered)
            assert "Rajesh Das" in rendered_str
            assert "HYP3R00T" in rendered_str
            assert "MIT License" in rendered_str

    async def test_modal_has_close_button(self) -> None:
        """Modal has close button."""
        app = InfoModalTestApp()
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, InfoModal)

            close_btn = modal.query_one("#info-close-btn", Button)
            assert close_btn.label == "Close"

    async def test_close_button_dismisses_modal(self) -> None:
        """Clicking close button dismisses modal."""
        app = InfoModalTestApp()
        async with app.run_test() as pilot:
            modal = app.screen_stack[-1]
            assert isinstance(modal, InfoModal)

            await pilot.click("#info-close-btn")
            await pilot.pause()

            assert app.dismissed is True

    async def test_escape_key_dismisses_modal(self) -> None:
        """Pressing escape key dismisses modal."""
        app = InfoModalTestApp()
        async with app.run_test() as pilot:
            modal = app.screen_stack[-1]
            assert isinstance(modal, InfoModal)

            await pilot.press("escape")
            await pilot.pause()

            assert app.dismissed is True

    async def test_i_key_dismisses_modal(self) -> None:
        """Pressing 'i' key dismisses modal."""
        app = InfoModalTestApp()
        async with app.run_test() as pilot:
            modal = app.screen_stack[-1]
            assert isinstance(modal, InfoModal)

            await pilot.press("i")
            await pilot.pause()

            assert app.dismissed is True

    async def test_modal_structure_has_dialog_container(self) -> None:
        """Modal has proper dialog container structure."""
        app = InfoModalTestApp()
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, InfoModal)

            dialog = modal.query_one("#info-dialog", Static)
            assert dialog is not None

    async def test_modal_has_dividers(self) -> None:
        """Modal has divider elements."""
        app = InfoModalTestApp()
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, InfoModal)

            divider1 = modal.query_one("#info-divider", Static)
            divider2 = modal.query_one("#info-divider2", Static)
            assert divider1 is not None
            assert divider2 is not None

    async def test_modal_has_links_container(self) -> None:
        """Modal has links container."""
        app = InfoModalTestApp()
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, InfoModal)

            links = modal.query_one("#info-links", Static)
            assert links is not None

            # Check both links are in container
            github_link = modal.query_one("#github-link", Link)
            sponsor_link = modal.query_one("#sponsor-link", Link)
            assert github_link.parent == links
            assert sponsor_link.parent == links

    async def test_modal_has_link_separator(self) -> None:
        """Modal has separator between links."""
        app = InfoModalTestApp()
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, InfoModal)

            separators = modal.query(".link-separator")
            assert len(separators) >= 1

    async def test_bindings_are_defined(self) -> None:
        """Modal has correct key bindings defined."""
        modal = InfoModal()

        # Check bindings exist
        binding_keys = [b.key for b in modal.BINDINGS]
        assert "escape" in binding_keys
        assert "i" in binding_keys

    async def test_compose_yields_correct_structure(self) -> None:
        """compose() yields correct widget structure."""
        app = InfoModalTestApp()
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, InfoModal)

            # Check dialog container exists
            dialog = modal.query_one("#info-dialog", Static)
            assert dialog is not None

    async def test_on_button_pressed_dismisses(self) -> None:
        """on_button_pressed handler dismisses modal."""
        app = InfoModalTestApp()
        async with app.run_test() as pilot:
            modal = app.screen_stack[-1]
            assert isinstance(modal, InfoModal)

            # Simulate button press event
            close_btn = modal.query_one("#info-close-btn", Button)
            close_btn.press()
            await pilot.pause()

            assert app.dismissed is True

    async def test_modal_displays_version_in_meta(self) -> None:
        """Modal displays version information in metadata."""
        app = InfoModalTestApp()
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, InfoModal)

            meta = modal.query_one("#info-meta", Static)
            rendered = meta.render()
            rendered_str = str(rendered)
            # Should contain either "v" followed by version or "dev"
            assert ("v" in rendered_str) or ("dev" in rendered_str)
