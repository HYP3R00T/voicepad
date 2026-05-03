"""Tests for DeleteConfirmModal."""

from __future__ import annotations

from textual.app import App
from textual.widgets import Button, Static
from voicepad.tui.modals.delete_modal import DeleteConfirmModal


class DeleteModalTestApp(App[None]):
    """Test app for DeleteConfirmModal."""

    def __init__(self, entry_name: str = "test_recording.wav") -> None:
        super().__init__()
        self.entry_name = entry_name
        self.result: bool | None = None

    def on_mount(self) -> None:
        def callback(result: bool | None) -> None:
            self.result = result if result is not None else False

        self.push_screen(DeleteConfirmModal(self.entry_name), callback)


class TestDeleteConfirmModal:
    """Test suite for DeleteConfirmModal."""

    def test_init_stores_entry_name(self) -> None:
        """DeleteConfirmModal stores the entry name."""
        modal = DeleteConfirmModal("my_recording.wav")
        assert modal._entry_name == "my_recording.wav"

    async def test_modal_displays_title(self) -> None:
        """Modal displays delete confirmation title."""
        app = DeleteModalTestApp("test.wav")
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, DeleteConfirmModal)

            title = modal.query_one("#delete-title", Static)
            rendered = title.render()
            assert "Delete recording" in str(rendered)

    async def test_modal_displays_entry_name(self) -> None:
        """Modal displays the entry name to be deleted."""
        app = DeleteModalTestApp("important_recording.wav")
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, DeleteConfirmModal)

            name_label = modal.query_one("#delete-name", Static)
            rendered = name_label.render()
            assert "important_recording.wav" in str(rendered)

    async def test_modal_displays_warning_message(self) -> None:
        """Modal displays warning about permanent deletion."""
        app = DeleteModalTestApp("test.wav")
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, DeleteConfirmModal)

            body = modal.query_one("#delete-body", Static)
            rendered = body.render()
            rendered_str = str(rendered)
            assert "permanently delete" in rendered_str
            assert "WAV file" in rendered_str
            assert "transcription markdown" in rendered_str

    async def test_modal_has_cancel_button(self) -> None:
        """Modal has a cancel button."""
        app = DeleteModalTestApp("test.wav")
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, DeleteConfirmModal)

            cancel_btn = modal.query_one("#delete-cancel", Button)
            assert cancel_btn.label == "Cancel"

    async def test_modal_has_delete_button(self) -> None:
        """Modal has a delete button."""
        app = DeleteModalTestApp("test.wav")
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, DeleteConfirmModal)

            delete_btn = modal.query_one("#delete-confirm", Button)
            assert delete_btn.label == "Delete"

    async def test_cancel_button_dismisses_with_false(self) -> None:
        """Clicking cancel button dismisses modal with False."""
        app = DeleteModalTestApp("test.wav")
        async with app.run_test() as pilot:
            modal = app.screen_stack[-1]
            assert isinstance(modal, DeleteConfirmModal)

            modal.query_one("#delete-cancel", Button)
            await pilot.click("#delete-cancel")
            await pilot.pause()

            assert app.result is False

    async def test_delete_button_dismisses_with_true(self) -> None:
        """Clicking delete button dismisses modal with True."""
        app = DeleteModalTestApp("test.wav")
        async with app.run_test() as pilot:
            modal = app.screen_stack[-1]
            assert isinstance(modal, DeleteConfirmModal)

            await pilot.click("#delete-confirm")
            await pilot.pause()

            assert app.result is True

    async def test_escape_key_dismisses_with_false(self) -> None:
        """Pressing escape key dismisses modal with False."""
        app = DeleteModalTestApp("test.wav")
        async with app.run_test() as pilot:
            modal = app.screen_stack[-1]
            assert isinstance(modal, DeleteConfirmModal)

            await pilot.press("escape")
            await pilot.pause()

            assert app.result is False

    async def test_n_key_dismisses_with_false(self) -> None:
        """Pressing 'n' key dismisses modal with False."""
        app = DeleteModalTestApp("test.wav")
        async with app.run_test() as pilot:
            modal = app.screen_stack[-1]
            assert isinstance(modal, DeleteConfirmModal)

            await pilot.press("n")
            await pilot.pause()

            assert app.result is False

    async def test_action_dismiss_false(self) -> None:
        """action_dismiss_false dismisses with False."""
        app = DeleteModalTestApp("test.wav")
        async with app.run_test() as pilot:
            modal = app.screen_stack[-1]
            assert isinstance(modal, DeleteConfirmModal)

            modal.action_dismiss_false()
            await pilot.pause()

            assert app.result is False

    async def test_modal_structure_has_dialog_container(self) -> None:
        """Modal has proper dialog container structure."""
        app = DeleteModalTestApp("test.wav")
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, DeleteConfirmModal)

            dialog = modal.query_one("#delete-dialog", Static)
            assert dialog is not None

    async def test_modal_structure_has_nav_container(self) -> None:
        """Modal has navigation container with buttons."""
        app = DeleteModalTestApp("test.wav")
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, DeleteConfirmModal)

            nav = modal.query_one("#delete-nav", Static)
            assert nav is not None

            # Check buttons are in nav
            cancel_btn = modal.query_one("#delete-cancel", Button)
            delete_btn = modal.query_one("#delete-confirm", Button)
            assert cancel_btn.parent == nav
            assert delete_btn.parent == nav

    async def test_modal_has_spacer_between_buttons(self) -> None:
        """Modal has spacer element between buttons."""
        app = DeleteModalTestApp("test.wav")
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, DeleteConfirmModal)

            spacer = modal.query_one("#delete-spacer", Static)
            assert spacer is not None

    async def test_different_entry_names(self) -> None:
        """Modal correctly displays different entry names."""
        test_names = [
            "recording_001.wav",
            "meeting_notes.wav",
            "voice_memo_2024.wav",
        ]

        for name in test_names:
            app = DeleteModalTestApp(name)
            async with app.run_test():
                modal = app.screen_stack[-1]
                assert isinstance(modal, DeleteConfirmModal)

                name_label = modal.query_one("#delete-name", Static)
                rendered = name_label.render()
                assert name in str(rendered)

    async def test_bindings_are_defined(self) -> None:
        """Modal has correct key bindings defined."""
        modal = DeleteConfirmModal("test.wav")

        # Check bindings exist
        binding_keys = [b.key for b in modal.BINDINGS]
        assert "escape" in binding_keys
        assert "n" in binding_keys

    async def test_compose_yields_correct_structure(self) -> None:
        """compose() yields correct widget structure."""
        app = DeleteModalTestApp("test.wav")
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, DeleteConfirmModal)

            # Check dialog container exists
            dialog = modal.query_one("#delete-dialog", Static)
            assert dialog is not None
