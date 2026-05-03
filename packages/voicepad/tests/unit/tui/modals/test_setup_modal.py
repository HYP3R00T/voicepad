"""Tests for SetupModal."""

from __future__ import annotations

from unittest.mock import MagicMock

from textual.app import App
from textual.widgets import Button, DataTable, ProgressBar, Select, Static
from voicepad.tui.modals.setup_modal import SetupModal


class SetupModalTestApp(App[None]):
    """Test app for SetupModal."""

    def __init__(self, config: MagicMock) -> None:
        super().__init__()
        self.config = config
        self.result: tuple[str, int | None] | None = None

    def on_mount(self) -> None:
        def callback(result: tuple[str, int | None] | None) -> None:
            self.result = result

        self.push_screen(SetupModal(self.config), callback)


def create_mock_config() -> MagicMock:
    """Create a mock config object."""
    config = MagicMock()
    config.transcription_model = "turbo"
    config.input_device_index = None
    return config


class TestSetupModal:
    """Test suite for SetupModal."""

    def test_init_stores_config(self) -> None:
        """SetupModal stores the config object."""
        config = create_mock_config()
        modal = SetupModal(config)
        assert modal._config == config

    def test_init_sets_initial_step_to_zero(self) -> None:
        """SetupModal initializes with step 0."""
        config = create_mock_config()
        modal = SetupModal(config)
        assert modal._step == 0

    def test_init_sets_chosen_model_from_config(self) -> None:
        """SetupModal initializes chosen model from config."""
        config = create_mock_config()
        config.transcription_model = "base.en"
        modal = SetupModal(config)
        assert modal._chosen_model == "base.en"

    def test_init_sets_chosen_device_from_config(self) -> None:
        """SetupModal initializes chosen device from config."""
        config = create_mock_config()
        config.input_device_index = 2
        modal = SetupModal(config)
        assert modal._chosen_device_index == 2

    def test_init_sets_downloading_to_false(self) -> None:
        """SetupModal initializes with downloading flag as False."""
        config = create_mock_config()
        modal = SetupModal(config)
        assert modal._downloading is False

    async def test_modal_displays_step_indicator(self) -> None:
        """Modal displays step indicator."""
        config = create_mock_config()
        app = SetupModalTestApp(config)
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, SetupModal)

            indicator = modal.query_one("#wizard-step-indicator", Static)
            rendered = indicator.render()
            assert "Step" in str(rendered)

    async def test_modal_has_wizard_body(self) -> None:
        """Modal has wizard body container."""
        config = create_mock_config()
        app = SetupModalTestApp(config)
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, SetupModal)

            body = modal.query_one("#wizard-body", Static)
            assert body is not None

    async def test_modal_has_back_button(self) -> None:
        """Modal has back button."""
        config = create_mock_config()
        app = SetupModalTestApp(config)
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, SetupModal)

            back_btn = modal.query_one("#wizard-back", Button)
            assert "Back" in str(back_btn.label)

    async def test_modal_has_next_button(self) -> None:
        """Modal has next button."""
        config = create_mock_config()
        app = SetupModalTestApp(config)
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, SetupModal)

            next_btn = modal.query_one("#wizard-next", Button)
            assert "Next" in str(next_btn.label)

    async def test_step_0_displays_welcome(self) -> None:
        """Step 0 displays welcome screen."""
        config = create_mock_config()
        app = SetupModalTestApp(config)
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, SetupModal)
            assert modal._step == 0

            # Check for welcome content
            titles = modal.query(".wizard-title")
            assert len(titles) > 0
            title_text = str(titles[0].render())
            assert "VoicePad" in title_text

    async def test_step_0_back_button_hidden(self) -> None:
        """Step 0 hides back button."""
        config = create_mock_config()
        app = SetupModalTestApp(config)
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, SetupModal)
            assert modal._step == 0

            back_btn = modal.query_one("#wizard-back", Button)
            assert back_btn.display is False

    async def test_step_0_displays_features(self) -> None:
        """Step 0 displays feature list."""
        config = create_mock_config()
        app = SetupModalTestApp(config)
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, SetupModal)

            features = modal.query(".wizard-features")
            if features:
                feature_text = str(features[0].render())
                assert "local" in feature_text or "GPU" in feature_text

    async def test_next_button_advances_step(self) -> None:
        """Clicking next button advances to next step."""
        config = create_mock_config()
        app = SetupModalTestApp(config)
        async with app.run_test() as pilot:
            modal = app.screen_stack[-1]
            assert isinstance(modal, SetupModal)
            assert modal._step == 0

            await pilot.click("#wizard-next")
            await pilot.pause()

            assert modal._step == 1

    async def test_step_1_displays_model_selection(self) -> None:
        """Step 1 displays model selection."""
        config = create_mock_config()
        app = SetupModalTestApp(config)
        async with app.run_test() as pilot:
            modal = app.screen_stack[-1]
            assert isinstance(modal, SetupModal)

            # Advance to step 1
            await pilot.click("#wizard-next")
            await pilot.pause()

            # Check for model select
            model_select = modal.query_one("#wizard-model-select", Select)
            assert model_select is not None

    async def test_step_1_has_model_hint(self) -> None:
        """Step 1 has model hint display."""
        config = create_mock_config()
        app = SetupModalTestApp(config)
        async with app.run_test() as pilot:
            modal = app.screen_stack[-1]
            assert isinstance(modal, SetupModal)

            await pilot.click("#wizard-next")
            await pilot.pause()

            hint = modal.query_one("#wizard-model-hint", Static)
            assert hint is not None

    async def test_step_1_has_download_status(self) -> None:
        """Step 1 has download status display."""
        config = create_mock_config()
        app = SetupModalTestApp(config)
        async with app.run_test() as pilot:
            modal = app.screen_stack[-1]
            assert isinstance(modal, SetupModal)

            await pilot.click("#wizard-next")
            await pilot.pause()

            status = modal.query_one("#wizard-download-status", Static)
            assert status is not None

    async def test_step_1_has_progress_bar(self) -> None:
        """Step 1 has progress bar."""
        config = create_mock_config()
        app = SetupModalTestApp(config)
        async with app.run_test() as pilot:
            modal = app.screen_stack[-1]
            assert isinstance(modal, SetupModal)

            await pilot.click("#wizard-next")
            await pilot.pause()

            progress = modal.query_one("#wizard-progress", ProgressBar)
            assert progress is not None
            # Progress bar should be hidden initially
            assert progress.display is False

    async def test_step_1_next_button_shows_download_text(self) -> None:
        """Step 1 next button shows download text."""
        config = create_mock_config()
        app = SetupModalTestApp(config)
        async with app.run_test() as pilot:
            modal = app.screen_stack[-1]
            assert isinstance(modal, SetupModal)

            await pilot.click("#wizard-next")
            await pilot.pause()

            next_btn = modal.query_one("#wizard-next", Button)
            assert "Download" in str(next_btn.label)

    async def test_back_button_goes_to_previous_step(self) -> None:
        """Clicking back button goes to previous step."""
        config = create_mock_config()
        app = SetupModalTestApp(config)
        async with app.run_test() as pilot:
            modal = app.screen_stack[-1]
            assert isinstance(modal, SetupModal)

            # Go to step 1
            await pilot.click("#wizard-next")
            await pilot.pause()
            assert modal._step == 1

            # Go back to step 0
            await pilot.click("#wizard-back")
            await pilot.pause()
            assert modal._step == 0

    async def test_step_indicator_updates_with_step(self) -> None:
        """Step indicator updates as steps change."""
        config = create_mock_config()
        app = SetupModalTestApp(config)
        async with app.run_test() as pilot:
            modal = app.screen_stack[-1]
            assert isinstance(modal, SetupModal)

            indicator = modal.query_one("#wizard-step-indicator", Static)

            # Step 0
            rendered = indicator.render()
            assert "Step 1 of 4" in str(rendered)

            # Advance to step 1
            await pilot.click("#wizard-next")
            await pilot.pause()

            rendered = indicator.render()
            assert "Step 2 of 4" in str(rendered)

    async def test_modal_structure_has_dialog_container(self) -> None:
        """Modal has proper dialog container structure."""
        config = create_mock_config()
        app = SetupModalTestApp(config)
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, SetupModal)

            dialog = modal.query_one("#wizard-dialog", Static)
            assert dialog is not None

    async def test_modal_structure_has_nav_container(self) -> None:
        """Modal has navigation container."""
        config = create_mock_config()
        app = SetupModalTestApp(config)
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, SetupModal)

            nav = modal.query_one("#wizard-nav", Static)
            assert nav is not None

    async def test_modal_has_spacer_in_nav(self) -> None:
        """Modal has spacer in navigation."""
        config = create_mock_config()
        app = SetupModalTestApp(config)
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, SetupModal)

            spacer = modal.query_one("#wizard-spacer", Static)
            assert spacer is not None

    async def test_bindings_list_is_empty(self) -> None:
        """SetupModal has empty bindings list (no escape to dismiss)."""
        config = create_mock_config()
        modal = SetupModal(config)
        assert modal.BINDINGS == []

    async def test_compose_yields_correct_structure(self) -> None:
        """compose() yields correct widget structure."""
        config = create_mock_config()
        app = SetupModalTestApp(config)
        async with app.run_test():
            modal = app.screen_stack[-1]
            assert isinstance(modal, SetupModal)

            # Check dialog container exists
            dialog = modal.query_one("#wizard-dialog", Static)
            assert dialog is not None

    async def test_on_model_changed_updates_chosen_model(self) -> None:
        """Changing model select updates chosen model."""
        config = create_mock_config()
        app = SetupModalTestApp(config)
        async with app.run_test() as pilot:
            modal = app.screen_stack[-1]
            assert isinstance(modal, SetupModal)

            # Go to step 1
            await pilot.click("#wizard-next")
            await pilot.pause()

            # Change model selection
            model_select = modal.query_one("#wizard-model-select", Select)
            model_select.value = "base.en"
            await pilot.pause()

            # Model should be updated
            assert modal._chosen_model == "base.en"

    async def test_set_download_status_updates_status_label(self) -> None:
        """_set_download_status updates the status label."""
        config = create_mock_config()
        app = SetupModalTestApp(config)
        async with app.run_test() as pilot:
            modal = app.screen_stack[-1]
            assert isinstance(modal, SetupModal)

            # Go to step 1
            await pilot.click("#wizard-next")
            await pilot.pause()

            # Set download status
            modal._set_download_status("Test status message")
            await pilot.pause()

            status = modal.query_one("#wizard-download-status", Static)
            rendered = status.render()
            assert "Test status message" in str(rendered)

    async def test_show_progress_bar_makes_bar_visible(self) -> None:
        """_show_progress_bar makes progress bar visible."""
        config = create_mock_config()
        app = SetupModalTestApp(config)
        async with app.run_test() as pilot:
            modal = app.screen_stack[-1]
            assert isinstance(modal, SetupModal)

            # Go to step 1
            await pilot.click("#wizard-next")
            await pilot.pause()

            progress = modal.query_one("#wizard-progress", ProgressBar)
            assert progress.display is False

            # Show progress bar
            modal._show_progress_bar()
            await pilot.pause()

            assert progress.display is True

    async def test_different_initial_models(self) -> None:
        """Modal correctly initializes with different models."""
        test_models = ["tiny", "base", "small", "turbo"]

        for model in test_models:
            config = create_mock_config()
            config.transcription_model = model
            modal = SetupModal(config)
            assert modal._chosen_model == model

    async def test_different_initial_device_indices(self) -> None:
        """Modal correctly initializes with different device indices."""
        test_indices = [None, 0, 1, 2, 5]

        for index in test_indices:
            config = create_mock_config()
            config.input_device_index = index
            modal = SetupModal(config)
            assert modal._chosen_device_index == index

    async def test_step_3_has_keybindings_table(self) -> None:
        """Step 3 displays keybindings table."""
        config = create_mock_config()
        app = SetupModalTestApp(config)
        async with app.run_test() as pilot:
            modal = app.screen_stack[-1]
            assert isinstance(modal, SetupModal)

            # Manually set to step 3
            modal._step = 3
            modal._render_step()
            await pilot.pause()

            table = modal.query_one("#wizard-keybindings", DataTable)
            assert table is not None

    async def test_step_3_next_button_shows_finish(self) -> None:
        """Step 3 next button shows 'Finish'."""
        config = create_mock_config()
        app = SetupModalTestApp(config)
        async with app.run_test() as pilot:
            modal = app.screen_stack[-1]
            assert isinstance(modal, SetupModal)

            # Manually set to step 3
            modal._step = 3
            modal._render_step()
            await pilot.pause()

            next_btn = modal.query_one("#wizard-next", Button)
            assert "Finish" in str(next_btn.label)
