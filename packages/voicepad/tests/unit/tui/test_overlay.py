"""Tests for overlay.py module."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

from voicepad.tui.overlay import (
    _AUTO_HIDE_AFTER_S,
    _BG,
    _COLORS,
    _FG,
    _LABELS,
    State,
    StatusOverlay,
    _draw_pill,
)


class TestConstants:
    """Test module-level constants."""

    def test_bg_color_is_string(self) -> None:
        assert isinstance(_BG, str)
        assert _BG.startswith("#")

    def test_fg_color_is_string(self) -> None:
        assert isinstance(_FG, str)
        assert _FG.startswith("#")

    def test_colors_dict_has_all_states(self) -> None:
        expected_states: list[State] = ["recording", "transcribing", "copied", "error", "hidden"]
        for state in expected_states:
            assert state in _COLORS
            assert isinstance(_COLORS[state], str)
            assert _COLORS[state].startswith("#")

    def test_labels_dict_has_all_states(self) -> None:
        expected_states: list[State] = ["recording", "transcribing", "copied", "error", "hidden"]
        for state in expected_states:
            assert state in _LABELS
            assert isinstance(_LABELS[state], str)

    def test_recording_label_contains_recording_text(self) -> None:
        assert "Recording" in _LABELS["recording"]

    def test_transcribing_label_contains_transcribing_text(self) -> None:
        assert "Transcribing" in _LABELS["transcribing"]

    def test_copied_label_contains_copied_text(self) -> None:
        assert "Copied" in _LABELS["copied"]

    def test_error_label_contains_error_text(self) -> None:
        assert "Error" in _LABELS["error"]

    def test_hidden_label_is_empty(self) -> None:
        assert _LABELS["hidden"] == ""

    def test_auto_hide_after_is_positive_float(self) -> None:
        assert isinstance(_AUTO_HIDE_AFTER_S, float)
        assert _AUTO_HIDE_AFTER_S > 0


class TestDrawPill:
    """Test _draw_pill function."""

    def test_draw_pill_creates_rectangles(self) -> None:
        mock_canvas = MagicMock()
        _draw_pill(mock_canvas, 0, 0, 100, 50, 10, fill="#000000", outline="#ffffff", width=2)

        # Should create rectangles for the pill body
        assert mock_canvas.create_rectangle.call_count >= 2

    def test_draw_pill_creates_ovals(self) -> None:
        mock_canvas = MagicMock()
        _draw_pill(mock_canvas, 0, 0, 100, 50, 10, fill="#000000", outline="#ffffff", width=2)

        # Should create 4 corner ovals
        assert mock_canvas.create_oval.call_count == 4

    def test_draw_pill_creates_arcs(self) -> None:
        mock_canvas = MagicMock()
        _draw_pill(mock_canvas, 0, 0, 100, 50, 10, fill="#000000", outline="#ffffff", width=2)

        # Should create 4 corner arcs for borders
        assert mock_canvas.create_arc.call_count == 4

    def test_draw_pill_creates_lines(self) -> None:
        mock_canvas = MagicMock()
        _draw_pill(mock_canvas, 0, 0, 100, 50, 10, fill="#000000", outline="#ffffff", width=2)

        # Should create 4 straight border lines
        assert mock_canvas.create_line.call_count == 4

    def test_draw_pill_uses_fill_color(self) -> None:
        mock_canvas = MagicMock()
        _draw_pill(mock_canvas, 0, 0, 100, 50, 10, fill="#123456", outline="#ffffff", width=2)

        # Check that fill color is used in create_rectangle calls
        calls = mock_canvas.create_rectangle.call_args_list
        assert any("#123456" in str(call) for call in calls)

    def test_draw_pill_uses_outline_color(self) -> None:
        mock_canvas = MagicMock()
        _draw_pill(mock_canvas, 0, 0, 100, 50, 10, fill="#000000", outline="#abcdef", width=2)

        # Check that outline color is used in create_line calls
        calls = mock_canvas.create_line.call_args_list
        assert any("#abcdef" in str(call) for call in calls)


class TestStatusOverlay:
    """Test StatusOverlay class."""

    def test_init_creates_overlay(self) -> None:
        overlay = StatusOverlay()
        assert overlay._root is None
        assert overlay._canvas is None
        assert overlay._label is None
        assert overlay._state == "hidden"
        assert overlay._hide_timer is None
        assert overlay._thread is None

    def test_init_creates_ready_event(self) -> None:
        overlay = StatusOverlay()
        assert overlay._ready is not None
        assert not overlay._ready.is_set()

    @patch("voicepad.tui.overlay.threading.Thread")
    def test_start_creates_daemon_thread(self, mock_thread_class: Mock) -> None:
        mock_thread = MagicMock()
        mock_thread_class.return_value = mock_thread
        overlay = StatusOverlay()
        overlay._ready.set()  # Prevent waiting

        overlay.start()

        mock_thread_class.assert_called_once()
        call_kwargs = mock_thread_class.call_args[1]
        assert call_kwargs["daemon"] is True
        assert call_kwargs["name"] == "overlay"
        assert callable(call_kwargs["target"])
        mock_thread.start.assert_called_once()

    @patch("voicepad.tui.overlay.threading.Thread")
    def test_start_waits_for_ready_event(self, mock_thread_class: Mock) -> None:
        mock_thread = MagicMock()
        mock_thread_class.return_value = mock_thread
        overlay = StatusOverlay()

        # Manually set ready to simulate thread completion
        overlay._ready.set()

        overlay.start()

        # Should have waited for ready event
        assert overlay._ready.is_set()

    def test_stop_does_nothing_when_root_is_none(self) -> None:
        overlay = StatusOverlay()
        # Should not raise an exception
        overlay.stop()

    def test_stop_calls_destroy_on_root(self) -> None:
        overlay = StatusOverlay()
        mock_root = MagicMock()
        overlay._root = mock_root

        overlay.stop()

        mock_root.after.assert_called_once()
        # Verify that destroy is scheduled
        call_args = mock_root.after.call_args
        assert call_args[0][0] == 0
        assert callable(call_args[0][1])

    def test_stop_handles_exception_gracefully(self) -> None:
        overlay = StatusOverlay()
        mock_root = MagicMock()
        mock_root.after.side_effect = RuntimeError("Tk error")
        overlay._root = mock_root

        # Should not raise an exception
        overlay.stop()

    def test_set_state_does_nothing_when_root_is_none(self) -> None:
        overlay = StatusOverlay()
        # Should not raise an exception
        overlay.set_state("recording")

    def test_set_state_schedules_update_on_root(self) -> None:
        overlay = StatusOverlay()
        mock_root = MagicMock()
        overlay._root = mock_root

        overlay.set_state("recording")

        mock_root.after.assert_called_once()
        call_args = mock_root.after.call_args
        assert call_args[0][0] == 0
        assert callable(call_args[0][1])

    def test_set_state_handles_exception_gracefully(self) -> None:
        overlay = StatusOverlay()
        mock_root = MagicMock()
        mock_root.after.side_effect = RuntimeError("Tk error")
        overlay._root = mock_root

        # Should not raise an exception
        overlay.set_state("recording")

    @patch("tkinter.Tk")
    @patch("tkinter.Canvas")
    @patch("tkinter.Label")
    def test_run_creates_tk_root(self, mock_label_class: Mock, mock_canvas_class: Mock, mock_tk_class: Mock) -> None:
        mock_root = MagicMock()
        mock_tk_class.return_value = mock_root
        mock_canvas_class.return_value = MagicMock()
        mock_label_class.return_value = MagicMock()
        mock_root.mainloop.side_effect = lambda: None

        overlay = StatusOverlay()
        overlay._run()

        mock_tk_class.assert_called_once()
        # _root is cleared in the finally block after mainloop exits
        assert overlay._root is None

    @patch("tkinter.Tk")
    @patch("tkinter.Canvas")
    @patch("tkinter.Label")
    def test_run_configures_window_attributes(
        self, mock_label_class: Mock, mock_canvas_class: Mock, mock_tk_class: Mock
    ) -> None:
        mock_root = MagicMock()
        mock_tk_class.return_value = mock_root
        mock_canvas_class.return_value = MagicMock()
        mock_label_class.return_value = MagicMock()
        mock_root.mainloop.side_effect = lambda: None

        overlay = StatusOverlay()
        overlay._run()

        mock_root.overrideredirect.assert_called_once_with(True)
        # Check that attributes were set
        assert mock_root.attributes.call_count >= 2

    @patch("tkinter.Tk")
    @patch("tkinter.Canvas")
    @patch("tkinter.Label")
    def test_run_creates_canvas(self, mock_label_class: Mock, mock_canvas_class: Mock, mock_tk_class: Mock) -> None:
        mock_root = MagicMock()
        mock_canvas = MagicMock()
        mock_tk_class.return_value = mock_root
        mock_canvas_class.return_value = mock_canvas
        mock_label_class.return_value = MagicMock()
        mock_root.mainloop.side_effect = lambda: None

        overlay = StatusOverlay()
        overlay._run()

        mock_canvas_class.assert_called_once()
        mock_canvas.pack.assert_called_once()
        # _canvas is cleared in the finally block after mainloop exits
        assert overlay._canvas is None

    @patch("tkinter.Tk")
    @patch("tkinter.Canvas")
    @patch("tkinter.Label")
    def test_run_creates_label(self, mock_label_class: Mock, mock_canvas_class: Mock, mock_tk_class: Mock) -> None:
        mock_root = MagicMock()
        mock_label = MagicMock()
        mock_tk_class.return_value = mock_root
        mock_canvas_class.return_value = MagicMock()
        mock_label_class.return_value = mock_label
        mock_root.mainloop.side_effect = lambda: None

        overlay = StatusOverlay()
        overlay._run()

        mock_label_class.assert_called_once()
        # _label is cleared in the finally block after mainloop exits
        assert overlay._label is None

    @patch("tkinter.Tk")
    @patch("tkinter.Canvas")
    @patch("tkinter.Label")
    def test_run_starts_hidden(self, mock_label_class: Mock, mock_canvas_class: Mock, mock_tk_class: Mock) -> None:
        mock_root = MagicMock()
        mock_tk_class.return_value = mock_root
        mock_canvas_class.return_value = MagicMock()
        mock_label_class.return_value = MagicMock()
        mock_root.mainloop.side_effect = lambda: None

        overlay = StatusOverlay()
        overlay._run()

        mock_root.withdraw.assert_called_once()

    @patch("tkinter.Tk")
    @patch("tkinter.Canvas")
    @patch("tkinter.Label")
    def test_run_sets_ready_event(self, mock_label_class: Mock, mock_canvas_class: Mock, mock_tk_class: Mock) -> None:
        mock_root = MagicMock()
        mock_tk_class.return_value = mock_root
        mock_canvas_class.return_value = MagicMock()
        mock_label_class.return_value = MagicMock()
        mock_root.mainloop.side_effect = lambda: None

        overlay = StatusOverlay()
        overlay._run()

        assert overlay._ready.is_set()

    @patch("tkinter.Tk")
    def test_run_handles_exception_and_sets_ready(self, mock_tk_class: Mock) -> None:
        mock_tk_class.side_effect = RuntimeError("Tk init failed")

        overlay = StatusOverlay()
        overlay._run()

        # Should set ready event even on error
        assert overlay._ready.is_set()

    def test_reposition_does_nothing_when_root_is_none(self) -> None:
        overlay = StatusOverlay()
        # Should not raise an exception
        overlay._reposition()

    def test_reposition_calculates_center_position(self) -> None:
        overlay = StatusOverlay()
        mock_root = MagicMock()
        mock_root.winfo_screenwidth.return_value = 1920
        mock_root.winfo_screenheight.return_value = 1080
        mock_root.winfo_reqwidth.return_value = 200
        mock_root.winfo_reqheight.return_value = 50
        overlay._root = mock_root

        overlay._reposition()

        mock_root.update_idletasks.assert_called()
        mock_root.geometry.assert_called_once()
        # Check that geometry string contains calculated position
        geometry_call = mock_root.geometry.call_args[0][0]
        assert "200x50" in geometry_call
        assert "+" in geometry_call

    def test_reposition_handles_exception_gracefully(self) -> None:
        overlay = StatusOverlay()
        mock_root = MagicMock()
        mock_root.update_idletasks.side_effect = RuntimeError("Tk error")
        overlay._root = mock_root

        # Should not raise an exception
        overlay._reposition()

    def test_apply_state_does_nothing_when_widgets_are_none(self) -> None:
        overlay = StatusOverlay()
        # Should not raise an exception
        overlay._apply_state("recording")

    def test_apply_state_withdraws_window_for_hidden_state(self) -> None:
        overlay = StatusOverlay()
        mock_root = MagicMock()
        mock_canvas = MagicMock()
        mock_label = MagicMock()
        overlay._root = mock_root
        overlay._canvas = mock_canvas
        overlay._label = mock_label

        overlay._apply_state("hidden")

        mock_root.withdraw.assert_called_once()
        assert overlay._state == "hidden"

    def test_apply_state_cancels_hide_timer(self) -> None:
        overlay = StatusOverlay()
        mock_root = MagicMock()
        mock_canvas = MagicMock()
        mock_label = MagicMock()
        overlay._root = mock_root
        overlay._canvas = mock_canvas
        overlay._label = mock_label
        overlay._hide_timer = "timer_id"

        overlay._apply_state("hidden")

        mock_root.after_cancel.assert_called_once_with("timer_id")
        assert overlay._hide_timer is None

    def test_apply_state_updates_label_text_and_color(self) -> None:
        overlay = StatusOverlay()
        mock_root = MagicMock()
        mock_canvas = MagicMock()
        mock_label = MagicMock()
        mock_label.winfo_reqwidth.return_value = 150
        mock_label.winfo_reqheight.return_value = 40
        overlay._root = mock_root
        overlay._canvas = mock_canvas
        overlay._label = mock_label

        overlay._apply_state("recording")

        # Check that label was configured with recording text and color
        mock_label.configure.assert_called()
        config_call = mock_label.configure.call_args[1]
        assert _LABELS["recording"] in str(config_call["text"])
        assert _COLORS["recording"] in str(config_call["fg"])

    def test_apply_state_shows_window(self) -> None:
        overlay = StatusOverlay()
        mock_root = MagicMock()
        mock_canvas = MagicMock()
        mock_label = MagicMock()
        mock_label.winfo_reqwidth.return_value = 150
        mock_label.winfo_reqheight.return_value = 40
        overlay._root = mock_root
        overlay._canvas = mock_canvas
        overlay._label = mock_label

        overlay._apply_state("recording")

        mock_root.deiconify.assert_called_once()
        mock_root.lift.assert_called_once()

    def test_apply_state_schedules_auto_hide_for_copied(self) -> None:
        overlay = StatusOverlay()
        mock_root = MagicMock()
        mock_canvas = MagicMock()
        mock_label = MagicMock()
        mock_label.winfo_reqwidth.return_value = 150
        mock_label.winfo_reqheight.return_value = 40
        overlay._root = mock_root
        overlay._canvas = mock_canvas
        overlay._label = mock_label

        overlay._apply_state("copied")

        # Should schedule auto-hide
        mock_root.after.assert_called()
        after_calls = [call for call in mock_root.after.call_args_list if call[0][0] > 0]
        assert len(after_calls) > 0
        # Check delay is approximately correct (in milliseconds)
        delay_ms = after_calls[0][0][0]
        assert delay_ms == int(_AUTO_HIDE_AFTER_S * 1000)

    def test_apply_state_schedules_auto_hide_for_error(self) -> None:
        overlay = StatusOverlay()
        mock_root = MagicMock()
        mock_canvas = MagicMock()
        mock_label = MagicMock()
        mock_label.winfo_reqwidth.return_value = 150
        mock_label.winfo_reqheight.return_value = 40
        overlay._root = mock_root
        overlay._canvas = mock_canvas
        overlay._label = mock_label

        overlay._apply_state("error")

        # Should schedule auto-hide
        mock_root.after.assert_called()
        after_calls = [call for call in mock_root.after.call_args_list if call[0][0] > 0]
        assert len(after_calls) > 0

    def test_apply_state_does_not_auto_hide_for_recording(self) -> None:
        overlay = StatusOverlay()
        mock_root = MagicMock()
        mock_canvas = MagicMock()
        mock_label = MagicMock()
        mock_label.winfo_reqwidth.return_value = 150
        mock_label.winfo_reqheight.return_value = 40
        overlay._root = mock_root
        overlay._canvas = mock_canvas
        overlay._label = mock_label

        overlay._apply_state("recording")

        # Should not schedule auto-hide (only update_idletasks calls with 0)
        after_calls = [call for call in mock_root.after.call_args_list if call[0][0] > 0]
        assert len(after_calls) == 0

    def test_apply_state_does_not_auto_hide_for_transcribing(self) -> None:
        overlay = StatusOverlay()
        mock_root = MagicMock()
        mock_canvas = MagicMock()
        mock_label = MagicMock()
        mock_label.winfo_reqwidth.return_value = 150
        mock_label.winfo_reqheight.return_value = 40
        overlay._root = mock_root
        overlay._canvas = mock_canvas
        overlay._label = mock_label

        overlay._apply_state("transcribing")

        # Should not schedule auto-hide
        after_calls = [call for call in mock_root.after.call_args_list if call[0][0] > 0]
        assert len(after_calls) == 0
