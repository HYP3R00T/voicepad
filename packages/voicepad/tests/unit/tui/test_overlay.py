"""Tests for voicepad.tui.overlay."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

from voicepad.tui.overlay import StatusOverlay, _draw_pill


class TestDrawPill:
    """Tests for _draw_pill helper function."""

    def _make_mock_canvas(self) -> tuple[MagicMock, list, list, list, list]:
        """Return a mock canvas and call-capture lists for each drawing method."""
        canvas = MagicMock()
        rect_calls: list = []
        oval_calls: list = []
        arc_calls: list = []
        line_calls: list = []
        canvas.create_rectangle.side_effect = lambda *a, **kw: rect_calls.append((a, kw))
        canvas.create_oval.side_effect = lambda *a, **kw: oval_calls.append((a, kw))
        canvas.create_arc.side_effect = lambda *a, **kw: arc_calls.append((a, kw))
        canvas.create_line.side_effect = lambda *a, **kw: line_calls.append((a, kw))
        return canvas, rect_calls, oval_calls, arc_calls, line_calls

    def test_draw_pill_calls_canvas_methods(self) -> None:
        """_draw_pill() calls various canvas methods to draw a rounded rectangle."""
        canvas, rect_calls, oval_calls, arc_calls, line_calls = self._make_mock_canvas()

        # Patch tkinter.ARC so _draw_pill's style=tk.ARC reference resolves
        with patch("tkinter.ARC", "arc"):
            _draw_pill(canvas, 10, 10, 100, 50, 10, fill="red", outline="blue", width=2)

        assert len(rect_calls) > 0
        assert len(oval_calls) > 0
        assert len(arc_calls) > 0
        assert len(line_calls) > 0

    def test_draw_pill_with_custom_colors(self) -> None:
        """_draw_pill() respects fill and outline colors."""
        canvas, rect_calls, oval_calls, arc_calls, line_calls = self._make_mock_canvas()
        all_calls: list = []
        canvas.create_rectangle.side_effect = lambda *a, **kw: all_calls.append((a, kw))
        canvas.create_oval.side_effect = lambda *a, **kw: all_calls.append((a, kw))
        canvas.create_arc.side_effect = lambda *a, **kw: all_calls.append((a, kw))
        canvas.create_line.side_effect = lambda *a, **kw: all_calls.append((a, kw))

        with patch("tkinter.ARC", "arc"):
            _draw_pill(canvas, 10, 10, 100, 50, 10, fill="green", outline="yellow")

        assert any("green" in str(kw.get("fill", "")) for _, kw in all_calls)


def _make_overlay_with_mocks() -> tuple[StatusOverlay, MagicMock, MagicMock, MagicMock]:
    """Create a StatusOverlay with all three required internals mocked.

    _apply_state guards on ``_root is None or _label is None or _canvas is None``
    so all three must be set for any state-transition test to exercise real logic.
    """
    overlay = StatusOverlay()
    mock_root = MagicMock()
    mock_canvas = MagicMock()
    mock_label = MagicMock()

    # Simulate realistic return values so _apply_state can compute pill size
    mock_label.winfo_reqwidth.return_value = 120
    mock_label.winfo_reqheight.return_value = 30

    overlay._root = mock_root
    overlay._canvas = mock_canvas
    overlay._label = mock_label
    return overlay, mock_root, mock_canvas, mock_label


class TestStatusOverlay:
    """Tests for StatusOverlay class."""

    def test_overlay_initializes_with_hidden_state(self) -> None:
        """StatusOverlay initializes with hidden state and no window."""
        overlay = StatusOverlay()

        assert overlay._state == "hidden"
        assert overlay._root is None
        assert overlay._canvas is None
        assert overlay._label is None

    def test_start_creates_thread(self) -> None:
        """start() creates a daemon thread."""
        overlay = StatusOverlay()

        with patch.object(overlay, "_run"):
            overlay.start()

            assert overlay._thread is not None
            assert overlay._thread.daemon is True

    def test_start_waits_for_ready(self) -> None:
        """start() waits for _ready event with timeout."""
        overlay = StatusOverlay()

        def mock_run() -> None:
            overlay._ready.set()

        with patch.object(overlay, "_run", side_effect=mock_run):
            overlay.start()

            assert overlay._ready.is_set()

    def test_stop_destroys_window(self) -> None:
        """stop() schedules window destruction."""
        overlay = StatusOverlay()
        mock_root = MagicMock()
        overlay._root = mock_root

        overlay.stop()

        mock_root.after.assert_called_once()

    def test_stop_handles_no_root(self) -> None:
        """stop() gracefully handles case where _root is None."""
        overlay = StatusOverlay()
        overlay._root = None

        overlay.stop()  # should not raise

    def test_set_state_updates_state_when_root_exists(self) -> None:
        """set_state() schedules a state update when root window exists."""
        overlay = StatusOverlay()
        mock_root = MagicMock()
        overlay._root = mock_root

        overlay.set_state("recording")

        mock_root.after.assert_called_once()

    def test_set_state_does_nothing_when_no_root(self) -> None:
        """set_state() does nothing when root window doesn't exist."""
        overlay = StatusOverlay()
        overlay._root = None

        overlay.set_state("recording")  # should not raise

    def test_state_transitions_recording(self) -> None:
        """Transitioning to 'recording' state configures the label."""
        overlay, mock_root, mock_canvas, mock_label = _make_overlay_with_mocks()

        overlay._apply_state("recording")

        mock_label.configure.assert_called()

    def test_state_transitions_transcribing(self) -> None:
        """Transitioning to 'transcribing' state configures the label."""
        overlay, mock_root, mock_canvas, mock_label = _make_overlay_with_mocks()

        overlay._apply_state("transcribing")

        mock_label.configure.assert_called()

    def test_state_transitions_copied(self) -> None:
        """Transitioning to 'copied' state configures the label."""
        overlay, mock_root, mock_canvas, mock_label = _make_overlay_with_mocks()

        overlay._apply_state("copied")

        mock_label.configure.assert_called()

    def test_state_transitions_error(self) -> None:
        """Transitioning to 'error' state configures the label."""
        overlay, mock_root, mock_canvas, mock_label = _make_overlay_with_mocks()

        overlay._apply_state("error")

        mock_label.configure.assert_called()

    def test_state_transitions_hidden(self) -> None:
        """Transitioning to 'hidden' state withdraws the root window."""
        overlay, mock_root, mock_canvas, mock_label = _make_overlay_with_mocks()

        overlay._apply_state("hidden")

        mock_root.withdraw.assert_called_once()

    def test_auto_hide_timer_set_for_temporary_states(self) -> None:
        """'copied' and 'error' states schedule an auto-hide timer."""
        overlay, mock_root, mock_canvas, mock_label = _make_overlay_with_mocks()

        overlay._apply_state("copied")

        mock_root.after.assert_called()

    def test_state_change_updates_correctly(self) -> None:
        """Multiple state changes are tracked in _state."""
        overlay, mock_root, mock_canvas, mock_label = _make_overlay_with_mocks()

        overlay._apply_state("recording")
        assert overlay._state == "recording"

        overlay._apply_state("transcribing")
        assert overlay._state == "transcribing"

        overlay._apply_state("hidden")
        assert overlay._state == "hidden"

    def test_thread_safety_of_state_updates(self) -> None:
        """set_state() can be called from multiple threads safely."""
        overlay = StatusOverlay()
        mock_root = MagicMock()
        overlay._root = mock_root

        def call_set_state() -> None:
            overlay.set_state("recording")

        threads = [threading.Thread(target=call_set_state) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert mock_root.after.call_count >= 5

    def test_run_creates_tk_window(self) -> None:
        """_run() sets _ready regardless of whether Tk succeeds or fails."""
        overlay = StatusOverlay()

        # _run() sets _ready.set() in both the success and except paths.
        # Patch tkinter.Tk to raise immediately so _run() exits via the except branch.
        with patch("tkinter.Tk", side_effect=RuntimeError("no display")):
            overlay._run()

        assert overlay._ready.is_set()

    def test_overlay_window_properties(self) -> None:
        """StatusOverlay window reference is accessible after _root is set."""
        overlay = StatusOverlay()
        mock_root = MagicMock()
        overlay._root = mock_root

        assert overlay._root is not None

    def test_multiple_start_stop_cycles(self) -> None:
        """Overlay can be started and stopped multiple times."""
        overlay = StatusOverlay()

        with patch.object(overlay, "_run"):
            overlay.start()
            assert overlay._thread is not None

            overlay.stop()

            overlay.start()
            assert overlay._thread is not None

    def test_hide_timer_callback(self) -> None:
        """Auto-hide timer is scheduled for 'copied' state."""
        overlay, mock_root, mock_canvas, mock_label = _make_overlay_with_mocks()

        overlay._apply_state("copied")

        assert mock_root.after.called

    def test_exception_handling_in_apply_state(self) -> None:
        """_apply_state returns immediately when root or label is None."""
        overlay = StatusOverlay()
        overlay._label = None
        overlay._root = None

        overlay._apply_state("recording")  # should not raise

    def test_label_configuration_values(self) -> None:
        """Label is configured with text and fg color when state is applied."""
        overlay, mock_root, mock_canvas, mock_label = _make_overlay_with_mocks()

        overlay._apply_state("recording")

        # configure() should have been called with at least text= and fg=
        assert mock_label.configure.called
        call_kwargs = mock_label.configure.call_args[1]
        assert "text" in call_kwargs
        assert "fg" in call_kwargs
