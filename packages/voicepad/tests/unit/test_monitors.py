from __future__ import annotations

from voicepad.tui.monitors import Monitor, bottom_center_position, parse_active_monitors, select_monitor


def test_disabled_laptop_gap_does_not_shift_enabled_monitor_overlay() -> None:
    output = """Monitors: 1
 0: +HDMI-A-1 1920/530x1080/300+1920+0  HDMI-A-1
"""

    monitors = parse_active_monitors(output)
    selected = select_monitor(monitors, (2500, 500))

    assert selected == Monitor("HDMI-A-1", 1920, 0, 1920, 1080)
    assert bottom_center_position(selected, (200, 40)) == (2780, 980)


def test_pointer_selects_enabled_monitor_with_negative_coordinates() -> None:
    output = """Monitors: 2
 0: +*DP-1 2560/600x1440/340+0+0  DP-1
 1: +HDMI-1 1920/530x1080/300-1920+180  HDMI-1
"""

    selected = select_monitor(parse_active_monitors(output), (-500, 400))

    assert selected == Monitor("HDMI-1", -1920, 180, 1920, 1080)
    assert bottom_center_position(selected, (240, 40)) == (-1080, 1160)


def test_primary_monitor_is_fallback_when_pointer_is_outside_desktop() -> None:
    monitors = (
        Monitor("left", -1920, 0, 1920, 1080),
        Monitor("primary", 0, 0, 2560, 1440, primary=True),
    )

    assert select_monitor(monitors, (-1, -1)) == monitors[1]
