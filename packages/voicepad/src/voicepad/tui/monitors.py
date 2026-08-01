from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

_MONITOR_PATTERN = re.compile(
    r"^\s*\d+:\s+(?P<flags>[+*]*)(?P<name>\S+)\s+"
    r"(?P<width>\d+)(?:/\d+)?x(?P<height>\d+)(?:/\d+)?"
    r"(?P<x>[+-]\d+)(?P<y>[+-]\d+)"
)


@dataclass(frozen=True, slots=True)
class Monitor:
    name: str
    x: int
    y: int
    width: int
    height: int
    primary: bool = False

    def contains(self, point: tuple[int, int]) -> bool:
        point_x, point_y = point
        return self.x <= point_x < self.x + self.width and self.y <= point_y < self.y + self.height


def parse_active_monitors(output: str) -> tuple[Monitor, ...]:
    monitors = []
    for line in output.splitlines():
        match = _MONITOR_PATTERN.match(line)
        if match is None:
            continue
        monitors.append(
            Monitor(
                name=match.group("name"),
                x=int(match.group("x")),
                y=int(match.group("y")),
                width=int(match.group("width")),
                height=int(match.group("height")),
                primary="*" in match.group("flags"),
            )
        )
    return tuple(monitors)


def active_monitors() -> tuple[Monitor, ...]:
    try:
        completed = subprocess.run(
            ["xrandr", "--listactivemonitors"],
            capture_output=True,
            check=True,
            text=True,
            timeout=1,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    return parse_active_monitors(completed.stdout)


def select_monitor(monitors: tuple[Monitor, ...], pointer: tuple[int, int]) -> Monitor | None:
    for monitor in monitors:
        if monitor.contains(pointer):
            return monitor
    return next((monitor for monitor in monitors if monitor.primary), monitors[0] if monitors else None)


def bottom_center_position(monitor: Monitor, window: tuple[int, int], margin: int = 60) -> tuple[int, int]:
    window_width, window_height = window
    x = monitor.x + (monitor.width - window_width) // 2
    y = monitor.y + monitor.height - window_height - margin
    return x, y
