"""Root conftest for the voicepad test suite.

GPU and display tests are skipped by default.
Pass --run-gpu to include GPU tests, --run-display to include display tests.

Usage:
    pytest packages                          # fast — skips GPU and display tests
    pytest packages --run-gpu                # includes GPU tests
    pytest packages --run-display            # includes display tests
    pytest packages --run-gpu --run-display  # includes all tests
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="Include tests marked with @pytest.mark.gpu (requires CUDA).",
    )
    parser.addoption(
        "--run-display",
        action="store_true",
        default=False,
        help="Include tests marked with @pytest.mark.display (requires X11/Wayland).",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_gpu = pytest.mark.skip(reason="GPU test — pass --run-gpu to run")
    skip_display = pytest.mark.skip(reason="Display test — pass --run-display to run")

    run_gpu = config.getoption("--run-gpu")
    run_display = config.getoption("--run-display")

    for item in items:
        if not run_gpu and item.get_closest_marker("gpu"):
            item.add_marker(skip_gpu)
        if not run_display and item.get_closest_marker("display"):
            item.add_marker(skip_display)
