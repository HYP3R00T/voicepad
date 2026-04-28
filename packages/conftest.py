"""Root conftest for the voicepad test suite.

GPU tests are skipped by default. Pass --run-gpu to include them.

Usage:
    pytest packages              # fast — skips GPU tests
    pytest packages --run-gpu    # full — includes GPU tests
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


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-gpu"):
        return  # run everything

    skip_gpu = pytest.mark.skip(reason="GPU test — pass --run-gpu to run")
    for item in items:
        if item.get_closest_marker("gpu"):
            item.add_marker(skip_gpu)
