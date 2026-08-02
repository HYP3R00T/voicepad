import os
from pathlib import Path

import numpy as np
import pytest
from voicepad_core.artifacts import ArtifactStore
from voicepad_core.deployments import PARAKEET_V3_CUDA, PARAKEET_V3_MANIFEST
from voicepad_core.inference import (
    CancellationToken,
    TranscriptionIntent,
    TransformersParakeetTDTSession,
    admit_cuda_device,
)
from voicepad_core.preprocessing import PreprocessedAudio


@pytest.mark.gpu
def test_official_parakeet_session_loads_warms_and_reuses_offline() -> None:
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    snapshot = ArtifactStore(cache_home / "voicepad-v2" / "artifacts").verify(PARAKEET_V3_MANIFEST)
    device = admit_cuda_device(PARAKEET_V3_CUDA)
    session = TransformersParakeetTDTSession(PARAKEET_V3_CUDA, PARAKEET_V3_MANIFEST, snapshot, device)
    silence = PreprocessedAudio(np.zeros(16_000, dtype=np.float32), sample_rate=16_000, channels=1)

    try:
        session.warm()
        first = session.transcribe(silence, TranscriptionIntent(), CancellationToken())
        second = session.transcribe(silence, TranscriptionIntent(), CancellationToken())
    finally:
        session.close()

    assert first == second
    assert first.cancelled is False
