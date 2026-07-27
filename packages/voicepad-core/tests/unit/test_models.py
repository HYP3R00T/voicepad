from pathlib import Path

import pytest
from voicepad_core.models import MODELS, ModelCompatibilityError, get_model, model_options, validate_model


def test_catalog_contains_only_curated_models() -> None:
    """The catalogue exposes the four deliberately supported models."""
    assert tuple(MODELS) == ("turbo", "small", "distil-large-v3.5", "parakeet-tdt-0.6b-v3")


def test_model_binds_artifact_to_backend() -> None:
    """Each catalogue entry contains its repository and runtime binding."""
    model = get_model("parakeet-tdt-0.6b-v3")
    assert (model.backend, model.precision, model.repo) == (
        "sherpa-onnx",
        "int8",
        "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
    )


def test_parakeet_requires_only_its_primary_artifact() -> None:
    """Parakeet no longer requires a companion BPE vocabulary for greedy decoding."""
    assert get_model("parakeet-tdt-0.6b-v3").files == (
        "decoder.int8.onnx",
        "encoder.int8.onnx",
        "joiner.int8.onnx",
        "tokens.txt",
    )


def test_unknown_model_is_rejected() -> None:
    """Unknown identifiers fail at the catalogue boundary."""
    with pytest.raises(ValueError, match="Unknown transcription model"):
        get_model("unknown")


def test_model_options_match_catalogue_order() -> None:
    """UI options are derived directly from the catalogue."""
    assert tuple(value for _, value in model_options()) == tuple(MODELS)


def test_validate_model_accepts_complete_directory(tmp_path: Path) -> None:
    """A directory containing every declared file is accepted."""
    model = get_model("small")
    for filename in model.files:
        (tmp_path / filename).write_text("model")
    assert validate_model(tmp_path, model) == tmp_path


def test_validate_model_lists_missing_files(tmp_path: Path) -> None:
    """An incomplete artifact reports its missing files."""
    with pytest.raises(ModelCompatibilityError, match="model.bin"):
        validate_model(tmp_path, get_model("small"))
