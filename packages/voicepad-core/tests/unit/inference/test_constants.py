from voicepad_core.inference.constants import (
    BEAM_SIZE,
    COMPUTE_TYPE,
    CPU_COMPUTE_TYPE,
    CUDA_ERROR_KEYWORDS,
    DEVICE,
    LANGUAGE,
)


def test_inference_defaults_are_valid() -> None:
    assert DEVICE == "cuda"
    assert COMPUTE_TYPE == "float16"
    assert CPU_COMPUTE_TYPE == "int8"
    assert BEAM_SIZE > 0
    assert LANGUAGE == "en"


def test_cuda_error_keywords_are_normalized() -> None:
    assert {"cublas", "cuda", "cudnn"}.issubset(CUDA_ERROR_KEYWORDS)
    assert all(keyword == keyword.lower() for keyword in CUDA_ERROR_KEYWORDS)
