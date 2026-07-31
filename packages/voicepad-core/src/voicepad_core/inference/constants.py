DEVICE = "cuda"
COMPUTE_TYPE = "float16"
CPU_COMPUTE_TYPE = "int8"
BEAM_SIZE = 5
LANGUAGE = "en"
CUDA_ERROR_KEYWORDS: tuple[str, ...] = (
    "cublas",
    "cuda",
    "cudnn",
    "nvrtc",
    "cufft",
    "curand",
)
