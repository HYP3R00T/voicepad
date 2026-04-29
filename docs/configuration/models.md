---
icon: lucide/brain
---

# Whisper Models

VoicePad uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper), a CTranslate2-optimised implementation of OpenAI Whisper.

## Default Model

The default model is **turbo**, a fine-tuned version of large-v3 that is significantly faster while maintaining high accuracy. It is the recommended model for most users with an NVIDIA GPU.

## Model Comparison

| Model | Parameters | VRAM | Speed | Accuracy | Language |
|---|---|---|---|---|---|
| `tiny` | 39 M | < 1 GB | Very fast | Low | Multilingual |
| `base` | 74 M | < 1 GB | Fast | Fair | Multilingual |
| `small` | 244 M | ~1 GB | Moderate | Good | Multilingual |
| `medium` | 769 M | ~2 GB | Slow | Very good | Multilingual |
| `large-v2` | 1.5 B | ~5 GB | Very slow | Excellent | Multilingual |
| `large-v3` | 1.5 B | ~5 GB | Very slow | Excellent | Multilingual |
| **`turbo`** | 809 M | ~3 GB | Moderate | Excellent | Multilingual |
| `distil-small.en` | 134 M | < 1 GB | Fast | Good | English only |
| `distil-medium.en` | 394 M | ~1 GB | Moderate | Very good | English only |
| `distil-large-v2` | 756 M | ~3 GB | Slow | Excellent | English only |
| `distil-large-v3` | 756 M | ~3 GB | Slow | Excellent | English only |

## Which Model Should I Use?

| Your Hardware | Recommended Model |
|---|---|
| NVIDIA GPU, 4 GB VRAM | `turbo` |
| NVIDIA GPU, 6-8 GB VRAM | `turbo` or `large-v3` |
| NVIDIA GPU, < 4 GB VRAM | `small` or `distil-small.en` |
| CPU only | `small` or `base` |

## Changing the Model

Open the **Settings** tab in VoicePad, select a model from the dropdown, and press **Save**. VoicePad reloads the model immediately.

The first time you use a new model, it downloads from HuggingFace. Download sizes range from ~75 MB (`tiny`) to ~3 GB (`large-v3`). After the first download, the model is cached locally and no network access is required.
