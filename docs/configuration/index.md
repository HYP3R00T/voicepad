---
icon: lucide/settings
---

# Configuration

VoicePad stores its configuration in a YAML file. You can edit this file directly or change settings from the **Settings** tab in the interface.

## Config File Location

The global config file lives at:

```sh
~/.config/voicepad/voicepad.yaml
```

On Windows, this expands to `C:\Users\<YourName>\.config\voicepad\voicepad.yaml`.

This file is created automatically when you save settings from the interface.

## Settings

```yaml
recordings_path: ~/.config/voicepad/data/recordings
markdown_path: ~/.config/voicepad/data/markdown
model_cache_path: ~/.config/voicepad/models
vad_model_path: ~/.config/voicepad/models/vad
logs_path: ~/.config/voicepad/logs
log_level: INFO

input_device_index: null
recording_prefix: recording

global_hotkey: <ctrl>+<alt>+v

transcription_model: turbo
transcription_device: auto
transcription_compute_type: auto
language: en
beam_size: 1
transcription_vad_filter: false
initial_prompt: "Hello. This is a transcription with proper punctuation, capitalization, and grammar."
no_speech_threshold: 0.6
hallucination_silence_threshold: 2.0
hallucination_max_repetitions: 3
min_audio_duration_s: 0.5
trim_trailing_silence_rms_threshold: 0.01
trim_trailing_silence_frame_ms: 20

silence_threshold_ms: 1000
min_chunk_s: 15.0
max_chunk_s: 29.0
overlap_s: 0.5
stream_poll_interval_s: 0.3
stream_context_chars: 200

dedup_prev_tail_words: 50
dedup_full_duplicate_threshold: 0.8
dedup_min_overlap_words_for_partial: 3
dedup_partial_lead_words: 5

vad_threshold: 0.5
vad_min_speech_duration_ms: 250
vad_speech_pad_ms: 30
vad_model_filename: silero_vad_v6.onnx
vad_model_url: https://raw.githubusercontent.com/SYSTRAN/faster-whisper/master/faster_whisper/assets/silero_vad_v6.onnx
vad_download_chunk_size: 8192

local_agreement_mic: false
local_agreement_file: true

model_warmup_enabled: true
model_warmup_duration_s: 0.5
model_warmup_language: en
model_warmup_beam_size: 1
model_warmup_vad_filter: false
```

!!! tip "Global Hotkey Format"
    The `global_hotkey` setting uses natural hotkey strings like `ctrl+shift+space`, `alt+v`, or `f9`.
    The hotkey works system-wide on Windows, allowing you to start/stop recording from any application.

!!! note "Settings visibility"
    Not all settings are shown in the Settings tab interface. Advanced runtime tuning such as inference thresholds, VAD behavior, overlap deduplication, warm-up behavior, and download locations can be changed by editing the YAML file directly.

For details on individual settings, see:

- [Whisper Models](configuration/models/): choose the right model for your hardware
- [Input Device](configuration/input-device/): select which microphone to use
- [Output Paths](configuration/output-paths/): change where recordings and transcriptions are saved
- [GPU Acceleration](configuration/gpu/): NVIDIA GPU requirements and performance

## Changing Settings

The easiest way to configure VoicePad is through the interface:

1. Open VoicePad: `uvx voicepad`
2. Go to the **Settings** tab
3. Change any value
4. Press **Save**

Changes take effect immediately. If you change the transcription model, VoicePad reloads it automatically.
