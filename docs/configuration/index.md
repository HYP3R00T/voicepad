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
vad_model_path: ~/.config/voicepad/data/models
input_device_index: null
recording_prefix: recording
transcription_model: turbo
transcription_device: auto
transcription_compute_type: auto
global_hotkey: <ctrl>+<alt>+v
language: null
log_level: INFO
logs_path: ~/.config/voicepad/logs
```

!!! note "Settings visibility"
    Not all settings are shown in the Settings tab interface. Advanced settings like `transcription_device`, `transcription_compute_type`, `language`, `log_level`, and `logs_path` can only be changed by editing the YAML file directly.

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
