# Configuration

VoicePad uses UtilityHub Config to load and validate its TOML configuration at:

```text
~/.config/voicepad/voicepad.toml
```

Inspect or initialize it with:

```bash
voicepad config path
voicepad config show
voicepad config init
```

Supported fields are:

```toml
deployment_id = "parakeet-v3.transformers-fp16-cuda"
recordings_path = "~/.config/voicepad/data/recordings"
markdown_path = "~/.config/voicepad/data/markdown"
artifact_cache_path = "~/.cache/voicepad/artifacts"
recording_prefix = "recording"
copy_complete_text = true
theme = "tokyo-night"
```

Unknown fields and invalid values fail with an actionable error. VoicePad does
not rewrite invalid configuration. Omit `input_device_index` to use the system
default microphone, because TOML has no null value.
