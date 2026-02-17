---
icon: lucide/terminal
---

# CLI Reference

Voicepad's command-line interface provides easy access to recording, transcription, and configuration management.

## Command Structure

```bash
voicepad [COMMAND] [SUBCOMMAND] [OPTIONS]
```

**Available commands:**

- **`record`** - Audio recording and management
- **`config`** - Configuration and system information

## Global Options

```bash
--help, -h        # Show help and exit
--version         # Show version and exit
```

## Commands

### Recording Commands

**`voicepad record`** - Manage audio recordings

- **[`record start`](record.md#start)** - Start a new recording
- **[`record info`](record.md#info)** - Show recording configuration

[Complete recording reference →](record.md)

### Configuration Commands

**`voicepad config`** - View and manage configuration

- **[`config input`](config.md#input)** - List audio input devices
- **[`config system`](config.md#system)** - Display system information
- **[`config transcription`](config.md#transcription)** - Show transcription settings
- **[`config recommend`](config.md#recommend)** - Get model recommendation
- **[`config models`](config.md#models)** - List available Whisper models

[Complete configuration reference →](config.md)

## Common Usage Patterns

### Quick Recording

```bash
# Basic recording (press Ctrl+C to stop)
voicepad record start

# With custom filename
voicepad record start --prefix meeting_notes

# Fixed duration
voicepad record start --duration 300
```

### Long Recording with VAD

```bash
# Enable smart chunking for real-time transcription
voicepad record start --vad --min-chunk-duration 60

# Adjust sensitivity
voicepad record start --vad --vad-threshold 0.45
```

### Configuration Check

```bash
# Check current setup
voicepad record info
voicepad config transcription

# Get recommendations
voicepad config recommend
voicepad config system
```

## Exit Codes

- `0` - Success
- `1` - General error
- `2` - Invalid configuration
- `130` - Interrupted by user (Ctrl+C - expected during recording)

## Environment Variables

Override configuration using `VOICEPAD_*` environment variables:

```bash
# Override transcription model
export VOICEPAD_TRANSCRIPTION_MODEL=tiny

# Run with override
voicepad record start
```

See [Configuration Guide](../configuration.md#environment-variables) for details.

## Next Steps

- **[Recording Commands](record.md)** - Detailed `voicepad record` reference
- **[Config Commands](config.md)** - Detailed `voicepad config` reference
- **[Configuration](../configuration.md)** - YAML configuration file
- **[Features](../features/index.md)** - Feature documentation
