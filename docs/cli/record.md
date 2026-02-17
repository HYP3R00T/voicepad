---
icon: lucide/mic
---

# record commands

Audio recording commands for capturing and transcribing voice input.

## voicepad record start {#start}

Start recording audio from the configured input device.

### Synopsis

```bash
voicepad record start [OPTIONS]
```

### Description

Begins audio recording from your microphone. Press **Ctrl+C** to stop recording. By default, the recording is automatically transcribed after stopping unless `--no-transcribe` is used.

When VAD chunking is enabled (`--vad`), transcription happens in real-time as chunks are detected, and the markdown file updates live while recording continues.

### Options

#### `--prefix TEXT, -p TEXT`

Custom filename prefix (overrides `recording_prefix` in config).

**Default:** Value from config file (typically `recording`)

**Example:**

```bash
voicepad record start --prefix team_meeting
# Creates: team_meeting_20260218_033000.wav
```

#### `--duration FLOAT, -d FLOAT`

Record for a fixed number of seconds, then stop automatically.

**Default:** `None` (record until Ctrl+C)

**Example:**

```bash
voicepad record start --duration 300
# Records for exactly 5 minutes
```

#### `--transcribe / --no-transcribe`

Enable or disable automatic transcription after recording.

**Default:** `--transcribe` (enabled)

**Example:**

```bash
# Record without transcribing
voicepad record start --no-transcribe

# Explicitly enable (redundant, but allowed)
voicepad record start --transcribe
```

!!! note "VAD Mode"
    When `--vad` is enabled, transcription always happens during recording.
    The `--transcribe` flag only affects post-recording transcription.

#### `--vad / --no-vad`

Enable or disable VAD-based chunking for smart audio splitting.

**Default:** `--no-vad` (disabled)

**Example:**

```bash
# Enable VAD chunking
voicepad record start --vad

# Explicitly disable (redundant)
voicepad record start --no-vad
```

**See:** [VAD Chunking Guide](../features/vad-chunking.md)

#### `--min-chunk-duration FLOAT`

Minimum duration (seconds) before allowing chunk splits (VAD mode only).

**Default:** `60.0` seconds

**Range:** `10.0` - `600.0` seconds

**Example:**

```bash
# Larger chunks for lectures
voicepad record start --vad --min-chunk-duration 120

# Smaller chunks for Q&A
voicepad record start --vad --min-chunk-duration 30
```

**See:** [VAD Settings Reference](../reference/vad.md#min-chunk-duration)

#### `--vad-threshold FLOAT`

Speech detection sensitivity threshold (VAD mode only).

**Default:** `0.5`

**Range:** `0.0` - `1.0`

**Example:**

```bash
# More sensitive (quiet environments)
voicepad record start --vad --vad-threshold 0.4

# Less sensitive (noisy environments)
voicepad record start --vad --vad-threshold 0.6
```

**See:** [VAD Settings Reference](../reference/vad.md#threshold)

### Examples

#### Basic Recording

```bash
voicepad record start
# Press Ctrl+C when done
```

Output:

- `data/recordings/recording_20260218_033000.wav`
- `data/markdown/recording_20260218_033000.md`

#### Meeting with VAD

```bash
voicepad record start \
  --prefix standup_meeting \
  --vad \
  --min-chunk-duration 60
```

Output (live updates):

- `data/recordings/standup_meeting_20260218_093000.wav`
- `data/markdown/standup_meeting_20260218_093000.md` (updates in real-time)

#### Fixed Duration, Audio Only

```bash
voicepad record start \
  --duration 180 \
  --no-transcribe \
  --prefix voice_memo
```

Output:

- `data/recordings/voice_memo_20260218_103000.wav` (only)

#### Sensitive VAD for Quiet Room

```bash
voicepad record start \
  --vad \
  --vad-threshold 0.4 \
  --min-chunk-duration 45
```

### Output Files

**Audio file:**

- Format: WAV (16-bit PCM, 16kHz mono)
- Location: `{recordings_path}/{prefix}_{timestamp}.wav`
- Created: When recording stops (VAD mode) or immediately (non-VAD mode)

**Transcription file:**

- Format: Markdown with metadata
- Location: `{markdown_path}/{prefix}_{timestamp}.md`
- Created: After recording (non-VAD) or during recording (VAD mode)

### Stopping Recording

Press **Ctrl+C** to stop recording:

```text
^C
[INFO] Stopping recording...
[INFO] Waiting for chunk worker to complete...
[INFO] Saving 5 accumulated chunks...
[INFO] Recording saved to: data/recordings/recording_20260218_033000.wav
```

### Exit Codes

- `0` - Recording completed successfully
- `1` - Recording failed (device error, permission denied, etc.)
- `130` - Interrupted by user (Ctrl+C - this is normal)

---

## voicepad record info {#info}

Display the current recording configuration.

### Synopsis

```bash
voicepad record info
```

### Description

Shows the active recording settings from your configuration file, including:

- Input device
- Output directories
- Filename prefix
- Number of existing recordings

### Output Example

```text
Current Recording Configuration
============================================================
Input device index: 2 (Microphone - High Definition Audio)
Recordings directory: data/recordings
Markdown directory: data/markdown
Filename prefix: recording
============================================================
[OK] Recordings directory exists
   12 recording(s) found

Tip: Use 'voicepad config input' to view and configure audio devices
```

### Exit Codes

- `0` - Always succeeds

---

## Common Workflows

### Standard Meeting Recording

```bash
# 1. Check setup
voicepad record info
voicepad config system

# 2. Start recording with VAD
voicepad record start --prefix team_meeting --vad

# 3. Let it run, watch markdown file update

# 4. Press Ctrl+C when done
```

### Quick Voice Memo

```bash
voicepad record start --prefix memo --duration 60 --no-transcribe
```

### Optimize for Accuracy

```bash
# Use larger chunks, adjust sensitivity
voicepad record start \
  --vad \
  --min-chunk-duration 90 \
  --vad-threshold 0.5 \
  --prefix lecture
```

## See Also

- **[VAD Chunking](../features/vad-chunking.md)** - Smart audio splitting
- **[Background Transcription](../features/background-transcription.md)** - Real-time processing
- **[Configuration](../configuration.md)** - Setting defaults in YAML
- **[VAD Settings Reference](../reference/vad.md)** - Parameter details
