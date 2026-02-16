# voicepad

Command-line interface for configuration inspection. Provides user-friendly commands for checking and updating audio input defaults.

## Overview

`voicepad` is a CLI wrapper around `voicepad-core` built with Typer. It exposes configuration functionality through simple terminal commands.

**Key Features:**

- List audio input devices
- Show the configured default input device
- Provide guidance for persisting `input_device_index`

## Installation

```bash
pip install voicepad
```

**Requirements:** Python 3.13+

**Dependencies:** `voicepad-core`, `typer`, `textual`

## Command Structure

```sh
voicepad
└── config                  Configuration management commands
  └── input               List audio input devices and configured default
```

This structure keeps configuration tasks grouped under a single command.

## Commands

### `voicepad config input`

List available audio input devices and show the configured default.

```bash
voicepad config input
```

## Example Workflow

```bash
# List available devices and current default
voicepad config input
```
