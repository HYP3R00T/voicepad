"""Markdown formatting and parsing for VoicePad transcription files.

File format (YAML front matter + transcription blocks):

    ---
    file: recording_xxx.wav
    transcriptions:
      - n: 2
        model: turbo · cuda / int8
        language: en (98.3%)
        duration: 31.5s
        latency: 1791ms
        timestamp: 2026-04-30 10:05
      - n: 1
        ...
    ---

    ## Transcription 2

    Latest text here.

    ## Transcription 1

    Original text here.
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voicepad_core import ChunkResult

    from voicepad.tui.models import SessionEntry


def format_markdown(audio_path: Path, result: object, model_name: str = "") -> str:
    """Create a new markdown file with the first transcription."""
    ts = time.strftime("%Y-%m-%d %H:%M")
    device = getattr(result, "device", "unknown")
    compute_type = getattr(result, "compute_type", "unknown")
    language = getattr(result, "language", "en")
    language_probability = getattr(result, "language_probability", 1.0)
    duration_s = getattr(result, "duration_s", 0.0)
    latency_ms = getattr(result, "latency_ms", 0.0)
    text = getattr(result, "text", "") or "*(no speech detected)*"
    segments = getattr(result, "segments", []) or []

    model_str = f"{model_name} · {device} / {compute_type}" if model_name else f"{device} / {compute_type}"
    lines = [
        "---",
        f"file: {audio_path.name}",
        "transcriptions:",
        "  - n: 1",
        f"    model: {model_str}",
        f"    language: {language} ({language_probability * 100:.1f}%)",
        f"    duration: {duration_s:.1f}s",
        f"    latency: {latency_ms:.0f}ms",
        f"    timestamp: {ts}",
        "---",
        "",
        "## Transcription 1",
        "",
        text,
        "",
    ]

    if segments:
        lines += ["## Segments", ""]
        for seg in segments:
            start = getattr(seg, "start", 0.0)
            end = getattr(seg, "end", 0.0)
            seg_text = getattr(seg, "text", "")
            lines.append(f"- **{start:.1f}s – {end:.1f}s** {seg_text}")
        lines.append("")

    return "\n".join(lines)


def format_markdown_streaming(
    wav_path: Path,
    text: str,
    duration_s: float,
    chunks: list[ChunkResult],
    model_name: str = "",
) -> str:
    """Create a new markdown file for a streaming transcription."""
    latest_chunk = next((chunk for chunk in reversed(chunks) if chunk.text), None)
    device = latest_chunk.device if latest_chunk else "unknown"
    language = latest_chunk.language if latest_chunk else "en"
    language_probability = latest_chunk.language_probability if latest_chunk else 0.0
    latency_ms = sum(chunk.latency_ms for chunk in chunks)
    ts = time.strftime("%Y-%m-%d %H:%M")
    model_str = f"{model_name} · {device} / streaming" if model_name else f"{device} / streaming"

    lines = [
        "---",
        f"file: {wav_path.name}",
        "transcriptions:",
        "  - n: 1",
        f"    model: {model_str}",
        f"    language: {language} ({language_probability * 100:.1f}%)",
        f"    duration: {duration_s:.1f}s",
        f"    latency: {latency_ms:.0f}ms",
        f"    timestamp: {ts}",
        "---",
        "",
        "## Transcription 1",
        "",
        text or "*(no speech detected)*",
        "",
    ]

    # Collect all segments from all chunks
    all_segments = []
    for chunk in chunks:
        segs = getattr(chunk, "segments", None) or []
        all_segments.extend(segs)

    if all_segments:
        lines += ["## Segments", ""]
        for seg in all_segments:
            start = getattr(seg, "start", 0.0)
            end = getattr(seg, "end", 0.0)
            seg_text = getattr(seg, "text", "")
            lines.append(f"- **{start:.1f}s – {end:.1f}s** {seg_text}")
        lines.append("")

    return "\n".join(lines)


def prepend_retranscription(md_path: Path, result: object, model_name: str = "") -> str:
    """Prepend a new transcription to an existing markdown file.

    Reads the existing file, increments the transcription count,
    adds new metadata entry at the top of the array, and prepends
    the new text block before the existing ones.
    """
    ts = time.strftime("%Y-%m-%d %H:%M")
    device = getattr(result, "device", "unknown")
    compute_type = getattr(result, "compute_type", "unknown")
    language = getattr(result, "language", "en")
    language_probability = getattr(result, "language_probability", 1.0)
    duration_s = getattr(result, "duration_s", 0.0)
    latency_ms = getattr(result, "latency_ms", 0.0)
    text = getattr(result, "text", "") or "*(no speech detected)*"

    try:
        existing = md_path.read_text(encoding="utf-8")
    except Exception:
        existing = ""

    lines = existing.splitlines()
    max_n = 0
    if lines and lines[0].strip() == "---":
        fm_end = next((idx for idx, ln in enumerate(lines[1:], 1) if ln.strip() == "---"), None)
        if fm_end is not None:
            for fl in lines[1:fm_end]:
                if fl.strip().startswith("- n:"):
                    with contextlib.suppress(Exception):
                        max_n = max(max_n, int(fl.strip().split(":")[-1].strip()))

    new_n = max_n + 1
    model_str = f"{model_name} · {device} / {compute_type}" if model_name else f"{device} / {compute_type}"
    new_fm_entry = [
        f"  - n: {new_n}",
        f"    model: {model_str}",
        f"    language: {language} ({language_probability * 100:.1f}%)",
        f"    duration: {duration_s:.1f}s",
        f"    latency: {latency_ms:.0f}ms",
        f"    timestamp: {ts}",
    ]

    new_lines: list[str] = []
    injected = False
    if lines and lines[0].strip() == "---":
        fm_end = next((idx for idx, ln in enumerate(lines[1:], 1) if ln.strip() == "---"), None)
        if fm_end is not None:
            for line in lines[: fm_end + 1]:
                new_lines.append(line)
                if not injected and line.strip() == "transcriptions:":
                    new_lines.extend(new_fm_entry)
                    injected = True
            body = lines[fm_end + 1 :]
        else:
            new_lines = lines
            body = []
    else:
        new_lines = ["---", f"file: {md_path.stem}.wav", "transcriptions:"] + new_fm_entry + ["---"]
        body = lines

    new_block = ["", f"## Transcription {new_n}", "", text, ""]
    all_lines = new_lines + new_block + ([""] if body and body[0] != "" else []) + body
    return "\n".join(all_lines)


def parse_markdown_entry(
    md_path: Path,
    index: int,
    recordings_path: Path | None = None,
) -> SessionEntry | None:
    """Parse a transcription markdown file into a SessionEntry.

    Uses the latest transcription (highest n) for preview text and metadata.
    """
    from voicepad.tui.models import SessionEntry

    try:
        content = md_path.read_text(encoding="utf-8")
    except Exception:
        return None

    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return None

    fm_end = next((idx for idx, ln in enumerate(lines[1:], 1) if ln.strip() == "---"), None)
    if fm_end is None:
        return None

    wav_name: str | None = None
    entries: list[dict] = []
    current: dict | None = None
    for fl in lines[1:fm_end]:
        stripped = fl.strip()
        if stripped.startswith("- n:"):
            if current is not None:
                entries.append(current)
            with contextlib.suppress(Exception):
                current = {"n": int(stripped.split(":")[-1].strip())}
        elif stripped.startswith("file:"):
            wav_name = stripped.split(":", 1)[-1].strip()
        elif current is not None and ":" in stripped:
            key, _, val = stripped.partition(":")
            current[key.strip()] = val.strip()
    if current is not None:
        entries.append(current)

    if not entries:
        return None

    latest = max(entries, key=lambda e: e.get("n", 0))
    duration_s = 0.0
    latency_ms = 0.0
    device = "unknown"
    with contextlib.suppress(Exception):
        duration_s = float(latest.get("duration", "0s").rstrip("s"))
    with contextlib.suppress(Exception):
        latency_ms = float(latest.get("latency", "0ms").rstrip("ms"))
    with contextlib.suppress(Exception):
        device = latest.get("model", "unknown").split("/")[0].strip()

    latest_n = latest.get("n", 1)
    marker = f"## Transcription {latest_n}"
    body = lines[fm_end + 1 :]
    text_lines: list[str] = []
    in_block = False
    for line in body:
        if line.strip() == marker:
            in_block = True
            continue
        if in_block:
            if line.strip().startswith("## Transcription "):
                break
            text_lines.append(line)
    text = " ".join(ln for ln in text_lines if ln.strip() and ln.strip() != "*(no speech detected)*").strip()

    if not text:
        return None

    wav_path: Path | None = None
    if wav_name:
        candidates: list[Path] = []
        if recordings_path is not None:
            candidates.append(recordings_path / wav_name)
        candidates.append(md_path.parent.parent / "recordings" / wav_name)
        for candidate in candidates:
            if candidate.exists():
                wav_path = candidate
                break

    timestamp = latest.get("timestamp", "")
    if not timestamp:
        parts = md_path.stem.split("_")
        if len(parts) >= 3:
            date_part = parts[-2]
            time_part = parts[-1]
            with contextlib.suppress(Exception):
                timestamp = f"{date_part[0:4]}-{date_part[4:6]}-{date_part[6:8]} {time_part[0:2]}:{time_part[2:4]}"

    return SessionEntry(
        index=index,
        wav_path=wav_path,
        md_path=md_path,
        duration_s=duration_s,
        text=text,
        latency_ms=latency_ms,
        device=device,
        timestamp=timestamp,
    )
