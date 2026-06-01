"""VoicePad transcription benchmark.

Measures accuracy (WER) and speed (RTF) for each fixture in fixtures/fixtures.json.

Usage:
    uv run python benchmarks/run_benchmark.py
    uv run python benchmarks/run_benchmark.py --model turbo
    uv run python benchmarks/run_benchmark.py --model tiny --device cpu
    uv run python benchmarks/run_benchmark.py --save

Metrics:
    WER  (Word Error Rate)      — lower is better; 0% = perfect match
    RTF  (Real-Time Factor)     — latency / audio_duration; < 1.0 = faster than real time
    CER  (Character Error Rate) — finer-grained than WER; useful for punctuation accuracy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BENCHMARKS_DIR = Path(__file__).parent
FIXTURES_DIR = BENCHMARKS_DIR / "fixtures"
RESULTS_DIR = BENCHMARKS_DIR / "results"
FIXTURES_FILE = FIXTURES_DIR / "fixtures.json"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    fixture_id: str
    model: str
    device: str
    compute_type: str
    audio_duration_s: float
    latency_ms: float
    rtf: float
    wer: float
    cer: float
    hypothesis: str
    reference: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def rtf_label(self) -> str:
        return f"{self.rtf:.3f}x"

    @property
    def wer_label(self) -> str:
        return f"{self.wer * 100:.1f}%"

    @property
    def cer_label(self) -> str:
        return f"{self.cer * 100:.1f}%"

    @property
    def latency_label(self) -> str:
        return f"{self.latency_ms:.0f}ms"

    @property
    def duration_label(self) -> str:
        return f"{self.audio_duration_s:.1f}s"


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------


def run_fixture(fixture: dict, model_name: str, device: str) -> BenchmarkResult:
    """Run transcription on one fixture and return metrics."""
    from jiwer import cer, wer
    from voicepad_core import transcribe
    from voicepad_core.config import get_config

    wav_path = FIXTURES_DIR / fixture["wav"]
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV not found: {wav_path}")

    reference = fixture["ground_truth"].strip()

    # Build a config pointing at the requested model/device.
    # Config is a frozen Pydantic model — use model_copy to override fields.
    config = get_config().model_copy(
        update={
            "transcription_model": model_name,
            "transcription_device": device,
        }
    )

    print(f"  [{fixture['id']}] transcribing with {model_name} on {device}...", end=" ", flush=True)
    t0 = time.perf_counter()
    import soundfile as sf

    audio, _ = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    result = transcribe(
        audio,
        model_name=config.transcription_model,
        device=config.transcription_device,
        compute_type=config.transcription_compute_type,
        language=config.language,
        word_timestamps=False,
    )
    wall_ms = (time.perf_counter() - t0) * 1000
    print(f"done ({wall_ms:.0f}ms)")

    hypothesis = result.text.strip()
    rtf = result.latency_ms / 1000 / result.duration_s if result.duration_s > 0 else 0.0

    # Normalise both strings the same way before scoring
    ref_norm = _normalise(reference)
    hyp_norm = _normalise(hypothesis)

    return BenchmarkResult(
        fixture_id=fixture["id"],
        model=model_name,
        device=result.device,
        compute_type=result.compute_type,
        audio_duration_s=result.duration_s,
        latency_ms=result.latency_ms,
        rtf=rtf,
        wer=wer(ref_norm, hyp_norm),
        cer=cer(ref_norm, hyp_norm),
        hypothesis=hypothesis,
        reference=reference,
    )


def _normalise(text: str) -> str:
    """Lowercase and strip punctuation for fair WER comparison."""
    import re

    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

COL_WIDTHS = {
    "fixture": 16,
    "model": 14,
    "device": 6,
    "duration": 9,
    "latency": 9,
    "rtf": 7,
    "wer": 7,
    "cer": 7,
}

HEADER = (
    f"{'fixture':<{COL_WIDTHS['fixture']}}"
    f"{'model':<{COL_WIDTHS['model']}}"
    f"{'device':<{COL_WIDTHS['device']}}"
    f"{'duration':>{COL_WIDTHS['duration']}}"
    f"{'latency':>{COL_WIDTHS['latency']}}"
    f"{'RTF':>{COL_WIDTHS['rtf']}}"
    f"{'WER':>{COL_WIDTHS['wer']}}"
    f"{'CER':>{COL_WIDTHS['cer']}}"
)
SEPARATOR = "-" * len(HEADER)


def print_result(r: BenchmarkResult) -> None:
    print(
        f"{r.fixture_id:<{COL_WIDTHS['fixture']}}"
        f"{r.model:<{COL_WIDTHS['model']}}"
        f"{r.device:<{COL_WIDTHS['device']}}"
        f"{r.duration_label:>{COL_WIDTHS['duration']}}"
        f"{r.latency_label:>{COL_WIDTHS['latency']}}"
        f"{r.rtf_label:>{COL_WIDTHS['rtf']}}"
        f"{r.wer_label:>{COL_WIDTHS['wer']}}"
        f"{r.cer_label:>{COL_WIDTHS['cer']}}"
    )


def print_diff(r: BenchmarkResult) -> None:
    """Show reference vs hypothesis side by side."""
    print(f"\n  reference : {r.reference}")
    print(f"  hypothesis: {r.hypothesis}")


def save_results(results: list[BenchmarkResult]) -> Path:
    """Save results as JSON for historical comparison."""
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"benchmark_{ts}.json"
    data = [
        {
            "fixture_id": r.fixture_id,
            "model": r.model,
            "device": r.device,
            "compute_type": r.compute_type,
            "audio_duration_s": r.audio_duration_s,
            "latency_ms": r.latency_ms,
            "rtf": r.rtf,
            "wer": r.wer,
            "cer": r.cer,
            "hypothesis": r.hypothesis,
            "reference": r.reference,
            "timestamp": r.timestamp,
        }
        for r in results
    ]
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out


def compare_with_last(results: list[BenchmarkResult]) -> None:
    """Print a delta table comparing current run against the most recent saved run."""
    if not RESULTS_DIR.exists():
        return
    past_files = sorted(RESULTS_DIR.glob("benchmark_*.json"))
    if not past_files:
        return

    past_data = json.loads(past_files[-1].read_text(encoding="utf-8"))
    past_by_key: dict[tuple[str, str], dict] = {(r["fixture_id"], r["model"]): r for r in past_data}

    print("\nDelta vs last saved run:")
    print(SEPARATOR)
    print(f"  {'fixture':<16}{'model':<14}{'ΔWER':>8}{'ΔCER':>8}{'ΔRTF':>8}")
    print(SEPARATOR)
    for r in results:
        key = (r.fixture_id, r.model)
        if key not in past_by_key:
            continue
        prev = past_by_key[key]
        d_wer = (r.wer - prev["wer"]) * 100
        d_cer = (r.cer - prev["cer"]) * 100
        d_rtf = r.rtf - prev["rtf"]
        wer_str = f"{d_wer:+.1f}%"
        cer_str = f"{d_cer:+.1f}%"
        rtf_str = f"{d_rtf:+.3f}x"
        print(f"  {r.fixture_id:<16}{r.model:<14}{wer_str:>8}{cer_str:>8}{rtf_str:>8}")
    print(SEPARATOR)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark VoicePad transcription accuracy and speed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        default="turbo",
        help="Whisper model name (default: turbo)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device (default: auto)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to benchmarks/results/ for historical comparison",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Print reference vs hypothesis for each fixture",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare current run against the last saved run",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not FIXTURES_FILE.exists():
        print(f"ERROR: fixtures file not found: {FIXTURES_FILE}", file=sys.stderr)
        return 1

    fixtures = json.loads(FIXTURES_FILE.read_text(encoding="utf-8"))
    if not fixtures:
        print("No fixtures defined.", file=sys.stderr)
        return 1

    print(f"\nVoicePad Benchmark — model={args.model}  device={args.device}")
    print(SEPARATOR)
    print(HEADER)
    print(SEPARATOR)

    results: list[BenchmarkResult] = []
    errors: list[str] = []

    for fixture in fixtures:
        try:
            r = run_fixture(fixture, model_name=args.model, device=args.device)
            results.append(r)
            print_result(r)
            if args.diff:
                print_diff(r)
        except Exception as e:
            errors.append(f"{fixture['id']}: {e}")
            print(f"  ERROR — {fixture['id']}: {e}")

    print(SEPARATOR)

    if results:
        avg_wer = sum(r.wer for r in results) / len(results)
        avg_cer = sum(r.cer for r in results) / len(results)
        avg_rtf = sum(r.rtf for r in results) / len(results)
        print(
            f"{'average':<{COL_WIDTHS['fixture']}}"
            f"{'':<{COL_WIDTHS['model']}}"
            f"{'':<{COL_WIDTHS['device']}}"
            f"{'':{COL_WIDTHS['duration']}}"
            f"{'':{COL_WIDTHS['latency']}}"
            f"{avg_rtf:.3f}x{'':{COL_WIDTHS['rtf'] - 6}}"
            f"{avg_wer * 100:.1f}%{'':{COL_WIDTHS['wer'] - 5}}"
            f"{avg_cer * 100:.1f}%"
        )
        print(SEPARATOR)

    if args.compare:
        compare_with_last(results)

    if args.save and results:
        out = save_results(results)
        print(f"\nResults saved → {out}")

    if errors:
        print(f"\n{len(errors)} fixture(s) failed.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
