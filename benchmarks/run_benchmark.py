"""Developer-grade local benchmark runner for VoicePad.

This tool is intended for contributors running on real machines, especially GPU
machines, to evaluate quality/speed tradeoffs and compare results across
hardware.

Core ideas:
- fixed fixture suite with ground truth
- explicit profile grid describing decoding/model settings
- repeated runs for more stable latency/RTF measurements
- machine/environment capture, including GPU details when available
- report bundles that can be compared across machines or commits

Examples:
    uv run python benchmarks/run_benchmark.py run
    uv run python benchmarks/run_benchmark.py run --suite benchmarks/fixtures/fixtures.json --profiles benchmarks/profiles/default_profiles.json
    uv run python benchmarks/run_benchmark.py run --repetitions 3 --save
    uv run python benchmarks/run_benchmark.py compare --left benchmarks/results/20260621_120000/report.json --right benchmarks/results/20260621_130000/report.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BENCHMARKS_DIR = Path(__file__).parent
FIXTURES_FILE = BENCHMARKS_DIR / "fixtures" / "fixtures.json"
PROFILES_FILE = BENCHMARKS_DIR / "profiles" / "default_profiles.json"
RESULTS_DIR = BENCHMARKS_DIR / "results"


@dataclass(frozen=True)
class Fixture:
    id: str
    wav: str
    ground_truth: str
    language: str = "en"
    notes: str = ""


@dataclass(frozen=True)
class Profile:
    id: str
    model: str
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 5
    vad_filter: bool = False
    initial_prompt: str | None = None
    category: str = "custom"
    notes: str = ""


@dataclass
class RunMeasurement:
    latency_ms: float
    wall_ms: float
    rtf: float
    hypothesis: str
    actual_device: str
    actual_compute_type: str
    fallback_to_cpu: bool


@dataclass
class CaseResult:
    fixture_id: str
    profile_id: str
    language: str
    model_cache_path: str
    model_cached_before_run: bool
    model_cached_after_run: bool
    audio_duration_s: float
    wer: float
    cer: float
    avg_latency_ms: float
    median_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    avg_wall_ms: float
    avg_rtf: float
    hypothesis: str
    reference: str
    actual_device: str
    actual_compute_type: str
    fallback_to_cpu: bool
    repetitions: int


@dataclass
class ProfileSummary:
    profile_id: str
    category: str
    model: str
    device: str
    compute_type: str
    beam_size: int
    vad_filter: bool
    notes: str
    avg_wer: float
    avg_cer: float
    avg_latency_ms: float
    avg_rtf: float
    balanced_score: float
    fixture_count: int


@dataclass
class EnvironmentInfo:
    hostname: str
    platform: str
    python_version: str
    timestamp_utc: str
    git_commit: str | None
    model_cache_path: str
    gpu: list[dict[str, Any]]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalise(text: str) -> str:
    import re

    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_fixture_suite(path: Path) -> tuple[str, list[Fixture]]:
    raw = _load_json(path)
    if isinstance(raw, list):
        fixtures = [Fixture(**item) for item in raw]
        return path.stem, fixtures
    if isinstance(raw, dict) and "fixtures" in raw:
        fixtures = [Fixture(**item) for item in raw["fixtures"]]
        return str(raw.get("name", path.stem)), fixtures
    raise ValueError(f"Unsupported suite format: {path}")


def _parse_profiles(path: Path) -> list[Profile]:
    raw = _load_json(path)
    if isinstance(raw, list):
        return [Profile(**item) for item in raw]
    if isinstance(raw, dict) and "profiles" in raw:
        return [Profile(**item) for item in raw["profiles"]]
    raise ValueError(f"Unsupported profiles format: {path}")


def _capture_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _capture_gpu_info() -> list[dict[str, Any]]:
    """Capture GPU metadata via nvidia-smi when available.

    This avoids importing heavyweight ML runtimes just to inspect the machine.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []

    gpu_info: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        index, name, memory_mb, driver_version = parts
        try:
            total_memory_mb = int(memory_mb)
        except ValueError:
            total_memory_mb = None
        gpu_info.append({
            "index": int(index),
            "name": name,
            "total_memory_mb": total_memory_mb,
            "driver_version": driver_version,
        })

    return gpu_info


def capture_environment() -> EnvironmentInfo:
    return EnvironmentInfo(
        hostname=platform.node(),
        platform=platform.platform(),
        python_version=platform.python_version(),
        timestamp_utc=_utc_now(),
        git_commit=_capture_git_commit(),
        model_cache_path=str(_get_runtime_model_cache_path()),
        gpu=_capture_gpu_info(),
    )


def _read_audio(wav_path: Path):
    import soundfile as sf

    audio, _ = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    return audio


def _get_runtime_model_cache_path() -> Path:
    from voicepad_core.config import get_config

    return get_config().model_cache_path


def _ensure_profile_model_ready(profile: Profile) -> tuple[Path, bool, bool]:
    from voicepad_core import model_is_ready, prepare_model

    model_cache_path = _get_runtime_model_cache_path()
    cached_before = model_is_ready(profile.model)
    if not cached_before:
        print(f"    cache miss: downloading {profile.model} into {model_cache_path}", flush=True)
    prepare_model(profile.model)
    cached_after = model_is_ready(profile.model)
    return model_cache_path, cached_before, cached_after


def run_case(
    fixture: Fixture,
    profile: Profile,
    fixture_root: Path,
    repetitions: int,
) -> CaseResult:
    from jiwer import cer, wer
    from voicepad_core import transcribe

    wav_path = fixture_root / fixture.wav
    if not wav_path.exists():
        raise FileNotFoundError(f"Fixture WAV not found: {wav_path}")

    audio = _read_audio(wav_path)
    model_cache_path, cached_before, cached_after = _ensure_profile_model_ready(profile)
    measurements: list[RunMeasurement] = []

    print(f"  [{fixture.id}] {profile.id} x{repetitions}", flush=True)
    print(f"    model cache: {model_cache_path}", flush=True)
    print(f"    cached before run: {cached_before}", flush=True)
    for attempt in range(1, repetitions + 1):
        started = time.perf_counter()
        result = transcribe(
            audio,
            model_name=profile.model,
            device=profile.device,
            compute_type=profile.compute_type,
            language=fixture.language,
            word_timestamps=False,
            initial_prompt=profile.initial_prompt,
            beam_size=profile.beam_size,
            vad_filter=profile.vad_filter,
        )
        wall_ms = (time.perf_counter() - started) * 1000
        rtf = result.latency_ms / 1000 / result.duration_s if result.duration_s > 0 else 0.0
        measurements.append(
            RunMeasurement(
                latency_ms=result.latency_ms,
                wall_ms=wall_ms,
                rtf=rtf,
                hypothesis=result.text.strip(),
                actual_device=result.device,
                actual_compute_type=result.compute_type,
                fallback_to_cpu=result.fallback_to_cpu,
            )
        )
        print(
            f"    run {attempt}: latency={result.latency_ms:.0f}ms wall={wall_ms:.0f}ms rtf={rtf:.3f}x device={result.device}",
            flush=True,
        )

    reference = fixture.ground_truth.strip()
    hypothesis = measurements[min(len(measurements) - 1, repetitions - 1)].hypothesis
    ref_norm = _normalise(reference)
    hyp_norm = _normalise(hypothesis)

    latencies = [m.latency_ms for m in measurements]
    wall_times = [m.wall_ms for m in measurements]
    rtfs = [m.rtf for m in measurements]
    last = measurements[-1]

    return CaseResult(
        fixture_id=fixture.id,
        profile_id=profile.id,
        language=fixture.language,
        model_cache_path=str(model_cache_path),
        model_cached_before_run=cached_before,
        model_cached_after_run=cached_after,
        audio_duration_s=len(audio) / 16000,
        wer=wer(ref_norm, hyp_norm),
        cer=cer(ref_norm, hyp_norm),
        avg_latency_ms=statistics.mean(latencies),
        median_latency_ms=statistics.median(latencies),
        min_latency_ms=min(latencies),
        max_latency_ms=max(latencies),
        avg_wall_ms=statistics.mean(wall_times),
        avg_rtf=statistics.mean(rtfs),
        hypothesis=hypothesis,
        reference=reference,
        actual_device=last.actual_device,
        actual_compute_type=last.actual_compute_type,
        fallback_to_cpu=last.fallback_to_cpu,
        repetitions=repetitions,
    )


def summarise_profiles(
    case_results: list[CaseResult],
    profiles: list[Profile],
    *,
    wer_weight: float,
    cer_weight: float,
    rtf_weight: float,
) -> list[ProfileSummary]:
    grouped: dict[str, list[CaseResult]] = {}
    for result in case_results:
        grouped.setdefault(result.profile_id, []).append(result)

    profile_map = {profile.id: profile for profile in profiles}
    raw_rows: list[tuple[Profile, list[CaseResult]]] = []
    for profile_id, rows in grouped.items():
        raw_rows.append((profile_map[profile_id], rows))

    def normalise(values: list[float]) -> list[float]:
        low = min(values)
        high = max(values)
        if high == low:
            return [0.0 for _ in values]
        return [(value - low) / (high - low) for value in values]

    avg_wers = [statistics.mean(row.wer for row in rows) for _, rows in raw_rows]
    avg_cers = [statistics.mean(row.cer for row in rows) for _, rows in raw_rows]
    avg_rtfs = [statistics.mean(row.avg_rtf for row in rows) for _, rows in raw_rows]

    wer_norm = normalise(avg_wers)
    cer_norm = normalise(avg_cers)
    rtf_norm = normalise(avg_rtfs)

    summaries: list[ProfileSummary] = []
    for (profile, rows), wn, cn, rn in zip(raw_rows, wer_norm, cer_norm, rtf_norm, strict=True):
        summaries.append(
            ProfileSummary(
                profile_id=profile.id,
                category=profile.category,
                model=profile.model,
                device=profile.device,
                compute_type=profile.compute_type,
                beam_size=profile.beam_size,
                vad_filter=profile.vad_filter,
                notes=profile.notes,
                avg_wer=statistics.mean(row.wer for row in rows),
                avg_cer=statistics.mean(row.cer for row in rows),
                avg_latency_ms=statistics.mean(row.avg_latency_ms for row in rows),
                avg_rtf=statistics.mean(row.avg_rtf for row in rows),
                balanced_score=(wer_weight * wn) + (cer_weight * cn) + (rtf_weight * rn),
                fixture_count=len(rows),
            )
        )

    return sorted(summaries, key=lambda item: item.balanced_score)


def _report_dir(base_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / stamp


def _write_summary_markdown(
    target: Path,
    *,
    suite_name: str,
    environment: EnvironmentInfo,
    summaries: list[ProfileSummary],
) -> None:
    lines = [
        f"# Benchmark Report: {suite_name}",
        "",
        f"- Timestamp: `{environment.timestamp_utc}`",
        f"- Host: `{environment.hostname}`",
        f"- Platform: `{environment.platform}`",
        f"- Python: `{environment.python_version}`",
        f"- Git commit: `{environment.git_commit or 'unknown'}`",
        f"- Model cache: `{environment.model_cache_path}`",
        "",
        "## GPUs",
        "",
    ]
    if environment.gpu:
        lines.extend([
            f"- GPU {gpu['index']}: `{gpu['name']}` ({gpu.get('total_memory_mb', 'unknown')} MB, driver {gpu.get('driver_version', 'unknown')})"
            for gpu in environment.gpu
        ])
    else:
        lines.append("- No CUDA GPU detected")

    lines.extend([
        "",
        "## Ranking",
        "",
        "| Rank | Category | Profile | WER | CER | RTF | Avg latency | Score |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for index, summary in enumerate(summaries, start=1):
        lines.append(
            f"| {index} | `{summary.category}` | `{summary.profile_id}` | {summary.avg_wer * 100:.1f}% | {summary.avg_cer * 100:.1f}% | {summary.avg_rtf:.3f}x | {summary.avg_latency_ms:.0f}ms | {summary.balanced_score:.3f} |"
        )

    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_report(
    *,
    suite_path: Path,
    suite_name: str,
    profiles_path: Path,
    environment: EnvironmentInfo,
    case_results: list[CaseResult],
    summaries: list[ProfileSummary],
    weights: dict[str, float],
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_dir = _report_dir(RESULTS_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": 1,
        "suite": {
            "name": suite_name,
            "path": str(suite_path),
        },
        "profiles": {
            "path": str(profiles_path),
        },
        "environment": asdict(environment),
        "weights": weights,
        "summaries": [asdict(item) for item in summaries],
        "cases": [asdict(item) for item in case_results],
    }

    report_path = report_dir / "report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_summary_markdown(
        report_dir / "SUMMARY.md", suite_name=suite_name, environment=environment, summaries=summaries
    )
    return report_path


def print_case_table(case_results: list[CaseResult]) -> None:
    print("\nPer-fixture results")
    print("-" * 120)
    print(f"{'fixture':<16}{'profile':<34}{'WER':>8}{'CER':>8}{'RTF':>9}{'latency':>10}{'device':>12}")
    print("-" * 120)
    for result in case_results:
        print(
            f"{result.fixture_id:<16}{result.profile_id:<34}{result.wer * 100:>7.1f}%{result.cer * 100:>7.1f}%{result.avg_rtf:>8.3f}x{result.avg_latency_ms:>9.0f}ms{result.actual_device:>12}"
        )
    print("-" * 120)


def print_summary_table(summaries: list[ProfileSummary]) -> None:
    print("\nProfile ranking")
    print("-" * 120)
    print(f"{'rank':<6}{'category':<12}{'profile':<34}{'WER':>8}{'CER':>8}{'RTF':>9}{'latency':>10}{'score':>9}")
    print("-" * 120)
    for index, summary in enumerate(summaries, start=1):
        print(
            f"{index:<6}{summary.category:<12}{summary.profile_id:<34}{summary.avg_wer * 100:>7.1f}%{summary.avg_cer * 100:>7.1f}%{summary.avg_rtf:>8.3f}x{summary.avg_latency_ms:>9.0f}ms{summary.balanced_score:>9.3f}"
        )
    print("-" * 120)
    winner = summaries[0]
    print(f"Winner: {winner.profile_id}")


def compare_reports(left: Path, right: Path) -> int:
    left_payload = _load_json(left)
    right_payload = _load_json(right)

    left_map = {item["profile_id"]: item for item in left_payload["summaries"]}
    right_map = {item["profile_id"]: item for item in right_payload["summaries"]}
    common = sorted(set(left_map) & set(right_map))
    if not common:
        print("No shared profiles between the two reports.", file=sys.stderr)
        return 1

    print("\nReport comparison")
    print("-" * 96)
    print(f"{'profile':<34}{'ΔWER':>10}{'ΔCER':>10}{'ΔRTF':>10}{'Δlatency':>12}{'Δscore':>10}")
    print("-" * 96)
    for profile_id in common:
        left_row = left_map[profile_id]
        right_row = right_map[profile_id]
        print(
            f"{profile_id:<34}"
            f"{(right_row['avg_wer'] - left_row['avg_wer']) * 100:>9.1f}%"
            f"{(right_row['avg_cer'] - left_row['avg_cer']) * 100:>9.1f}%"
            f"{right_row['avg_rtf'] - left_row['avg_rtf']:>9.3f}x"
            f"{right_row['avg_latency_ms'] - left_row['avg_latency_ms']:>11.0f}ms"
            f"{right_row['balanced_score'] - left_row['balanced_score']:>10.3f}"
        )
    print("-" * 96)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local developer benchmark runner for VoicePad")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a benchmark suite")
    run_parser.add_argument("--suite", type=Path, default=FIXTURES_FILE, help="Fixture suite JSON path")
    run_parser.add_argument("--profiles", type=Path, default=PROFILES_FILE, help="Profiles JSON path")
    run_parser.add_argument(
        "--repetitions", type=int, default=3, help="How many times to run each fixture/profile pair"
    )
    run_parser.add_argument("--wer-weight", type=float, default=0.6, help="Balanced-score WER weight")
    run_parser.add_argument("--cer-weight", type=float, default=0.2, help="Balanced-score CER weight")
    run_parser.add_argument("--rtf-weight", type=float, default=0.2, help="Balanced-score RTF weight")
    run_parser.add_argument("--save", action="store_true", help="Save report bundle under benchmarks/results/")

    compare_parser = subparsers.add_parser("compare", help="Compare two saved reports")
    compare_parser.add_argument("--left", type=Path, required=True, help="Older or baseline report.json")
    compare_parser.add_argument("--right", type=Path, required=True, help="Newer or candidate report.json")

    return parser.parse_args()


def run_command(args: argparse.Namespace) -> int:
    total_weight = args.wer_weight + args.cer_weight + args.rtf_weight
    if total_weight <= 0:
        print("Score weights must sum to a positive value.", file=sys.stderr)
        return 1
    if args.repetitions <= 0:
        print("Repetitions must be at least 1.", file=sys.stderr)
        return 1

    suite_name, fixtures = _parse_fixture_suite(args.suite)
    profiles = _parse_profiles(args.profiles)
    environment = capture_environment()

    print(f"\nVoicePad benchmark suite: {suite_name}")
    print(f"Fixtures: {len(fixtures)}")
    print(f"Profiles: {len(profiles)}")
    print(f"Repetitions: {args.repetitions}")
    if environment.gpu:
        print("GPUs:")
        for gpu in environment.gpu:
            print(
                f"  - {gpu['name']} ({gpu.get('total_memory_mb', 'unknown')} MB, driver {gpu.get('driver_version', 'unknown')})"
            )
    else:
        print("GPUs: none detected")

    fixture_root = args.suite.parent
    case_results: list[CaseResult] = []
    for profile in profiles:
        case_results.extend(run_case(fixture, profile, fixture_root, args.repetitions) for fixture in fixtures)

    summaries = summarise_profiles(
        case_results,
        profiles,
        wer_weight=args.wer_weight / total_weight,
        cer_weight=args.cer_weight / total_weight,
        rtf_weight=args.rtf_weight / total_weight,
    )

    print_case_table(case_results)
    print_summary_table(summaries)

    if args.save:
        report = save_report(
            suite_path=args.suite,
            suite_name=suite_name,
            profiles_path=args.profiles,
            environment=environment,
            case_results=case_results,
            summaries=summaries,
            weights={
                "wer": args.wer_weight / total_weight,
                "cer": args.cer_weight / total_weight,
                "rtf": args.rtf_weight / total_weight,
            },
        )
        print(f"\nSaved report -> {report}")

    return 0


def main() -> int:
    args = parse_args()
    if args.command == "run":
        return run_command(args)
    if args.command == "compare":
        return compare_reports(args.left, args.right)
    print(f"Unknown command: {args.command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
