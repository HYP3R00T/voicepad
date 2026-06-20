# VoicePad Developer Benchmarks

This benchmarking system is for contributors running VoicePad locally on real hardware.
It is intentionally **not** a CI benchmark because CI does not have the same GPU profile.

The goal is to answer:

- which model/profile gives the best quality on our fixed audio corpus?
- how much latency/RTF does that profile cost on this machine?
- did a code change improve or regress quality/speed?
- how does one machine compare with another?

## Design

The system is built around four pieces:

1. **Fixture suite**: fixed WAV files with exact ground truth.
2. **Profile grid**: explicit decoding/model configurations to test.
3. **Repeated runs**: multiple repetitions per fixture/profile for more stable timing.
4. **Saved report bundles**: machine-readable JSON plus a human-readable summary.

## Metrics

| Metric | Meaning | Goal |
|---|---|---|
| **WER** | Word Error Rate | Lower is better |
| **CER** | Character Error Rate | Lower is better |
| **RTF** | Real-Time Factor (`latency / audio_duration`) | Lower is better |
| **Balanced score** | Weighted mix of WER, CER, and RTF | Lower is better |

Default score weights:

- `WER`: `0.6`
- `CER`: `0.2`
- `RTF`: `0.2`

That means quality dominates, but speed still matters.

## Files

- Suite manifest: `benchmarks/fixtures/fixtures.json`
- Default profile grid: `benchmarks/profiles/default_profiles.json`
- Saved reports: `benchmarks/results/<timestamp>/report.json`
- Human summary: `benchmarks/results/<timestamp>/SUMMARY.md`

## Run a benchmark

```sh
uv run python benchmarks/run_benchmark.py run
```

Use a specific suite/profile set:

```sh
uv run python benchmarks/run_benchmark.py run \
  --suite benchmarks/fixtures/fixtures.json \
  --profiles benchmarks/profiles/default_profiles.json \
  --repetitions 3 \
  --save
```

Tune the quality/speed balance used for ranking:

```sh
uv run python benchmarks/run_benchmark.py run \
  --wer-weight 0.5 \
  --cer-weight 0.2 \
  --rtf-weight 0.3 \
  --save
```

## Compare two reports

```sh
uv run python benchmarks/run_benchmark.py compare \
  --left benchmarks/results/20260621_120000/report.json \
  --right benchmarks/results/20260621_130000/report.json
```

## Suite format

You can keep the suite simple:

```json
[
  {
    "id": "dictation_short",
    "wav": "dictation_short.wav",
    "ground_truth": "The exact spoken text goes here.",
    "language": "en",
    "notes": "Quiet room, laptop mic"
  }
]
```

Or use the richer wrapped format:

```json
{
  "name": "developer-suite-v1",
  "fixtures": [
    {
      "id": "dictation_short",
      "wav": "dictation_short.wav",
      "ground_truth": "The exact spoken text goes here.",
      "language": "en",
      "notes": "Quiet room, laptop mic"
    }
  ]
}
```

## Profile format

```json
{
  "name": "developer-defaults",
  "profiles": [
    {
      "id": "turbo-auto-beam5-vadoff",
      "model": "turbo",
      "device": "auto",
      "compute_type": "auto",
      "beam_size": 5,
      "vad_filter": false
    }
  ]
}
```

## Recommended fixture corpus

Build a corpus that reflects real usage:

- short dictation
- long dictation
- proper nouns
- numbers and symbols
- mild background noise
- one harder clip that often exposes regressions

Try to keep each clip between 15 and 60 seconds.
Do not keep replacing fixtures casually; a stable corpus makes historical comparisons meaningful.

## Machine metadata

Each saved report captures:

- hostname
- OS/platform
- Python version
- git commit
- detected CUDA GPUs
- GPU memory and CUDA version when available

That makes it possible to compare results across contributor machines.

## Suggested contributor workflow

1. Add or update fixture WAV files and ground truth.
2. Run the benchmark suite locally on your GPU.
3. Save the report bundle.
4. Compare it against your previous baseline.
5. Optimize code/config and rerun.
6. Choose defaults based on the best profile score, not anecdotal perception.

This is the mechanism we should use going forward instead of repeated live dictation checks.
