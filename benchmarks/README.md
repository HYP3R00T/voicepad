# VoicePad Benchmarks

Measures transcription **accuracy** (WER, CER) and **speed** (RTF) against known reference audio.

## Metrics

| Metric | Meaning | Goal |
|---|---|---|
| **WER** | Word Error Rate — % of words wrong | Lower is better; 0% = perfect |
| **CER** | Character Error Rate — finer than WER | Lower is better |
| **RTF** | Real-Time Factor — latency / audio duration | < 1.0 = faster than real time |

## Running

```sh
# Basic run (turbo model, auto device)
uv run python benchmarks/run_benchmark.py

# Different model or device
uv run python benchmarks/run_benchmark.py --model tiny --device cpu

# Show reference vs hypothesis for each fixture
uv run python benchmarks/run_benchmark.py --diff

# Save results for historical comparison
uv run python benchmarks/run_benchmark.py --save

# Compare current run against last saved run
uv run python benchmarks/run_benchmark.py --save --compare
```

## Adding fixtures

1. Record a WAV file and place it in `benchmarks/fixtures/`
2. Add an entry to `benchmarks/fixtures/fixtures.json`:

```json
{
  "id": "my_fixture",
  "wav": "my_fixture.wav",
  "ground_truth": "The exact words spoken in the recording."
}
```

1. Run the benchmark — the new fixture is picked up automatically.

## Guidelines for good fixtures

- Read the text at a natural, conversational pace — not slow or exaggerated
- Include a mix of: common words, proper nouns, numbers, punctuation boundaries
- Keep each clip between 15–60 seconds
- Record in the same environment you normally use VoicePad (same mic, same room)

## Results history

Saved runs are stored in `benchmarks/results/benchmark_YYYYMMDD_HHMMSS.json`.
Use `--compare` to see deltas between the current run and the last saved run.
This makes regressions visible immediately after adding a new feature.
