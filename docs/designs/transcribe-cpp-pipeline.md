---
icon: lucide/workflow
---

# transcribe.cpp GGUF transcription pipeline

> **Status:** Approved migration design
>
> **Tracking issue:** [#31](https://github.com/HYP3R00T/voicepad/issues/31)
>
> This document describes the target architecture. Until the migration issues
> are merged, the current implementation remains the authority for current
> runtime behavior.

## Summary

VoicePad will replace Faster Whisper, CTranslate2, Sherpa-ONNX, and the current
split transcription paths with one local GGUF pipeline built around
`transcribe.cpp`. Recording audio remains continuously persisted to disk while
VoicePad plans bounded, VAD-aware ranges and sends them sequentially through one
reused model session. The assembled chunk transcript is authoritative; stopping
a recording never triggers an unbounded full-file retry.

This is an intentional compatibility break. Existing APIs, model IDs,
configuration fields, cache layouts, runtime abstractions, and Markdown metadata
need not remain compatible. Existing WAV files and legacy model caches remain
untouched by automatic migration behavior.

## Problem and success criteria

The current architecture uses different paths for provisional recording chunks
and final file transcription. Its Sherpa path exposes text without native word
timing and fills the shared result shape with synthetic timing. Recording stop
performs a separate unbounded full-file pass, and the application uses a
text-length heuristic when that result appears shorter than provisional text.
A private 141.792-second regression recording demonstrated that a backend can
return nonempty text while omitting later speech.

The migration succeeds when:

- live recording, finite files, history, CLI, foreground TUI, and global hotkey
  transcription use the same planner, executor, assembler, and result contract;
- every runtime input is at most 60 seconds including overlap;
- recording persistence is independent of inference speed and success;
- real word timestamps drive overlap handling and source-coverage checks;
- incomplete, cancelled, truncated, or failed work is never reported as a
  complete transcript;
- existing WAV files and old caches are not modified or deleted automatically;
- a clean installation contains none of the replaced inference or CUDA Python
  dependencies; and
- required repository checks and the private long-recording regression pass.

ASR cannot prove that every spoken word was recognized correctly. In this
design, `complete` means that VoicePad processed every material speech region
without a known execution, truncation, timestamp-coverage, or assembly failure.
It is not a semantic accuracy guarantee.

## Scope and non-goals

The initial migration includes two curated Parakeet v3 quantizations, verified
artifact acquisition, CPU/Vulkan/Metal execution, direct Silero VAD, finite and
growing sources, bounded planning, timestamp assembly, strict configuration,
and complete application cutover.

The initial migration does not include:

- arbitrary community GGUF files;
- cloud transcription or user-side model conversion;
- CUDA provider packaging;
- translation or vocabulary biasing for the initial deployment;
- native model streaming;
- multiple simultaneous inference calls on one model;
- additional model families without a separate evaluation; or
- package publication.

## System boundaries

```text
voicepad application
  owns configuration, TUI/CLI, history, Markdown, clipboard, and presentation
                              |
                              v
TranscriptionEngine (voicepad-core)
  owns deployment resolution, jobs, events, completeness, and lifecycle
       |                 |                  |
       v                 v                  v
ArtifactStore      Audio pipeline       Runtime registry
       |          source -> VAD ->       TranscribeCppDriver
       |          planner -> executor           |
       |          -> assembler                  v
       v                                  transcribe.cpp
Hugging Face
```

### Application ownership

The `voicepad` package owns user configuration, model setup screens, TUI state,
hotkeys, clipboard behavior, history presentation, Markdown rendering, and
operator-visible status. It consumes typed engine events and results. It does
not choose boundaries, call native runtime APIs, inspect GGUF metadata, merge
text, or infer completeness from text length.

### Core ownership

`voicepad-core` owns the offline catalogue, artifact verification, runtime and
device selection, source preparation, VAD, chunk planning, sequential
execution, overlap assembly, progress events, cancellation, and final result
state. It does not import Textual, access global UI state, render Markdown, or
write clipboard content.

### Runtime-adapter ownership

A runtime adapter converts canonical VoicePad requests to one native runtime.
It owns native model/session lifecycle, device enumeration, capability
inspection, result conversion, cancellation, and exception conversion. It does
not acquire artifacts, run VAD, plan chunks, assemble text, or update the UI.

## Principal contracts

The final names may move between modules, but these meanings and dependency
directions are fixed. Implementations must not collapse model, artifact,
deployment, runtime, and device identity into one string.

```python
@dataclass(frozen=True, slots=True)
class ModelDefinition:
    id: str
    name: str
    family: str
    source_repository: str
    source_revision: str | None
    languages: tuple[str, ...]
    license: str


@dataclass(frozen=True, slots=True)
class ArtifactDefinition:
    id: str
    repository: str
    revision: str
    filename: str
    size_bytes: int
    sha256: str
    quantization: str


@dataclass(frozen=True, slots=True)
class DeploymentDefinition:
    id: str
    model_id: str
    artifact_id: str
    runtime_id: str
    declared_capabilities: DeclaredCapabilities
    processing_profile: ProcessingProfile
    default_policy: ChunkPolicy
    recommended: bool
```

Capabilities preserve important distinctions rather than exposing one coarse
feature boolean:

```python
@dataclass(frozen=True, slots=True)
class EffectiveCapabilities:
    native_sample_rate: int
    languages: tuple[str, ...]
    accepts_language_hint: bool
    automatic_language_routing: bool
    returns_detected_language: bool
    timestamp_granularity: TimestampGranularity | None
    streaming: StreamingMode
    translation_targets: tuple[str, ...]
    max_audio_s: float | None
    input_limit_kind: InputLimitKind


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    id: str
    runtime_id: str
    backend: str
    name: str
    device_type: str
    stable_hardware_id: str | None
    memory_total_bytes: int | None
    selectable: bool
```

The adapter boundary is blocking and cancellation-safe. Applications place
blocking operations in their own workers; the core does not depend on an event
loop.

```python
class RuntimeDriver(Protocol):
    @property
    def id(self) -> str: ...

    def enumerate_devices(self) -> tuple[DeviceInfo, ...]: ...
    def inspect(self, artifact: Path) -> EffectiveCapabilities: ...
    def open(self, artifact: Path, device: DeviceInfo) -> RuntimeSession: ...


class RuntimeSession(Protocol):
    @property
    def capabilities(self) -> EffectiveCapabilities: ...

    def warm_up(self) -> None: ...
    def transcribe(
        self,
        audio: PreparedAudio,
        intent: TranscriptionIntent,
    ) -> BackendResult: ...
    def cancel(self) -> None: ...
    def close(self) -> None: ...
```

Requests contain semantic intent only. The initial deployment accepts an
optional listed language and rejects nonempty vocabulary.

```python
@dataclass(frozen=True, slots=True)
class TranscriptionIntent:
    language: str | None = None
    vocabulary: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TimedWord:
    text: str
    start_sample: int
    end_sample: int
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class BackendResult:
    text: str
    words: tuple[TimedWord, ...] | None
    language: str | None
    latency_ms: float
    truncated: bool
```

A chunk separates the audio sent to the runtime from the range it owns:

```python
@dataclass(frozen=True, slots=True)
class AudioChunk:
    index: int
    source_start_sample: int
    source_end_sample: int
    logical_start_sample: int
    logical_end_sample: int
    left_boundary_reason: BoundaryReason | None
    end_boundary_reason: BoundaryReason
```

The engine is the application-facing composition root:

```python
engine = TranscriptionEngine(
    catalog=catalog,
    artifact_store=artifact_store,
    runtimes=runtime_registry,
)

engine.list_deployments()
engine.list_devices(runtime_id="transcribe-cpp")
engine.prepare(deployment_id, on_progress=None, cancel=None)
engine.activate(deployment_id, device_id="auto")
engine.transcribe_file(path, intent, policy=None, on_event=None)
engine.start_incremental(source, intent, policy=None, on_event=None)
engine.deactivate()
```

`transcribe_file()` blocks until one result is available. The application runs
it in a worker. `start_incremental()` returns one `PipelineJob`:

```python
job.finalize() -> TranscriptionResult
job.cancel() -> None
job.result() -> TranscriptionResult | None
```

`finalize()` means no more source samples will arrive and drains terminal work;
it is not cancellation. Repeated finalization and cancellation are idempotent.

```python
@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    text: str
    words: tuple[TimedWord, ...] | None
    duration_s: float
    inference_ms: float
    deployment_id: str
    runtime_id: str
    device: DeviceInfo
    complete: bool
    chunks: tuple[ChunkOutcome, ...]
    failures: tuple[ChunkFailure, ...]
    warnings: tuple[PipelineWarning, ...]
```

Callbacks receive typed events serially outside internal locks. Event callbacks
are presentation observers: exceptions are logged without transcript content
and do not alter pipeline success or ordering.

## Initial deployment baseline

### Runtime

```text
transcribe-cpp==0.1.3
transcribe-cpp-native==0.1.3
```

The native provider supplies CPU plus Vulkan on Linux and Windows, and CPU plus
Metal on macOS. CUDA is not required.

### Model and artifacts

```text
Model ID:          parakeet-tdt-0.6b-v3
Base repository:   nvidia/parakeet-tdt-0.6b-v3
Base revision:     6d590f77001d318fb17a0b5bf7ee329a91b52598
Model license:     CC-BY-4.0

GGUF repository:   handy-computer/parakeet-tdt-0.6b-v3-gguf
GGUF revision:     85ac09ea12fc4b1112fa76810059364bc6adc9de
```

| Deployment ID | Artifact | Bytes | SHA-256 | Role |
|---|---|---:|---|---|
| `parakeet-v3.gguf-q8` | `parakeet-tdt-0.6b-v3-Q8_0.gguf` | 739,508,576 | `5859f77944efcd8eafa23a6350731960b2b55b2203df51f319665c807d802cc7` | Default |
| `parakeet-v3.gguf-q4` | `parakeet-tdt-0.6b-v3-Q4_K_M.gguf` | 485,425,504 | `b68557be1e3c40207fd7c4bd9d63f1d3316b963f15325bfb0cc16a8bb0ffd181` | Smaller alternative |

A model identifies the neural model, an artifact identifies immutable bytes,
and a deployment identifies the selectable model/artifact/runtime combination.
Hosting coordinates are acquisition data, not durable user identity.

## Catalogue and capabilities

A versioned JSON catalogue is a package resource and loads without network
access. It contains model, artifact, and deployment records. Startup and tests
reject duplicate IDs, dangling references, malformed hashes, nonpositive sizes,
unknown runtime IDs, an invalid default, and inconsistent required capability
claims.

The catalogue records reviewed policy. Loaded GGUF metadata records runtime
fact. Activation reconciles both:

- architecture, variant, native sample rate, artifact identity, and required
  timestamp support must match or activation fails;
- optional runtime capabilities may narrow catalogue claims and emit a warning;
- deployment policy may deliberately expose less than the runtime supports; and
- application code receives only effective capabilities, never model-family
  conditionals.

The initial effective capability shape distinguishes language hints, automatic
language routing, and returned detected-language metadata. Parakeet may route
without a hint but returns no useful detected-language value. An empty native
language string becomes `None`. Vocabulary intent is rejected because this
deployment does not support it. Word timing is required by its processing
profile; nonempty text without usable native words is a protocol failure.

## Artifact trust and cache lifecycle

The managed cache layout is deployment-specific:

```text
~/.config/voicepad/models/deployments/<deployment-id>/
  model.gguf
  artifact.json
```

The packaged catalogue is the trust anchor. Repository responses and downloaded
bytes are untrusted until exact verification succeeds.

Acquisition follows this contract:

1. Resolve one curated artifact from the offline catalogue.
2. If `model.gguf` exists, verify exact bytes and SHA-256 before readiness.
3. A valid model with missing metadata is reverified before metadata recovery.
4. Download into a unique operation-owned staging directory.
5. Reject a conflicting `Content-Length` before streaming.
6. Stop immediately after reading `expected_size + 1` bytes.
7. Report typed progress and observe cancellation during streaming and hashing.
8. Verify exact size and SHA-256 in staging.
9. Flush and atomically replace the managed `model.gguf` only with verified
   bytes.
10. Write `artifact.json` atomically.
11. Mark the deployment ready only when the model verifies and metadata is
    consistent.

An invalid destination remains in place until a verified replacement is ready.
Concurrent processes may use separate staging paths; because deployment IDs
refer to immutable bytes, verified identical promotions are safe. A process
that observes a model without metadata revalidates it instead of trusting or
deleting it. Failed and cancelled operations remove only their own staging
content. Disk exhaustion is a typed acquisition failure.

Legacy Sherpa and CTranslate2 caches are never inspected, migrated, or deleted
by the new system.

## Device and runtime lifecycle

Device enumeration returns runtime ID, backend kind, display name, device type,
stable hardware identity when available, memory when reported, and whether the
device can be selected explicitly. Runtime array indices are process-local and
never persisted.

Stable IDs use semantic or hardware identity, for example:

```text
transcribe-cpp:vulkan:pci:0000:01:00.0
transcribe-cpp:metal:default
transcribe-cpp:cpu
```

`auto` ranks discrete GPU, integrated GPU, then CPU. It falls through only when
a class is unavailable. A load, memory, driver, or inference failure on an
available selected device is reported; it does not trigger a silent retry on a
different device.

Explicit selection is strict. VoicePad resolves the stored stable ID to current
runtime data, opens the model, and compares the actual native device with the
request. A mismatch closes the model and fails activation. A device that the
0.x binding can enumerate but cannot address unambiguously is visible but not
explicitly selectable.

One engine owns one active model and session. Activation:

1. verifies the artifact;
2. resolves the device;
3. opens the model and one session;
4. reconciles capabilities;
5. executes a one-second zero-PCM warm-up using the deployment's timestamp
   request; and
6. reports ready only after warm-up succeeds.

One job and one native inference call may be active at a time. Starting a second
job, switching deployment or device, or deactivating during work raises
`EngineBusyError`. Calls are serialized even if a future API creates multiple
sessions over one model.

`Session.cancel()` is called from a control thread. `Aborted` preserves its
native partial result, marks the job incomplete, and invalidates the active
session. Unknown native execution failures do the same. Reuse after cancellation
may be enabled only after an integration contract proves it safe for the pinned
runtime. Closing model and session objects is idempotent.

On shutdown, capture and audio finalization happen before inference
cancellation. VoicePad waits up to ten seconds for native cancellation. If it
does not return, VoicePad does not free native state underneath an active call;
shutdown remains blocked and reports that force termination is the remaining
operator action.

## Audio sources and preparation

A source exposes native sample rate, channels, committed sample count, final
state, final sample count when known, and bounded sample-range reads. Immutable
sources are final at construction. Growing sources expose only writer-committed
samples and become final after capture stops and persistence drains.

WAV, FLAC, and Ogg inputs use SoundFile. MP3, M4A, and MP4 inputs are converted
by a non-shell FFmpeg argument vector using a local filesystem path,
`-nostdin`, mono 16 kHz output, and operation-owned temporary storage. Conversion
is cancellable; VoicePad asks FFmpeg to terminate, waits five seconds, then
kills it and waits five more seconds. Decoded staging output is capped at 4 GiB
and disk exhaustion is reported distinctly. Temporary conversion content is
removed idempotently when the source closes. Existing source files are opened
read-only and never replaced or removed.

Preparation reads only a descriptor's source range, converts integer PCM to
float32 using its numeric scale, downmixes, resamples with
`scipy.signal.resample_poly`, and returns a contiguous one-dimensional 16 kHz
array. It does not peak-normalize. Each prepared range retains an integer,
rational source/runtime mapping. Native timestamp starts map with floor and ends
with ceiling, then clamp to the descriptor, preventing cumulative floating-point
drift.

## Direct Silero VAD

VoicePad bundles one reviewed VAD artifact:

```text
Artifact: silero_vad_v6.onnx
Bytes:    1,249,744
SHA-256:  914fd98ac0a73d69ba1e70c9b1d66acb740eff90500dfde08b89a961b168a6a9
Runtime:  CPU ONNX Runtime
```

The artifact matches the Silero v6 model pinned by Faster Whisper at
`ed9a06cd89a93e47838f564998a6c09b655d7f43`. Distribution includes Silero and
SYSTRAN/Faster Whisper provenance and applicable MIT notices. The duplicate
legacy VAD asset and network downloader are removed only after this adapter is
proven.

The detector retains recurrent state while scanning sequential frames, resets
at job start and after terminal failure/cancellation, and emits absolute source
speech regions. It does not know about inference chunks. VAD always uses CPU so
it does not compete for transcription GPU memory.

## Deterministic chunk planning

Default policy:

```text
minimum boundary: 20 seconds
preferred target: 30 seconds
lookback:          10 seconds
hard input cap:    60 seconds, including left overlap
minimum silence:   500 milliseconds
natural overlap:   0.5 seconds
forced overlap:    2.0 seconds
```

Durations convert once to integer source samples for each source. A silence
qualifies after its complete duration reaches 500 ms. Its boundary is the
integer midpoint of the confirmed silence.

For each logical range:

1. Ignore boundaries before `logical_start + minimum`.
2. At the preferred target, choose the latest qualifying midpoint in
   `[preferred - lookback, preferred]` relative to the logical start.
3. If none exists, wait and choose the first subsequently confirmed qualifying
   midpoint.
4. Before waiting further, calculate the forced endpoint from the current
   source start so `source_end - source_start` can never exceed 60 seconds.
5. Emit at that forced endpoint if no natural endpoint occurs first.
6. On finalization, emit remaining material speech; classify an all-silent tail
   as excluded without inference.

Left overlap belongs to the boundary that begins the new logical range: 0.5
seconds after a natural boundary and 2 seconds after a forced boundary, clamped
to source sample zero. The first range has no left overlap. Consequently, after
a forced split, a range normally owns at most 58 seconds of new audio while its
complete runtime input remains at most 60 seconds.

Logical ranges are ordered, half-open, and never overlap. Source ranges may
overlap only by the declared left context. Every committed sample is classified
exactly once as logical ownership or VAD-confirmed excluded non-speech. Planner
finalization is idempotent and all positions remain integer source samples.

## Execution and bounded resources

```text
microphone callback
  -> bounded 256-buffer persistence queue
  -> growing float WAV and committed cursor

planner worker
  -> sequential VAD scan
  -> bounded queue of 8 AudioChunk descriptors

inference worker
  -> bounded disk read
  -> preparation
  -> one serialized session call
  -> assembler
```

Queues never contain accumulated prepared audio. At most one prepared runtime
array exists. A slow runtime creates a disk-backed descriptor backlog; planner
backpressure may pause scanning but cannot block the writer. Disk use grows with
recording duration and cached artifacts. Disk or writer failure stops capture
loudly. If PCM finalization fails, the float WAV spool remains available for
recovery and its path is reported without transcript content.

Recording stop is not job cancellation. It stops microphone callbacks, drains
the writer, atomically finalizes the PCM WAV, marks the source final, plans the
remaining material speech, and waits for all descriptors to reach terminal
outcomes. No full-file retry follows.

Pipeline events are ordered and delivered outside internal locks:

```text
ArtifactProgress, ModelLoading, ModelReady,
ChunkPlanned, ChunkStarted, ChunkCompleted, ChunkFailed,
TranscriptUpdated, PipelineFinalizing,
PipelineCompleted, PipelineCancelled, PipelineFailed
```

Presentation callback failures are logged without transcript content and do not
mutate pipeline state.

## Timestamp assembly

Native word text, including punctuation, is preserved. Word confidence remains
`None`; token probability is not promoted to word confidence. For the initial
processing profile, output text must map to the native word sequence. Unmapped
non-whitespace text or nonempty text without words is a protocol failure.

For adjacent chunks, only words whose absolute source intervals intersect the
known audio overlap, expanded by 250 ms for rounding and alignment jitter, are
merge candidates. Comparison normalization applies Unicode NFKC, case folding,
and removal of leading/trailing Unicode punctuation while retaining internal
apostrophes and hyphens. Original text is always used for output.

Two candidates match only when their normalized forms are equal and their
absolute midpoints differ by at most 750 ms. A dynamic-programming alignment
selects the monotonic maximum-cardinality matching, then minimum total midpoint
distance. This prevents repeated words from matching out of order.

A duplicate may be removed only when either:

- it belongs to a run of at least two consecutive matched words; or
- it is one normalized token of at least four characters whose midpoint differs
  by no more than 250 ms.

Matched duplicates collapse to the observation farther from its physical input
edge; ties retain the earlier chunk's form. Unmatched words from both chunks are
preserved and ordered by absolute midpoint. Preserving competing overlap text
emits `UncertainOverlapWarning`. No fuzzy lexical substitution permits deletion.
This intentionally prefers a visible duplicate over silent loss.

The assembler commits words before the unresolved overlap and keeps only the
current overlap tail provisional. The next chunk resolves that tail. Job
finalization applies the same rules once and commits everything. Rendering joins
the preserved Parakeet word forms with spaces, which retains punctuation attached
to native words.

## Speech-coverage validation

Coverage is evaluated per logical range after assembly. It detects gross silent
omission; it does not estimate semantic accuracy.

Each timed word supplies an evidence interval expanded by one second on each
side and clamped to the logical range. The validator intersects those intervals
with VAD-confirmed speech. It records `CoverageGap` when either:

- total VAD-confirmed speech is at least one second and the range has no words;
  or
- more than two contiguous seconds of VAD-confirmed speech remains outside all
  word evidence intervals.

A coverage gap, native truncation, chunk failure, cancellation, protocol
failure, or unprocessed material speech makes the final result incomplete. VAD
false positives may conservatively mark a result incomplete; the preserved WAV
allows retranscription. Coverage thresholds are versioned processing-profile
policy and require benchmark evidence and design review before change.

## Results and application behavior

The final result includes text, optional real words, source duration, summed
inference wall time, deployment/runtime/device identity, `complete`, chunk
outcomes, failures, and warnings. Activation and VAD time are reported
separately from inference time. Missing optional metadata stays `None`.

One strict application configuration owns:

```text
deployment_id, device_id, language,
chunk_minimum_s, chunk_preferred_s, chunk_lookback_s, chunk_maximum_s,
silence_ms, natural_overlap_s, forced_overlap_s
```

Unknown and obsolete fields fail with the configuration path and field names.
VoicePad does not ignore them, overwrite the file, or regenerate defaults over
an existing configuration. Deployment or device changes are allowed only while
idle.

For a complete nonempty result, the application atomically writes Markdown,
updates history from `result.text`, and copies that exact text. For an incomplete
nonempty result, it atomically records the partial transcript with
`complete: false`, displays the failure, and does not auto-copy. With no partial
text, it writes metadata-only failure information. Complete no-speech output
preserves the WAV and metadata but copies nothing.

New Markdown metadata records schema version, WAV filename, deployment, runtime,
stable device identity and display name, completeness, duration, inference
latency, chunks, failures, warnings, and timestamp. Backward compatibility with
old metadata is not required. History retranscription opens the original WAV
read-only.

## Failure and recovery matrix

| Failure | Result | Recovery |
|---|---|---|
| Invalid catalogue | Engine construction fails | Install corrected package |
| Invalid configuration | Startup fails with fields/path | User edits configuration |
| Download interruption/cancellation | Not ready; own staging removed | Retry preparation |
| Hash/size mismatch | Never loaded; existing destination retained | Retry or inspect network/storage |
| Explicit device unavailable/mismatched | Activation fails without fallback | Select `auto` or another device |
| Warm-up/capability failure | Partial session closes; inactive | Correct artifact/runtime/device |
| VAD/planner invariant failure | Job incomplete; WAV preserved | Retranscribe after correction |
| Native truncation | Partial chunk retained; job incomplete | Retranscribe or adjust reviewed policy |
| Cancellation | Partial result retained; session invalidated | Reactivate for another job |
| Native memory/unknown failure | Remaining work stops; session invalidated | Select another device/reactivate |
| Uncertain overlap | Preserve competing text; warning | User can edit or retranscribe |
| Coverage gap | Preserve available text; incomplete | Retranscribe from WAV |
| Writer/disk failure | Capture stops; recoverable spool retained when possible | Free space and recover spool |
| Markdown failure | WAV and result remain; no false save success | Retry persistence |

VoicePad performs no cross-device inference retry and no unbounded final retry.
Cleanup operations are idempotent and never target old caches or existing WAVs.

## Security, privacy, and licensing

Only catalogue-approved, hash-verified GGUF bytes reach the native parser.
Arbitrary paths and community files are rejected initially. Public artifact
acquisition requires no credential. VoicePad never logs audio, transcript text,
word text, credentials, tokens, or private fixture paths. Operational logs may
contain deployment/device IDs, non-sensitive sample ranges, durations, timings,
hashes, and typed failures.

User recordings, transcripts, private logs, model binaries, and caches are not
committed, attached to issues, or uploaded for CI. Private regression evidence
records only non-sensitive measurements and pass/fail coverage.

Distribution documentation and notices retain:

- Parakeet and GGUF model CC-BY-4.0 attribution;
- `transcribe.cpp` and Python binding MIT attribution;
- Silero VAD MIT attribution; and
- Faster Whisper/SYSTRAN provenance and MIT attribution for the bundled pinned
  VAD asset.

Package publication remains frozen and the release workflow remains disabled.

## Migration and rollback

Replacement is delivered behind focused issues while `main` remains buildable.
Internal legacy and new paths may coexist only until all application consumers
cut over. The final state contains no compatibility adapter. Legacy code and
dependencies are removed only after file and growing-source application paths
are proven.

No automatic data migration runs. Rolling back code therefore leaves existing
WAV files and old caches untouched. New strict configuration and Markdown may
not load in an old release; compatibility is explicitly out of scope. A design
change discovered during implementation must be reviewed and merged before
dependent implementation continues.

## Verification strategy

### Deterministic CI proof

Unit and property tests cover catalogue validation, artifact bounds and atomic
promotion, device selection, capability and exception conversion, sample maps,
Silero state, boundary invariants, queue bounds, cancellation, timestamp
alignment, uncertainty preservation, coverage gaps, completeness, strict
configuration, and output identity.

Runtime contract tests use a fake binding but exercise the adapter contract,
including unsupported intent, silent truncation, cancellation, idempotent close,
and serialization. Concurrency tests use barriers rather than timing-only
sleeps.

Every migration pull request runs:

```sh
uv run ruff check
uv run ruff format --check
uv run ty check
uv run pytest packages --cov=voicepad --cov=voicepad_core --cov-fail-under=70
uv run zensical build --clean
```

### Optional local integration proof

Checks marked for local model or GPU assets cover Q8 Vulkan, Q4 CPU, actual
discrete-device selection, real word timestamps, native cancellation, and cold
and warm activation. Unavailable hardware or assets are reported as unavailable,
not passing.

The private long-recording regression must use bounded calls, produce the known
complete-scale transcript, and show timed-word coverage through final speech.
The recording and transcript remain outside Git and GitHub.

### Real application proof

Before migration closure, exercise finite file transcription, history
retranscription, CLI recording, foreground TUI recording, global hotkey
recording, stop with backlog, shutdown during recording, model switching while
idle, strict configuration failure, complete/incomplete/no-speech results,
Markdown persistence, and clipboard equality.

Windows Vulkan and macOS Metal behavior remain unverified until exercised on
those platforms and must be documented honestly.

## Final migration acceptance

The tracking issue may close only when all required child issues are merged and:

1. exact package and artifact pins are present;
2. only verified GGUF reaches the runtime;
3. discrete, integrated, and CPU selection behave honestly;
4. one warmed session is reused sequentially;
5. every runtime range remains within the hard cap including overlap;
6. persistence survives slow or failed inference;
7. real timestamps drive conservative assembly and coverage;
8. stop drains the final speech-bearing tail without a full-file retry;
9. all application surfaces use the shared pipeline;
10. incomplete output is persisted honestly and never auto-copied;
11. legacy inference code and dependencies are absent;
12. old WAV files and caches remain untouched;
13. licensing and operator documentation are complete; and
14. required quality, local integration, private regression, and real-surface
    evidence is recorded without exposing sensitive data.
