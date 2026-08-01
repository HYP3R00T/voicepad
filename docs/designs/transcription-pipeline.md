---
icon: lucide/workflow
---

# Resident transcription pipeline

> **Status:** Proposed for human review
>
> **Primary target:** Linux x86_64 with an NVIDIA GPU
>
> **Initial deployment:** Official NVIDIA Parakeet TDT 0.6B v3,
> Transformers, PyTorch FP16, and CUDA
>
> This document defines the target architecture. Current production code remains
> authoritative until the migration is implemented and verified.

## Executive decision

VoicePad will use one resident, warmed transcription deployment and one bounded,
disk-backed audio pipeline. The application will load the selected deployment
once, keep it ready across recording sessions, transcribe VAD-planned chunks
while recording continues, drain the final speech-bearing tail on stop, persist
one honest result, copy complete text, and return to ready without restarting the
TUI.

The first production deployment is deliberately optimized for the maintainer's
actual system:

```text
Operating system: Linux x86_64
Device:           NVIDIA CUDA GPU
Tested memory:    4 GiB VRAM
Model:            NVIDIA Parakeet TDT 0.6B v3
Artifact:         official model.safetensors
Runtime:          Transformers + PyTorch
Precision:        FP16
```

The pipeline is not coupled to Parakeet or PyTorch. Applications select a
curated deployment. A deployment binds a model, immutable artifacts, one
adapter, capabilities, a resource profile, and a processing profile. New model
families or runtimes add deployments and adapters without changing recording,
VAD, planning, persistence, history, Markdown, clipboard, or TUI contracts.

VoicePad will not initially ship several speculative runtimes. Official
Parakeet/PyTorch is the one production path. Other artifacts such as GGUF,
ONNX, or TensorRT become separate deployments only when a measured user need
and quality evaluation justify them.

## Design principles

- Prefer official model artifacts, repositories, released packages, and public
  APIs over conversions, forks, copied private internals, or compatibility
  wrappers.
- Pin every external version and immutable artifact; verify downloaded bytes.
- Keep one obvious production path until evidence justifies another deployment.
- Keep model/runtime specifics behind the deployment adapter, not spread across
  application code.
- Preserve user audio first; inference and derived text are replaceable.
- Fail unsupported requests explicitly rather than ignore them.
- Prefer bounded, deterministic state and visible recovery over silent fallback.
- Add configuration only for behavior users can understand and verify.
- Do not optimize already sub-second work through a more complex runtime without
  an observable product benefit.

## Clean replacement policy

Backward compatibility is not a requirement. The migration does not preserve
legacy core APIs, model IDs, backend names, configuration fields, cache layouts,
Markdown schemas, keyword arguments, or synthetic result shapes. It introduces
no compatibility adapters, deprecation shims, dual-write formats, or permanent
legacy branches.

Obsolete source files, tests, dependencies, documentation, and configuration
may be deleted or rewritten when their final consumer cuts over. Temporary
coexistence is allowed only to keep `main` functional across focused pull
requests; it is not part of the final architecture. Tests that assert obsolete
behavior are removed with that behavior rather than weakening new contracts to
keep them passing.

Data retention is separate from API compatibility. Existing WAV files remain
immutable and legacy model caches remain on disk until the user deletes them,
but new code ignores old cache layouts. Obsolete configuration fails with an
actionable clean-break message and is never silently migrated or overwritten.
Old Markdown compatibility is not guaranteed; the original WAV remains the
source for a new transcription.

## Goals

- Make the official Parakeet model reliable on the target Linux/NVIDIA system.
- Keep the model loaded and warmed between recordings.
- Preserve microphone audio independently of inference.
- Process bounded ranges while recording continues.
- Prefer natural VAD boundaries and force a safe boundary during continuous
  speech.
- Produce real token/word timing and conservative overlap assembly.
- Use the same pipeline for live recording, files, history, CLI, TUI, and the
  global hotkey.
- Represent unsupported model features honestly.
- Permit a future model family or runtime without application-level model
  conditionals.
- Preserve existing WAV files and leave obsolete caches untouched.

## Non-goals for the first deployment

- Supporting every operating system or GPU.
- macOS support.
- Claiming Windows support before a Windows/NVIDIA validation.
- Shipping several runtime engines merely for theoretical flexibility.
- Training or fine-tuning Parakeet.
- Creating or hosting a converted model before a measured need exists.
- Native model streaming.
- Concurrent inference calls against one loaded model.
- Cloud transcription.
- Translation.
- Claiming native hotword support that the model does not provide.
- PyPI publication during the migration freeze.

## Verified research basis

All private audio remained local. Only aggregate measurements are recorded.

### Official Transformers/PyTorch FP16

Test environment:

```text
Python:       3.13.14
GPU:          NVIDIA GeForce RTX 3050 Laptop GPU, 4 GiB
Driver:       595.84
PyTorch:      2.13.0+cu130
Transformers: 5.14.1
Accelerate:   1.14.0
Precision:    FP16
```

Measured behavior:

| Case | Result |
|---|---:|
| Cached model load | 4.57 s in the first measured process; 0.76-1.31 s after unload/reload in-process |
| 30 s zero-audio warm-up | 1.41 s |
| First real 30 s after same-shape warm-up | 0.193 s |
| Repeated warm 30 s | 0.168-0.234 s, 0.181 s mean over 49 runs |
| Complete 141.792 s inference | 1.43 s |
| Complete transcript size | 1,705 characters, 529 timestamp records, 321 aggregated words |
| Resident 30 s GPU use | about 1.46 GiB reported by `nvidia-smi` |
| Complete-file GPU peak | about 1.90 GiB reported by `nvidia-smi` |
| Fifty repeated jobs | one deterministic output hash; no allocator growth |
| Explicit unload | GPU use fell to the CUDA-context floor, about 148 MiB |
| Offline reload | succeeded and reproduced the same output |
| Generation cancellation | returned after encoder completion and the model remained reusable |

A bounded 30-second call succeeded under an artificial 1.43 GiB PyTorch
allocator cap. A complete 142-second call required about 1.67 GiB reserved and
failed under a 1.64 GiB cap. These allocator experiments are not a substitute
for testing a physical lower-memory GPU. The initial supported claim remains the
actually tested 4 GiB class.

### Overlapping chunks

A five-chunk fixed-boundary evaluation used source ranges approximately
`0-30`, `28-60`, `58-90`, `88-120`, and `118-end`. The resident model processed
all chunks in about two seconds total. Transformers timestamp records aggregated
exactly back to each chunk's native text.

A conservative timestamp assembler:

- identified 18 duplicate overlap words;
- preserved two uncertain overlap observations;
- retained final timing through the final recognized content; and
- reached about 95.2% word-sequence agreement with direct full-file inference.

Direct full-file output is a comparison oracle, not ground truth. Independent
chunks legitimately differ near artificial boundaries. Production planning must
prefer VAD-confirmed pauses, and assembly must preserve uncertainty rather than
delete text to imitate the full-file result.

### Alternative runtime findings

- Full-precision ONNX CUDA was extremely fast but consumed about 3.6-3.75 GiB
  for bounded calls and failed on the complete recording with 4 GiB VRAM.
- The tested ONNX int8 conversion was slower, used extensive CPU/GPU copies,
  and produced materially less text on the private recording.
- GGUF Q8 through Vulkan was complete, compact, cancellable, and fast, but used
  a third-party conversion and pre-1.0 runtime. It remains a researched fallback,
  not the initial production deployment.
- PyTorch FP16 was at least as fast as the tested ONNX bounded path while using
  much less VRAM and using NVIDIA's official artifact.

### Official Silero VAD

VoicePad will use the official [`snakers4/silero-vad`](https://github.com/snakers4/silero-vad)
v6.2.1 release, not the older model inherited through Faster Whisper.

```text
Repository:     snakers4/silero-vad
Release:        v6.2.1
Commit:         7e30209a3e901f9842f81b225f3e93d8199902b1
License:        MIT, Copyright 2020-present Silero Team
Distribution:   silero-vad 6.2.1 official PyPI wheel
Wheel bytes:    9,146,242
Wheel SHA-256:  09de93c4d874bb19c53e62a47dd38be5f163cedad2b5599583231f2a84ef79cb
Wheel entry:    silero_vad/data/silero_vad.onnx
Model bytes:    2,327,524
Model SHA-256:  1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3
```

The official model ran through ONNX Runtime 1.28 CPU with 512-sample 16 kHz
frames, the required rolling 64-sample context, and recurrent `[2, 1, 128]`
state. It scanned 141.792 seconds in 0.457 seconds, about 310x real time, and
returned the expected speech-probability range.

The released Silero Python package currently requires `torchaudio`, while no
matching torchaudio 2.13 release exists for the selected PyTorch 2.13 stack.
VoicePad therefore does not install or import that package. The artifact store
verifies the official wheel, safely extracts only the exact ONNX entry, verifies
the extracted bytes, and caches them with provenance metadata. Runtime uses a
small independent CPU ONNX session. This keeps the official model while avoiding
mismatched binary dependencies.

## Initial artifact manifest

Official source:

```text
Repository: nvidia/parakeet-tdt-0.6b-v3
Revision:   7c35754d166cca382ad1e53e68b01e7c575f3a1d
License:    CC-BY-4.0
```

Required files:

| File | Bytes | SHA-256 |
|---|---:|---|
| `model.safetensors` | 2,508,311,120 | `3a2026366188c8c68598edbbff92f8d11590a08e0ae2e6775544e7b07d6a5e11` |
| `config.json` | 1,153 | `e747b85e1bdfd300c8b8ac63bac8dd5221f8fe9bc275b48d06c735fcd6971b6e` |
| `generation_config.json` | 289 | `b141de6ec6d7f982ece13f98f604e3fe1807ea9c0e839185d0ab7064604209d0` |
| `processor_config.json` | 392 | `8346a93a3b987fa1dec57a78f045cd0817d21786589a5a096b41a57a446fd1d7` |
| `tokenizer.json` | 1,159,960 | `bd321b096832a3f270bd3b2a88823957920f1a5c5ada71114a26ea729d0cbe91` |
| `tokenizer_config.json` | 290 | `0b2fe0037599ee335f0b972fa682bf0ece74e4ccfec755cb7daa3405d3d3e874` |

VoicePad loads the official stored artifact as FP16 without publishing a
converted model:

```python
AutoModelForTDT.from_pretrained(
    local_snapshot,
    dtype=torch.float16,
    device_map="cuda",
    local_files_only=True,
)
```

## Dependency baseline

Initial direct runtime dependencies are expected to include exact reviewed
versions of:

```text
torch==2.13.0
transformers==5.14.1
accelerate==1.14.0
librosa==0.11.0
huggingface-hub
onnxruntime==1.28.0  # CPU VAD only
```

Existing NumPy, SciPy, SoundFile, and SoundDevice dependencies remain where
used. The lock file pins all transitive CUDA, tokenizer, safetensors, and audio
processor packages. A separate CUDA toolkit installation is not required; the
NVIDIA driver remains a system prerequisite.

`torchaudio` is not part of the initial dependency set. Faster Whisper,
CTranslate2, Sherpa-ONNX, their CUDA wheel configuration, and obsolete NVIDIA
runtime packages are removed only after every application path has cut over.

The tested environment occupied about 5.1 GiB for Python packages plus 2.4 GiB
for the official model snapshot. This one-time storage cost is accepted for the
initial target.

## Deployment-oriented architecture

The application selects a deployment, not a model filename, runtime index, or
model family.

```python
@dataclass(frozen=True, slots=True)
class DeploymentDefinition:
    id: str
    model_id: str
    artifact_manifest_id: str
    adapter_id: str
    precision: Precision
    capabilities: DeclaredCapabilities
    resources: ResourceProfile
    processing: ProcessingProfile
    recommended: bool
```

The initial record is conceptually:

```text
id:                 parakeet-v3.transformers-fp16-cuda
model_id:           nvidia-parakeet-tdt-0.6b-v3
artifact_manifest:  official-parakeet-v3-safetensors
adapter_id:         transformers-parakeet-tdt
precision:          fp16
required_device:    cuda
initial_platform:   linux-x86_64
```

### Identity separation

- A model identifies the neural model.
- An artifact manifest identifies immutable files and source revision.
- An adapter identifies one execution and output contract.
- A deployment combines those with precision, resources, and processing.
- A device identifies actual hardware independently of a transient CUDA index.

Hosting coordinates are acquisition data and do not become user-facing IDs.
Changing hosting without changing verified bytes does not change deployment
identity.

### Model and runtime independence

The TUI, CLI, history, and pipeline depend only on a `TranscriptionSession`:

```python
class TranscriptionSession(Protocol):
    @property
    def deployment(self) -> ActiveDeployment: ...

    @property
    def capabilities(self) -> EffectiveCapabilities: ...

    def transcribe(
        self,
        audio: PreparedAudio,
        intent: TranscriptionIntent,
        cancellation: CancellationToken,
    ) -> BackendResult: ...

    def close(self) -> None: ...
```

The initial `TransformersParakeetTDTAdapter` alone knows about
`AutoProcessor`, `AutoModelForTDT`, FP16 tensors, TDT duration records, and
Metaspace token aggregation.

A future Transformers Whisper model may reuse PyTorch device and artifact
infrastructure but use a different adapter. A future runtime such as TensorRT
or GGUF adds another adapter and deployment. Model-family branching is confined
to adapter registration and never appears in application or pipeline code.

The abstraction does not claim every model has the same features. Effective
capabilities describe what the loaded deployment can actually accept and
return.

## Capability contract

```python
@dataclass(frozen=True, slots=True)
class EffectiveCapabilities:
    native_sample_rate: int
    languages: tuple[str, ...]
    accepts_language_hint: bool
    returns_detected_language: bool
    timestamps: TimestampGranularity | None
    native_streaming: bool
    translation_targets: tuple[str, ...]
    context_biasing: ContextBiasingMode
    cancellation: CancellationMode
```

Initial Parakeet capabilities:

```text
sample rate:                16 kHz
languages:                  25 documented European languages
automatic multilingual ASR: yes
language hint:              not exposed by this adapter
detected-language output:   no
punctuation/capitalization:  yes
timestamps:                 token + predicted duration
native streaming:           no
translation:                no
beam search:                no; greedy only
native prompt/hotwords:     no
cancellation:               after encoder, during generation
batching:                   runtime-capable, unused for one live job
```

Unsupported intent is rejected or explicitly reported. It is never silently
ignored.

## Resident model lifecycle

The application owns one engine with this observable state machine:

```text
UNPREPARED
  -> PREPARING_ARTIFACT
  -> LOADING
  -> WARMING
  -> READY
  -> ACTIVE_JOB
  -> FINALIZING
  -> READY

READY -> UNLOADING -> UNPREPARED
READY -> switching deployment -> UNLOADING -> LOADING -> WARMING -> READY
any state -> FAILED when recovery cannot preserve the active contract
```

Application startup prepares the selected deployment, loads it on CUDA, and
warms it with a discarded 30-second zero waveform. The same-shape warm-up was
measured to reduce the next real 30-second call to the stable warm range. Ready
is emitted only after artifact verification, CUDA placement, and warm-up pass.

The model remains resident across recordings. Each job creates fresh decoder,
VAD, planner, assembler, cancellation, and result state. No decoder state leaks
between recordings. Stop finalizes one job and returns the engine to ready; it
does not unload the model or close the TUI.

Only one job and one model call run at a time. Deployment or device switching is
rejected during an active job. An explicit unload action releases model tensors,
runs garbage collection, empties the CUDA allocator cache, verifies release as
far as PyTorch can report it, and leaves the CUDA context available for later
reload. Automatic idle unload is not enabled initially because immediate
recording readiness is the product priority.

## Artifact preparation and offline behavior

The catalogue is a packaged offline resource. Preparation resolves only a
curated immutable manifest. Each required file is downloaded into
operation-owned staging storage, size-limited, SHA-256 verified, flushed, and
atomically promoted into a deployment snapshot. Readiness requires all files.

A failed or cancelled acquisition never appears ready. An existing verified
snapshot remains available after another preparation failure. Partial staging
is removable; verified snapshots and legacy caches are not automatically
deleted.

The official Silero wheel is a second curated artifact manifest. Extraction is
data-only: reject path traversal, duplicate entries, a missing or compressed-size
mismatch, and any entry other than the exact declared ONNX path. Neither wheel
code nor package metadata is imported or executed. Readiness requires both wheel
and extracted-model hashes to match.

Runtime load uses a local path and `local_files_only=True`. The tested model
loaded, inferred, unloaded, and reloaded successfully with `HF_HUB_OFFLINE=1`.
VoicePad does not depend on a network request after preparation.

## Audio source and persistence

One source protocol covers immutable files and growing recordings:

```python
class AudioSource(Protocol):
    sample_rate: int
    channels: int

    def committed_samples(self) -> int: ...
    def is_final(self) -> bool: ...
    def read(self, start_sample: int, end_sample: int) -> AudioWindow: ...
```

The microphone callback copies buffers only into the bounded persistence queue.
The writer appends a float WAV spool, publishes committed sample positions, and
atomically finalizes the user-visible PCM WAV. Inference never executes on the
microphone callback and cannot block persistence.

Writer backpressure or disk failure stops capture loudly. A finalization failure
retains recoverable spool data. Existing WAV files are opened read-only and are
never modified or deleted automatically.

File inputs use SoundFile where supported. Any retained FFmpeg conversion uses
an argument vector rather than a shell, operation-owned disk storage, explicit
cancellation, output bounds, and idempotent cleanup. Supported formats are a
product decision recorded by the application, not inferred from whichever
libraries happen to be installed.

## Canonical audio preparation

Every ASR range becomes:

```text
mono
16,000 Hz
float32 source representation
contiguous one-dimensional array
```

The adapter's processor creates model features and moves them to CUDA FP16.
VoicePad downmixes and resamples with deterministic source/runtime sample
mapping. It does not peak-normalize by default. Timestamps map back to absolute
source samples without cumulative floating-point drift.

## CPU VAD and adaptive planning

One official Silero v6.2.1 ONNX session stays on CPU and retains rolling audio
context plus recurrent state while scanning sequential committed frames. It
emits absolute speech/silence regions and does not know about model chunks. Job
start and terminal teardown reset both context and recurrent state; ordinary
frame and chunk boundaries do not.

Initial planning policy remains evidence-driven and configurable within safe
bounds:

```text
minimum useful boundary:       20 seconds
preferred duration:            30 seconds
lookback before target:        10 seconds
minimum qualifying silence:    500 milliseconds
natural overlap:               previous pause-to-pause speech unit
maximum semantic overlap:      12 seconds
forced/fallback overlap:       2 seconds
hard runtime input:            60 seconds including overlap
```

A natural breakpoint is the midpoint of a confirmed qualifying silence. The
planner retains an ordered history of breakpoints. At the preferred target, it
selects the most recent qualifying breakpoint inside the lookback. If none
exists, it waits for the first later qualifying breakpoint but forces a boundary
before the complete runtime source range, including left context, exceeds the
hard maximum.

Natural overlap uses two consecutive breakpoints:

```text
                 shared semantic context
pause A  <-------------------------------->  pause B
          complete speech unit between pauses

chunk N:      ... ----------------------------> B
chunk N+1:             A ----------------------------> ...
logical N+1:                                  B -----> ...
```

When chunk N ends at pause B, chunk N+1 starts its source at the immediately
preceding natural pause A while its new logical ownership starts at B. Both
inferences therefore hear the complete speech unit between A and B, including
half of each boundary silence. The overlap contains real shared words rather
than only silence, allowing timestamp assembly to compare equivalent speech and
preserve continuity.

This does not require waiting for an additional future pause: A is already in
breakpoint history when B is selected. In the private official-Silero analysis,
pause-to-pause units had a 3.43-second median, 7.65-second 90th percentile, and
10.27-second maximum. The initial 12-second cap covers all observed units with
headroom.

If no previous natural breakpoint exists, or A-to-B exceeds 12 seconds, source
context is capped to the final 12 seconds and marked as capped semantic context.
A forced boundary uses the final two seconds instead. In every case the planner
reduces new logical duration as necessary so the complete source range remains
at most 60 seconds.

Logical ownership never overlaps. Source ranges overlap only through declared
context. Text before pause A can be committed after chunk N; the A-to-B speech
unit remains provisional until chunk N+1 confirms it or job finalization accepts
its only observation.

Finalization classifies and drains remaining material speech. If recording stops
before another ordinary chunk is ready, the final descriptor starts from the
last applicable semantic-context point so the previous tail can still be
reconciled. A VAD-confirmed silent tail is classified without inference. Every
committed sample is classified exactly once as logical ownership or excluded
non-speech.

## Incremental execution

```text
microphone callback
  -> bounded persistence queue
  -> growing WAV + committed cursor

planner worker
  -> CPU VAD
  -> bounded AudioChunk descriptor queue

inference worker
  -> bounded disk read
  -> canonical preparation
  -> resident PyTorch session under torch.inference_mode()
  -> timestamp conversion
  -> assembler
```

Queues store descriptors, not prepared audio arrays. At most one prepared model
input exists. Slow inference creates a disk-backed backlog. The measured warm
throughput is far above real time, but correctness does not rely on that speed.

Recording stop means stop capture, drain persistence, mark the source final,
plan the remaining speech, process terminal descriptors, assemble, persist, and
return to ready. It is not cancellation and does not trigger another unbounded
full-file inference.

## Timestamp conversion and words

Transformers returns token strings with start/end times derived from TDT
predicted durations. Parakeet's tokenizer uses Metaspace; decoded timestamp
pieces beginning with a space start a new word, and following pieces extend that
word. Punctuation pieces remain attached. Aggregating this way reproduced the
native decoded text exactly in the fixed-chunk evaluation.

The adapter returns original token records and timed words. It does not invent
confidence. The final token may have zero duration and trailing encoder duration
may extend beyond the final emitted token, so source coverage uses VAD evidence
and tolerances rather than assuming the last token end equals audio end.

## Conservative overlap assembly

Only previous-tail and current-prefix words whose absolute times intersect the
known pause-to-pause semantic overlap are candidates. A natural overlap carries
the complete shared speech unit A-to-B; forced or capped overlap carries the
explicit trailing source interval. Comparison normalization is used only for
matching; original text is preserved.

The assembler uses monotonic timestamp-compatible sequence alignment. It may
collapse a duplicate only when text and timing provide sufficient evidence that
both observations represent the same source speech. For a confirmed duplicate,
it keeps the observation farther from its physical inference edge. Unmatched or
conflicting overlap text is preserved with a warning.

Text before the unresolved overlap is committed. The current tail remains
provisional until the next chunk or finalization. The initial evaluation proves
that this mechanism is viable but also shows that independent chunks differ at
artificial boundaries. VAD-selected boundaries and local regression fixtures
must validate the final thresholds before implementation is considered closed.

## Completeness

`complete=True` means:

- all material logical speech ranges reached terminal processing;
- no native generation limit or cancellation truncated required work;
- no chunk failed;
- no material VAD-confirmed speech lacks plausible timestamp evidence; and
- assembly did not encounter an unrecoverable protocol error.

It does not claim semantic ASR perfection.

A normal native return is not sufficient by itself. Coverage compares VAD speech
with token/word timing and records conservative gaps. A false-positive VAD gap
may mark a transcript incomplete; the WAV remains the durable source for
retranscription.

## Cancellation and failure recovery

A cancellation token is passed to every job. Transformers stopping criteria can
stop TDT generation, but only after encoder execution completes. Bounded chunks
therefore provide the practical cancellation bound. Cancellation preserves
completed chunks and any valid partial generation, marks the result incomplete,
and resets job state. The resident model may remain ready when a post-encoder
cancel completes normally; CUDA or unknown runtime failures require unloading
and reactivation.

Shutdown during recording follows this order:

1. stop microphone callbacks;
2. drain/finalize or preserve recoverable audio;
3. cancel planning and generation;
4. persist honest partial metadata/text when available;
5. release workers; and
6. unload the model only when application shutdown continues.

VoicePad does not free CUDA tensors underneath active model work.

## Proper nouns and vocabulary intent

The official Parakeet Transformers implementation supports greedy search only
and exposes no native prompt, hotword, or contextual-bias API. VoicePad will not
claim native biasing or pass ignored vocabulary options.

The model-neutral request may still express vocabulary intent:

```python
@dataclass(frozen=True, slots=True)
class TranscriptionIntent:
    language: str | None = None
    vocabulary: tuple[str, ...] = ()
```

Initial supported proper-noun behavior is deterministic alias correction after
timestamp assembly. Configuration associates a canonical spelling only with
explicit alternatives approved by the user. Replacement is word/phrase aware,
preserves timing provenance, records the correction, and never performs a broad
fuzzy rewrite.

Example shape:

```yaml
proper_nouns:
  - canonical: VoicePad
    aliases: [voice pad]
```

Genuine decoder bias remains a research capability. A safe TDT implementation
would tokenize configured phrases, maintain a prefix trie during decoding, and
jointly alter token selection without desynchronizing predicted duration/frame
advancement. The generic Transformers `LogitsProcessor` is not accepted without
proof because TDT token and duration decisions are coupled. Fine-tuning is not
used for dynamic user vocabulary.

Contextual bias may become a separate Parakeet adapter capability only after
local samples show improved names without ordinary-speech regressions. Until
then the effective capability is `aliases`, not `native`.

## Application result behavior

The final result includes text, timed words/tokens when available, duration,
latency, deployment/model/artifact/runtime/device identity, completeness, chunk
outcomes, warnings, failures, and applied corrections.

For complete nonempty text, the application atomically writes Markdown, updates
history from the same result text, and copies that exact text. For incomplete
nonempty text, it persists `complete: false`, displays the reason, and does not
auto-copy. With no text, it stores metadata-only failure information. Complete
no-speech output preserves the WAV and metadata but copies nothing.

After every terminal job outcome the model returns to ready unless the failure
invalidated CUDA/model state. The user can immediately begin another recording
without restarting the TUI.

## Configuration ownership

The application owns strict user configuration. Initial fields include:

```text
deployment_id
device_id
language
chunk minimum/preferred/lookback/maximum
silence duration
natural/forced overlap
proper-noun aliases
```

Unknown and obsolete fields fail with actionable field/path information. The
application does not silently ignore them or overwrite an existing configuration
with defaults. Device/deployment changes are rejected during an active job.

## Device and resource admission

The initial adapter requires a CUDA-capable NVIDIA GPU and validates:

- PyTorch CUDA availability;
- selected stable device identity;
- FP16 support;
- total and currently free memory; and
- actual model parameter placement after load.

Transient CUDA indices are not durable IDs. The deployment resource profile
records measured GPU, peak memory, precision, chunk policy, and confidence level
of the requirement. Initial public guidance claims the 4 GB physical class tested. The RTX 3050
reports 4,096 MiB through `nvidia-smi` but exposes 3,953,393,664 bytes to
PyTorch, so executable admission uses a 3,900,000,000-byte floor rather than an
incorrect binary 4 GiB threshold. Lower-memory support requires physical
hardware evidence; allocator-cap experiments are supporting evidence only.

No silent CPU fallback occurs for the CUDA deployment. A future CPU or
lower-memory deployment must be explicitly represented and tested.

## Windows extension

Windows is a future target, not an initial claim. PyTorch 2.13 provides a
CPython 3.13 Windows AMD64 wheel, Transformers is pure Python, and tokenizer and
safetensors packages provide Windows ABI wheels. That establishes packaging
feasibility, not runtime proof.

Windows support requires a Windows/NVIDIA test for CUDA library resolution,
model load/inference, memory, cache paths, microphone persistence, worker
shutdown, clipboard, and global hotkey behavior. If the same adapter contract
passes, Windows adds a validated platform entry rather than a parallel pipeline.
macOS remains out of scope.

## Adding another model or runtime

A new model within an existing adapter requires:

1. immutable official artifact provenance;
2. a deployment catalogue entry;
3. capability and resource profiles;
4. adapter compatibility proof;
5. timestamp and completeness contract tests;
6. public quality evidence; and
7. local representative regression checks.

A model with a different processor/decoder adds a focused adapter. A new runtime
adds a deployment and adapter but reuses source, VAD, planning, events,
assembly, results, application lifecycle, and persistence.

VoicePad creates a converted model only to solve a measured problem. Conversion
must be reproducible from an official pinned source, preserve licensing and
provenance, publish exact hashes, pass public WER/numerical comparison, and pass
private representative checks. A conversion is always a distinct deployment;
it never silently replaces official bytes.

## Observability and privacy

Logs may include deployment/device IDs, state transitions, queue depth, sample
ranges, timings, memory, hashes, and typed failures. They do not include audio,
transcript text, token/word text, credentials, private fixture names, or private
paths by default.

Model download is public and requires no token. User WAV files, transcripts,
private logs, model binaries, and caches never enter Git or GitHub. Model and
Silero licensing/provenance notices are retained. Package publication remains
frozen.

## Verification strategy

### Deterministic CI

- catalogue and artifact manifest validation;
- bounded download, cancellation, hashing, and atomic promotion;
- adapter capability and unsupported-intent contracts;
- token-to-word timestamp aggregation;
- planner property tests for complete sample classification;
- overlap alignment, repeated words, punctuation, mismatch preservation, and
  final-tail tests;
- coverage and completeness tests including silent native truncation;
- resident state-machine and single-job tests;
- controlled worker/backpressure/cancellation tests;
- strict configuration, Markdown, history, and clipboard tests.

### Local Linux/NVIDIA

- official pinned model offline load;
- FP16 CUDA placement;
- warm-up and first real chunk;
- repeated session stability;
- model unload/reload;
- official Silero wheel/model verification, state continuity, and reset;
- occupied-memory and out-of-memory handling;
- private long-recording complete and chunked paths;
- real recording, stop, immediate next recording, and application shutdown.

### Required repository gate

```sh
uv run ruff check
uv run ruff format --check
uv run ty check
uv run pytest packages --cov=voicepad --cov=voicepad_core --cov-fail-under=70
uv run zensical build --clean
```

Optional hardware checks are reported as passed, failed, or unavailable. They
never masquerade as CI proof.

## Remaining evidence before approval

The architecture is technically coherent and the primary runtime is proven on
the target machine. These items remain explicit human/research gates:

1. Review differing regions between official FP16 and GGUF outputs against the
   source audio; neither output is ground truth.
2. Validate the 12-second semantic-overlap cap and final alignment thresholds
   with VAD-selected natural boundaries, not only fixed synthetic boundaries.
3. Obtain representative local recordings for configured proper nouns and
   compare baseline, explicit alias correction, and any experimental TDT
   contextual bias.
4. Exercise startup, active recording, stop/drain/copy, immediate second
   recording, unload/reload, and shutdown through the actual TUI after
   implementation.
5. Test Windows on real Windows/NVIDIA hardware before adding support.

## Migration sequencing

1. Approve this design and replace the obsolete draft architecture.
2. Re-plan tracker issues around deployment-neutral contracts and the initial
   PyTorch adapter.
3. Add catalogue/artifact preparation together with the first consumer.
4. Add the resident official Parakeet session and prove Linux/NVIDIA lifecycle.
5. Add direct CPU Silero VAD, deterministic planning, timestamp aggregation,
   assembly, and finite-file transcription.
6. Add growing-source execution and events.
7. Cut over file/history/CLI, then TUI/hotkey recording.
8. Remove legacy backends, dependencies, heuristics, and obsolete configuration.
9. Complete documentation, privacy/licensing notices, real-surface checks, and
   migration-wide evidence.

Each step leaves `main` buildable. Legacy code is removed only after its final
consumer cuts over. Existing WAV files and old caches are never automatically
migrated or deleted.
