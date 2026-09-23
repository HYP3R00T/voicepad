"""Deterministic contention tests; all audio is synthetic."""

from __future__ import annotations

import queue
from collections.abc import Iterator
from pathlib import Path
from threading import Event, Thread, current_thread
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
from voicepad_core.audio.errors import AudioStreamStateError, AudioWriteBackpressureError
from voicepad_core.audio.live_recording import LiveWavRecording, _ReadRequest
from voicepad_core.audio.types import AudioWindow
from voicepad_core.audio.wav_persistence import WavArtifact


@pytest.fixture
def full_writer(tmp_path: Path) -> Iterator[tuple[LiveWavRecording, Event]]:
    """Hold the first write while one more accepted audio block fills the queue."""
    recording = LiveWavRecording(tmp_path / "recording.wav", 4, 1, max_pending_chunks=1)
    entered = Event()
    release = Event()
    original_write = sf.SoundFile.write

    def write(file: sf.SoundFile, data: np.ndarray) -> None:
        if current_thread().name == "audio-writer" and not entered.is_set():
            entered.set()
            assert release.wait(3), "test did not release the writer"
        original_write(file, data)

    with patch.object(sf.SoundFile, "write", write):
        recording.start()
        recording.append(np.array([0.25], dtype=np.float32))
        assert entered.wait(1)
        recording.append(np.array([0.5], dtype=np.float32))
        try:
            yield recording, release
        finally:
            release.set()
            recording.abort()


@pytest.mark.parametrize("operation", ["read", "finish"])
def test_full_queue_control_operations_do_not_hold_writer_state_lock(
    full_writer: tuple[LiveWavRecording, Event], operation: str
) -> None:
    recording, release = full_writer
    attempting = Event()
    results: list[AudioWindow | WavArtifact] = []
    errors: list[Exception] = []
    original_put = recording._queue.put
    writer = recording._thread
    assert writer is not None
    original_join = writer.join

    def put(item: np.ndarray | _ReadRequest, block: bool = True, timeout: float | None = None) -> None:
        if not isinstance(item, np.ndarray):
            attempting.set()
        if block and recording._queue.full():
            unlocked = recording._state_lock.acquire(blocking=False)
            if unlocked:
                recording._state_lock.release()
            assert unlocked, "blocking queue put holds the lock the writer needs to drain it"
        original_put(item, block=block, timeout=timeout)

    def join(timeout: float | None = None) -> None:
        attempting.set()
        original_join(timeout)

    def operate() -> None:
        try:
            results.append(recording.read_from(0) if operation == "read" else recording.finish())
        except Exception as error:
            errors.append(error)

    with patch.object(recording._queue, "put", put), patch.object(writer, "join", join):
        caller = Thread(target=operate, daemon=True)
        caller.start()
        try:
            assert attempting.wait(1)
        finally:
            release.set()
            caller.join(timeout=2)

    assert not caller.is_alive()
    assert errors == []
    if operation == "read":
        assert isinstance(results[0], AudioWindow)
        np.testing.assert_array_equal(results[0].samples, [0.25, 0.5])
        artifact = recording.finish()
    else:
        assert isinstance(results[0], WavArtifact)
        artifact = results[0]
    samples, _ = sf.read(artifact.path, dtype="float32")
    np.testing.assert_array_equal(samples, [0.25, 0.5])


def test_full_queue_append_fails_promptly_and_accepted_audio_is_preserved(
    full_writer: tuple[LiveWavRecording, Event],
) -> None:
    recording, release = full_writer
    with pytest.raises(AudioWriteBackpressureError):
        recording.append(np.array([0.75], dtype=np.float32))
    release.set()
    assert recording.wait_for_update(1, timeout=1)[0] == 2
    artifact = recording.finish()
    samples, _ = sf.read(artifact.path, dtype="float32")
    np.testing.assert_array_equal(samples, [0.25, 0.5])


def test_read_queue_timeout_does_not_enqueue_or_discard_audio(full_writer: tuple[LiveWavRecording, Event]) -> None:
    recording, release = full_writer
    with (
        patch("voicepad_core.audio.live_recording.time.monotonic", side_effect=[100.0, 111.0]),
        pytest.raises(AudioStreamStateError, match="Timed out while queueing"),
    ):
        recording.read_from(0)
    assert recording._queue.qsize() == 1
    release.set()
    artifact = recording.finish()
    samples, _ = sf.read(artifact.path, dtype="float32")
    np.testing.assert_array_equal(samples, [0.25, 0.5])


def test_finalization_failure_wakes_update_waiters_and_retains_spool(tmp_path: Path) -> None:
    recording = LiveWavRecording(tmp_path / "recording.wav", 4, 1)
    recording.start()
    recording.append(np.array([0.25], dtype=np.float32))
    recording.read_from(0)
    waiting = Event()
    done = Event()
    errors: list[Exception] = []
    original_wait = recording._committed_condition.wait

    def wait(timeout: float | None = None) -> bool:
        waiting.set()
        return original_wait(timeout)

    def read_update() -> None:
        try:
            recording.wait_for_update(1, timeout=None)
        except AudioStreamStateError as error:
            errors.append(error)
        finally:
            done.set()

    with patch.object(recording._committed_condition, "wait", wait):
        reader = Thread(target=read_update, daemon=True)
        reader.start()
        assert waiting.wait(1)
        try:
            with (
                patch("voicepad_core.audio.live_recording._finalize_live_wav", side_effect=OSError("disk failed")),
                pytest.raises(OSError, match="disk failed"),
            ):
                recording.finish()
            assert done.wait(1), "finalization failure left update waiter asleep"
        finally:
            with recording._committed_condition:
                recording._committed_condition.notify_all()
            reader.join(timeout=1)

    assert len(errors) == 1
    assert recording._spool_path is not None and recording._spool_path.exists()
    recording.abort()
    assert recording._spool_path.exists(), "cleanup deleted recoverable audio after failure"


def test_writer_failure_notifies_queued_read_and_retains_spool_on_abort(tmp_path: Path) -> None:
    recording = LiveWavRecording(tmp_path / "recording.wav", 4, 1)
    recording.start()
    recording.append(np.array([0.25], dtype=np.float32))
    recording.read_from(0)  # Ensure some recoverable audio is present.
    writing = Event()
    fail = Event()
    requested = Event()
    errors: list[Exception] = []
    original_put = recording._queue.put

    def failing_write(_file: sf.SoundFile, _data: np.ndarray) -> None:
        writing.set()
        assert fail.wait(2)
        raise OSError("disk full")

    def put(item: np.ndarray | _ReadRequest, block: bool = True, timeout: float | None = None) -> None:
        original_put(item, block=block, timeout=timeout)
        if not isinstance(item, np.ndarray):
            requested.set()

    def read() -> None:
        try:
            recording.read_from(0)
        except AudioStreamStateError as error:
            errors.append(error)

    with patch.object(sf.SoundFile, "write", failing_write), patch.object(recording._queue, "put", put):
        recording.append(np.array([0.5], dtype=np.float32))
        assert writing.wait(1)
        reader = Thread(target=read, daemon=True)
        reader.start()
        try:
            assert requested.wait(1)
        finally:
            fail.set()
            reader.join(timeout=1)
    assert not reader.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0].__cause__, OSError)
    with pytest.raises(AudioStreamStateError, match="writer failed"):
        recording.finish()
    recording.abort()
    assert recording._spool_path is not None and recording._spool_path.exists()
    samples, _ = sf.read(recording._spool_path, dtype="float32")
    np.testing.assert_array_equal(samples, [0.25])


def test_read_waiting_for_queue_space_switches_to_final_artifact_when_stopped(
    full_writer: tuple[LiveWavRecording, Event],
) -> None:
    recording, release = full_writer
    waiting = Event()
    stopping = Event()
    errors: list[Exception] = []
    windows: list[AudioWindow] = []
    artifacts: list[WavArtifact] = []
    original_wait = recording._committed_condition.wait
    writer = recording._thread
    assert writer is not None
    original_join = writer.join

    def wait(timeout: float | None = None) -> bool:
        waiting.set()
        return original_wait(timeout)

    def join(timeout: float | None = None) -> None:
        stopping.set()
        original_join(timeout)

    def read() -> None:
        try:
            windows.append(recording.read_from(0))
        except Exception as error:
            errors.append(error)

    def finish() -> None:
        try:
            artifacts.append(recording.finish())
        except Exception as error:
            errors.append(error)

    with patch.object(recording._committed_condition, "wait", wait), patch.object(writer, "join", join):
        reader = Thread(target=read, daemon=True)
        finisher = Thread(target=finish, daemon=True)
        reader.start()
        try:
            assert waiting.wait(1)
            finisher.start()
            assert stopping.wait(1)
        finally:
            release.set()
            reader.join(timeout=2)
            if finisher.ident is not None:
                finisher.join(timeout=2)

    assert not reader.is_alive() and not finisher.is_alive()
    assert errors == []
    assert artifacts[0].frame_count == 2
    np.testing.assert_array_equal(windows[0].samples, [0.25, 0.5])
    assert recording._queue.unfinished_tasks == 0


def test_abort_drains_full_queue_and_retains_accepted_audio(full_writer: tuple[LiveWavRecording, Event]) -> None:
    recording, release = full_writer
    stopping = Event()
    writer = recording._thread
    assert writer is not None
    original_join = writer.join

    def join(timeout: float | None = None) -> None:
        stopping.set()
        original_join(timeout)

    with patch.object(writer, "join", join):
        aborter = Thread(target=recording.abort, daemon=True)
        aborter.start()
        try:
            assert stopping.wait(1)
        finally:
            release.set()
            aborter.join(timeout=2)

    assert not aborter.is_alive()
    assert not writer.is_alive()
    assert recording._spool_path is not None and recording._spool_path.exists()
    samples, _ = sf.read(recording._spool_path, dtype="float32")
    np.testing.assert_array_equal(samples, [0.25, 0.5])
    with pytest.raises(AudioStreamStateError, match="aborted"):
        recording.wait_for_update(2, timeout=None)


def test_finish_timeout_is_terminal_and_writer_exits_after_stalled_io_recovers(
    full_writer: tuple[LiveWavRecording, Event],
) -> None:
    recording, release = full_writer
    writer = recording._thread
    assert writer is not None
    with patch.object(writer, "join"), pytest.raises(AudioStreamStateError, match="Timed out"):
        recording.finish()

    assert recording._spool_path is not None and recording._spool_path.exists()
    with pytest.raises(AudioStreamStateError, match="aborted"):
        recording.wait_for_update(0, timeout=None)
    with pytest.raises(AudioStreamStateError):
        recording.append(np.zeros(1, dtype=np.float32))
    release.set()
    writer.join(timeout=2)
    assert not writer.is_alive()
    recording.abort()
    samples, _ = sf.read(recording._spool_path, dtype="float32")
    np.testing.assert_array_equal(samples, [0.25, 0.5])


def test_stop_does_not_lose_append_racing_with_empty_queue_timeout(tmp_path: Path) -> None:
    recording = LiveWavRecording(tmp_path / "recording.wav", 4, 1)
    empty_observed = Event()
    stopping = Event()
    release = Event()
    original_get = recording._queue.get
    results: list[WavArtifact] = []
    errors: list[Exception] = []

    def get(block: bool = True, timeout: float | None = None) -> np.ndarray | _ReadRequest:
        if not empty_observed.is_set():
            empty_observed.set()
            assert release.wait(2)
            raise queue.Empty  # Empty was observed just before the last append.
        return original_get(block=block, timeout=timeout)

    def finish() -> None:
        try:
            results.append(recording.finish())
        except Exception as error:
            errors.append(error)

    with patch.object(recording._queue, "get", get):
        recording.start()
        assert empty_observed.wait(1)
        recording.append(np.array([0.25], dtype=np.float32))
        writer = recording._thread
        assert writer is not None
        original_join = writer.join

        def join(timeout: float | None = None) -> None:
            stopping.set()
            original_join(timeout)

        with patch.object(writer, "join", join):
            finisher = Thread(target=finish, daemon=True)
            finisher.start()
            try:
                assert stopping.wait(1)
            finally:
                release.set()
                finisher.join(timeout=2)

    assert not finisher.is_alive()
    assert errors == []
    assert results[0].frame_count == 1
    samples, _ = sf.read(results[0].path, dtype="float32")
    np.testing.assert_array_equal(samples, [0.25])
    assert recording._queue.unfinished_tasks == 0


def test_start_timeout_requests_shutdown_of_late_starting_writer(tmp_path: Path) -> None:
    recording = LiveWavRecording(tmp_path / "recording.wav", 4, 1)
    opening = Event()
    release = Event()
    original_file = sf.SoundFile

    def open_file(file: str, *, mode: str, samplerate: int, channels: int, subtype: str, format: str) -> sf.SoundFile:
        opening.set()
        assert release.wait(2)
        return original_file(file, mode=mode, samplerate=samplerate, channels=channels, subtype=subtype, format=format)

    def timed_out(timeout: float | None = None) -> bool:
        del timeout
        assert opening.wait(1)
        return False

    with (
        patch("voicepad_core.audio.live_recording.sf.SoundFile", side_effect=open_file),
        patch.object(recording._ready, "wait", side_effect=timed_out),
        patch.object(Thread, "join"),
    ):
        try:
            with pytest.raises(AudioStreamStateError, match="Timed out while starting"):
                recording.start()
        finally:
            release.set()

    writer = recording._thread
    assert writer is not None
    writer.join(timeout=2)
    assert not writer.is_alive()
    assert recording._finished.is_set()
    recording.abort()
    assert recording._spool_path is not None and not recording._spool_path.exists()
