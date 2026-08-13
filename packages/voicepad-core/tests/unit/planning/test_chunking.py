from voicepad_core.planning import AdaptiveChunkPlanner, OverlapKind
from voicepad_core.vad import NaturalPause

SR = 16_000


def pause(seconds: float) -> NaturalPause:
    midpoint = round(seconds * SR)
    return NaturalPause(midpoint - SR // 4, midpoint + SR // 4)


def test_planner_selects_latest_natural_pause_before_preferred_target() -> None:
    planner = AdaptiveChunkPlanner()
    for seconds in (18, 23, 27):
        planner.add_pause(pause(seconds))

    chunks = planner.poll(30 * SR)

    assert len(chunks) == 1
    assert chunks[0].source_start_sample == 0
    assert chunks[0].logical_start_sample == 0
    assert chunks[0].logical_end_sample == 27 * SR


def test_next_chunk_overlaps_complete_previous_pause_to_pause_unit() -> None:
    planner = AdaptiveChunkPlanner()
    for seconds in (18, 27):
        planner.add_pause(pause(seconds))
    planner.poll(30 * SR)
    for seconds in (40, 48, 52):
        planner.add_pause(pause(seconds))

    chunks = planner.poll(57 * SR)

    assert len(chunks) == 1
    assert chunks[0].source_start_sample == 18 * SR
    assert chunks[0].logical_start_sample == 27 * SR
    assert chunks[0].logical_end_sample == 52 * SR
    assert chunks[0].overlap is OverlapKind.NATURAL


def test_semantic_overlap_is_capped_at_twelve_seconds() -> None:
    planner = AdaptiveChunkPlanner()
    planner.add_pause(pause(5))
    planner.add_pause(pause(27))
    planner.poll(30 * SR)

    terminal = planner.poll(35 * SR, final=True)

    assert terminal[0].source_start_sample == 15 * SR
    assert terminal[0].logical_start_sample == 27 * SR
    assert terminal[0].overlap is OverlapKind.CAPPED_NATURAL


def test_backlog_still_prefers_available_natural_boundary_before_forcing() -> None:
    planner = AdaptiveChunkPlanner()
    planner.add_pause(pause(27))

    chunks = planner.poll(100 * SR)

    assert chunks[0].logical_end_sample == 27 * SR
    assert chunks[0].overlap is OverlapKind.NONE


def test_continuous_speech_forces_at_preferred_duration_with_two_second_overlap() -> None:
    planner = AdaptiveChunkPlanner()

    assert planner.poll(29 * SR) == ()
    chunks = planner.poll(65 * SR)
    tail = planner.poll(70 * SR, final=True)

    assert [chunk.logical_end_sample for chunk in chunks] == [30 * SR, 60 * SR]
    assert chunks[0].source_start_sample == 0
    assert chunks[1].source_start_sample == 28 * SR
    assert chunks[1].overlap is OverlapKind.FORCED
    assert tail[0].source_start_sample == 58 * SR
    assert tail[0].logical_start_sample == 60 * SR
    assert tail[0].overlap is OverlapKind.FORCED
    assert tail[0].terminal is True


def test_planner_dispatches_first_confirmed_pause_after_25_seconds() -> None:
    planner = AdaptiveChunkPlanner()
    planner.add_pause(pause(26))

    chunks = planner.poll(27 * SR)

    assert chunks[0].logical_end_sample == 26 * SR


def test_planner_does_not_wait_past_preferred_duration_for_later_pause() -> None:
    planner = AdaptiveChunkPlanner()
    planner.add_pause(pause(34))

    chunks = planner.poll(30 * SR)

    assert chunks[0].logical_end_sample == 30 * SR
    assert chunks[0].overlap is OverlapKind.NONE
