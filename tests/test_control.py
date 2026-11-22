"""Contract tests for the block-based AGC control state machine.

Every test exercises one of the ten hidden state/invariant defects plus the
five end-to-end invariants required by the task.  The module imports cleanly
while the implementation is still a set of stubs; failures happen only when a
test calls a stub.
"""

from __future__ import annotations

import math
from copy import deepcopy

import pytest

from agc.control import (
    agc_batch,
    apply_blocks,
    noise_floor,
    process_block,
    rms_db,
    smooth,
    target_gain,
    validate,
)


def make_config(**overrides: object) -> dict[str, object]:
    """Build a small, fast, deterministic AGC configuration."""

    config: dict[str, object] = {
        "sample_rate": 1000.0,
        "target_db": -18.0,
        "max_gain": 40.0,
        "attack_ms": 10.0,
        "release_ms": 200.0,
        "noise_percentile": 20.0,
        "floor_window": 8,
        "block": 8,
    }
    config.update(overrides)
    return config


def make_state(
    prev_gain: float = 0.0,
    floor_db: float = -100.0,
    levels: list[float] | None = None,
) -> dict[str, object]:
    """Build the documented three-field AGC state mapping."""

    return {
        "prev_gain": float(prev_gain),
        "floor_db": float(floor_db),
        "levels": [] if levels is None else list(levels),
    }


# ---------------------------------------------------------------------------
# The ten targeted defects
# ---------------------------------------------------------------------------


def test_smooth_release_step_uses_release_tau() -> None:
    """Defect 1: release must not reuse the fast attack time constant."""

    attack_ms = 10.0
    release_ms = 200.0
    block_ms = 20.0

    falling = smooth(0.0, 12.0, attack_ms, release_ms, block_ms)
    rising = smooth(12.0, 0.0, attack_ms, release_ms, block_ms)

    assert falling == pytest.approx(12.0 * math.exp(-2.0), abs=1e-9)
    assert rising == pytest.approx(
        12.0 * (1.0 - math.exp(-0.1)), abs=1e-9
    )

    attack_delta = 12.0 - falling
    release_delta = rising
    assert attack_delta > 5.0 * release_delta


def test_process_block_records_output_gain_and_cross_block_continuity() -> None:
    """Defect 2: prev_gain is the applied output gain, not the input gain."""

    config = make_config(
        sample_rate=1000.0,
        target_db=-18.0,
        max_gain=40.0,
        attack_ms=10.0,
        release_ms=200.0,
        floor_window=8,
        noise_percentile=20.0,
    )
    block = [0.001] * 10
    initial = make_state(prev_gain=0.0, floor_db=-80.0)
    initial_snapshot = deepcopy(initial)

    out1, state1 = process_block(block, initial, config)
    applied_db = 20.0 * math.log10(out1[0] / block[0])

    assert state1["prev_gain"] == pytest.approx(applied_db, abs=1e-9)

    out2, state2 = process_block(block, state1, config)
    assert state2["prev_gain"] > state1["prev_gain"] + 1e-9
    assert out2[0] > out1[0] + 1e-9

    out_again, state_again = process_block(block, initial, config)
    assert initial == initial_snapshot
    assert out_again == pytest.approx(out1, abs=1e-12)
    assert state_again["prev_gain"] == pytest.approx(
        state1["prev_gain"], abs=1e-12
    )


def test_noise_floor_percentile_preserves_quiet_floor_and_gain_monotonicity() -> None:
    """Defect 3: mean-tracking raises the floor and lowers all later gains."""

    levels = [-60.0] * 7 + [-20.0] * 3
    tracked_floor = noise_floor(levels, 20.0)

    assert tracked_floor == pytest.approx(-60.0, abs=1e-9)
    assert tracked_floor != pytest.approx(sum(levels) / len(levels), abs=1e-9)

    quiet_gain = target_gain(-70.0, tracked_floor, -18.0, 60.0)
    inflated_gain = target_gain(-70.0, -48.0, -18.0, 60.0)

    assert quiet_gain == pytest.approx(42.0, abs=1e-9)
    assert inflated_gain == pytest.approx(30.0, abs=1e-9)
    assert quiet_gain > inflated_gain + 1e-9


def test_target_gain_clamps_at_zero_db() -> None:
    """Defect 4: a loud block must clamp to 0 dB, never a negative gain."""

    loud = target_gain(0.0, -80.0, -18.0, 24.0)
    above_target = target_gain(-6.0, -80.0, -18.0, 24.0)
    bounded = target_gain(-30.0, -40.0, -18.0, 24.0)
    capped = target_gain(-80.0, -80.0, -18.0, 24.0)

    assert loud == 0.0
    assert above_target == 0.0
    assert loud >= 0.0 and above_target >= 0.0
    assert bounded == pytest.approx(12.0, abs=1e-9)
    assert capped == pytest.approx(24.0, abs=1e-9)


def test_rms_db_silence_is_minus_infinity() -> None:
    """Defect 5: an all-zero block is silence, not 0 dBFS full scale."""

    assert rms_db([0.0] * 32) == float("-inf")
    assert rms_db([]) == float("-inf")
    assert rms_db([1.0]) == pytest.approx(0.0, abs=1e-9)
    assert rms_db([0.5, -0.5] * 8) == pytest.approx(
        20.0 * math.log10(0.5), abs=1e-9
    )


def test_apply_blocks_keeps_incomplete_tail() -> None:
    """Defect 6: floor division would drop the two samples after the last full block."""

    samples = [0.25] * 10
    output = apply_blocks(samples, make_config(), 4)

    assert len(output) == len(samples)
    assert output[-1] == pytest.approx(0.25, abs=1e-9)


def test_smooth_time_constant_is_ms_not_block_count() -> None:
    """Defect 7: two 10 ms steps equal one 20 ms step in continuous time."""

    attack_ms = 5.0
    release_ms = 100.0
    target = 12.0

    first = smooth(target, 0.0, attack_ms, release_ms, 10.0)
    two_steps = smooth(target, first, attack_ms, release_ms, 10.0)
    one_step = smooth(target, 0.0, attack_ms, release_ms, 20.0)

    assert two_steps == pytest.approx(one_step, abs=1e-9)
    assert smooth(target, 0.0, attack_ms, release_ms, 100.0) == pytest.approx(
        target * (1.0 - math.exp(-1.0)), abs=1e-9
    )


def test_validate_reports_gain_floor_mismatch() -> None:
    """Defect 8: ranges alone cannot detect a gain/floor inconsistency."""

    config = make_config(
        target_db=-18.0,
        max_gain=60.0,
        attack_ms=10.0,
        release_ms=200.0,
        floor_window=8,
        noise_percentile=20.0,
    )
    valid_state = make_state(
        prev_gain=22.0,
        floor_db=-40.0,
        levels=[-40.0] * 8,
    )
    dirty_state = make_state(
        prev_gain=22.0,
        floor_db=-20.0,
        levels=[-20.0] * 8,
    )

    assert validate(valid_state, config) == []

    snapshot = deepcopy(dirty_state)
    problems = validate(dirty_state, config)
    assert problems
    assert validate(dirty_state, config) == problems
    assert dirty_state == snapshot


def test_process_block_gain_is_flat_within_block() -> None:
    """Defect 9: per-sample state updates make the in-block ratio drift."""

    config = make_config(
        sample_rate=1000.0,
        target_db=-18.0,
        max_gain=40.0,
        attack_ms=10.0,
        release_ms=100.0,
    )
    samples = [
        0.01,
        -0.02,
        0.03,
        -0.04,
        0.05,
        -0.06,
        0.07,
        -0.08,
        0.09,
        -0.10,
    ]
    initial = make_state(prev_gain=12.0, floor_db=-80.0)

    output, new_state = process_block(samples, initial, config)
    ratios = [out / sample for out, sample in zip(output, samples)]

    assert max(ratios) - min(ratios) <= 1e-12
    expected_linear = 10.0 ** (new_state["prev_gain"] / 20.0)
    assert ratios[0] == pytest.approx(expected_linear, abs=1e-12)
    assert ratios[-1] == pytest.approx(expected_linear, abs=1e-12)


def test_agc_batch_resets_state_for_each_track() -> None:
    """Defect 10: a shared shallow state leaks the first track into the second."""

    config = make_config(
        block=8,
        target_db=-18.0,
        max_gain=24.0,
        attack_ms=5.0,
        release_ms=50.0,
        floor_window=8,
    )
    quiet = [0.001] * 32
    loud = [0.5] * 32

    batch = agc_batch({"quiet": quiet, "loud": loud}, config)
    solo_loud = apply_blocks(loud, config, 8)

    assert batch["loud"] == pytest.approx(solo_loud, abs=1e-12)


# ---------------------------------------------------------------------------
# The five required end-to-end invariants
# ---------------------------------------------------------------------------


def test_constant_level_converges_to_constant_gain() -> None:
    """At steady state, a constant level produces a constant block gain."""

    config = make_config(
        sample_rate=1000.0,
        target_db=-18.0,
        max_gain=60.0,
        attack_ms=5.0,
        release_ms=40.0,
        floor_window=8,
        noise_percentile=20.0,
    )
    samples = [0.01] * 32
    state = make_state(floor_db=-100.0)
    gains: list[float] = []
    output: list[float] = []

    for _ in range(300):
        output, state = process_block(samples, state, config)
        gains.append(float(state["prev_gain"]))

    assert gains[-1] == pytest.approx(gains[-2], abs=1e-9)
    assert output[0] / samples[0] == pytest.approx(
        10.0 ** (gains[-1] / 20.0), abs=1e-12
    )


def test_attack_progress_exceeds_release_progress() -> None:
    """Attack reaches 63.2% of its step far sooner than release does."""

    target = 12.0
    criterion = 1.0 - math.exp(-1.0)

    gain = target
    attack_blocks = 0
    while 1.0 - gain / target < criterion:
        gain = smooth(0.0, gain, 5.0, 200.0, 1.0)
        attack_blocks += 1
        assert attack_blocks < 1000

    gain = 0.0
    release_blocks = 0
    while gain / target < criterion:
        gain = smooth(target, gain, 5.0, 200.0, 1.0)
        release_blocks += 1
        assert release_blocks < 5000

    assert attack_blocks <= 8
    assert release_blocks >= 150
    assert release_blocks > 10 * attack_blocks


def test_output_length_equals_input_length() -> None:
    """Output length is preserved for empty, partial, and exact blocks."""

    config = make_config()
    for length, block in ((0, 5), (1, 5), (7, 5), (16, 8), (37, 8)):
        samples = [0.001] * length
        output = apply_blocks(samples, config, block)
        assert len(output) == len(samples)


def test_steady_state_gain_independent_of_block_size() -> None:
    """The same signal settles to the same gain at block sizes 8 and 32."""

    config = make_config(
        sample_rate=1000.0,
        target_db=-18.0,
        max_gain=60.0,
        attack_ms=5.0,
        release_ms=20.0,
        floor_window=8,
        noise_percentile=20.0,
    )
    samples = [0.01] * 4800

    output_8 = apply_blocks(samples, config, 8)
    output_32 = apply_blocks(samples, config, 32)

    gain_8 = 20.0 * math.log10(abs(output_8[-1] / samples[-1]))
    gain_32 = 20.0 * math.log10(abs(output_32[-1] / samples[-1]))

    assert gain_8 == pytest.approx(gain_32, abs=0.01)


def test_batch_tracks_with_opposite_levels_differ() -> None:
    """Two tracks with opposite levels keep independent outputs."""

    config = make_config(
        block=16,
        target_db=-18.0,
        max_gain=24.0,
        attack_ms=5.0,
        release_ms=50.0,
        floor_window=8,
    )
    quiet = [0.001] * 64
    loud = [0.5] * 64

    batch = agc_batch({"quiet": quiet, "loud": loud}, config)

    assert set(batch) == {"quiet", "loud"}
    assert len(batch["quiet"]) == len(quiet)
    assert len(batch["loud"]) == len(loud)
    assert any(
        abs(quiet_sample - loud_sample) > 1e-6
        for quiet_sample, loud_sample in zip(batch["quiet"], batch["loud"])
    )
