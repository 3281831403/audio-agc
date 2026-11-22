"""Block-based automatic gain control (AGC) control surface.

All sample values are normalized to full scale in ``[-1.0, 1.0]``.  Levels are
expressed in dBFS as ``20 * log10(rms)``; digital silence is ``-inf``.  Gains,
noise floors, and target levels are expressed in dB.  Times are milliseconds.
The block size passed to :func:`apply_blocks` and ``config["block"]`` is a
number of samples.  Unless stated otherwise, comparisons of floating point
results should use an absolute tolerance of ``1e-9``.

Configuration mapping used by the stateful helpers::

    {
        "sample_rate": 16000.0,      # Hz, > 0
        "target_db": -18.0,          # desired dBFS level
        "max_gain": 24.0,            # maximum boost in dB, >= 0
        "attack_ms": 5.0,            # gain-reduction time constant, > 0
        "release_ms": 200.0,         # gain-increase time constant, > 0
        "noise_percentile": 20.0,    # percentile used by noise_floor, 0..100
        "floor_window": 8,           # number of block levels kept in history
        "block": 256,                # samples per block, only for agc_batch
    }

State mapping used by :func:`process_block`, :func:`validate`, and
:func:`agc_batch`::

    {
        "prev_gain": 0.0,            # dB gain applied to the previous block
        "floor_db": -100.0,          # tracked noise floor in dBFS
        "levels": [],                # recent block levels in dBFS
    }

The fresh state used by :func:`apply_blocks` and :func:`agc_batch` is
``{"prev_gain": 0.0, "floor_db": -100.0, "levels": []}``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

_MINUS_INF = float("-inf")


def _to_float(value: Any) -> float | None:
    """Best-effort numeric conversion used by :func:`validate`."""

    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    """Compare two floats, treating identical infinities as equal."""

    if a == b:
        return True
    try:
        return abs(a - b) <= tol
    except (TypeError, ValueError):
        return False


def rms_db(samples: Sequence[float]) -> float:
    """Return the block RMS level in dBFS.

    The RMS is ``sqrt(sum(x * x) / len(samples))`` and the result is
    ``20 * log10(rms)``.  An empty block and an all-zero block are digital
    silence and must return ``float("-inf")``, never ``0.0``.  No clipping or
    normalization is applied.
    """

    values = list(samples)
    if not values:
        return _MINUS_INF
    mean_square = sum(float(x) * float(x) for x in values) / len(values)
    if mean_square <= 0.0:
        return _MINUS_INF
    return 20.0 * math.log10(math.sqrt(mean_square))


def noise_floor(levels: Sequence[float], percentile: float) -> float:
    """Estimate the noise floor from block levels in dBFS.

    Use the nearest-rank percentile: sort the levels ascending, clamp
    ``percentile`` to ``[0, 100]``, take
    ``k = max(0, ceil(percentile / 100 * n) - 1)`` (with ``k = n - 1`` for the
    100th percentile), and return the element at index ``k``.  The 0th
    percentile is the smallest element.  ``-inf`` entries are ordinary low
    values.  An empty input returns ``-inf``.  This estimate must be robust to
    occasional loud speech blocks: it is a percentile, not the arithmetic mean.
    """

    ordered = sorted(float(level) for level in levels)
    count = len(ordered)
    if count == 0:
        return _MINUS_INF

    pct = min(100.0, max(0.0, float(percentile)))
    if pct >= 100.0:
        index = count - 1
    else:
        index = max(0, math.ceil(pct / 100.0 * count) - 1)
        index = min(index, count - 1)
    return ordered[index]


def target_gain(
    level: float,
    floor: float,
    target_db: float,
    max_gain: float,
) -> float:
    """Return the desired gain in dB for the current block.

    The reference level is ``max(level, floor)``, so the controller never
    boosts a block further than what the tracked noise floor allows.  The raw
    gain is ``target_db - reference`` and is clamped to ``[0.0, max_gain]``.
    A negative dB gain is therefore clamped to 0 dB instead of being applied
    as a negative multiplier.  Return a finite float within the closed range;
    ``max_gain`` is measured in dB.
    """

    reference = max(float(level), float(floor))
    raw = float(target_db) - reference
    ceiling = max(0.0, float(max_gain))
    return min(ceiling, max(0.0, raw))


def smooth(
    gain: float,
    prev: float,
    attack_ms: float,
    release_ms: float,
    block_ms: float,
) -> float:
    """Move a gain in dB one block toward ``gain`` with asymmetric smoothing.

    A one-pole exponential step is used, with the time constant selected by the
    direction of travel:

    * ``gain < prev`` (reducing boost) uses ``attack_ms`` and coefficient
      ``1 - exp(-block_ms / attack_ms)``;
    * ``gain > prev`` (increasing boost) uses ``release_ms`` and coefficient
      ``1 - exp(-block_ms / release_ms)``;
    * ``gain == prev`` returns ``prev`` unchanged.

    The returned value is ``prev + alpha * (gain - prev)`` in dB.  Time is
    converted from milliseconds, never from a block count: splitting the same
    elapsed time into two equal blocks must produce the same result as one
    block of that total duration (within ``1e-9``).
    """

    gain = float(gain)
    prev = float(prev)
    if gain == prev:
        return prev

    tau = float(attack_ms) if gain < prev else float(release_ms)
    duration = max(0.0, float(block_ms))
    if tau <= 0.0:
        # Degenerate time constant: the filter tracks the target immediately.
        return gain
    alpha = 1.0 - math.exp(-duration / tau)
    return prev + alpha * (gain - prev)



def _fresh_state() -> dict[str, Any]:
    """Return a brand-new state mapping (no shared mutable containers)."""

    return {"prev_gain": 0.0, "floor_db": -100.0, "levels": []}


def _copy_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Shallow-copy a state mapping, duplicating the level history."""

    levels = state.get("levels", [])
    return {
        "prev_gain": float(state.get("prev_gain", 0.0)),
        "floor_db": float(state.get("floor_db", -100.0)),
        "levels": [float(level) for level in levels],
    }


def process_block(
    samples: Sequence[float],
    state: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[float], dict[str, Any]]:
    """Apply AGC to one block and return ``(output, new_state)``.

    The block duration is ``1000 * len(samples) / config["sample_rate"]``
    milliseconds.  For a non-silent block:

    1. measure the block with :func:`rms_db`;
    2. append the level to ``state["levels"]``, keep at most
       ``config["floor_window"]`` recent levels, and re-estimate the floor with
       :func:`noise_floor` and ``config["noise_percentile"]``;
    3. get the desired dB gain from :func:`target_gain`;
    4. smooth it from ``state["prev_gain"]`` with :func:`smooth`.

    The same dB gain is applied to every sample in the block, converted to a
    linear multiplier with ``10 ** (gain_db / 20)``.  The new state must record
    the gain actually applied to this block as ``"prev_gain"`` (the output
    gain, not the block's input/previous gain), together with the new floor and
    level history.  ``len(output) == len(samples)``.

    Digital silence (empty or all-zero samples) holds the state: it returns
    zeros of the same length and a copy of the state with the same gain,
    floor, and history.  The caller's ``state`` must not be mutated.
    """

    values = [float(sample) for sample in samples]

    # Measure first: an empty or all-zero block yields -inf, i.e. digital
    # silence, which holds the state.
    level = rms_db(values)
    if level == _MINUS_INF:
        return [0.0] * len(values), _copy_state(state)

    sample_rate = float(config["sample_rate"])
    block_ms = 1000.0 * len(values) / sample_rate

    floor_window = int(config["floor_window"])
    levels = [float(entry) for entry in state.get("levels", [])]
    levels.append(level)
    if floor_window > 0:
        levels = levels[-floor_window:]

    floor_db = noise_floor(levels, float(config["noise_percentile"]))
    desired = target_gain(
        level,
        floor_db,
        float(config["target_db"]),
        float(config["max_gain"]),
    )
    gain_db = smooth(
        desired,
        float(state.get("prev_gain", 0.0)),
        float(config["attack_ms"]),
        float(config["release_ms"]),
        block_ms,
    )

    multiplier = 10.0 ** (gain_db / 20.0)
    output = [sample * multiplier for sample in values]

    new_state = {
        "prev_gain": gain_db,
        "floor_db": floor_db,
        "levels": levels,
    }
    return output, new_state


def apply_blocks(
    samples: Sequence[float],
    config: Mapping[str, Any],
    block: int,
) -> list[float]:
    """Process ``samples`` in blocks of ``block`` samples and concatenate.

    A fresh state is created for the call.  The final partial block is
    processed, not discarded, so the output length always equals
    ``len(samples)``; an empty input returns an empty list.  ``block`` must be
    a positive integer.  This function must agree sample-for-sample with
    calling :func:`process_block` sequentially with the returned state.
    """

    if isinstance(block, bool) or not isinstance(block, int):
        raise TypeError("block must be a positive integer")
    if block <= 0:
        raise ValueError("block must be a positive integer")

    values = list(samples)
    state = _fresh_state()
    output: list[float] = []

    # range() stops after the final partial block, which is processed too.
    for start in range(0, len(values), block):
        chunk = values[start : start + block]
        chunk_output, state = process_block(chunk, state, config)
        output.extend(chunk_output)

    return output


def validate(
    state: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[str]:
    """Return a list of consistency problems; an empty list means valid.

    The check is read-only and idempotent.  It must verify:

    * ``prev_gain`` is finite (or zero) and lies in ``[0, max_gain]``;
    * ``floor_db`` is not NaN and is at most 0 dBFS;
    * ``attack_ms``, ``release_ms`` and ``floor_window`` are usable, with
      ``0 < attack_ms <= release_ms``;
    * when ``prev_gain > 0`` and ``floor_db`` is finite, the gain is
      self-consistent with the floor:
      ``abs(prev_gain - (target_db - floor_db)) <=
      config.get("consistency_tol_db", 0.5)``;
    * when ``levels`` is non-empty, ``floor_db`` equals the percentile
      estimate of the retained history (absolute tolerance ``1e-9``).

    A dirty state that violates one of these invariants must produce a
    non-empty list rather than being silently accepted.
    """

    problems: list[str] = []

    max_gain = _to_float(config.get("max_gain", 0.0))
    target_db = _to_float(config.get("target_db", 0.0))

    # -- prev_gain -----------------------------------------------------
    prev_gain = _to_float(state.get("prev_gain"))
    if prev_gain is None or not math.isfinite(prev_gain):
        problems.append("prev_gain must be a finite dB value")
    else:
        if prev_gain < 0.0:
            problems.append(f"prev_gain {prev_gain} is below 0 dB")
        if max_gain is None or not math.isfinite(max_gain) or max_gain < 0.0:
            problems.append("max_gain must be a finite, non-negative dB value")
        elif prev_gain > max_gain + 1e-9:
            problems.append(
                f"prev_gain {prev_gain} exceeds max_gain {max_gain}"
            )

    # -- floor_db ------------------------------------------------------
    floor_db = _to_float(state.get("floor_db"))
    if floor_db is None or math.isnan(floor_db):
        problems.append("floor_db must not be NaN")
    elif floor_db > 0.0:
        problems.append(f"floor_db {floor_db} is above 0 dBFS")

    # -- timing / window configuration --------------------------------
    attack_ms = _to_float(config.get("attack_ms"))
    release_ms = _to_float(config.get("release_ms"))
    if attack_ms is None or not math.isfinite(attack_ms) or attack_ms <= 0.0:
        problems.append("attack_ms must be finite and > 0")
    if (
        release_ms is None
        or not math.isfinite(release_ms)
        or release_ms <= 0.0
    ):
        problems.append("release_ms must be finite and > 0")
    if (
        attack_ms is not None
        and release_ms is not None
        and math.isfinite(attack_ms)
        and math.isfinite(release_ms)
        and attack_ms > release_ms
    ):
        problems.append("attack_ms must not exceed release_ms")

    floor_window = config.get("floor_window")
    if (
        isinstance(floor_window, bool)
        or not isinstance(floor_window, int)
        or floor_window < 1
    ):
        problems.append("floor_window must be a positive integer")

    # -- gain / floor self-consistency --------------------------------
    tolerance = _to_float(config.get("consistency_tol_db", 0.5))
    if tolerance is None or tolerance < 0.0:
        tolerance = 0.5
    if (
        prev_gain is not None
        and math.isfinite(prev_gain)
        and prev_gain > 0.0
        and floor_db is not None
        and math.isfinite(floor_db)
        and target_db is not None
        and math.isfinite(target_db)
    ):
        expected = target_db - floor_db
        if abs(prev_gain - expected) > tolerance:
            problems.append(
                "prev_gain is inconsistent with target_db - floor_db "
                f"({prev_gain} vs {expected})"
            )

    # -- tracked floor matches the retained level history -------------
    levels = state.get("levels", [])
    if levels:
        history = [float(level) for level in levels]
        if (
            not isinstance(floor_window, bool)
            and isinstance(floor_window, int)
            and floor_window > 0
        ):
            history = history[-floor_window:]
        percentile = _to_float(config.get("noise_percentile", 20.0))
        if percentile is None:
            percentile = 20.0
        expected_floor = noise_floor(history, percentile)
        if floor_db is None or not _close(floor_db, expected_floor, 1e-9):
            problems.append(
                "floor_db does not match the tracked level history "
                f"({floor_db} vs {expected_floor})"
            )

    return problems


def agc_batch(
    tracks: Mapping[str, Sequence[float]],
    config: Mapping[str, Any],
) -> dict[str, list[float]]:
    """Process a mapping of track names to samples and return outputs.

    Every track starts from its own fresh state; no state, level history, or
    gain may leak from one track to another.  The block size is
    ``config["block"]`` samples and each track is equivalent to calling
    :func:`apply_blocks` on that track alone.  Return a new dict with the same
    keys and insertion order; values are output sample lists of the same length
    as their input track.
    """

    block = config["block"]
    return {
        name: apply_blocks(samples, config, block)
        for name, samples in tracks.items()
    }

