"""audio-agc: block-based automatic gain control."""

from .control import (
    agc_batch,
    apply_blocks,
    noise_floor,
    process_block,
    rms_db,
    smooth,
    target_gain,
    validate,
)

__all__ = [
    "agc_batch",
    "apply_blocks",
    "noise_floor",
    "process_block",
    "rms_db",
    "smooth",
    "target_gain",
    "validate",
]
