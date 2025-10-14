from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray

# Helpful aliases
UInt32Array = NDArray[np.uint32]
Float64Array = NDArray[np.float64]

__all__ = [
    "LoggerCore",
    "ScalePair",
    "decode_segment_file_with_scales",
    "UInt32Array",
    "Float64Array",
]


class ScalePair:
    """Holds per-stream scales used by the decoder."""
    epoch_scale: float
    value_scale: float

    def __init__(self) -> None: ...


class LoggerCore:
    """
    Low-level logger backend.
    Writes zstd-framed segments and accepts sparse vectors per stream.
    """

    def __init__(
            self,
            dir: str,
            ring_bytes: int = ...,
            rotate_bytes: int = ...,
            zstd_level: int = ...,
    ) -> None: ...

    def register_stream(self, epoch_scale: float, value_scale: float) -> int: ...

    def record(self, stream_id: int, epoch: float, indices: UInt32Array, values: Float64Array) -> None: ...

    def close(self) -> None: ...


def decode_segment_file_with_scales(
        path: str,
        scales: Dict[int, ScalePair],
) -> List[Tuple[int, float, UInt32Array, Float64Array]]: ...
