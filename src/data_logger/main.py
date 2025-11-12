"""
data_logger.experiment
======================

High-performance, stream-based sparse data logger for Disco simulations.

This module wraps the compiled `data_logger._core` extension
and provides convenient Python-level management:
  - Register labeled streams
  - Record sparse vectors (indices + values)
  - Decode logged data into Arrow/Parquet for postprocessing

Author: Michiel Jansen
License: MIT
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Tuple

import numpy as np

# Import the pybind11 core module
from ._core import LoggerCore, ScalePair, decode_segment_file_with_scales
from .periodic import PeriodicVectorStream


class DataLogger:
    """
    Manages stream registration, segment recording, and Parquet export
    for sparse numerical logs from discrete-event or Monte Carlo simulations.
    """

    def __init__(
        self,
        segments_dir: str | Path,
        ring_bytes: int = 1 << 27,     # 128 MB ring buffer
        rotate_bytes: int = 256 << 20, # 256 MB per segment file
        zstd_level: int = 1,           # lightweight compression
    ):
        self._segdir = Path(segments_dir)
        self._segdir.mkdir(parents=True, exist_ok=True)
        (self._segdir / "streams").mkdir(exist_ok=True)

        # C++ backend
        self._core = LoggerCore(
            str(self._segdir),
            ring_bytes,
            rotate_bytes,
            zstd_level,
        )

        # Internal registries
        self._scales: dict[int, tuple[float, float]] = {}
        self._labels: dict[int, dict[str, Any]] = {}

    # ----------------------------------------------------------------------
    # Stream management
    # ----------------------------------------------------------------------

    def register_stream(
        self,
        labels: Dict[str, Any],
        *,
        epoch_scale: float,
        value_scale: float,
    ) -> int:
        """
        Register a new labeled data stream.

        Parameters
        ----------
        labels : dict
            Identifiers describing the stream (e.g. organisation, model,
            experiment, run, replication, entity, measure, logging_type, ...).
        epoch_scale : float
            Time quantization step (seconds per tick).
        value_scale : float
            Fixed-point quantization for values (units per LSB).

        Returns
        -------
        stream_id : int
            Unique stream identifier for use in `record()`.
        """
        sid = self._core.register_stream(epoch_scale, value_scale)
        self._scales[sid] = (epoch_scale, value_scale)
        self._labels[sid] = dict(labels)

        meta = {
            **labels,
            "stream_id": sid,
            "epoch_scale": epoch_scale,
            "value_scale": value_scale,
        }
        with open(self._segdir / "streams" / f"{sid:08d}.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, separators=(",", ":"))

        return sid

    # ----------------------------------------------------------------------
    # Recording
    # ----------------------------------------------------------------------

    def record(
        self,
        stream_id: int,
        epoch: float,
        indices: np.ndarray,
        values: np.ndarray,
    ) -> None:
        """
        Record one sparse vector (indices + values) for a stream.

        Parameters
        ----------
        stream_id : int
            Stream identifier obtained from `register_stream`.
        epoch : float
            Simulation time (double).
        indices : np.ndarray[np.uint32]
            Array of indices of changed elements.
        values : np.ndarray[np.float64]
            Corresponding delta values.
        """
        # Copy the buffers so callers retain ownership of their inputs. The
        # C++ core releases the GIL (see ``py::gil_scoped_release`` in
        # ``_core.cpp``) while it encodes the payload and hands it to the
        # background segment writer. During that window, other Python threads
        # can run and mutate the original arrays. Taking copies gives the core a
        # stable snapshot even when callers reuse or mutate their buffers right
        # after invoking ``record``.
        indices = np.array(indices, dtype=np.uint32, copy=True, order="C")
        values = np.array(values, dtype=np.float64, copy=True, order="C")

        if indices.shape != values.shape:
            raise ValueError("indices and values must have the same length")

        # Optional safety check: ensure indices strictly increasing
        if indices.size > 1 and not np.all(indices[1:] > indices[:-1]):
            order = np.argsort(indices, kind="stable")
            indices, values = indices[order], values[order]

        self._core.record(int(stream_id), float(epoch), indices, values)

    # ----------------------------------------------------------------------
    # Closing and flushing
    # ----------------------------------------------------------------------

    def close(self) -> None:
        """Flush remaining buffers and close the C++ writer thread."""
        self._core.close()

    # ----------------------------------------------------------------------
    # Decoding & Parquet export
    # ----------------------------------------------------------------------

    def _scales_map_for_cpp(self) -> dict[int, ScalePair]:
        m = {}
        for sid, (e, v) in self._scales.items():
            sp = ScalePair()
            sp.epoch_scale = e
            sp.value_scale = v
            m[sid] = sp
        return m

    def decode_all_segments(self) -> Iterable[Tuple[int, float, np.ndarray, np.ndarray]]:
        """
        Stream over all decoded records across all segments.
        Returns tuples (stream_id, epoch, indices, values).
        """
        for seg in sorted(self._segdir.glob("*.seg.zst")):
            recs = decode_segment_file_with_scales(str(seg), self._scales_map_for_cpp())
            for sid, epoch, idx, vals in recs:
                yield int(sid), float(epoch), np.asarray(idx), np.asarray(vals)

    def register_periodic_stream(
        self,
        labels: Dict[str, Any],
        *,
        epoch_scale: float,
        value_scale: float,
        periodicity: float = 1.0,
        kind: Literal["state", "accumulator"] = "state",
    ) -> "PeriodicVectorStream":
        """Register a stream and wrap it in a :class:`PeriodicVectorStream` helper."""

        labels_with_meta = {**labels, "periodicity": periodicity, "kind": kind}
        stream_id = self.register_stream(
            labels_with_meta,
            epoch_scale=epoch_scale,
            value_scale=value_scale,
        )
        return PeriodicVectorStream(self, stream_id, periodicity=periodicity, kind=kind)

    def to_parquet(self, out_path: str | Path) -> None:
        """
        Decode all segments into a Parquet file with schema:
            stream_id : uint32
            epoch     : double
            indices   : list<uint32>
            values    : list<double>
        """

        try:
            import pyarrow as _pa  # type: ignore[import-untyped]
            import pyarrow.parquet as _pq  # type: ignore[import-untyped]
        except Exception as exc:
            raise RuntimeError(
                "pyarrow is required for Parquet export. "
                "Install with `pip install disco-data-logger[parquet]`."
            ) from exc

        sids, epochs, idx_list, val_list = [], [], [], []
        for sid, epoch, idx, vals in self.decode_all_segments():
            sids.append(np.uint32(sid))
            epochs.append(np.float64(epoch))
            idx_list.append(idx.astype(np.uint32, copy=False))
            val_list.append(vals.astype(np.float64, copy=False))

        # build Arrow arrays/table using _pa / _pq (not pa/pq)
        table = _pa.table({
            "stream_id": _pa.array(sids, type=_pa.uint32()),
            "epoch": _pa.array(epochs, type=_pa.float64()),
            "indices": _pa.array(idx_list, type=_pa.list_(_pa.uint32())),
            "values": _pa.array(val_list, type=_pa.list_(_pa.float64())),
        })

        _pq.write_table(table, out_path, compression="zstd", row_group_size=1_000_000)
        print(f"Wrote Parquet file: {out_path}")

    # ----------------------------------------------------------------------
    # Context manager convenience
    # ----------------------------------------------------------------------

    def __enter__(self) -> DataLogger:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()



