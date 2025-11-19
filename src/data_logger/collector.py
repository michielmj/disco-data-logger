"""Utilities for decoding loggers into Arrow record batches."""

from __future__ import annotations

import json
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Protocol, Sequence

import numpy as np
import pyarrow as pa
from pyarrow.ipc import RecordBatchFileWriter, RecordBatchStreamWriter
from tools import label_selector
from tools.label_selector import Rule

from ._core import ScalePair, decode_segment_file_with_scales

DONE_MARKER = "_DONE"
DEFAULT_BATCH_SIZE = 2048

class _ArrowWriter(Protocol):
    schema: pa.Schema | None

    def write_batch(self, batch: pa.RecordBatch) -> None:
        ...


class Collector:
    """Collect stream records from one or more logger directories."""

    def __init__(self, paths: Sequence[str | Path]) -> None:
        if not paths:
            raise ValueError("at least one path must be provided")
        self._paths = [Path(p) for p in paths]
        self._batch_size = DEFAULT_BATCH_SIZE

    def collect(
        self,
        writer: _ArrowWriter,
        *,
        rule: Rule | None = None,
        columns: Sequence[str] | None = None,
        backoff: int = 100,
        timeout: int | None = 60_000,
    ) -> bool:
        """Decode selected streams and write them to ``writer``."""

        all_streams, streams_by_path = self._load_streams()
        selected_streams = list(label_selector.select(all_streams, rule)) if rule else list(all_streams)
        if not selected_streams:
            return True

        deduped_columns = tuple(dict.fromkeys(columns or []))
        selected_by_path = self._group_streams_by_path(selected_streams)

        if selected_by_path and not self._wait_for_done(selected_by_path.keys(), backoff, timeout):
            return False

        buffer = _BatchBuffer(deduped_columns, self._batch_size)
        for path, stream_ids in selected_by_path.items():
            meta_map = streams_by_path.get(path, {})
            self._decode_logger(path, stream_ids, meta_map, writer, buffer)
        self._flush_buffer(writer, buffer)
        return True

    def cleanup(
        self,
        *,
        keep_meta: bool = True,
        wait_for_done: bool = True,
        backoff: int = 100,
        timeout: int | None = 60_000,
    ) -> bool:
        """Remove segment files (and optionally metadata) for all logger paths."""

        if wait_for_done and not self._wait_for_done(self._paths, backoff, timeout):
            return False

        for base in self._paths:
            self._remove_segments(base)
            if not keep_meta:
                self._remove_metadata(base)
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_streams(self) -> tuple[list[dict[str, Any]], dict[Path, dict[int, dict[str, Any]]]]:
        streams: list[dict[str, Any]] = []
        per_path: dict[Path, dict[int, dict[str, Any]]] = {}
        for base in self._paths:
            streams_dir = base / "streams"
            if not streams_dir.is_dir():
                continue
            base_map = per_path.setdefault(base, {})
            for label_file in sorted(streams_dir.glob("*.json")):
                with label_file.open("r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                if "stream_id" not in meta:
                    raise ValueError(f"stream metadata missing stream_id: {label_file}")
                sid = int(meta["stream_id"])
                meta["stream_id"] = sid
                meta.setdefault("_logger_path", str(base))
                streams.append(meta)
                base_map[sid] = meta
        return streams, per_path

    def _group_streams_by_path(self, streams: Iterable[dict[str, Any]]) -> dict[Path, set[int]]:
        grouped: dict[Path, set[int]] = defaultdict(set)
        for meta in streams:
            logger_path = meta.get("_logger_path")
            if logger_path is None:
                raise ValueError("stream metadata missing _logger_path")
            grouped[Path(logger_path)].add(int(meta["stream_id"]))
        return grouped

    def _wait_for_done(self, paths: Iterable[Path], backoff: int, timeout: int | None) -> bool:
        pending = {Path(p) for p in paths}
        if not pending:
            return True
        sleep_s = max(backoff, 0) / 1000.0
        deadline = None if timeout is None else time.monotonic() + max(timeout, 0) / 1000.0
        while pending:
            ready = {path for path in pending if (path / DONE_MARKER).exists()}
            pending -= ready
            if not pending:
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(sleep_s)
        return True

    def _remove_segments(self, base: Path) -> None:
        if not base.exists():
            return
        for seg_file in base.glob("*.seg.zst"):
            try:
                seg_file.unlink()
            except FileNotFoundError:
                continue

    def _remove_metadata(self, base: Path) -> None:
        streams_dir = base / "streams"
        if streams_dir.is_dir():
            shutil.rmtree(streams_dir, ignore_errors=True)

    def _decode_logger(
        self,
        base: Path,
        selected_ids: Iterable[int],
        meta_map: Dict[int, Dict[str, Any]],
        writer: _ArrowWriter,
        buffer: "_BatchBuffer",
    ) -> None:
        selected_set = set(selected_ids)
        if not selected_set:
            return
        if not meta_map:
            raise ValueError(f"no metadata available for logger {base}")
        scales = self._scale_pairs(meta_map)
        for seg_file in sorted(base.glob("*.seg.zst")):
            records = decode_segment_file_with_scales(str(seg_file), scales)
            for sid, epoch, idx, vals in records:
                sid_int = int(sid)
                if sid_int not in selected_set:
                    continue
                meta = meta_map.get(sid_int)
                if meta is None:
                    continue
                buffer.append(
                    sid_int,
                    float(epoch),
                    np.asarray(idx, dtype=np.int64),
                    np.asarray(vals, dtype=np.float64),
                    meta,
                )
                if buffer.is_full:
                    self._flush_buffer(writer, buffer)

    def _scale_pairs(self, meta_map: Dict[int, Dict[str, Any]]) -> Dict[int, ScalePair]:
        scales: dict[int, ScalePair] = {}
        for sid, meta in meta_map.items():
            if "epoch_scale" not in meta or "value_scale" not in meta:
                raise ValueError(f"stream metadata missing scales for stream {sid}")
            sp = ScalePair()
            sp.epoch_scale = float(meta["epoch_scale"])
            sp.value_scale = float(meta["value_scale"])
            scales[int(sid)] = sp
        return scales

    def _flush_buffer(self, writer: _ArrowWriter, buffer: "_BatchBuffer") -> None:
        if buffer.size == 0:
            return
        arrays = buffer.to_arrow_arrays()
        schema = getattr(writer, "schema", None)
        batch = pa.record_batch(arrays, schema=schema)
        writer.write_batch(batch)
        buffer.clear()


@dataclass(slots=True)
class _BatchBuffer:
    columns: Sequence[str]
    capacity: int
    stream_ids: list[int] = field(default_factory=list)
    epochs: list[float] = field(default_factory=list)
    indices: list[np.ndarray] = field(default_factory=list)
    values: list[np.ndarray] = field(default_factory=list)
    extras: dict[str, list[Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.columns, tuple):
            self.columns = tuple(self.columns)
        if not self.extras:
            self.extras = {col: [] for col in self.columns}

    @property
    def size(self) -> int:
        return len(self.stream_ids)

    @property
    def is_full(self) -> bool:
        return self.size >= self.capacity > 0

    def append(
        self,
        stream_id: int,
        epoch: float,
        indices: np.ndarray,
        values: np.ndarray,
        meta: dict[str, Any],
    ) -> None:
        self.stream_ids.append(stream_id)
        self.epochs.append(epoch)
        self.indices.append(indices)
        self.values.append(values)
        for col in self.columns:
            self.extras[col].append(meta.get(col))

    def clear(self) -> None:
        self.stream_ids.clear()
        self.epochs.clear()
        self.indices.clear()
        self.values.clear()
        for values in self.extras.values():
            values.clear()

    def to_arrow_arrays(self) -> dict[str, pa.Array]:
        data = {
            "stream_id": pa.array(self.stream_ids, type=pa.uint32()),
            "epoch": pa.array(self.epochs, type=pa.float64()),
            "indices": pa.array(self.indices, type=pa.list_(pa.int64())),
            "values": pa.array(self.values, type=pa.list_(pa.float64())),
        }
        for col, values in self.extras.items():
            data[col] = pa.array(values)
        return data
