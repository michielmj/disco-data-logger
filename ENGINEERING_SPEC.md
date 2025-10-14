# disco-data-logger — Engineering Spec

## 0. Context / Intent
High-performance logging library for Disco simulations. Records **sparse vectors** (indices: uint32, values: float64) across many **streams** (entities/measures). Writes **zstd-compressed log segments** with **fixed-point quantization** to minimize overhead and size. Optional **Parquet** export.

## 1. Goals (Requirements)
- **Hot path speed**: zero/low-GIL path, buffered writer thread, minimal syscalls.
- **Sparse vectors**: (indices: `np.uint32`, values: `np.float64`) per record.
- **Streams**: multiple independent streams per run, each identified by labels (organisation, model, experiment, run, replication, entity, measure, logging_type, …).
- **API (Python)**:
  - `DataLogger.register_stream(labels, epoch_scale, value_scale) -> stream_id`
  - `DataLogger.record(stream_id, epoch: float, indices: uint32[], values: float64[])`
  - `DataLogger.to_parquet(path)` (optional feature)
- **On-disk format**: rotating `.seg.zst` files (zstd-framed). One directory per run. Sidecar JSON files for stream labels and scales: `streams/<stream_id>.json`.
- **Quantization**: fixed-point: `q = round(value / value_scale)` -> varint(ZigZag(q)). Epochs quantized by `epoch_scale` to integer ticks, then **delta** encoded varint(ZigZag(delta)).
- **Indices encoding**: strictly increasing per record; encode `n`, then `first_idx`, then gaps `(idx[i] - idx[i-1] - 1)` as varints.
- **Decoding**: C++ helper returns `(stream_id, epoch: float, indices: uint32[], values: float64[])`.
- **Storage lifecycle**: data is ephemeral; after reduction -> discard.
- **Packaging**: PyPI project name `disco-data-logger`, import `data_logger`, source in `src/data_logger`.
- **Build**: `scikit-build-core` + CMake; **vendored zstd** via `FetchContent` (static); optional system zstd.
- **Type checking**: mypy-clean; `pyarrow` is **optional** and imported lazily only inside `to_parquet()`.

## 2. Non-goals
- No external cloud storage/DB as primary sink.
- No exact IEEE754 preservation for values (we rely on quantized fixed-point; doubles are reconstructed).
- No mid-run fan-out streaming (Kafka/Redpanda) in v1.

## 3. Architecture

### 3.1 Components
- **LoggerCore (C++ / pybind11)**:
  - `register_stream(epoch_scale, value_scale) -> stream_id`
  - `record(stream_id, epoch, indices, values)` — releases GIL, encodes to a **frame** (varints + zstd) and pushes into a **lock-free ring buffer**.
  - Background **SegmentWriter** thread:
    - pops raw frames
    - compresses each frame into a **single zstd frame**
    - appends to current `.seg.zst`
    - rotates file at `rotate_bytes`
- **DataLogger (Python)**:
  - Manages labels + writes `streams/<sid>.json`
  - Wraps LoggerCore, ensures `indices` sorted (if needed)
  - Decoding/parquet utilities

### 3.2 Frame format (per record)
```
[stream_id varint]
[epoch_delta_ticks ZigZag+varint]
[n varint][first_idx varint][gap_1 varint]...[gap_{n-1} varint]
[q_0 ZigZag+varint] ... [q_{n-1} ZigZag+varint]
```
Where:
- `tick = round(epoch / epoch_scale)`
- `epoch_delta_ticks = tick - last_tick_for_stream` (first record uses delta from 0)
- `q_i = round(value_i / value_scale)`

### 3.3 Segment file
- Path: `<segments_dir>/<NNNNN>.seg.zst`
- Append-only zstd frames. Each pushed record becomes one frame (current impl). (Future: batch multiple records per frame.)
- Rotation at `rotate_bytes`. Close → `fsync` (portable).

### 3.4 Metadata
- One JSON per stream: `streams/<stream_id>.json` with:
  - labels (organisation, model, experiment, run, replication, entity, measure, logging_type, …)
  - `epoch_scale`, `value_scale`, `stream_id`

## 4. Public API (Python)
```python
class DataLogger:
    def __init__(segments_dir: str|Path, ring_bytes=1<<27, rotate_bytes=256<<20, zstd_level=1): ...

    def register_stream(labels: dict, *, epoch_scale: float, value_scale: float) -> int: ...

    def record(stream_id: int, epoch: float, indices: np.ndarray[uint32], values: np.ndarray[float64]) -> None: ...

    def to_parquet(self, out_path: str | Path) -> None:  # optional; lazy import pyarrow
        # writes columns: stream_id:uint32, epoch:float64,
        # indices:list<uint32>, values:list<float64>

    def decode_all_segments(self) -> Iterable[tuple[int, float, np.ndarray, np.ndarray]]: ...
```

## 5. C++/pybind11 surface
```cpp
// module: data_logger._core
class LoggerCore {
  LoggerCore(std::string dir, size_t ring_bytes, size_t rotate_bytes, int zstd_level);
  uint32_t register_stream(double epoch_scale, double value_scale);
  void record(uint32_t stream_id, double epoch,
              uint32_t* indices, size_t n_idx,
              double* values, size_t n_val); // pybind takes NumPy arrays
  void close();
};

struct ScalePair { double epoch_scale; double value_scale; };

std::vector<std::tuple<uint32_t,double,py::array,py::array>>
decode_segment_file_with_scales(std::string path,
                                std::unordered_map<uint32_t,ScalePair> scales);
```

## 6. Key implementation details / decisions
- **Quantization**: fixed-point; ZigZag+varint stores small magnitude integers compactly.
- **Indices**: strictly increasing; encoded as gaps to shrink size.
- **Epochs**: per-stream delta ticks to reduce entropy.
- **Compression**: zstd **level 1** by default; one record → one zstd frame (simple & crash-safe).
- **Portability**: macOS uses `F_FULLFSYNC` fallback to `fsync`; Linux uses `fdatasync` if available; Windows uses `_commit`.
- **Vendoring zstd**: `FetchContent` with `SOURCE_SUBDIR build/cmake`; prefer static lib; option to use system zstd.
- **Optional PyArrow**: import inside `to_parquet()`; no top-level imports; mypy ignores untyped external package by avoiding import at module level.

## 7. Packaging / Build
- `pyproject.toml`: scikit-build-core; pybind11; numpy headers; setuptools-scm versioning.
- Module name: `_core` (`PYBIND11_MODULE(_core, m)`).
- `CMakeLists.txt`: creates `_core` target; includes `src/data_logger/cpp`, links zstd target.
- Wheels: built by `cibuildwheel` (Linux + macOS ARM/Intel). sdist built separately.

## 8. CI/CD
- **test.yml**: Linux (3.11–3.13), Ninja generator, runs `pytest` + `mypy`.
- **build.yml**: wheel matrix (ubuntu, macos-13, macos-14) + sdist.
- **release-publish.yml**: on GitHub release → build wheels/sdist → publish to PyPI via OIDC (Trusted Publisher).

## 9. Test plan (essentials)
- Register + record minimal stream; verify `.seg.zst` exists and label JSON written.
- Decode segments: assert tuple counts and round-trip `epoch` and indices; values within correct quantization tolerance (or reconstruct integers for exactness).
- Backpressure: fill ring buffer with many records; ensure no crashes/deadlocks.
- Parquet (when optional dep installed): file schema as expected, row counts match.

## 10. Performance targets (initial)
- Single-thread record() sustained: ≥ 1–5 million records/minute on a modern laptop (zstd lvl 1, 256 MB rotate).
- Segment writer throughput close to NVMe append rate for 1–2 cores.
- Decoder throughput: ≥ 100 MB/s per core.

## 11. Future work
- Batch multiple records per zstd frame to reduce per-frame overhead.
- `values_q`/Decimal128 Parquet columns for exact decimals.
- Parallel decode and merge across many segments.
- Optional RWX volume test harness.
- Stream schema/version header per segment.

## 12. Known constraints
- Requires indices strictly increasing per record (Python wrapper sorts if needed).
- First record per stream uses tick relative to 0 (not absolute epoch in file header).
- PyArrow optional; to_parquet raises informative error if missing.
