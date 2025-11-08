# 🧾 disco-data-logger

**High-performance, C++/NumPy-backed data logger**  
for **Disco** discrete-event and Monte Carlo simulation programs.

[![PyPI](https://img.shields.io/pypi/v/disco-data-logger.svg)](https://pypi.org/project/disco-data-logger/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build](https://github.com/michielmj/disco-data-logger/actions/workflows/build.yml/badge.svg)](https://github.com/michielmj/disco-data-logger/actions)
[![Tests](https://github.com/michielmj/disco-data-logger/actions/workflows/test.yml/badge.svg)](https://github.com/michielmj/disco-data-logger/actions)

---

## Overview

`disco-data-logger` provides a **fast, compressed, and lightweight data recording layer**  
for large-scale **Disco** simulations and other computational experiments.  

It is optimized for capturing **sparse numerical state updates** and **accumulators** during simulation runs,  
and writing them efficiently to disk as **Zstandard-compressed segment files**.  
Each simulation entity or measurement can log its data independently through labeled streams.

It combines:
- A **C++/pybind11 core** for high-throughput buffering and compression.
- **Python API** for easy stream registration and control.
- Optional **Parquet export** for analysis and aggregation after runs.

---

## ✨ Features

- **Sparse vector logging** powered by [`sparse_array.Vector`](https://pypi.org/project/disco-sparse-array/).
- **Fixed-point quantization** for compact and deterministic encoding.
- **Buffered, lock-free write path** (ring buffer + writer thread).
- **Zstandard compression** (vendored, no external dependencies).
- **Segment rotation** for large simulation outputs.
- **JSON metadata** for each stream (`organisation`, `model`, `experiment`, …).
- **Periodic vector streams** that emit state snapshots or accumulator sums once per period.
- **Optional Parquet export** for post-run analytics.
- **MIT-licensed** and designed for in-cluster (on-disk/in-memory) use.

---

## 🚀 Installation

```bash
pip install disco-data-logger
```

For Parquet export support:

```bash
pip install "disco-data-logger[parquet]"
```

---

## 📚 Documentation

- [Periodic vector stream logging](docs/periodic_vector_stream.md) – step-by-step guide for
  configuring `periodicity`, choosing between `state` and `accumulator` modes, and verifying
  the emitted sparse data.
- [ENGINEERING_SPEC.md](ENGINEERING_SPEC.md) – project history, motivation, and architectural
  overview.