# Periodic Vector Stream Logging

Periodic vector streams make it easy to capture **sparse simulation measurements** exactly
once per logical period even when your system emits many intermediate updates. Each logged
period stores a [`graphblas.Vector`](https://pypi.org/project/python-graphblas/) so the
segment files stay compact while preserving sparsity information.

This guide walks through:

1. The difference between *state* and *accumulator* periodic streams.
2. How the logger slices epochs into periods.
3. How to register, record, and close periodic vector streams safely.
4. Strategies for validating the resulting data.

---

## 1. Concepts

### Periodicity

`periodicity` describes the width of each logical period. With the default value `1.0`,
period boundaries occur at epochs `[0.0, 1.0, 2.0, …]`. A vector recorded at epoch `3.6`
belongs to the period `[3.0, 4.0)`.

### Kinds

Periodic streams operate in one of two modes:

| Kind | Behaviour | Typical use cases |
| --- | --- | --- |
| `"state"` | Emits the **last** sparse vector recorded at or before the end of each period. | Final state snapshots, control variables, counters. |
| `"accumulator"` | Emits the **sum** of all vectors recorded within each period. | Event tallies, per-step energy usage, cumulative rewards. |

The logger stores both `periodicity` and `kind` in the stream metadata, allowing downstream
systems to understand how each record was produced.

---

## 2. How epochs map to periods

Each call to `record_vector(epoch, vector)` computes the **period index** as
`floor(epoch / periodicity)`.

- For **state** streams, the vector is cached and flushed when the next period boundary is
  crossed. A period with no updates inherits the most recent cached state.
- For **accumulator** streams, vectors are added together within the same period using
  `Vector.__iadd__`. Empty periods emit an empty vector so gaps stay visible.

Closing the stream with `close(final_epoch)` ensures every relevant period boundary is
flushed. States emit through the period containing `final_epoch`; accumulators emit every
period strictly before that boundary.

---

## 3. End-to-end usage

```python
from data_logger import DataLogger
from graphblas import Vector
import numpy as np

logger = DataLogger("/tmp/run")
stream = logger.register_periodic_stream(
    labels={"entity": "temperature"},
    epoch_scale=1.0,
    value_scale=0.01,
    periodicity=1.0,
    kind="state",  # or "accumulator"
)

# Record sparse measurements
vector = Vector.from_coo(
    indices=np.array([0, 3], dtype=np.int64),
    values=np.array([2500, 7050], dtype=np.float64),
    size=4,  # at least one plus the max index
)
stream.record_vector(epoch=0.3, vector=vector)

# Later…
stream.close(final_epoch=10.0)
logger.close()
```

**Key takeaways:**

- Prefer `record_vector` when working with `graphblas.Vector` inputs. Use `record` if you
  already have NumPy `indices` (``int64``) and `values` (``float64``) arrays and want the
  helper to construct the vector for you.
- State streams perform a single `.dup()` when the period boundary advances. Accumulator
  streams reuse the vector reference whenever possible and rely on `Vector.__iadd__` for
  efficient summation.
- Call `close` even if the final period received no updates. The logger writes empty
  vectors as needed so downstream analytics can distinguish missing data from zeros.

---

## 4. Inspecting the results

Each periodic record is written with:

- The **epoch** equal to the period boundary (e.g. `3.0` for the period `[3, 4)`).
- The sparse vector **indices** and **values** from the state snapshot or accumulator sum.
- Stream metadata in the JSON manifest that now includes:
  - `periodicity`
  - `kind`

To verify your setup:

1. Iterate over the emitted segment files and confirm that the epoch spacing matches your
   configured periodicity.
2. For accumulator streams, compare the logged values with an independent sum of your raw
   measurements.
3. For state streams, check that the logged vectors match the last update from each period.

---

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Missing final period in the output | `close(final_epoch)` was not called or the epoch was too small. | Invoke `close` with the simulation's final epoch so the last boundary is flushed. |
| Empty vectors where data was expected | All updates happened after the period boundary. | Ensure your epochs are scaled correctly relative to `periodicity`. |
| Mismatched indices during accumulation | Raw data is not a `Vector`. | Wrap measurements in `Vector` before recording so addition can align sparse entries. |

By following the patterns above, you can convert bursty sparse updates into predictable,
one-record-per-period logs without sacrificing accuracy.

