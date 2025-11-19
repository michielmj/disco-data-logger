# Collector

The collector utility reads completed logger directories, filters their streams via
[`disco-tools`](https://pypi.org/project/disco-tools/) label selectors, and writes the decoded
records directly into any PyArrow IPC writer. It is designed for high-volume post-processing
jobs where iterating over rows in Python would be prohibitively expensive.

## When to use it

Use `data_logger.collector.Collector` when you need to:

1. Scan one or more logger directories on disk.
2. Filter the available streams based on their label JSON using `label_selector.Rule` objects.
3. Wait until the loggers have finished writing (presence of the `_DONE` marker file).
4. Decode the `.seg.zst` segments and flush them as PyArrow `RecordBatch` objects into
   a `RecordBatchFileWriter` or `RecordBatchStreamWriter` (or any compatible writer exposing
   `schema` and `write_batch`).
5. Augment each row with additional metadata columns that come from the stream labels.

The collector batches records column-wise so Arrow arrays are created without Python row
iteration, keeping the pipeline efficient even when hundreds of thousands of records are
involved.

## Basic usage

```python
from pathlib import Path
import pyarrow.ipc as pa_ipc
from data_logger.collector import Collector
from tools.label_selector import Rule

paths = ["/mnt/run_01", "/mnt/run_02"]
rule = Rule.all_of([Rule.equals("model", "supply_chain"), Rule.equals("measure", "inventory")])

collector = Collector(paths)
with pa_ipc.new_file("inventory.batches", schema=None) as writer:
    collector.collect(writer, rule=rule, columns=["organisation", "experiment"])
```

Notes:
- Pass `columns` to add extra label fields to the Arrow output. They are pulled from the
  stream label JSON files. Missing keys become `null` values.
- The collector automatically deduplicates column names to avoid duplicates, so you can safely
  concatenate multiple lists.
- Set `backoff` (milliseconds) to control how frequently `_DONE` markers are polled while waiting
  for loggers to finish. Increase this if you have many directories on a slow filesystem.
- Set `timeout` (milliseconds) if you need collection to stop after a certain time if the loggers
  never finish.

## Output schema

The collector always emits the following base columns:

| Column     | Type               | Description                               |
|------------|--------------------|-------------------------------------------|
| stream_id  | `uint32`           | Stream identifier from the logger labels. |
| epoch      | `float64`          | Dequantized epoch value.                  |
| indices    | `list<int64>`      | Per-record sparse indices.                |
| values     | `list<float64>`    | Per-record sparse values.                 |

Columns listed via the `columns` argument are appended after the base columns in the order they
are provided.

## Selecting streams

The collector loads every `streams/<stream_id>.json` file under each provided path into a list of
Python dictionaries. When you pass a `label_selector.Rule`, it filters that list using
`label_selector.select(streams, rule)`. Each matching stream dictionary is augmented with its
`_logger_path`, so the collector knows which directory to read.

If no rule is provided, every discovered stream is collected.

## Monitoring logger completion

Before decoding any `.seg.zst` files the collector waits for a `_DONE` file to appear in each
selected logger directory. The wait uses the `backoff` interval between checks and respects the
`timeout` limit. Passing `timeout=None` disables the deadline entirely.

## Cleaning up collected runs

After a run has been collected you can reclaim disk space using `Collector.cleanup`. By default it
removes only the `.seg.zst` segment files (keeping label JSON and the `streams/` directory) once all
paths contain the `_DONE` marker. The method accepts the same `backoff` and `timeout` arguments as
`collect`, plus two cleanup-specific options:

- `keep_meta=True` keeps the metadata and only removes the segment files. Set it to `False` if you
  also want the label JSON files and the entire `streams/` directory removed.
- `wait_for_done=True` ensures cleanup does not start until every logger path contains a `_DONE`
  marker. Set it to `False` when you need to purge incomplete runs regardless of their state.

The method returns `True` if every path was processed (and, when `wait_for_done=True`, the `_DONE`
markers were found before hitting the timeout) and `False` otherwise.

## Writers

`Collector.collect` accepts any object that exposes a `schema` attribute (or `None`) and a
`write_batch(batch: pa.RecordBatch)` method. Both `RecordBatchFileWriter` and
`RecordBatchStreamWriter` satisfy this protocol, but custom writers can be used as long as they
support those two attributes.
