import numpy as np
import pyarrow as pa
from tools.label_selector import Label

from data_logger import Collector, DataLogger


def _make_writer(extra_columns=None):
    extra_columns = extra_columns or []
    fields = [
        pa.field("stream_id", pa.uint32()),
        pa.field("epoch", pa.float64()),
        pa.field("indices", pa.list_(pa.int64())),
        pa.field("values", pa.list_(pa.float64())),
    ]
    for col in extra_columns:
        fields.append(pa.field(col, pa.string()))
    schema = pa.schema(fields)
    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, schema)
    return writer, sink


def test_collector_filters_and_writes(tmp_path):
    base = tmp_path / "segments"
    elog = DataLogger(base)
    sid_a = elog.register_stream({"entity": "A"}, epoch_scale=1.0, value_scale=1.0)
    sid_b = elog.register_stream({"entity": "B"}, epoch_scale=1.0, value_scale=1.0)
    elog.record(sid_a, 0.1, np.array([0], dtype=np.int64), np.array([1.0], dtype=np.float64))
    elog.record(sid_b, 0.2, np.array([1], dtype=np.int64), np.array([2.0], dtype=np.float64))
    elog.close()
    (base / "_DONE").touch()

    writer, sink = _make_writer(["entity"])
    collector = Collector([base])
    rule = Label("entity") == "A"
    assert collector.collect(writer, rule=rule, columns=["entity"]) is True
    writer.close()

    reader = pa.ipc.open_stream(sink.getvalue())
    table = reader.read_all()
    assert table.num_rows == 1
    assert table.column("stream_id")[0].as_py() == sid_a
    assert table.column("entity")[0].as_py() == "A"


def test_collector_times_out_without_done(tmp_path):
    base = tmp_path / "segments"
    elog = DataLogger(base)
    sid = elog.register_stream({"entity": "wait"}, epoch_scale=1.0, value_scale=1.0)
    elog.record(sid, 0.1, np.array([0], dtype=np.int64), np.array([1.0], dtype=np.float64))
    elog.close()

    writer, sink = _make_writer()
    collector = Collector([base])
    assert collector.collect(writer, timeout=10, backoff=1) is False
    writer.close()
    reader = pa.ipc.open_stream(sink.getvalue())
    assert reader.read_all().num_rows == 0
