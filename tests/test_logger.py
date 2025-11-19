import numpy as np
from data_logger import DataLogger


def test_basic(tmp_path):
    elog = DataLogger(tmp_path / "segments")
    sid = elog.register_stream(
        labels={"entity":"test"},
        epoch_scale=1e-3, value_scale=1e-6
    )
    for t in np.arange(0.0, 0.01, 0.001):
        idx = np.array([1,2,3], dtype=np.uint32)
        val = np.array([0.1, 0.2, -0.1], dtype=np.float64)
        elog.record(sid, t, idx, val)
    elog.close()
    elog.to_parquet(tmp_path / "out.parquet")


def test_done_sentinel_written(tmp_path):
    segdir = tmp_path / "segments"
    elog = DataLogger(segdir)
    sid = elog.register_stream(
        labels={"entity": "done-sentinel"},
        epoch_scale=1e-3,
        value_scale=1e-6,
    )
    idx = np.array([0], dtype=np.uint32)
    val = np.array([1.0], dtype=np.float64)
    elog.record(sid, 0.0, idx, val)
    elog.close()

    done_file = segdir / "_DONE"
    assert done_file.exists(), "_DONE sentinel should be written when closing the logger"
    assert done_file.stat().st_size == 0, "_DONE sentinel must be empty"
