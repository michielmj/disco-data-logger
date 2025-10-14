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
