import json

import numpy as np
import pytest

from data_logger import DataLogger
from sparse_array import Vector


def test_periodic_vector_stream_state_and_accumulator(tmp_path):
    elog = DataLogger(tmp_path / "segments_periodic")

    state = elog.register_periodic_stream(
        labels={"entity": "state"},
        epoch_scale=1.0,
        value_scale=0.1,
        periodicity=1.0,
        kind="state",
    )
    accum = elog.register_periodic_stream(
        labels={"entity": "acc"},
        epoch_scale=1.0,
        value_scale=0.1,
        periodicity=1.0,
        kind="accumulator",
    )

    events = [
        0.0,
        0.3,
        1.1,
        1.4,
        1.8,
        3.4,
        3.6,
        4.1,
    ]

    for t in events:
        indices = np.array([0], dtype=np.uint32)
        values = np.array([t], dtype=np.float64)
        vector = Vector(indices, values)
        state.record(t, vector)
        accum.record(t, vector)

    state.close(final_epoch=5.0)
    accum.close(final_epoch=5.0)
    elog.close()

    decoded = list(elog.decode_all_segments())
    decoded_state = [row for row in decoded if row[0] == state.stream_id]
    decoded_acc = [row for row in decoded if row[0] == accum.stream_id]

    # State stream: one sample per period, carrying the last seen value before the boundary.
    expected_state_epochs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    expected_state_values = [0.0, 0.3, 1.8, 1.8, 3.6, 4.1]

    assert [float(epoch) for _, epoch, *_ in decoded_state] == expected_state_epochs
    assert [float(vals[0]) if len(vals) else None for *_, vals in decoded_state] == pytest.approx(expected_state_values)

    # Accumulator stream: sums per period, empty period has an empty sparse vector.
    expected_acc_epochs = [0.0, 1.0, 2.0, 3.0, 4.0]
    expected_acc_values = [
        0.0 + 0.3,
        1.1 + 1.4 + 1.8,
        0.0,
        3.4 + 3.6,
        4.1,
    ]

    assert [float(epoch) for _, epoch, *_ in decoded_acc] == expected_acc_epochs
    acc_values = [float(vals[0]) if len(vals) else 0.0 for *_, vals in decoded_acc]
    assert acc_values == pytest.approx(expected_acc_values)

    # Metadata emitted during registration includes periodicity and kind markers.
    meta_path = tmp_path / "segments_periodic" / "streams" / f"{state.stream_id:08d}.json"
    with meta_path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    assert metadata["periodicity"] == pytest.approx(1.0)
    assert metadata["kind"] == "state"
