import numpy as np

from data_logger import DataLogger
from data_logger.periodic import PeriodicVectorStream
from sparse_array import Vector


def _collect_records(logger: DataLogger, stream: PeriodicVectorStream | None = None):
    decoded = list(logger.decode_all_segments())
    if stream is None:
        return decoded
    return [row for row in decoded if row[0] == stream.stream_id]


def test_data_logger_does_not_share_array_memory(tmp_path):
    logger = DataLogger(tmp_path / "segments_mut")
    stream_id = logger.register_stream(
        labels={"entity": "mutable"},
        epoch_scale=1.0,
        value_scale=1.0,
    )

    indices = np.array([1, 2, 3], dtype=np.uint32)
    values = np.array([2.0, -3.0, 1.0], dtype=np.float64)

    logger.record(stream_id, 0.25, indices, values)

    # Mutate the arrays after recording; the logged values must remain intact.
    indices[:] = 0
    values[:] = 0.0

    logger.close()

    [(sid, epoch, logged_indices, logged_values)] = list(logger.decode_all_segments())
    assert sid == stream_id
    np.testing.assert_array_equal(logged_indices, np.array([1, 2, 3], dtype=np.uint32))
    np.testing.assert_array_equal(
        logged_values, np.array([2.0, -3.0, 1.0], dtype=np.float64)
    )


def test_periodic_state_copies_vectors(tmp_path):
    logger = DataLogger(tmp_path / "segments_state")
    stream = logger.register_periodic_stream(
        labels={"entity": "state"},
        epoch_scale=1.0,
        value_scale=1.0,
        periodicity=1.0,
        kind="state",
    )

    base_indices = np.array([5], dtype=np.uint32)
    base_values = np.array([4.0], dtype=np.float64)
    vector = Vector(base_indices, base_values)

    stream.record(0.4, vector)

    # Mutate the underlying values; the cached state must not change.
    vector.values[:] = 99.0

    stream.close(final_epoch=2.0)
    logger.close()

    records = _collect_records(logger, stream)
    assert records, "Expected at least one flushed state record"
    _, _, idx, vals = records[0]
    np.testing.assert_array_equal(idx, np.array([5], dtype=np.uint32))
    np.testing.assert_array_equal(vals, np.array([4.0], dtype=np.float64))


def test_periodic_accumulator_copies_vectors(tmp_path):
    logger = DataLogger(tmp_path / "segments_acc")
    stream = logger.register_periodic_stream(
        labels={"entity": "acc"},
        epoch_scale=1.0,
        value_scale=1.0,
        periodicity=1.0,
        kind="accumulator",
    )

    first_indices = np.array([1], dtype=np.uint32)
    first_values = np.array([2.0], dtype=np.float64)
    first_vector = Vector(first_indices, first_values)

    stream.record(0.1, first_vector)

    # Mutate after recording; internal accumulator should keep original values.
    first_vector.values[:] = -5.0

    second_indices = np.array([1], dtype=np.uint32)
    second_values = np.array([3.0], dtype=np.float64)
    second_vector = Vector(second_indices, second_values)

    stream.record(0.9, second_vector)

    second_vector.values[:] = -7.0

    stream.close(final_epoch=1.0)
    logger.close()

    records = _collect_records(logger, stream)
    assert records, "Expected accumulator output"
    # First (and only) period sum should equal original unmutated vectors.
    _, _, idx, vals = records[0]
    np.testing.assert_array_equal(idx, np.array([1], dtype=np.uint32))
    np.testing.assert_array_equal(vals, np.array([5.0], dtype=np.float64))
