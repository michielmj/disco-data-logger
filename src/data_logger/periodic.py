"""Helpers for periodic sparse logging built on :mod:`graphblas`."""

from __future__ import annotations

from typing import Literal, TYPE_CHECKING

import numpy as np
from graphblas import Vector

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from .main import DataLogger


class PeriodicStream:
    """Reduce multiple sparse measurements to one record per period."""

    def __init__(
        self,
        logger: DataLogger,
        stream_id: int,
        *,
        periodicity: float = 1.0,
        kind: Literal["state", "accumulator"] = "state",
    ) -> None:
        if periodicity <= 0:
            raise ValueError("periodicity must be positive")
        if kind not in {"state", "accumulator"}:
            raise ValueError("kind must be either 'state' or 'accumulator'")

        self._logger = logger
        self._stream_id = stream_id
        self._periodicity = float(periodicity)
        self._kind = kind

        # State tracking
        self._next_state_period = 0
        self._state_vector: Vector | None = None

        # Accumulator tracking
        self._next_acc_period = 0
        self._acc_current_period: int | None = None
        self._accumulator_vector: Vector | None = None
        self._empty_vector = Vector.from_coo([], [], size=0, dtype=np.float64)

    @property
    def stream_id(self) -> int:
        """Return the wrapped stream identifier."""

        return self._stream_id

    def record(self, epoch: float, indices: np.ndarray, values: np.ndarray) -> None:
        """Record one measurement using arrays, mirroring :meth:`DataLogger.record`."""

        indices = np.array(indices, dtype=np.int64, copy=False)
        values = np.array(values, dtype=np.float64, copy=False)

        if indices.shape != values.shape:
            raise ValueError("indices and values must have the same length")

        if indices.size > 1 and not np.all(indices[1:] > indices[:-1]):
            order = np.argsort(indices, kind="stable")
            indices, values = indices[order], values[order]

        size = int(indices.max()) + 1 if indices.size else 0
        vector = Vector.from_coo(indices, values, size=size)
        self.record_vector(epoch, vector)

    def record_vector(self, epoch: float, vector: Vector) -> None:
        """Record one measurement and reduce it to the appropriate period."""

        if not isinstance(vector, Vector):
            raise TypeError("periodic streams expect graphblas.Vector inputs")

        if self._kind == "state":
            self._record_state(epoch, vector)
        else:
            self._record_accumulator(epoch, vector)

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _emit_state_up_to(self, epoch: float, include_equal: bool) -> None:
        """Flush cached state values for any completed periods."""

        if self._state_vector is None:
            return

        while True:
            boundary = self._next_state_period * self._periodicity
            if boundary > epoch or (not include_equal and boundary == epoch):
                break

            vector = self._state_vector
            idx, vals = vector.to_coo()
            self._logger.record(
                self._stream_id,
                boundary,
                np.asarray(idx, dtype=np.int64),
                np.asarray(vals, dtype=np.float64),
            )
            self._next_state_period += 1

    def _record_state(
        self,
        epoch: float,
        vector: Vector,
    ) -> None:
        """Cache the most recent state and emit completed periods."""

        self._emit_state_up_to(epoch, include_equal=False)
        self._state_vector = vector.dup()

    # ------------------------------------------------------------------
    # Accumulator helpers
    # ------------------------------------------------------------------

    def _emit_acc_periods_before(self, period_index: int) -> None:
        """Emit accumulator results for every period strictly before ``period_index``."""

        while self._next_acc_period < period_index:
            emit_period_index = self._next_acc_period
            if self._acc_current_period == emit_period_index:
                vector = self._materialize_accumulator()
                self._acc_current_period = None
                self._accumulator_vector = None
            else:
                vector = self._empty_vector

            idx, vals = vector.to_coo()
            self._logger.record(
                self._stream_id,
                emit_period_index * self._periodicity,
                np.asarray(idx, dtype=np.int64),
                np.asarray(vals, dtype=np.float64),
            )
            self._next_acc_period += 1

    def _materialize_accumulator(self) -> Vector:
        """Return the aggregated values for the current period as a Vector."""

        if self._accumulator_vector is None:
            return self._empty_vector

        return self._accumulator_vector

    def _record_accumulator(
        self,
        epoch: float,
        vector: Vector,
    ) -> None:
        """Add a measurement to the current accumulator period."""

        period_index = int(np.floor(epoch / self._periodicity))
        self._emit_acc_periods_before(period_index)

        if self._acc_current_period != period_index:
            self._acc_current_period = period_index
            self._accumulator_vector = vector.dup() if vector.nvals else None
            return

        if vector.nvals == 0:
            return

        if self._accumulator_vector is None:
            self._accumulator_vector = vector.dup()
            return

        self._accumulator_vector += vector

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def close(self, final_epoch: float) -> None:
        """Flush any pending period data up to ``final_epoch``."""

        if self._kind == "state":
            self._emit_state_up_to(final_epoch, include_equal=True)
        else:
            last_period_index = int(np.floor(final_epoch / self._periodicity))
            self._emit_acc_periods_before(last_period_index)
