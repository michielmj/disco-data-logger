# Agent Guidelines

- Keep `tests/test_logger.py` focused on the original regression coverage. Add any new logger or periodic vector stream tests in dedicated files instead.
- When extending periodic logging behaviour, prefer adding or updating tests in `tests/test_periodic_vector_stream.py`.
- Follow existing code style and avoid unnecessary data copies; rely on `sparse_array.Vector` utilities for sparse operations.
