"""
disco-data-logger
=================

High-performance, stream-based logger for sparse numerical data in discrete-event
and Monte Carlo simulations.
"""

from .main import DataLogger
from .periodic import PeriodicVectorStream

# Expose package version (set via setuptools-scm write_to)
try:
    from ._version import version as __version__
except Exception:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["DataLogger", "PeriodicVectorStream"]
