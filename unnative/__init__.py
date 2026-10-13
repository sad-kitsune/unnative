"""unnative — recover readable Java from JNIC/JVMTP native-translated jars.

See plan.md for the design. The package is importable with the standard library
alone; heavier stages (emulation, Ghidra) import their dependencies lazily so that
triage never fails because an optional back-end is absent.
"""

from .errors import UnnativeError

__version__ = "0.0.1"
__all__ = ["UnnativeError", "__version__"]
