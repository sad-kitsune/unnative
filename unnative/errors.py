"""The one error type the tool raises for shapes it refuses to guess about.

Design rule inherited from unseed: loud failure over silent guessing. Anything a
stage does not understand raises `UnnativeError` rather than returning a
plausible-looking answer. Callers rely on this being the *only* way a stage fails
for malformed or unrecognised input.
"""

from __future__ import annotations


class UnnativeError(Exception):
    """A shape this tool refuses to guess about."""
