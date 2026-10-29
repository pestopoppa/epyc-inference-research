"""Measurement-unit vocabulary shared without importing the serving loop.

A floor and an effect must name what was resampled: arm-scoped readings,
sessions, or separate process launches. These units are not
interchangeable; consumers retain their existing mismatch checks.
"""

UNIT_ARM = "arm"
UNIT_SESSION = "session"
UNIT_PROCESS = "process"
#: The only admissible units, narrowest first.
FLOOR_UNITS = (UNIT_ARM, UNIT_SESSION, UNIT_PROCESS)

__all__ = ["FLOOR_UNITS", "UNIT_ARM", "UNIT_PROCESS", "UNIT_SESSION"]
