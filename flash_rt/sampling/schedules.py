"""Scalar sampling clocks. Mapping a clock does not rescale neural velocity."""

from dataclasses import dataclass
import math
from numbers import Real


@dataclass(frozen=True)
class ShiftedSigmaSchedule:
    """Wan-style sigma(u), for Euler updates in actual sigma coordinates."""

    shift: float = 1.0

    def __post_init__(self):
        if (isinstance(self.shift, bool) or not isinstance(self.shift, Real)
                or not math.isfinite(self.shift) or self.shift <= 0):
            raise ValueError('sigma shift must be positive and finite')

    def __call__(self, clock: float) -> float:
        if not math.isfinite(clock) or not 0 <= clock <= 1:
            raise ValueError('sampling clock must be finite and in [0, 1]')
        scaled = self.shift * clock
        # Algebraically identical to 1+(shift-1)*clock, but retains a positive
        # endpoint denominator even when a tiny shift rounds shift-1 to -1.
        return scaled / ((1.0 - clock) + scaled)
