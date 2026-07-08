import math


class StepRounder:
    """Rounds a value to an exchange step/tick size (LOT_SIZE / PRICE_FILTER)."""

    __slots__ = ("step", "precision")

    def __init__(self, step: float):
        self.step = step
        self.precision = max(0, round(-math.log10(step))) if step > 0 else 0

    def __call__(self, value: float) -> float:
        if self.step <= 0:
            return value
        return round(round(value / self.step) * self.step, self.precision)

    @classmethod
    def apply(cls, value: float, step: float) -> float:
        return cls(step)(value)


def round_step(value: float, step: float) -> float:
    """Backwards-compatible shim around StepRounder."""
    return StepRounder.apply(value, step)