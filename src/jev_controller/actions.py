"""Discrete action sets shared by every controller."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActionSet:
    name: str
    labels: tuple[str, ...]
    forces: tuple[float, ...]  # newtons, positive pushes the cart toward +x (right)

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def neutral(self) -> int:
        """Index of the action whose force is closest to zero (lowest index on ties)."""
        return int(np.argmin(np.abs(self.forces)))

    def quantize(self, force: float) -> int:
        """Nearest action to a continuous force; saturates at the extremes, lowest index on ties."""
        return int(np.argmin(np.abs(np.asarray(self.forces) - force)))


TWO_ACTIONS = ActionSet("two", ("push_left", "push_right"), (-10.0, 10.0))
FIVE_ACTIONS = ActionSet(
    "five",
    ("strong_left", "left", "none", "right", "strong_right"),
    (-10.0, -5.0, 0.0, 5.0, 10.0),
)


def action_set(n_actions: int) -> ActionSet:
    match n_actions:
        case 2:
            return TWO_ACTIONS
        case 5:
            return FIVE_ACTIONS
    raise ValueError(f"unsupported action count: {n_actions}")
