"""The interface every controller implements."""

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from jev_controller.actions import ActionSet


@dataclass
class Decision:
    action: int
    latency_s: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)


class Policy(Protocol):
    name: str
    actions: ActionSet

    def reset(self, seed: int) -> None: ...

    def act(self, obs: np.ndarray) -> Decision: ...
