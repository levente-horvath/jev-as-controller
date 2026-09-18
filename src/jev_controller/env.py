"""CartPole dynamics identical to Gymnasium's CartPole-v1, with a configurable action set.

The dynamics are a pure function so the counterfactual evaluator can branch from any state.
`tests/test_env.py` checks them step-for-step against `gymnasium`'s reference implementation.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from jev_controller.actions import ActionSet, action_set

GRAVITY = 9.8
MASS_CART = 1.0
MASS_POLE = 0.1
TOTAL_MASS = MASS_CART + MASS_POLE
HALF_LENGTH = 0.5
POLEMASS_LENGTH = MASS_POLE * HALF_LENGTH
DT = 0.02
X_LIMIT = 2.4
THETA_LIMIT = 12 * 2 * math.pi / 360
MAX_STEPS = 500  # CartPole-v1 horizon


def dynamics(state: np.ndarray, force: float) -> np.ndarray:
    """One explicit-Euler step of the CartPole-v1 equations of motion."""
    x, x_dot, theta, theta_dot = state
    cos, sin = math.cos(theta), math.sin(theta)
    temp = (force + POLEMASS_LENGTH * theta_dot**2 * sin) / TOTAL_MASS
    theta_acc = (GRAVITY * sin - cos * temp) / (HALF_LENGTH * (4.0 / 3.0 - MASS_POLE * cos**2 / TOTAL_MASS))
    x_acc = temp - POLEMASS_LENGTH * theta_acc * cos / TOTAL_MASS
    return np.array(
        (x + DT * x_dot, x_dot + DT * x_acc, theta + DT * theta_dot, theta_dot + DT * theta_acc),
        dtype=np.float64,
    )


def failed(state: np.ndarray) -> bool:
    return abs(state[0]) > X_LIMIT or abs(state[2]) > THETA_LIMIT


@dataclass(frozen=True)
class Condition:
    """An evaluation condition: initial-state spread plus optional seeded impulse pushes."""

    name: str
    init_scale: float = 0.05  # CartPole-v1 samples every state variable from U(-0.05, 0.05)
    impulse_steps: tuple[int, ...] = ()
    impulse_min: float = 0.0  # |delta theta_dot| in rad/s
    impulse_max: float = 0.0

    def impulses(self, seed: int) -> dict[int, float]:
        """Seeded map of step -> angular-velocity kick, independent of the controller."""
        rng = np.random.default_rng([seed, 1])
        return {
            step: float(rng.choice((-1.0, 1.0)) * rng.uniform(self.impulse_min, self.impulse_max))
            for step in self.impulse_steps
        }


@dataclass
class CartPole:
    condition: Condition
    actions: ActionSet
    state: np.ndarray = field(init=False)
    t: int = field(init=False, default=0)
    _impulses: dict[int, float] = field(init=False, default_factory=dict)

    @classmethod
    def make(cls, condition: Condition, n_actions: int) -> "CartPole":
        return cls(condition, action_set(n_actions))

    def reset(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng([seed, 0])
        s = self.condition.init_scale
        self.state = rng.uniform(-s, s, size=4)
        self.t = 0
        self._impulses = self.condition.impulses(seed)
        return self.state.copy()

    def step(self, action: int) -> tuple[np.ndarray, bool, bool, float]:
        """Advance one 20 ms step. Returns (state, terminated, truncated, applied impulse)."""
        self.state = dynamics(self.state, self.actions.forces[action])
        self.t += 1
        terminated = failed(self.state)
        impulse = 0.0
        if not terminated and self.t in self._impulses:
            impulse = self._impulses[self.t]
            self.state[3] += impulse
        return self.state.copy(), terminated, self.t >= MAX_STEPS, impulse
