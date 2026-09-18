"""Jev as a direct controller: one `Choice` question per decision, always executed as returned."""

import hashlib
import json
import threading
import time
from typing import Any, Literal

import numpy as np
from typesafe_sdk import Choice, TypeSafeError

from jev_controller.actions import ActionSet
from jev_controller.env import DT, THETA_LIMIT, X_LIMIT
from jev_controller.policies.base import Decision

Variant = Literal["raw", "semantic"]

_TASK = (
    "You are the controller of a cart-pole system. A pole is hinged on top of a cart that moves along "
    "a horizontal track. Choose the force to apply to the cart for the next decision interval so that "
    "the pole stays upright and the cart stays inside the track limits for as long as possible. "
    "The episode ends if the pole angle leaves the angle limit or the cart position leaves the "
    "position limit."
)

# Instruction variants. "base" is the pre-registered v0 prompt; "priorities" adds how urgent each
# concern is, without stating any control rule.
INSTRUCTIONS = {
    "base": _TASK,
    "priorities": _TASK
    + (
        " Priorities: an unbalanced pole falls within about half a second, so the pole's angle and "
        "rotation are the urgent concern at every decision. The cart's position and velocity only "
        "matter over several seconds, unless the cart is close to a track limit. Each decision lasts a "
        "very short time and you will choose again immediately afterwards."
    ),
}

CONVENTIONS = {
    "cart_position_m": "Positive is right of the track centre, negative is left.",
    "cart_velocity_m_per_s": "Positive means the cart is moving right.",
    "pole_angle_rad": "Zero is perfectly upright. Positive means the top of the pole leans right.",
    "pole_angular_velocity_rad_per_s": "Positive means the pole is rotating clockwise, toward the right.",
    "force": "A positive force pushes the cart to the right, a negative force to the left.",
}

# Semantic-label thresholds. Frozen with the prompt: change them only before the test run.
LEAN_BANDS = ((0.01, "upright"), (0.05, "slight"), (0.12, "moderate"), (np.inf, "severe"))
SPIN_BANDS = ((0.05, "still"), (0.3, "slow"), (1.0, "moderate"), (np.inf, "fast"))
POSITION_BANDS = ((0.25, "centred"), (1.2, "off centre"), (1.9, "far off centre"), (np.inf, "near the edge"))
SPEED_BANDS = ((0.05, "stationary"), (0.5, "slow"), (1.5, "moderate"), (np.inf, "fast"))


def _band(value: float, bands: tuple[tuple[float, str], ...]) -> str:
    return next(label for limit, label in bands if abs(value) < limit)


def _side(value: float) -> str:
    return "right" if value > 0 else "left"


def semantic_labels(obs: np.ndarray) -> dict[str, str]:
    """Deterministic, code-computed descriptions of the state. Descriptive only, never advice."""
    x, x_dot, theta, theta_dot = (float(v) for v in obs)
    lean = _band(theta, LEAN_BANDS)
    spin = _band(theta_dot, SPIN_BANDS)
    position = _band(x, POSITION_BANDS)
    speed = _band(x_dot, SPEED_BANDS)
    return {
        "pole_lean": lean if lean == "upright" else f"{lean} lean to the {_side(theta)}",
        "pole_rotation": spin if spin == "still" else f"{spin} rotation toward the {_side(theta_dot)}",
        "pole_trend": (
            "steady"
            if theta_dot == 0
            else "crossing upright"
            if theta == 0
            else "lean increasing"
            if theta * theta_dot > 0
            else "lean decreasing"
        ),
        "cart_position": position if position == "centred" else f"{position}, on the {_side(x)}",
        "cart_motion": speed if speed == "stationary" else f"{speed}, moving {_side(x_dot)}",
        "cart_trend": (
            "holding position"
            if speed == "stationary"
            else "crossing centre"
            if x == 0
            else "moving away from centre"
            if x * x_dot > 0
            else "moving toward centre"
        ),
    }


def option_descriptions(actions: ActionSet, interval_s: float) -> dict[str, str]:
    out = {}
    for label, force in zip(actions.labels, actions.forces, strict=True):
        if force == 0:
            out[label] = f"Apply no force to the cart for the next {interval_s:g} s."
        else:
            size = "full" if abs(force) == max(actions.forces) else "half"
            direction = "right (positive)" if force > 0 else "left (negative)"
            out[label] = (
                f"Push the cart {direction} with {size} force, {abs(force):g} N, for the next {interval_s:g} s."
            )
    return out


class RateLimiter:
    """Thread-safe minimum spacing between requests, shared by every Jev policy in a run."""

    def __init__(self, max_per_second: float) -> None:
        self.interval = 1.0 / max_per_second
        self.lock = threading.Lock()
        self.next_slot = 0.0

    def wait(self) -> float:
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next_slot)
            self.next_slot = slot + self.interval
        delay = slot - now
        if delay > 0:
            time.sleep(delay)
        return delay


class JevPolicy:
    def __init__(
        self,
        client: Any,
        actions: ActionSet,
        variant: Variant,
        *,
        model: str,
        decision_steps: int = 1,
        timeout_s: float = 2.0,
        limiter: RateLimiter | None = None,
        instructions: str = "base",
    ) -> None:
        self.name = f"jev_{variant}"
        self.client = client
        self.actions = actions
        self.variant = variant
        self.model = model
        self.timeout_s = timeout_s
        self.limiter = limiter
        self.interval_s = decision_steps * DT
        self.question = Choice(
            instructions=INSTRUCTIONS[instructions], criteria=option_descriptions(actions, self.interval_s)
        )
        self.last_action: int | None = None

    @property
    def prompt_spec(self) -> dict[str, Any]:
        """Everything that is fixed across calls; hashed into the run config to freeze the prompt."""
        return {
            "variant": self.variant,
            "question": self.question.model_dump(),
            "state_template": self.build_state(np.zeros(4)),
            "semantic_thresholds": {
                "lean": LEAN_BANDS,
                "spin": SPIN_BANDS,
                "position": POSITION_BANDS,
                "speed": SPEED_BANDS,
            },
        }

    @property
    def prompt_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.prompt_spec, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def build_state(self, obs: np.ndarray) -> dict[str, Any]:
        x, x_dot, theta, theta_dot = (round(float(v), 4) for v in obs)
        state: dict[str, Any] = {
            "observation": {
                "cart_position_m": x,
                "cart_velocity_m_per_s": x_dot,
                "pole_angle_rad": theta,
                "pole_angular_velocity_rad_per_s": theta_dot,
            },
            "limits": {
                "cart_position_m": [-X_LIMIT, X_LIMIT],
                "pole_angle_rad": [round(-THETA_LIMIT, 4), round(THETA_LIMIT, 4)],
            },
            "conventions": CONVENTIONS,
            "decision_interval_s": self.interval_s,
        }
        if self.variant == "semantic":
            state["labels"] = semantic_labels(obs)
        return state

    def reset(self, seed: int) -> None:
        self.last_action = None

    def act(self, obs: np.ndarray) -> Decision:
        state = self.build_state(obs)
        start = time.perf_counter()
        wait = self.limiter.wait() if self.limiter else 0.0
        diag: dict[str, Any] = {"jev_state": state, "prompt_hash": self.prompt_hash, "rate_limit_wait_s": wait}
        try:
            response = self.client.system_one(
                state=state, questions={"action": self.question}, model=self.model, timeout=self.timeout_s
            )
        except TypeSafeError as error:
            latency = time.perf_counter() - start
            # Pre-declared fallback: hold the previous action (neutral if there is none yet).
            action = self.last_action if self.last_action is not None else self.actions.neutral
            diag |= {"error": type(error).__name__, "error_message": str(error)[:300], "fallback": True}
            return Decision(action, latency, diag)
        latency = time.perf_counter() - start
        answer = response.choices["action"]
        action = self.actions.labels.index(answer.choice)
        self.last_action = action
        diag |= {
            "choice": answer.choice,
            "confidence": answer.confidence,
            "probabilities": dict(answer.probabilities),
            "model": response.model,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "fallback": False,
        }
        return Decision(action, latency, diag)
