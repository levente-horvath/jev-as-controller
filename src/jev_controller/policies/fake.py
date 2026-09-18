"""A cost-free stand-in for the TypeSafe client, used by tests and `--fake-jev` dry runs.

It answers like a noisy heuristic controller and returns the same response shape as the SDK. Its
results say nothing about Jev; it only exercises the pipeline end to end.
"""

import threading
import time
from dataclasses import dataclass

import numpy as np
from typesafe_sdk import TypeSafeAPITimeoutError


@dataclass(frozen=True)
class _Answer:
    choice: str
    confidence: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class _Usage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class _Response:
    model: str
    usage: _Usage
    choices: dict[str, _Answer]


class FakeJevClient:
    def __init__(self, seed: int = 0, latency_s: float = 0.0, timeout_rate: float = 0.0) -> None:
        self.rng = np.random.default_rng(seed)
        self.latency_s = latency_s
        self.timeout_rate = timeout_rate
        self.calls = 0
        self.lock = threading.Lock()  # numpy generators are not thread-safe

    def system_one(self, *, state, questions, model=None, timeout=None):
        if self.latency_s:
            time.sleep(self.latency_s)
        with self.lock:
            self.calls += 1
            timed_out = self.rng.random() < self.timeout_rate
            noise = self.rng.normal(0, 0.5, len(questions["action"].criteria))
        if timed_out:
            raise TypeSafeAPITimeoutError(timeout or 2.0)
        obs = state["observation"]
        labels = list(questions["action"].criteria)
        lean = obs["pole_angle_rad"] + 0.5 * obs["pole_angular_velocity_rad_per_s"]
        # Soft preference for pushing toward the fall, strongest at the extreme actions.
        positions = np.linspace(-1.0, 1.0, len(labels))
        logits = 40.0 * lean * positions + noise
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        best = int(np.argmax(probs))
        return _Response(
            model="fake-jev",
            usage=_Usage(input_tokens=len(str(state)) // 4, output_tokens=1),
            choices={
                "action": _Answer(labels[best], float(probs[best]), dict(zip(labels, probs.tolist(), strict=True)))
            },
        )
