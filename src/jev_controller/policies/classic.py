"""Reference controllers: LQR (ceiling), a one-line heuristic (easy bar) and random (floor)."""

import time

import numpy as np
from scipy.linalg import solve_discrete_are

from jev_controller.actions import ActionSet
from jev_controller.env import dynamics
from jev_controller.policies.base import Decision

LQR_Q = np.diag([1.0, 1.0, 10.0, 1.0])
LQR_R = np.array([[0.01]])


def lqr_gain(q: np.ndarray = LQR_Q, r: np.ndarray = LQR_R, eps: float = 1e-6, decision_steps: int = 1) -> np.ndarray:
    """Discrete-time LQR gain from a finite-difference linearisation of one 20 ms step about upright."""
    x0 = np.zeros(4)
    a = np.column_stack([(dynamics(x0 + eps * e, 0.0) - dynamics(x0 - eps * e, 0.0)) / (2 * eps) for e in np.eye(4)])
    b = ((dynamics(x0, eps) - dynamics(x0, -eps)) / (2 * eps)).reshape(4, 1)
    if decision_steps < 1:
        raise ValueError("decision_steps must be positive")
    b = sum(np.linalg.matrix_power(a, i) @ b for i in range(decision_steps))
    a = np.linalg.matrix_power(a, decision_steps)
    p = solve_discrete_are(a, b, q, r)
    return np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)


class LQRPolicy:
    name = "lqr"

    def __init__(self, actions: ActionSet, decision_steps: int = 1) -> None:
        self.actions = actions
        self.k = lqr_gain(decision_steps=decision_steps)

    def reset(self, seed: int) -> None:
        pass

    def force(self, obs: np.ndarray) -> float:
        return float(-(self.k @ obs)[0])

    def act(self, obs: np.ndarray) -> Decision:
        start = time.perf_counter()
        u = self.force(obs)
        action = self.actions.quantize(u)
        return Decision(action, time.perf_counter() - start, {"unclipped_force": u})


class PIDPolicy:
    """Outer position PD sets tilt; inner angle PID sets force, with integral anti-windup."""

    name = "pid"

    def __init__(self, actions: ActionSet, decision_steps: int = 1) -> None:
        self.actions = actions
        self.dt = 0.02 * decision_steps
        self.integral = 0.0

    def reset(self, seed: int) -> None:
        self.integral = 0.0

    def act(self, obs: np.ndarray) -> Decision:
        start = time.perf_counter()
        x, velocity, theta, omega = obs
        target = float(np.clip(-0.04 * x - 0.08 * velocity, -0.12, 0.12))
        error = theta - target
        candidate = float(np.clip(self.integral + error * self.dt, -0.2, 0.2))
        force = float(60 * error + 15 * omega + 2 * candidate)
        if abs(force) <= max(abs(f) for f in self.actions.forces):
            self.integral = candidate
        return Decision(self.actions.quantize(force), time.perf_counter() - start, {"unclipped_force": force})


class HeuristicPolicy:
    """Push full force toward the side the pole is falling: sign(theta + k * theta_dot)."""

    name = "heuristic"

    def __init__(self, actions: ActionSet, k: float = 0.5) -> None:
        self.actions = actions
        self.k = k

    def reset(self, seed: int) -> None:
        pass

    def act(self, obs: np.ndarray) -> Decision:
        start = time.perf_counter()
        push_right = obs[2] + self.k * obs[3] > 0
        action = len(self.actions) - 1 if push_right else 0
        return Decision(action, time.perf_counter() - start)


class RandomPolicy:
    name = "random"

    def __init__(self, actions: ActionSet) -> None:
        self.actions = actions
        self.rng = np.random.default_rng(0)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng([seed, 2])

    def act(self, obs: np.ndarray) -> Decision:
        return Decision(int(self.rng.integers(len(self.actions))))


class FixedLatency:
    """Wraps a local controller so it reports a fixed latency; used for the latency-budget sweep."""

    def __init__(self, policy: LQRPolicy | HeuristicPolicy, latency_s: float) -> None:
        self.policy = policy
        self.latency_s = latency_s
        self.name = policy.name
        self.actions = policy.actions

    def reset(self, seed: int) -> None:
        self.policy.reset(seed)

    def act(self, obs: np.ndarray) -> Decision:
        return Decision(self.policy.act(obs).action, self.latency_s)
