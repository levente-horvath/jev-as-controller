"""Closed-loop episodes under the paused and real-time protocols."""

import math
from typing import Any

import numpy as np

from jev_controller.config import Protocol
from jev_controller.env import DT, MAX_STEPS, THETA_LIMIT, X_LIMIT, CartPole, Condition
from jev_controller.policies.base import Policy


def episode_key(policy: str, condition: str, protocol: str, seed: int) -> str:
    return f"{protocol}__{condition}__{policy}__{seed}"


def run_episode(policy: Policy, condition: Condition, protocol: Protocol, seed: int) -> dict[str, Any]:
    """Run one episode.

    Paused: the plant waits for every decision, which then applies immediately.
    Real-time: the plant keeps moving under the previous action while a decision is computed. A
    request issued at step t with latency L takes effect at step t + ceil(L / DT), using the call's
    measured latency, and the next request is issued once it has landed (and at least
    `decision_steps` after the last one). The first decision of an episode applies immediately: the
    episode starts when the controller is ready.
    """
    env = CartPole(condition, policy.actions)
    obs = env.reset(seed)
    policy.reset(seed)

    held: int | None = None
    pending: tuple[int, int] | None = None  # (apply_step, action)
    next_request = 0
    decisions: list[dict[str, Any]] = []
    states, actions, impulses = [obs.tolist()], [], []
    terminated = truncated = False

    while not (terminated or truncated):
        t = env.t
        if pending is not None and t >= pending[0]:  # a response landed: apply it, then ask again
            held, pending = pending[1], None
        if pending is None and t >= next_request:
            decision = policy.act(obs)
            delay = math.ceil(decision.latency_s / DT) if protocol.realtime and held is not None else 0
            pending = (t + delay, decision.action)
            next_request = t + max(delay, protocol.decision_steps)
            decisions.append(
                {
                    "step": t,
                    "apply_step": t + delay,
                    "observation": obs.tolist(),
                    "action": decision.action,
                    "action_label": policy.actions.labels[decision.action],
                    "latency_s": decision.latency_s,
                    **decision.diagnostics,
                }
            )
        if pending is not None and t >= pending[0]:  # zero-delay decisions apply this step
            held, pending = pending[1], None
        assert held is not None
        obs, terminated, truncated, impulse = env.step(held)
        states.append(obs.tolist())
        actions.append(held)
        if impulse:
            impulses.append([env.t, impulse])

    traj = np.array(states)
    if terminated:
        reason = "cart_position" if abs(traj[-1, 0]) > X_LIMIT else "pole_angle"
    else:
        reason = "horizon"
    latencies = np.array([d["latency_s"] for d in decisions])
    tokens = [d.get("input_tokens") for d in decisions if d.get("input_tokens") is not None]
    return {
        "policy": policy.name,
        "condition": condition.name,
        "protocol": protocol.name,
        "timing_mode": "measured_delay_simulation" if protocol.realtime else "paused",
        "decision_steps": protocol.decision_steps,
        "action_forces": list(policy.actions.forces),
        "seed": seed,
        "action_labels": list(policy.actions.labels),
        "steps": env.t,
        "return": float(env.t),
        "success": env.t >= MAX_STEPS and not terminated,
        "termination": reason,
        "rms_theta": float(np.sqrt(np.mean(traj[:, 2] ** 2))),
        "max_abs_theta": float(np.max(np.abs(traj[:, 2]))),
        "max_abs_theta_frac": float(np.max(np.abs(traj[:, 2])) / THETA_LIMIT),
        "rms_x": float(np.sqrt(np.mean(traj[:, 0] ** 2))),
        "max_abs_x": float(np.max(np.abs(traj[:, 0]))),
        "switches": int(np.count_nonzero(np.diff(actions))),
        "n_decisions": len(decisions),
        "n_fallbacks": sum(1 for d in decisions if d.get("fallback")),
        "latency_p50_s": float(np.percentile(latencies, 50)),
        "latency_p95_s": float(np.percentile(latencies, 95)),
        "mean_delay_steps": float(np.mean([d["apply_step"] - d["step"] for d in decisions])),
        "input_tokens": int(sum(tokens)),
        "models": sorted({d["model"] for d in decisions if "model" in d}),
        "states": states,
        "actions": actions,
        "impulses": impulses,
        "decisions": decisions,
    }
