"""Open-loop sanity check: does Jev pick a good action from fixed states?

States are sampled from reference-controller trajectories, then every legal action is scored by a
simulator-side counterfactual: commit to the action briefly, continue with a frozen LQR controller,
and total a pre-declared state cost. This never affects any closed-loop episode.
"""

from typing import Any

import numpy as np

from jev_controller.actions import ActionSet, action_set
from jev_controller.config import Protocol
from jev_controller.env import THETA_LIMIT, X_LIMIT, Condition, dynamics, failed
from jev_controller.policies.base import Policy
from jev_controller.policies.classic import HeuristicPolicy, LQRPolicy
from jev_controller.runner import run_episode

FAIL_PENALTY = 10.0  # cost per remaining step after a failure


def action_costs(state: np.ndarray, actions: ActionSet, commit_steps: int, horizon_steps: int) -> np.ndarray:
    continuation = LQRPolicy(actions)
    costs = np.zeros(len(actions))
    for a in range(len(actions)):
        s = np.asarray(state, dtype=np.float64)
        for i in range(horizon_steps):
            u = a if i < commit_steps else continuation.actions.quantize(continuation.force(s))
            s = dynamics(s, actions.forces[u])
            if failed(s):
                costs[a] += FAIL_PENALTY * (horizon_steps - i)
                break
            costs[a] += (s[2] / THETA_LIMIT) ** 2 + 0.25 * (s[0] / X_LIMIT) ** 2
    return costs


def acceptable(costs: np.ndarray, rel_tol: float = 0.05, abs_tol: float = 0.5) -> np.ndarray:
    """Actions whose cost is within tolerance of the best one."""
    best = costs.min()
    return costs <= best + max(rel_tol * best, abs_tol)


def sample_states(actions: ActionSet, condition: Condition, seeds: list[int], n: int, rng_seed: int) -> np.ndarray:
    """Pool states from LQR and heuristic episodes (alternating by seed) and sample n of them."""
    protocol = Protocol("paused", 1, False)
    pool = []
    for i, seed in enumerate(seeds):
        policy = LQRPolicy(actions) if i % 2 == 0 else HeuristicPolicy(actions)
        pool.extend(run_episode(policy, condition, protocol, seed)["states"][:-1])  # drop terminal states
    pool = np.array(pool)
    idx = np.random.default_rng(rng_seed).choice(len(pool), size=min(n, len(pool)), replace=False)
    return pool[idx]


def label_states(states: np.ndarray, actions: ActionSet, commit_steps: int, horizon_steps: int) -> list[dict[str, Any]]:
    labelled = []
    for s in states:
        costs = action_costs(s, actions, commit_steps, horizon_steps)
        ok = acceptable(costs)
        labelled.append(
            {
                "state": s.tolist(),
                "costs": costs.tolist(),
                "best": int(np.argmin(costs)),
                "acceptable": ok.tolist(),
                "decisive": bool(not ok.all()),  # at least one action is clearly worse
            }
        )
    return labelled


def query(policy: Policy, item: dict[str, Any]) -> dict[str, Any]:
    policy.reset(0)
    d = policy.act(np.array(item["state"]))
    return {
        "policy": policy.name,
        "action": d.action,
        "latency_s": d.latency_s,
        **{k: v for k, v in d.diagnostics.items() if k != "jev_state"},
    }


def summarize(
    items: list[dict[str, Any]], answers: dict[str, list[dict[str, Any] | None]], n_actions: int
) -> dict[str, Any]:
    """Agreement with the counterfactual labels, over all states and over decisive states only."""
    out: dict[str, Any] = {}
    for name, answered in answers.items():
        rows = [(it, a) for it, a in zip(items, answered, strict=True) if a is not None and not a.get("fallback")]
        decisive = [(it, a) for it, a in rows if it["decisive"]]
        stats: dict[str, Any] = {"n": len(rows), "n_decisive": len(decisive)}
        for label, subset in (("all", rows), ("decisive", decisive)):
            if not subset:
                continue
            stats[f"best_rate_{label}"] = float(np.mean([a["action"] == it["best"] for it, a in subset]))
            stats[f"acceptable_rate_{label}"] = float(np.mean([it["acceptable"][a["action"]] for it, a in subset]))
            probs = [a.get("probabilities") for _, a in subset]
            if all(probs):
                labels = action_set(n_actions).labels
                p_best = [p[labels[it["best"]]] for (it, _), p in zip(subset, probs, strict=True)]
                p_ok = [
                    sum(p[labels[i]] for i, ok in enumerate(it["acceptable"]) if ok)
                    for (it, _), p in zip(subset, probs, strict=True)
                ]
                stats[f"mean_prob_best_{label}"] = float(np.mean(p_best))
                stats[f"mean_prob_acceptable_{label}"] = float(np.mean(p_ok))
        out[name] = stats
    # Random-choice reference computed exactly from the labels.
    for label, subset in (("all", items), ("decisive", [it for it in items if it["decisive"]])):
        if subset:
            out.setdefault("random_expected", {})[f"best_rate_{label}"] = 1 / n_actions
            out["random_expected"][f"acceptable_rate_{label}"] = float(
                np.mean([np.mean(it["acceptable"]) for it in subset])
            )
    return out
