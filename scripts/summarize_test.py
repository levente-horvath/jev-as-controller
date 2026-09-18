"""Validate and summarize the completed frozen test run without making API calls."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from jev_controller.cli import ALL_POLICIES
from jev_controller.config import Config
from jev_controller.report import PRICE_PER_MTOK, bootstrap_mean, wilson


def main() -> None:
    cfg = Config.load("configs/cartpole-test.toml")
    root = Path("results") / cfg.name / "test"
    episodes = [json.loads(p.read_text()) for p in sorted((root / "paused/episodes").glob("*.json"))]
    expected = {(p, c, s) for p in ALL_POLICIES for c in ("nominal", "impulse") for s in cfg.seeds("test")}
    actual = [(e["policy"], e["condition"], e["seed"]) for e in episodes]
    assert len(actual) == len(set(actual)), "Duplicate episodes"
    assert set(actual) == expected, f"Incomplete run: {len(set(actual))}/{len(expected)} episodes"
    assert all(e["config_hash"] == cfg.hash for e in episodes), "Mixed configurations"
    assert all(len(e["states"]) == e["steps"] + 1 for e in episodes), "Incomplete trajectories"
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode["policy"], episode["condition"]].append(episode)
    rows = []
    for policy in ALL_POLICIES:
        for condition in ("nominal", "impulse"):
            es = groups[policy, condition]
            returns = np.array([e["steps"] for e in es])
            successes = sum(e["success"] for e in es)
            rows.append(
                {
                    "policy": policy,
                    "condition": condition,
                    "n": len(es),
                    "successes": successes,
                    "success_ci95": wilson(successes, len(es)),
                    "mean_steps": float(returns.mean()),
                    "mean_steps_ci95": bootstrap_mean(returns),
                }
            )
    paired = []
    for condition in ("nominal", "impulse"):
        for left, right in (("jev_semantic", "jev_raw"), ("jev_semantic", "lqr"), ("jev_raw", "lqr")):
            left_steps = {e["seed"]: e["steps"] for e in groups[left, condition]}
            right_steps = {e["seed"]: e["steps"] for e in groups[right, condition]}
            delta = np.array([left_steps[s] - right_steps[s] for s in cfg.seeds("test")])
            paired.append(
                {
                    "condition": condition,
                    "difference": f"{left} - {right}",
                    "mean_steps": float(delta.mean()),
                    "ci95": bootstrap_mean(delta),
                }
            )
    decisions = [d for e in episodes if e["policy"].startswith("jev") for d in e["decisions"]]
    tokens = sum(d.get("input_tokens", 0) for d in decisions)
    attempts = len(decisions)
    assert attempts <= 60000, "Request scope exceeded"
    models = sorted({d["model"] for d in decisions if "model" in d})
    assert models == [cfg.model], f"Unexpected models: {models}"
    cost = tokens * PRICE_PER_MTOK / 1e6
    fallbacks = sum(bool(d.get("fallback")) for d in decisions)
    output = {
        "config_hash": cfg.hash,
        "episodes": len(episodes),
        "api_attempts": attempts,
        "known_input_tokens": tokens,
        "estimated_cost_usd": cost,
        "fallbacks": fallbacks,
        "models": models,
        "results": rows,
        "paired_step_differences": paired,
        "latency_p50_p95_ms": np.percentile([d["latency_s"] * 1000 for d in decisions], [50, 95]).tolist(),
    }
    (root / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    lines = [
        "# Frozen CartPole test",
        "",
        f"{len(episodes)} episodes; 30 held-out seeds per condition and controller.",
        "Five actions, paused physics, 500 steps (10 simulated seconds).",
        "",
        "| Controller | Nominal success | Nominal mean steps | Impulse success | Impulse mean steps |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy in ALL_POLICIES:
        nominal, impulse = [r for r in rows if r["policy"] == policy]
        lines.append(
            f"| {policy} | {nominal['successes']}/30 | {nominal['mean_steps']:.1f} | "
            f"{impulse['successes']}/30 | {impulse['mean_steps']:.1f} |"
        )
    lines += [
        "",
        "## Paired differences",
        "",
        "Mean step differences on matching seeds; bootstrap 95% intervals.",
        "",
        "| Condition | Difference | Mean | 95% interval |",
        "|---|---|---:|---:|",
    ]
    for pair in paired:
        lo, hi = pair["ci95"]
        lines.append(
            f"| {pair['condition']} | {pair['difference']} | {pair['mean_steps']:.1f} | {lo:.1f} to {hi:.1f} |"
        )
    lines += [
        "",
        "## Cost and limitations",
        "",
        f"{attempts:,} API attempts, {tokens:,} known input tokens, estimated ${cost:.6f}.",
        f"{fallbacks} failed requests used the declared fallback. Their billing is unknown; costs are not an invoice.",
        "Automatic retries were disabled. The earlier pilot and smoke costs are excluded.",
        "",
        "Physics paused during calls. This evaluates decisions, not live real-time control.",
        "Some failures precede the first impulse; condition membership does not establish causation.",
        "No test-set prompt tuning was performed. RL was not included.",
        "",
        "See report.md for individual confidence intervals and additional metrics,",
        "summary.json for structured results, and cartpole-lab.html for all recorded trajectories.",
    ]
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
