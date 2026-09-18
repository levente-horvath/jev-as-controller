"""Summary table, pass/fail verdict and survival curves from a run directory."""

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from jev_controller.env import MAX_STEPS  # noqa: E402

PRICE_PER_MTOK = 0.042  # USD per million input tokens; output tokens are free
POLICY_ORDER = ("jev_raw", "jev_semantic", "lqr", "pid", "heuristic", "random")


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_mean(values: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def passes(returns: np.ndarray, successes: np.ndarray, pass_mark: dict[str, float]) -> bool:
    return bool(returns.mean() >= pass_mark["mean_return"] and successes.mean() >= pass_mark["success_rate"])


def summarize(episodes: list[dict[str, Any]], pass_mark: dict[str, float]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for e in episodes:
        groups[(e["protocol"], e["condition"], e["policy"])].append(e)
    order = {p: i for i, p in enumerate(POLICY_ORDER)}
    rows = []
    for (protocol, condition, policy), eps in sorted(
        groups.items(), key=lambda kv: (kv[0][0], kv[0][1], order.get(kv[0][2], 99))
    ):
        returns = np.array([e["return"] for e in eps])
        successes = np.array([e["success"] for e in eps])
        decisions = sum(e["n_decisions"] for e in eps)
        tokens = sum(e["input_tokens"] for e in eps)
        rows.append(
            {
                "protocol": protocol,
                "condition": condition,
                "policy": policy,
                "n": len(eps),
                "success_rate": float(successes.mean()),
                "success_ci": wilson(int(successes.sum()), len(eps)),
                "mean_return": float(returns.mean()),
                "return_ci": bootstrap_mean(returns),
                "passes": passes(returns, successes, pass_mark),
                "rms_theta_deg": float(np.degrees(np.mean([e["rms_theta"] for e in eps]))),
                "rms_x_m": float(np.mean([e["rms_x"] for e in eps])),
                "failures": dict(
                    sorted(
                        {t: sum(e["termination"] == t for e in eps) for t in {e["termination"] for e in eps}}.items()
                    )
                ),
                "decisions": decisions,
                "fallback_rate": sum(e["n_fallbacks"] for e in eps) / max(decisions, 1),
                "latency_p50_ms": 1000 * float(np.median([d["latency_s"] for e in eps for d in e["decisions"]])),
                "latency_p95_ms": 1000
                * float(np.percentile([d["latency_s"] for e in eps for d in e["decisions"]], 95)),
                "mean_delay_steps": float(np.mean([e["mean_delay_steps"] for e in eps])),
                "tokens_per_call": tokens / max(decisions, 1),
                "cost_usd": tokens * PRICE_PER_MTOK / 1e6,
                "cost_per_episode_usd": tokens * PRICE_PER_MTOK / 1e6 / len(eps),
                "models": sorted({m for e in eps for m in e["models"]}),
            }
        )
    return rows


def survival_plot(episodes: list[dict[str, Any]], out: Path) -> None:
    keys = sorted({(e["protocol"], e["condition"]) for e in episodes})
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3.6), squeeze=False, sharey=True)
    steps = np.arange(MAX_STEPS + 1)
    for ax, (protocol, condition) in zip(axes[0], keys, strict=True):
        subset = [e for e in episodes if e["protocol"] == protocol and e["condition"] == condition]
        for policy in [p for p in POLICY_ORDER if any(e["policy"] == p for e in subset)]:
            lengths = np.array([e["steps"] for e in subset if e["policy"] == policy])
            successes = np.array([e["success"] for e in subset if e["policy"] == policy])
            alive = [((lengths > s) | ((lengths == s) & successes)).mean() for s in steps]
            ax.step(steps * 0.02, alive, where="post", label=f"{policy} (n={len(lengths)})", lw=1.8)
        ax.set_title(f"{condition} · {protocol}")
        ax.set_xlabel("time balanced (s)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("fraction of episodes still balanced")
    axes[0][-1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def markdown(rows: list[dict[str, Any]], pass_mark: dict[str, float]) -> str:
    lines = [
        f"Pass mark: mean return ≥ {pass_mark['mean_return']:g} and success rate ≥ {pass_mark['success_rate']:.0%} "
        f"(success = all {MAX_STEPS} steps balanced).",
        "",
        "| protocol | condition | policy | n | success (95% CI) | mean return (95% CI) | pass | RMS θ (°) | RMS x (m) "
        "| failures | fallback | latency p50/p95 (ms) | delay (steps) | tokens/call | cost/episode |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lo, hi = r["success_ci"]
        rlo, rhi = r["return_ci"]
        fails = ", ".join(f"{k}: {v}" for k, v in r["failures"].items())
        is_jev = r["policy"].startswith("jev")
        lines.append(
            f"| {r['protocol']} | {r['condition']} | {r['policy']} | {r['n']} "
            f"| {r['success_rate']:.0%} ({lo:.0%}–{hi:.0%}) | {r['mean_return']:.0f} ({rlo:.0f}–{rhi:.0f}) "
            f"| {'✅' if r['passes'] else '❌'} | {r['rms_theta_deg']:.2f} | {r['rms_x_m']:.2f} | {fails} "
            f"| {r['fallback_rate']:.1%} | {r['latency_p50_ms']:.0f} / {r['latency_p95_ms']:.0f} "
            f"| {r['mean_delay_steps']:.1f} | {r['tokens_per_call']:.0f} "
            f"| {f'${r["cost_per_episode_usd"]:.4f}' if is_jev else '–'} |"
        )
    total = sum(r["cost_usd"] for r in rows)
    models = sorted({m for r in rows for m in r["models"]})
    lines += [
        "",
        f"Estimated cost of logged successful responses: ${total:.2f} at ${PRICE_PER_MTOK}/M input tokens. "
        f"Not a billing reconciliation; legacy retries may be missing. Models: {', '.join(models) or '–'}.",
    ]
    return "\n".join(lines)


def latency_budget_plot(
    curves: dict[str, list[tuple[float, float]]], out: Path, measured_ms: dict[str, float] | None = None
) -> None:
    """Success rate of local controllers against an artificial real-time latency."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for name, points in curves.items():
        ms, rate = zip(*points, strict=True)
        five = "five" in name
        ax.plot(ms, rate, marker="s" if five else "o", ls="--" if five else "-", lw=1.8, label=name, alpha=0.85)
    for label, value in (measured_ms or {}).items():
        ax.axvline(value, color="0.3", ls="--", lw=1)
        ax.text(value, 1.03, label, rotation=90, va="bottom", ha="right", fontsize=8)
    ax.set_xlabel("decision latency (ms); the plant keeps moving meanwhile")
    ax.set_ylabel("success rate (500 steps)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
