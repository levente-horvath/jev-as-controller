"""Build the self-contained replay page ("Jev at the Helm") from logged episodes. No API calls."""

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from jev_controller.actions import action_set
from jev_controller.env import DT, HALF_LENGTH, THETA_LIMIT, X_LIMIT
from jev_controller.report import PRICE_PER_MTOK


def _r(values: list[float], nd: int = 4) -> list[float]:
    return [round(v, nd) for v in values]


def _compact(episode: dict[str, Any], labels: list[str]) -> dict[str, Any]:
    decisions = []
    for d in episode["decisions"]:
        probs = d.get("probabilities")
        decisions.append(
            {
                "s": d["step"],
                "a": d["apply_step"],
                "act": d["action"],
                "l": round(d["latency_s"], 4),
                "o": _r(d["observation"]),
                "p": [round(probs[k], 4) for k in labels] if probs else None,
                "c": d.get("confidence"),
                "fb": bool(d.get("fallback")),
                "lab": d.get("jev_state", {}).get("labels"),
            }
        )
    return {
        "policy": episode["policy"],
        "protocol": episode["protocol"],
        "decision_steps": episode.get("decision_steps", 1),
        "condition": episode["condition"],
        "seed": episode["seed"],
        "steps": episode["steps"],
        "success": episode["success"],
        "termination": episode["termination"],
        "states": [_r(s) for s in episode["states"]],
        "actions": episode["actions"],
        "impulses": episode["impulses"],
        "tokens": episode["input_tokens"],
        "decisions": decisions,
    }


def build_player(
    jev_episodes: list[dict[str, Any]],
    compare: dict[tuple[str, int], dict[str, Any]],
    manifest: dict[str, Any],
    out: Path,
    fake: bool,
) -> None:
    labels = jev_episodes[0]["action_labels"]
    episodes = []
    for e in jev_episodes:
        item = _compact(e, labels)
        ref = compare.get((e["condition"], e["seed"]))
        if ref is not None and ref["policy"] == e["policy"]:
            ref = None
        item["compare"] = (
            {
                "policy": ref["policy"],
                "success": ref["success"],
                "states": [_r(s) for s in ref["states"]],
                "actions": ref["actions"],
                "steps": ref["steps"],
                "termination": ref["termination"],
                "impulses": ref["impulses"],
            }
            if ref
            else None
        )
        episodes.append(item)
    data = {
        "meta": {
            "policy": jev_episodes[0]["policy"],
            "models": sorted({m for e in jev_episodes for m in e["models"]}),
            "labels": labels,
            "forces": list(action_set(len(labels)).forces),
            "dt": DT,
            "x_limit": X_LIMIT,
            "theta_limit": THETA_LIMIT,
            "half_length": HALF_LENGTH,
            "price_per_mtok": PRICE_PER_MTOK,
            "config_hash": manifest.get("config_hash", "–"),
            "fake": fake,
            "legacy": not bool(manifest.get("source_hash")),
            "run_name": manifest.get("config", {}).get("run", {}).get("name", "unknown"),
        },
        "episodes": episodes,
    }
    template = files("jev_controller").joinpath("player_template.html").read_text()
    payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    html = template.replace("/*__DATA__*/null", payload)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
