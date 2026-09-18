"""Experiment configuration loaded from TOML, plus local secret loading."""

import hashlib
import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jev_controller.env import Condition

SPLITS = ("dev", "offline", "test")


@dataclass(frozen=True)
class Protocol:
    name: str
    decision_steps: int
    realtime: bool
    episodes: int | None = None  # cap on seeds per condition, e.g. for the short live demo


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, "rb") as f:
            cfg = cls(tomllib.load(f))
        if cfg.workers < 1 or cfg.max_rps <= 0:
            raise ValueError("workers and max_requests_per_second must be positive")
        if cfg.n_actions not in (2, 5):
            raise ValueError("n_actions must be 2 or 5")
        if any(p.decision_steps < 1 for p in cfg.protocols().values()):
            raise ValueError("decision_steps must be positive")
        if any(count < 1 or start < 0 for start, count in cfg.raw["seeds"].values()):
            raise ValueError("seeds require a nonnegative start and positive count")
        return cfg

    @property
    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.raw, sort_keys=True).encode()).hexdigest()[:12]

    @property
    def name(self) -> str:
        return self.raw["run"]["name"]

    @property
    def frozen(self) -> bool:
        return bool(self.raw["run"].get("frozen", False))

    @property
    def model(self) -> str:
        return self.raw["run"]["model"]

    @property
    def n_actions(self) -> int:
        return int(self.raw["env"]["n_actions"])

    @property
    def workers(self) -> int:
        return int(self.raw["run"].get("workers", 1))

    @property
    def max_rps(self) -> float:
        return float(self.raw["run"]["max_requests_per_second"])

    @property
    def jev(self) -> dict[str, Any]:
        return self.raw["jev"]

    @property
    def offline(self) -> dict[str, Any]:
        return self.raw["offline"]

    def seeds(self, split: str) -> list[int]:
        start, count = self.raw["seeds"][split]
        return list(range(start, start + count))

    def conditions(self) -> dict[str, Condition]:
        return {
            name: Condition(
                name=name,
                init_scale=c.get("init_scale", 0.05),
                impulse_steps=tuple(c.get("impulse_steps", ())),
                impulse_min=c.get("impulse_min", 0.0),
                impulse_max=c.get("impulse_max", 0.0),
            )
            for name, c in self.raw["conditions"].items()
        }

    def protocols(self) -> dict[str, Protocol]:
        return {
            name: Protocol(name, int(p["decision_steps"]), bool(p["realtime"]), p.get("episodes"))
            for name, p in self.raw["protocols"].items()
        }


def load_dotenv(path: str | Path = ".env") -> None:
    """Load KEY=VALUE lines into the process environment without overriding existing variables."""
    try:
        lines = Path(path).read_text().splitlines()
    except FileNotFoundError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
