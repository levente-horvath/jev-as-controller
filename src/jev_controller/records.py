"""Write-once result storage: one JSON file per episode, never overwritten.

An episode file is written to a temporary name and hard-linked into place, so a crash leaves either a
complete episode or none, and a resumed run skips finished episodes instead of duplicating them.
"""

import json
import os
from pathlib import Path
from typing import Any


class RunDir:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.episodes = self.path / "episodes"

    def bind(self, manifest: dict[str, Any]) -> None:
        """Create the run directory or confirm an existing one was made with the same frozen settings."""
        self.episodes.mkdir(parents=True, exist_ok=True)
        manifest_path = self.path / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            keys = ("config_hash", "prompt_hashes", "model", "source_hash", "fake")
            if any(existing.get(k) != manifest.get(k) for k in keys):
                raise RuntimeError(
                    f"{manifest_path} was created with different settings "
                    f"({ {k: existing.get(k) for k in keys} }); use a new run name instead of mixing results."
                )
            return
        write_once(manifest_path, manifest)

    def episode_path(self, key: str) -> Path:
        return self.episodes / f"{key}.json"

    def done(self, key: str) -> bool:
        return self.episode_path(key).exists()

    def write_episode(self, key: str, record: dict[str, Any]) -> None:
        write_once(self.episode_path(key), record)

    def load_episodes(self) -> list[dict[str, Any]]:
        return [json.loads(p.read_text()) for p in sorted(self.episodes.glob("*.json"))]


def write_once(path: Path, data: Any) -> None:
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    with open(tmp, "x") as f:
        json.dump(data, f, default=float)
    try:
        os.link(tmp, path)  # fails if the destination exists
    finally:
        tmp.unlink()
