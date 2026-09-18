import json

import numpy as np
import pytest

from jev_controller import cli
from jev_controller.actions import TWO_ACTIONS
from jev_controller.offline import acceptable, action_costs
from jev_controller.records import RunDir


def test_counterfactual_prefers_pushing_under_a_falling_pole():
    costs = action_costs(np.array([0.0, 0.0, 0.15, 1.0]), TWO_ACTIONS, commit_steps=5, horizon_steps=100)
    assert costs[1] < costs[0]
    assert acceptable(costs).tolist() == [False, True]


def test_run_dir_never_overwrites_and_rejects_changed_settings(tmp_path):
    run = RunDir(tmp_path / "r")
    run.bind({"config_hash": "a", "prompt_hashes": {}, "model": "m"})
    run.bind({"config_hash": "a", "prompt_hashes": {}, "model": "m"})
    with pytest.raises(RuntimeError):
        run.bind({"config_hash": "b", "prompt_hashes": {}, "model": "m"})
    run.write_episode("k", {"x": 1})
    with pytest.raises(FileExistsError):
        run.write_episode("k", {"x": 2})
    assert json.loads(run.episode_path("k").read_text()) == {"x": 1}
    assert not list(run.episodes.glob("*.tmp*"))


def test_fake_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "c.toml"
    src = open(cli.Path(__file__).parents[1] / "configs" / "v0.toml").read()
    src = src.replace("frozen = true", "frozen = false")
    config.write_text(
        src.replace("dev = [0, 10]", "dev = [0, 2]").replace("offline = [20000, 40]", "offline = [20000, 2]")
    )
    base = ["--config", str(config), "--fake-jev"]
    cli.main([*base, "run", "--policies", "jev_raw,lqr", "--conditions", "nominal"])
    cli.main([*base, "run", "--policies", "jev_raw,lqr", "--conditions", "nominal"])  # resume: nothing to do
    cli.main([*base, "run", "--protocol", "realtime", "--policies", "jev_semantic", "--episodes", "1"])
    cli.main([*base, "report"])
    root = tmp_path / "results" / "v0" / "dev-fake"
    assert len(list(root.glob("paused/episodes/*.json"))) == 4
    assert "jev_raw" in (root / "report.md").read_text()
    episode = next(root.glob("paused/episodes/*jev_raw*.json"))
    cli.main([*base, "render", str(episode), "--stride", "25"])
    assert episode.with_suffix(".gif").stat().st_size > 0
    cli.main([*base, "render", str(episode), "--stride", "25", "--out", str(tmp_path / "ep.mp4")])
    assert (tmp_path / "ep.mp4").stat().st_size > 0
    cli.main([*base, "player", "--policy", "jev_raw", "--out", str(tmp_path / "player.html")])
    html = (tmp_path / "player.html").read_text()
    assert "<title>CartPole</title>" in html and "/*__DATA__*/null" not in html and '"fake":true' in html
    cli.main([*base, "offline", "--n", "20"])
    assert list((tmp_path / "results" / "v0" / "dev-fake").glob("offline-*.json"))
    with pytest.raises(SystemExit):
        cli.main(["--config", str(config), "run", "--split", "test"])  # not frozen
