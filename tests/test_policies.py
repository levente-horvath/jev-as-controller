import numpy as np
import pytest

from jev_controller.actions import FIVE_ACTIONS, TWO_ACTIONS
from jev_controller.config import Protocol
from jev_controller.env import Condition
from jev_controller.policies.base import Decision
from jev_controller.policies.classic import HeuristicPolicy, LQRPolicy, RandomPolicy
from jev_controller.policies.fake import FakeJevClient
from jev_controller.policies.jev import JevPolicy, semantic_labels
from jev_controller.runner import run_episode

PAUSED = Protocol("paused", 1, False)
NOMINAL = Condition("nominal")


@pytest.mark.parametrize("actions", [TWO_ACTIONS, FIVE_ACTIONS])
def test_baselines_solve_nominal(actions):
    for policy in (LQRPolicy(actions), HeuristicPolicy(actions)):
        assert all(run_episode(policy, NOMINAL, PAUSED, s)["success"] for s in range(5))
    assert not any(run_episode(RandomPolicy(actions), NOMINAL, PAUSED, s)["success"] for s in range(5))


def test_semantic_labels_describe_without_advising():
    labels = semantic_labels(np.array([1.5, -0.8, 0.07, 0.4]))
    assert labels == {
        "pole_lean": "moderate lean to the right",
        "pole_rotation": "moderate rotation toward the right",
        "pole_trend": "lean increasing",
        "cart_position": "far off centre, on the right",
        "cart_motion": "moderate, moving left",
        "cart_trend": "moving toward centre",
    }
    text = " ".join(labels.values())
    assert "push" not in text and "should" not in text


def test_jev_policy_maps_choice_and_logs_diagnostics():
    policy = JevPolicy(FakeJevClient(), FIVE_ACTIONS, "semantic", model="m")
    d = policy.act(np.array([0.0, 0.0, 0.1, 0.5]))
    assert policy.actions.labels[d.action] == "strong_right"
    assert set(d.diagnostics["probabilities"]) == set(FIVE_ACTIONS.labels)
    assert d.diagnostics["fallback"] is False and d.diagnostics["input_tokens"] > 0
    assert "labels" in d.diagnostics["jev_state"]
    raw = JevPolicy(FakeJevClient(), FIVE_ACTIONS, "raw", model="m")
    assert "labels" not in raw.build_state(np.zeros(4))
    assert raw.prompt_hash != policy.prompt_hash


def test_jev_timeout_holds_last_action():
    client = FakeJevClient(timeout_rate=0.0)
    policy = JevPolicy(client, TWO_ACTIONS, "raw", model="m")
    first = policy.act(np.array([0.0, 0.0, 0.1, 0.5]))
    client.timeout_rate = 1.0
    d = policy.act(np.array([0.0, 0.0, -0.1, -0.5]))
    assert d.diagnostics["fallback"] and d.action == first.action
    policy.reset(0)
    assert policy.act(np.zeros(4)).action == TWO_ACTIONS.neutral


class _SlowLQR(LQRPolicy):
    name = "slow_lqr"

    def __init__(self, actions, latency_s):
        super().__init__(actions)
        self.latency_s = latency_s

    def act(self, obs):
        d = super().act(obs)
        return Decision(d.action, self.latency_s, d.diagnostics)


def test_realtime_protocol_applies_actions_after_latency():
    protocol = Protocol("realtime", 5, True)
    ep = run_episode(_SlowLQR(FIVE_ACTIONS, 0.13), NOMINAL, protocol, 0)  # 130 ms -> 7 steps
    first, second = ep["decisions"][:2]
    assert first["apply_step"] == 0  # the episode starts when the controller is ready
    assert second["step"] == 5 and second["apply_step"] == 12
    third = ep["decisions"][2]
    assert third["step"] == 12  # next request once the previous one landed
    fast = run_episode(_SlowLQR(FIVE_ACTIONS, 0.0), NOMINAL, protocol, 0)
    assert [d["step"] for d in fast["decisions"][:3]] == [0, 5, 10]
