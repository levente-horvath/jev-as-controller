from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from jev_controller import cli
from jev_controller.actions import FIVE_ACTIONS, TWO_ACTIONS
from jev_controller.config import Protocol
from jev_controller.costs import LimitedClient, RequestLimitReached
from jev_controller.env import Condition
from jev_controller.offline import summarize
from jev_controller.policies.base import Decision
from jev_controller.policies.classic import PIDPolicy, lqr_gain
from jev_controller.policies.fake import FakeJevClient
from jev_controller.policies.jev import JevPolicy, semantic_labels
from jev_controller.runner import run_episode


@pytest.mark.parametrize("command", [["smoke"], ["latency"], ["offline"], ["run"]])
def test_paid_commands_block_before_loading_secrets(command, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Must not construct a client before explicit approval")

    monkeypatch.setattr(cli, "make_client", forbidden)
    with pytest.raises(SystemExit, match="2"):
        cli.main(command)


def test_request_cap_holds_under_concurrency():
    class Client:
        def system_one(self, **kwargs):
            return True

    client = LimitedClient(Client(), 7)

    def send(_):
        try:
            return client.system_one()
        except RequestLimitReached:
            return False

    with ThreadPoolExecutor(8) as pool:
        assert sum(pool.map(send, range(40))) == 7
    assert client.calls == 7


def test_offline_probabilities_use_action_labels_not_dict_order():
    items = [{"best": 0, "acceptable": [True, False], "decisive": True}]
    answers = {"jev_raw": [{"action": 0, "probabilities": {"push_right": 0.1, "push_left": 0.9}}]}
    result = summarize(items, answers, 2)["jev_raw"]
    assert result["mean_prob_best_all"] == 0.9


def test_rate_limit_wait_included_in_decision_latency(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("jev_controller.policies.jev.time.perf_counter", lambda: clock[0])

    class Limiter:
        def wait(self):
            clock[0] += 0.12
            return 0.12

    policy = JevPolicy(FakeJevClient(), TWO_ACTIONS, "raw", model="fake", limiter=Limiter())
    assert policy.act(np.zeros(4)).latency_s == 0.12


def test_semantic_zero_crossings_are_not_recovery():
    labels = semantic_labels(np.array([0, 0.2, 0, 0.2]))
    assert labels["pole_trend"] == "crossing upright"
    assert labels["cart_trend"] == "crossing centre"
    assert semantic_labels(np.array([0, 0, 0.1, 0]))["pole_trend"] == "steady"


@pytest.mark.parametrize("actions", [TWO_ACTIONS, FIVE_ACTIONS])
def test_pid_stabilizes_nominal_local_fixture(actions):
    policy = PIDPolicy(actions)
    for seed in range(5):
        assert run_episode(policy, Condition("nominal"), Protocol("paused", 1, False), seed)["success"]


def test_lqr_gain_respects_action_hold_period():
    assert not np.allclose(lqr_gain(), lqr_gain(decision_steps=5))


def test_delay_record_never_changes_state_before_action_arrives():
    class Alternating:
        name = "fixture"
        actions = FIVE_ACTIONS

        def reset(self, seed):
            self.i = 0

        def act(self, obs):
            self.i += 1
            return Decision(2 if self.i == 1 else 4, 0.06)

    record = run_episode(Alternating(), Condition("nominal"), Protocol("realtime", 2, True), 0)
    assert record["decisions"][1]["apply_step"] == 5
    assert record["actions"][:5] == [2] * 5
    assert record["actions"][5] == 4
