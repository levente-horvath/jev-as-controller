import numpy as np
from gymnasium.envs.classic_control.cartpole import CartPoleEnv

from jev_controller.actions import FIVE_ACTIONS, TWO_ACTIONS
from jev_controller.env import CartPole, Condition, dynamics, failed


def test_dynamics_match_gymnasium_step_for_step():
    ref = CartPoleEnv()
    ref.reset(seed=3)
    rng = np.random.default_rng(0)
    state = ref.state.copy()
    for _ in range(200):
        action = int(rng.integers(2))
        ref.step(action)
        state = dynamics(state, 10.0 if action == 1 else -10.0)
        np.testing.assert_allclose(state, ref.state, rtol=0, atol=1e-12)
        if failed(state):
            break


def test_five_action_extremes_match_two_action_env():
    cond = Condition("nominal")
    two, five = CartPole(cond, TWO_ACTIONS), CartPole(cond, FIVE_ACTIONS)
    assert np.array_equal(two.reset(7), five.reset(7))
    rng = np.random.default_rng(1)
    for _ in range(100):
        a = int(rng.integers(2))
        s2, t2, *_ = two.step(a)
        s5, t5, *_ = five.step(0 if a == 0 else 4)
        assert np.array_equal(s2, s5) and t2 == t5
        if t2:
            break


def test_impulses_are_seeded_and_controller_independent():
    cond = Condition("impulse", impulse_steps=(3, 6), impulse_min=0.5, impulse_max=1.0)
    assert cond.impulses(5) == cond.impulses(5)
    assert cond.impulses(5) != cond.impulses(6)
    assert all(0.5 <= abs(v) <= 1.0 for v in cond.impulses(5).values())
    env = CartPole(cond, TWO_ACTIONS)
    env.reset(5)
    kicks = [env.step(0)[3] for _ in range(6)]
    assert kicks[2] == cond.impulses(5)[3] and kicks[5] == cond.impulses(5)[6]
    assert kicks.count(0.0) == 4


def test_quantizer():
    assert FIVE_ACTIONS.quantize(-100) == 0
    assert FIVE_ACTIONS.quantize(3.0) == 3
    assert FIVE_ACTIONS.quantize(0.4) == 2
    assert TWO_ACTIONS.quantize(0.1) == 1
    assert FIVE_ACTIONS.neutral == 2
