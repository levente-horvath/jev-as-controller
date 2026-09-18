# Frozen CartPole test

360 episodes; 30 held-out seeds per condition and controller.
Five actions, paused physics, 500 steps (10 simulated seconds).

| Controller | Nominal success | Nominal mean steps | Impulse success | Impulse mean steps |
|---|---:|---:|---:|---:|
| jev_raw | 3/30 | 158.4 | 0/30 | 123.3 |
| jev_semantic | 6/30 | 181.3 | 0/30 | 85.7 |
| lqr | 30/30 | 500.0 | 30/30 | 500.0 |
| pid | 30/30 | 500.0 | 30/30 | 500.0 |
| heuristic | 30/30 | 500.0 | 9/30 | 359.8 |
| random | 0/30 | 24.5 | 0/30 | 24.5 |

## Paired differences

Mean step differences on matching seeds; bootstrap 95% intervals.

| Condition | Difference | Mean | 95% interval |
|---|---|---:|---:|
| nominal | jev_semantic - jev_raw | 22.8 | -35.9 to 82.6 |
| nominal | jev_semantic - lqr | -318.7 | -378.4 to -255.7 |
| nominal | jev_raw - lqr | -341.6 | -388.9 to -286.8 |
| impulse | jev_semantic - jev_raw | -37.6 | -73.1 to -7.2 |
| impulse | jev_semantic - lqr | -414.3 | -425.7 to -402.9 |
| impulse | jev_raw - lqr | -376.7 | -408.5 to -340.3 |
