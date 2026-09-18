"""Replay a logged episode as a GIF: the cart-pole plus the controller's latest action probabilities.

Rendering uses only the stored trajectory, so it never calls the API.
"""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter  # noqa: E402

from jev_controller.env import DT, HALF_LENGTH, X_LIMIT  # noqa: E402


def render_episode(episode: dict[str, Any], out: Path, stride: int = 2) -> None:
    states = np.array(episode["states"])
    decisions = episode["decisions"]
    labels = episode["action_labels"]
    has_probs = any("probabilities" in d for d in decisions)
    impulses = {int(t): v for t, v in episode["impulses"]}
    # The decision in force at each step: the latest one whose apply_step has passed.
    applied = sorted(decisions, key=lambda d: d["apply_step"])

    fig, (ax, bars_ax) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [2.2, 1]})
    ax.set_xlim(-X_LIMIT - 0.3, X_LIMIT + 0.3)
    ax.set_ylim(-0.3, 1.4)
    ax.set_aspect("equal")
    ax.axhline(0, color="0.4", lw=1)
    for edge in (-X_LIMIT, X_LIMIT):
        ax.axvline(edge, color="tab:red", lw=1, ls="--", alpha=0.6)
    ax.set_yticks([])
    cart = plt.Rectangle((0, -0.1), 0.5, 0.2, color="0.25")
    ax.add_patch(cart)
    (pole,) = ax.plot([], [], lw=5, color="tab:orange", solid_capstyle="round")
    push_text = ax.text(0, 1.3, "", ha="center", fontsize=11, color="tab:red")
    title = ax.set_title("")

    bars = bars_ax.bar(range(len(labels)), np.zeros(len(labels)), color="tab:blue")
    bars_ax.set_xticks(range(len(labels)), [lab.replace("_", "\n") for lab in labels], fontsize=8)
    bars_ax.set_ylim(0, 1)
    bars_ax.set_title("Jev action probabilities" if has_probs else "action", fontsize=10)

    def frame(i: int):
        x, _, theta, _ = states[i]
        cart.set_x(x - 0.25)
        pole.set_data([x, x + 2 * HALF_LENGTH * np.sin(theta)], [0.1, 0.1 + 2 * HALF_LENGTH * np.cos(theta)])
        current = next((d for d in reversed(applied) if d["apply_step"] <= i), None)
        kick = next((v for t, v in impulses.items() if 0 <= i - t < 15), None)
        push_text.set_text("" if kick is None else f"push! {kick:+.2f} rad/s")
        latency = f" · {1000 * current['latency_s']:.0f} ms" if current else ""
        paused = " · sim paused while deciding" if episode["protocol"] == "paused" else ""
        title.set_text(f"{episode['policy']} · t = {i * DT:5.2f} s{latency}{paused}")
        for j, bar in enumerate(bars):
            if current and "probabilities" in current:
                bar.set_height(current["probabilities"][labels[j]])
            else:
                bar.set_height(1.0 if current and current["action"] == j else 0.0)
            bar.set_color("tab:green" if current and current["action"] == j else "tab:blue")
        return [cart, pole, push_text, title, *bars]

    frames = range(0, len(states), stride)
    anim = FuncAnimation(fig, frame, frames=frames, blit=False)
    fps = round(1 / (DT * stride))
    writer = FFMpegWriter(fps=fps, bitrate=2400) if out.suffix == ".mp4" else PillowWriter(fps=fps)
    anim.save(out, writer=writer, dpi=150 if out.suffix == ".mp4" else 100)
    plt.close(fig)
