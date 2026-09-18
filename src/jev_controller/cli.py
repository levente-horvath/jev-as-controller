"""`jev-demo` command line: smoke test, latency pilot, offline check, closed-loop runs, report, render."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from jev_controller import offline as offline_mod
from jev_controller import report as report_mod
from jev_controller.actions import FIVE_ACTIONS, TWO_ACTIONS, action_set
from jev_controller.config import SPLITS, Config, Protocol, load_dotenv
from jev_controller.costs import LimitedClient, estimate
from jev_controller.player import build_player
from jev_controller.policies.base import Policy
from jev_controller.policies.classic import FixedLatency, HeuristicPolicy, LQRPolicy, PIDPolicy, RandomPolicy
from jev_controller.policies.fake import FakeJevClient
from jev_controller.policies.jev import JevPolicy, RateLimiter
from jev_controller.records import RunDir, write_once
from jev_controller.render import render_episode
from jev_controller.runner import episode_key, run_episode

ALL_POLICIES = ("jev_raw", "jev_semantic", "lqr", "pid", "heuristic", "random")


def make_client(fake: bool, *, allow_paid: bool = False, max_calls: int | None = None) -> Any:
    if fake:
        return FakeJevClient(seed=0)
    if not allow_paid or max_calls is None or max_calls <= 0:
        sys.exit("Paid calls blocked. Review `estimate` first; approval requires --allow-paid --max-calls N.")
    load_dotenv()
    from typesafe_sdk import RetryPolicy, TypeSafeClient

    return LimitedClient(TypeSafeClient(retry=RetryPolicy(max_retries=0)), max_calls)


def client_for(args: argparse.Namespace) -> Any:
    return make_client(args.fake_jev, allow_paid=args.allow_paid, max_calls=args.max_calls)


def make_policy(name: str, cfg: Config, decision_steps: int, client: Any, limiter: RateLimiter | None) -> Policy:
    actions = action_set(cfg.n_actions)
    match name:
        case "lqr":
            return LQRPolicy(actions, decision_steps)
        case "pid":
            return PIDPolicy(actions, decision_steps)
        case "heuristic":
            return HeuristicPolicy(actions)
        case "random":
            return RandomPolicy(actions)
        case "jev_raw" | "jev_semantic":
            if isinstance(client, FakeJevClient):
                client = FakeJevClient()
            return JevPolicy(
                client,
                actions,
                name.removeprefix("jev_"),  # type: ignore[arg-type]
                model=cfg.model,
                decision_steps=decision_steps,
                timeout_s=cfg.jev["timeout_s"],
                limiter=limiter,
                instructions=cfg.jev.get("instructions", "base"),
            )
    raise ValueError(f"unknown policy {name!r}")


def prompt_hashes(cfg: Config, decision_steps: int) -> dict[str, str]:
    return {
        name: make_policy(name, cfg, decision_steps, None, None).prompt_hash  # type: ignore[attr-defined]
        for name in ("jev_raw", "jev_semantic")
    }


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def results_dir(cfg: Config, split: str, fake: bool) -> Path:
    return Path("results") / cfg.name / (f"{split}-fake" if fake else split)


def guard_split(cfg: Config, split: str, fake: bool) -> None:
    if split == "test" and not cfg.frozen and not fake:
        sys.exit("Refusing to use test seeds: set [run] frozen = true once prompts and thresholds are final.")


def csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        digest.update(str(path.relative_to(Path(__file__).parent)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def cmd_estimate(args: argparse.Namespace, cfg: Config) -> None:
    from math import ceil

    policies = csv(args.policies)
    conditions = csv(args.conditions)
    if set(policies) - set(ALL_POLICIES) or set(conditions) - set(cfg.conditions()):
        sys.exit("Unknown policy or condition; estimate refused.")
    period = cfg.protocols()[args.protocol].decision_steps
    calls = sum(p.startswith("jev") for p in policies) * len(conditions) * args.episodes * ceil(500 / period)
    print(estimate(calls))
    print(f"Scope: {args.episodes} seeds/condition, {','.join(conditions)}, {','.join(policies)}, {args.protocol}.")
    print("Fresh full-horizon episodes assumed; early failures/resume reduce calls. Smoke: 2 extra calls.")


# --- commands ---------------------------------------------------------------------------------


def cmd_smoke(args: argparse.Namespace, cfg: Config) -> None:
    """One call per Jev variant from a small tilt, printing everything the API returned."""
    client = client_for(args)
    obs = np.array([0.0, 0.0, 0.03, 0.1])
    for name in ("jev_raw", "jev_semantic"):
        policy = make_policy(name, cfg, 1, client, None)
        d = policy.act(obs)
        shown = {k: v for k, v in d.diagnostics.items() if k != "jev_state"}
        print(f"{name}: action={policy.actions.labels[d.action]} latency={1000 * d.latency_s:.0f} ms")
        print(json.dumps(shown, indent=2, default=str))
        if (tokens := d.diagnostics.get("input_tokens")) is not None:
            print(f"  cost of this call: ${tokens * report_mod.PRICE_PER_MTOK / 1e6:.8f}")
    if args.show_state:
        print(json.dumps(make_policy("jev_semantic", cfg, 1, None, None).build_state(obs), indent=2))  # type: ignore[attr-defined]


def cmd_models(args: argparse.Namespace, cfg: Config) -> None:
    load_dotenv()
    from typesafe_sdk import TypeSafeClient

    with TypeSafeClient() as client:
        for m in client.models.list().models:
            print(f"{m.name}\t{m.release_date}\t{m.description}")


def cmd_latency(args: argparse.Namespace, cfg: Config) -> None:
    """Sequential calls from varied dev states; suggests the real-time decision period."""
    client = client_for(args)
    policy = make_policy("jev_semantic", cfg, 1, client, None)
    rng = np.random.default_rng(12345)
    latencies, fallbacks = [], 0
    for _ in range(args.n):
        d = policy.act(rng.uniform(-0.1, 0.1, 4))
        latencies.append(d.latency_s)
        fallbacks += bool(d.diagnostics.get("fallback"))
    ms = 1000 * np.array(latencies)
    p50, p95, p99 = np.percentile(ms, [50, 95, 99])
    print(f"n={args.n} fallbacks={fallbacks} p50={p50:.0f} ms p95={p95:.0f} ms p99={p99:.0f} ms max={ms.max():.0f} ms")
    print(f"suggested real-time decision_steps (p95 / 20 ms, rounded up): {int(np.ceil(p95 / 20))}")
    out = Path("results") / cfg.name / f"latency-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_once(out, {"latencies_s": latencies, "fallbacks": fallbacks, "fake": args.fake_jev})
    print(f"wrote {out}")


def cmd_offline(args: argparse.Namespace, cfg: Config) -> None:
    guard_split(cfg, args.split, args.fake_jev)
    o = cfg.offline
    actions = action_set(cfg.n_actions)
    condition = cfg.conditions()[o["condition"]]
    seeds = cfg.seeds(args.split)
    n = args.n or o["states"]
    states = offline_mod.sample_states(actions, condition, seeds, n, rng_seed=seeds[0])
    items = offline_mod.label_states(states, actions, o["commit_steps"], o["horizon_steps"])
    print(f"{len(items)} states labelled, {sum(it['decisive'] for it in items)} decisive")

    client = client_for(args)
    limiter = None if args.fake_jev else RateLimiter(cfg.max_rps)
    names = csv(args.policies)
    answers: dict[str, list[dict[str, Any] | None]] = {}
    for name in names:
        policies = [make_policy(name, cfg, o["commit_steps"], client, limiter) for _ in range(cfg.workers)]
        with ThreadPoolExecutor(cfg.workers) as pool:
            chunks = [items[i :: cfg.workers] for i in range(cfg.workers)]
            done = pool.map(
                lambda pc: [offline_mod.query(pc[0], it) for it in pc[1]], zip(policies, chunks, strict=True)
            )
            flat: list[Any] = [None] * len(items)
            for w, results in enumerate(done):
                for j, r in enumerate(results):
                    flat[w + j * cfg.workers] = r
        answers[name] = flat
    repeats: dict[str, float] = {}
    n_rep = int(len(items) * (args.repeat_fraction if args.repeat_fraction is not None else o["repeat_fraction"]))
    for name in [n for n in names if n.startswith("jev")]:
        policy = make_policy(name, cfg, o["commit_steps"], client, limiter)
        again = [offline_mod.query(policy, it) for it in items[:n_rep]]
        same = [a["action"] == b["action"] for a, b in zip(again, answers[name][:n_rep], strict=True) if a and b]
        if same:
            repeats[name] = float(np.mean(same))

    summary = offline_mod.summarize(items, answers, cfg.n_actions)
    for name, stats in summary.items():
        print(name, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()}))
    for name, rate in repeats.items():
        print(f"{name}: same action on repeat {rate:.1%} (n={n_rep})")
    out = results_dir(cfg, args.split, args.fake_jev) / f"offline-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        out,
        {
            "config_hash": cfg.hash,
            "prompt_hashes": prompt_hashes(cfg, o["commit_steps"]),
            "items": items,
            "answers": answers,
            "summary": summary,
            "repeat_agreement": repeats,
        },
    )
    print(f"wrote {out}")


def cmd_run(args: argparse.Namespace, cfg: Config) -> None:
    guard_split(cfg, args.split, args.fake_jev)
    protocol = cfg.protocols()[args.protocol]
    conditions = {k: v for k, v in cfg.conditions().items() if k in csv(args.conditions)}
    seeds = cfg.seeds(args.split)[: args.episodes or protocol.episodes or None]
    names = csv(args.policies)
    if set(names) - set(ALL_POLICIES) or set(csv(args.conditions)) - set(cfg.conditions()):
        sys.exit("Unknown policy or condition; no episodes started.")

    run = RunDir(results_dir(cfg, args.split, args.fake_jev) / protocol.name)
    run.bind(
        {
            "config": cfg.raw,
            "config_hash": cfg.hash,
            "prompt_hashes": prompt_hashes(cfg, protocol.decision_steps),
            "model": cfg.model,
            "sdk_version": version("typesafe-sdk"),
            "python": platform.python_version(),
            "git_commit": git_commit(),
            "source_hash": source_hash(),
            "fake": args.fake_jev,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    jobs = [
        (name, cond, seed)
        for name in names
        for cond in conditions.values()
        for seed in seeds
        if not run.done(episode_key(name, cond.name, protocol.name, seed))
    ]
    jev_jobs = sum(n.startswith("jev") for n, _, _ in jobs)
    print(
        f"{len(jobs)} episodes to run ({jev_jobs} with Jev, up to ~{jev_jobs * 500 // protocol.decision_steps} calls)"
    )
    if not jobs:
        return
    client = client_for(args) if jev_jobs else None
    limiter = None if args.fake_jev else RateLimiter(cfg.max_rps)
    # Real-time runs one episode at a time so concurrent requests cannot inflate each other's latency.
    workers = 1 if protocol.realtime else cfg.workers

    def job(name: str, cond: Any, seed: int) -> dict[str, Any]:
        policy = make_policy(name, cfg, protocol.decision_steps, client, limiter)
        record = run_episode(policy, cond, protocol, seed)
        record["config_hash"] = cfg.hash
        run.write_episode(episode_key(name, cond.name, protocol.name, seed), record)
        return record

    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(job, *j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            print(
                f"[{i}/{len(jobs)}] {r['policy']:<13} {r['condition']:<8} seed={r['seed']:<6} steps={r['steps']:<4} "
                f"{r['termination']:<13} fallbacks={r['n_fallbacks']}"
            )


def cmd_report(args: argparse.Namespace, cfg: Config) -> None:
    root = results_dir(cfg, args.split, args.fake_jev)
    episodes = [e for d in sorted(root.glob("*/episodes")) for e in RunDir(d.parent).load_episodes()]
    if not episodes:
        sys.exit(f"no episodes under {root}")
    pass_mark = cfg.raw["pass_mark"]
    rows = report_mod.summarize(episodes, pass_mark)
    md = report_mod.markdown(rows, pass_mark)
    (root / "report.md").write_text(md + "\n")
    report_mod.survival_plot(episodes, root / "survival.png")
    print(md)
    print(f"\nwrote {root / 'report.md'} and {root / 'survival.png'}")


def cmd_budget(args: argparse.Namespace, cfg: Config) -> None:
    """How much latency can a perfect local controller tolerate? No API calls."""
    condition = cfg.conditions()["nominal"]
    protocol = Protocol("realtime", 1, True)
    seeds = cfg.seeds("dev")
    grid_ms = [0, 20, 40, 60, 80, 100, 150, 200, 300, 400]
    curves = {}
    for actions in (TWO_ACTIONS, FIVE_ACTIONS):
        for base in (LQRPolicy(actions), HeuristicPolicy(actions)):
            points = []
            for ms in grid_ms:
                wrapped = FixedLatency(base, ms / 1000)
                rate = np.mean([run_episode(wrapped, condition, protocol, s)["success"] for s in seeds])
                points.append((ms, float(rate)))
            curves[f"{base.name} ({actions.name} actions)"] = points
            print(f"{base.name:<9} {actions.name}: " + "  ".join(f"{ms}ms={r:.0%}" for ms, r in points))
    measured = {}
    for path in sorted((Path("results") / cfg.name).glob("latency-*.json")):
        data = json.loads(path.read_text())
        if not data["fake"]:
            measured["Jev p50"] = 1000 * float(np.median(data["latencies_s"]))
    out = Path("results") / cfg.name / "latency_budget.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    report_mod.latency_budget_plot(curves, out, measured)
    print(f"wrote {out}")


def cmd_player(args: argparse.Namespace, cfg: Config) -> None:
    """Self-contained replay page from logged paused episodes, with LQR on the same seeds alongside."""
    run = RunDir(results_dir(cfg, args.split, args.fake_jev) / "paused")
    episodes = run.load_episodes()
    jev = [e for e in episodes if args.policy == "all" or e["policy"] == args.policy]
    if not jev:
        sys.exit(f"no {args.policy} episodes in {run.path}")
    # Select by condition/seed, never by outcome. Zero means all episodes.
    jev.sort(key=lambda e: (e["condition"], e["seed"]))
    if args.max_episodes:
        jev = jev[: args.max_episodes]
    compare = {(e["condition"], e["seed"]): e for e in episodes if e["policy"] == args.compare}
    manifest = json.loads((run.path / "manifest.json").read_text())
    out = Path(args.out or run.path.parent / f"player-{args.policy}.html")
    build_player(jev, compare, manifest, out, fake=args.fake_jev)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(jev)} episodes)")


def cmd_render(args: argparse.Namespace, cfg: Config) -> None:
    episode = json.loads(Path(args.episode).read_text())
    out = Path(args.out or Path(args.episode).with_suffix(".gif"))
    render_episode(episode, out, stride=args.stride)
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="jev-demo", description=__doc__)
    parser.add_argument("--config", default="configs/cartpole.toml")
    parser.add_argument("--fake-jev", action="store_true", help="use a cost-free fake client (pipeline testing only)")
    parser.add_argument("--allow-paid", action="store_true", help="explicitly approve paid API calls")
    parser.add_argument("--max-calls", type=int, help="hard request ceiling; required with --allow-paid")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("smoke", help="one call per Jev variant")
    p.add_argument("--show-state", action="store_true")
    p.set_defaults(func=cmd_smoke)

    sub.add_parser("models", help="list available models").set_defaults(func=cmd_models)

    p = sub.add_parser("latency", help="latency pilot")
    p.add_argument("--n", type=int, default=40)
    p.set_defaults(func=cmd_latency)

    p = sub.add_parser("offline", help="open-loop agreement with counterfactual labels")
    p.add_argument("--split", choices=SPLITS, default="dev")
    p.add_argument("--n", type=int)
    p.add_argument("--policies", default="jev_raw,jev_semantic,lqr,heuristic")
    p.add_argument("--repeat-fraction", type=float)
    p.set_defaults(func=cmd_offline)

    p = sub.add_parser("run", help="closed-loop episodes")
    p.add_argument("--split", choices=("dev", "test"), default="dev")
    p.add_argument("--protocol", default="paused")
    p.add_argument("--policies", default=",".join(ALL_POLICIES))
    p.add_argument("--conditions", default="nominal,impulse")
    p.add_argument("--episodes", type=int, help="limit seeds per condition")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="summary table and survival curves")
    p.add_argument("--split", choices=("dev", "test"), default="dev")
    p.set_defaults(func=cmd_report)

    sub.add_parser("budget", help="latency tolerance of local controllers (no API calls)").set_defaults(func=cmd_budget)

    p = sub.add_parser("player", help="interactive replay page from logged episodes")
    p.add_argument("--split", choices=("dev", "test"), default="dev")
    p.add_argument("--policy", default="all")
    p.add_argument("--compare", default="lqr")
    p.add_argument("--max-episodes", type=int, default=0, help="0 includes all episodes; selection never uses outcome")
    p.add_argument("--out")
    p.set_defaults(func=cmd_player)

    p = sub.add_parser("render", help="GIF or MP4 of a logged episode")
    p.add_argument("episode")
    p.add_argument("--out")
    p.add_argument("--stride", type=int, default=2)
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("estimate", help="offline API cost estimate; no credentials or network required")
    p.add_argument("--episodes", type=int, default=5, help="seeds per condition")
    p.add_argument("--protocol", default="paused")
    p.add_argument("--conditions", default="nominal,impulse")
    p.add_argument("--policies", default="jev_raw,jev_semantic")
    p.set_defaults(func=cmd_estimate)

    args = parser.parse_args(argv)
    if getattr(args, "episodes", None) is not None and args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.command in ("smoke", "latency", "offline", "run"):
        uses_jev = args.command in ("smoke", "latency") or any(n.startswith("jev") for n in csv(args.policies))
        if uses_jev and not args.fake_jev and (not args.allow_paid or not args.max_calls or args.max_calls <= 0):
            parser.error("Paid calls blocked: review estimate, then use --allow-paid --max-calls N after approval.")
    args.func(args, Config.load(args.config))


if __name__ == "__main__":
    main()
