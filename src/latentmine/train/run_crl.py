"""Programmatic CRL training entrypoint.

Bypasses `run.py`'s tyro CLI and `jaxgcrl.utils.env.create_env` - both bake
upstream's env names into a `Literal` at import time - by constructing the env
and the `CRL` / `RunConfig` dataclasses ourselves and calling `train_fn`
directly (LLD section 4.2).

    python -m latentmine.train.run_crl --maze two_rooms --env simple --seed 1
    python -m latentmine.train.run_crl --maze spiral --env ant --dry-run

`--dry-run` resolves and validates the whole configuration, prints its derived
quantities and memory footprint, and exits **without importing JAX**. On a
laptop that is a one-second check that a multi-hour run is set up correctly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..mazes import layouts
from . import manifest
from .presets import ARCH_PRESETS, CONTRASTIVE_LOSS_FNS, ENERGY_FNS, ConfigError, RunSpec, make_run_spec

# Reference point for the utd ratio: upstream's own defaults. Ours is lower
# whenever we shorten episodes for the laptop budget, and seeing the two side
# by side is what makes that visible rather than accidental.
UPSTREAM_UTD_REFERENCE = (512 * 1000 * 1 / 256) / (512 * 62)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_crl",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--maze", required=True, choices=list(layouts.names()))
    p.add_argument("--env", default="simple", choices=("simple", "ant"))
    p.add_argument("--preset", default="deep", choices=sorted(ARCH_PRESETS))
    p.add_argument("--seed", type=int, default=1)

    b = p.add_argument_group("budget (defaults are provisional - see LLD 5.6)")
    b.add_argument("--total-env-steps", type=int)
    b.add_argument("--episode-length", type=int)
    b.add_argument("--num-envs", type=int)
    b.add_argument("--num-evals", type=int, help="also the checkpoint cadence: one per eval")
    b.add_argument("--num-eval-envs", type=int)

    a = p.add_argument_group("algorithm")
    a.add_argument("--energy-fn", choices=list(ENERGY_FNS))
    a.add_argument("--contrastive-loss-fn", choices=list(CONTRASTIVE_LOSS_FNS))
    a.add_argument("--batch-size", type=int)
    a.add_argument("--train-step-multiplier", type=int)
    a.add_argument("--backend", help="brax backend; defaults to spring for both maze envs")
    a.add_argument("--eval-goal-region", help="restrict eval goals to one region label")

    o = p.add_argument_group("output")
    o.add_argument("--runs-dir", type=Path, default=Path("runs"))
    o.add_argument("--wandb", action="store_true", help="log to wandb (offline by default)")
    o.add_argument("--wandb-mode", default="offline", choices=("online", "offline"))
    o.add_argument("--wandb-project", default="crl-latent-mining")
    o.add_argument("--dry-run", action="store_true", help="validate and report, importing no JAX")
    o.add_argument("--json", action="store_true", help="with --dry-run, emit the manifest as JSON")
    return p


def spec_from_args(args: argparse.Namespace) -> RunSpec:
    overrides = {
        k: getattr(args, k)
        for k in (
            "total_env_steps",
            "episode_length",
            "num_envs",
            "num_evals",
            "num_eval_envs",
            "energy_fn",
            "contrastive_loss_fn",
            "batch_size",
            "train_step_multiplier",
            "backend",
            "eval_goal_region",
        )
        if getattr(args, k) is not None
    }
    return make_run_spec(args.maze, args.env, args.preset, args.seed, **overrides)


def describe(spec: RunSpec) -> str:
    """Human-readable resolution of a run spec. What `--dry-run` prints."""
    arch, env_spec, maze = spec.arch, spec.env_spec, spec.maze_spec
    crash_pct = 100.0 * spec.env_steps_per_epoch / max(spec.actual_total_env_steps, 1)
    lines = [
        f"run_id          {spec.run_id}",
        "",
        f"maze            {maze.name}  {maze.shape[0]}x{maze.shape[1]} grid, "
        f"{len(maze.free_cells())} free cells, scaling {maze.scaling:g}",
        f"env             {env_spec.cls} on backend '{spec.resolved_backend}'  "
        f"(state {env_spec.state_dim}, action {env_spec.action_size}, "
        f"goal {env_spec.goal_size}, obs {env_spec.obs_size})",
        f"arch            {arch.name}: depth {arch.depth} x width {arch.width}, "
        f"repr_dim {arch.repr_dim}, skip {arch.skip_connections}, "
        f"layernorm {arch.use_ln}, {'relu' if arch.use_relu else 'swish'}",
        f"critic          energy '{spec.energy_fn}', loss '{spec.contrastive_loss_fn}', "
        f"gamma {spec.discounting}",
        "",
        f"total steps     {spec.actual_total_env_steps:,} actually run "
        f"(requested {spec.total_env_steps:,}; integer division truncates)",
        f"  prefill       {spec.num_prefill_env_steps:,} "
        f"({100.0 * spec.num_prefill_env_steps / max(spec.actual_total_env_steps, 1):.1f}%), "
        f"re-paid on every resume unless the buffer is checkpointed",
        f"  per epoch     {spec.env_steps_per_epoch:,} env steps "
        f"({spec.num_training_steps_per_epoch} train steps x "
        f"{spec.env_steps_per_actor_step:,})",
        f"  epochs        {spec.num_evals}  -> a crash costs at most one epoch, {crash_pct:.1f}% of the run",
        "",
        f"parallel envs   {spec.num_envs}  x episode_length {spec.episode_length}"
        f"   (constraint: {spec.num_envs} * {spec.episode_length - 1} "
        f"= {spec.num_envs * (spec.episode_length - 1):,} % {spec.batch_size} == 0 OK)",
        f"replay buffer   {spec.replay_buffer_bytes / 1e6:,.0f} MB "
        f"({spec.max_replay_size:,} x {spec.num_envs} x {env_spec.transition_floats} float32)",
        f"utd ratio       {spec.utd_ratio:.4f}   (upstream defaults give {UPSTREAM_UTD_REFERENCE:.4f})",
    ]
    if spec.utd_ratio < 0.75 * UPSTREAM_UTD_REFERENCE:
        lines.append(
            "                NOTE: well below upstream's. Shortening episodes cuts gradient\n"
            "                updates per env step; consider --train-step-multiplier 2."
        )
    if spec.eval_goal_region:
        lines.append(f"eval goals      restricted to region '{spec.eval_goal_region}'")
    return "\n".join(lines)


def progress_printer(spec: RunSpec, run_dir: Path, wandb_run: Any | None):
    """`progress_fn` for `train_fn`: prints, appends to metrics.jsonl, and
    forwards to wandb when enabled.

    Note the signature upstream calls this with - it receives the actor
    parameters and nothing else of the training state, which is exactly why
    resume cannot be bolted on from here (LLD section 5.5).
    """
    metrics_path = run_dir / "metrics.jsonl"

    def progress(step: int, metrics: dict, make_policy=None, params=None, env=None, do_render=False):
        flat = {k: float(v) for k, v in metrics.items() if _is_scalar(v)}
        record = {"step": int(step), **flat}
        with metrics_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

        pct = 100.0 * step / max(spec.actual_total_env_steps, 1)
        summary = "  ".join(
            f"{k.split('/')[-1]}={flat[k]:.4g}"
            for k in (
                "eval/episode_success",
                "eval/episode_dist",
                "training/critic_loss",
                "training/categorical_accuracy",
                "training/sps",
            )
            if k in flat
        )
        print(f"[{pct:5.1f}%] step {step:>12,}  {summary}", flush=True)

        if wandb_run is not None:
            wandb_run.log(record, step=int(step))

    return progress


def _is_scalar(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return getattr(value, "ndim", 0) == 0


def train(spec: RunSpec, runs_dir: Path, wandb_enabled: bool, wandb_mode: str, wandb_project: str) -> Path:
    """Run training. Imports JAX, so it is never reached under `--dry-run`."""
    from jaxgcrl.agents.crl import CRL
    from jaxgcrl.utils.config import RunConfig

    from .envs import build_envs

    run_dir = runs_dir / spec.run_id
    ckpt_dir = run_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_env, eval_env = build_envs(spec)
    arch = spec.arch

    agent = CRL(
        policy_lr=spec.policy_lr,
        critic_lr=spec.critic_lr,
        alpha_lr=spec.alpha_lr,
        batch_size=spec.batch_size,
        discounting=spec.discounting,
        logsumexp_penalty_coeff=spec.logsumexp_penalty_coeff,
        train_step_multiplier=spec.train_step_multiplier,
        max_replay_size=spec.max_replay_size,
        min_replay_size=spec.min_replay_size,
        unroll_length=spec.unroll_length,
        # Upstream's names invert the usual convention: n_hidden is depth,
        # h_dim is width.
        h_dim=arch.width,
        n_hidden=arch.depth,
        skip_connections=arch.skip_connections,
        use_relu=arch.use_relu,
        repr_dim=arch.repr_dim,
        use_ln=arch.use_ln,
        contrastive_loss_fn=spec.contrastive_loss_fn,
        energy_fn=spec.energy_fn,
    )
    # `RunConfig.env` is typed Literal[legal_envs] but flax.struct.dataclass
    # does not enforce annotations, and train_fn never reads the field - it
    # only reaches create_env, which we bypass. Passing our maze name keeps the
    # value honest for anything that logs the config.
    config = RunConfig(
        env=spec.maze,
        total_env_steps=spec.total_env_steps,
        episode_length=spec.episode_length,
        num_envs=spec.num_envs,
        num_eval_envs=spec.num_eval_envs,
        action_repeat=spec.action_repeat,
        num_evals=spec.num_evals,
        seed=spec.seed,
        exp_name=spec.run_id,
        log_wandb=False,  # we drive wandb ourselves; upstream's path assumes run.py
        visualization_interval=spec.visualization_interval,
        checkpoint_logdir=str(ckpt_dir),
    )
    agent.check_config(config)

    wandb_run = None
    if wandb_enabled:
        import wandb

        wandb_run = wandb.init(
            project=wandb_project,
            group=spec.maze,
            name=spec.run_id,
            mode=wandb_mode,
            config=manifest.build(spec, run_dir),
        )

    manifest.write(spec, run_dir, extra={"wandb_run_id": getattr(wandb_run, "id", None)})
    print(describe(spec))
    print(f"\nwriting to {run_dir}\n", flush=True)

    _, params, metrics = agent.train_fn(
        config=config,
        train_env=train_env,
        eval_env=eval_env,
        progress_fn=progress_printer(spec, run_dir, wandb_run),
    )
    if wandb_run is not None:
        wandb_run.finish()
    print(f"\ndone: {run_dir}")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        spec = spec_from_args(args)
    except ConfigError as exc:
        print(f"configuration error:\n{exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        if args.json:
            print(json.dumps(manifest.build(spec, args.runs_dir / spec.run_id), indent=2))
        else:
            print(describe(spec))
            print("\n(dry run - no JAX imported, nothing written)")
        return 0

    train(spec, args.runs_dir, args.wandb, args.wandb_mode, args.wandb_project)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
