"""Throughput probe: confirm the accelerator is real, then size `num_envs`.

Two jobs, in order of importance (LLD section 5.6):

1. **Confirm JAX is on the GPU.** It falls back to CPU silently, and a run
   that should take twenty minutes then takes a day - discovered only when it
   fails to finish. The probe prints the devices and says plainly whether an
   accelerator was found.
2. **Pick `num_envs` for the card.** Throughput is not monotone in it: too few
   underuses the GPU, too many spills memory. One measurement beats guessing,
   and the parameter multiplies the cost of every run afterwards.

    python -m latentmine.train.probe --env simple --num-envs 256,512,1024

The first epoch of each configuration is discarded, because it pays for JIT
compilation and would otherwise dominate a short measurement.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from ..mazes import layouts
from .presets import ARCH_PRESETS, BUDGET_PROFILES, ConfigError, RunSpec, make_run_spec, suggest_num_envs

# Epochs per configuration: the first is thrown away as compilation.
PROBE_EPOCHS = 4


@dataclass
class ProbeResult:
    num_envs: int
    steps_per_second: float
    env_steps: int
    seconds: float
    buffer_mb: float
    error: str | None = None


def describe_devices() -> tuple[str, bool]:
    """`(human-readable summary, an accelerator was found)`."""
    import jax

    devices = jax.devices()
    kinds = sorted({d.platform for d in devices})
    accelerated = any(d.platform != "cpu" for d in devices)
    summary = f"{len(devices)} device(s): " + ", ".join(f"{d.platform}:{d.id}" for d in devices)
    if not accelerated:
        summary += (
            "\n  WARNING: no accelerator found - JAX is on CPU.\n"
            "  A GPU run falls back to CPU silently. If this box has an NVIDIA card, the\n"
            "  CUDA jaxlib is probably not installed:\n"
            "    pip install -e third_party/JaxGCRL \\\n"
            "      -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
        )
    else:
        summary += f"   (platforms: {', '.join(kinds)})"
    return summary, accelerated


def probe_one(spec: RunSpec) -> ProbeResult:
    """Time `PROBE_EPOCHS` epochs of a real training loop."""
    from .crl_resumable import ResumableCRL
    from .envs import build_envs
    from .run_crl import build_agent, build_config

    train_env, eval_env = build_envs(spec)
    agent: ResumableCRL = build_agent(spec)
    config = build_config(spec, checkpoint_dir=None)

    timings: list[tuple[float, int]] = []
    last = {"t": time.time(), "step": 0}

    def progress(step, metrics, *args, **kwargs):
        now = time.time()
        timings.append((now - last["t"], int(step) - last["step"]))
        last["t"], last["step"] = now, int(step)

    agent.train_fn(config=config, train_env=train_env, eval_env=eval_env, progress_fn=progress)

    # Drop the first epoch: it carries JIT compilation.
    measured = timings[1:] or timings
    seconds = sum(t for t, _ in measured)
    steps = sum(s for _, s in measured)
    return ProbeResult(
        num_envs=spec.num_envs,
        steps_per_second=steps / seconds if seconds > 0 else float("nan"),
        env_steps=steps,
        seconds=seconds,
        buffer_mb=spec.replay_buffer_bytes / 1e6,
    )


def format_report(results: list[ProbeResult], spec_for_budget: RunSpec) -> str:
    budget = spec_for_budget.actual_total_env_steps
    lines = [
        "",
        f"{'num_envs':>9}  {'steps/s':>12}  {'buffer':>9}  {'est. run':>12}",
        f"{'-' * 9}  {'-' * 12}  {'-' * 9}  {'-' * 12}",
    ]
    for r in results:
        if r.error:
            lines.append(f"{r.num_envs:>9}  {r.error}")
            continue
        hours = budget / r.steps_per_second / 3600 if r.steps_per_second > 0 else float("inf")
        est = f"{hours * 60:.0f} min" if hours < 1.5 else f"{hours:.1f} h"
        lines.append(f"{r.num_envs:>9}  {r.steps_per_second:>12,.0f}  {r.buffer_mb:>7.0f} MB  {est:>12}")
    ok = [r for r in results if not r.error]
    if ok:
        best = max(ok, key=lambda r: r.steps_per_second)
        lines += [
            "",
            f"fastest: --num-envs {best.num_envs} at {best.steps_per_second:,.0f} steps/s",
            f"est. wall clock for the '{spec_for_budget.profile}' budget "
            f"({budget:,} steps): {budget / best.steps_per_second / 3600:.1f} h per run",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="probe", description=__doc__)
    parser.add_argument("--maze", default="two_rooms", choices=list(layouts.names()))
    parser.add_argument("--env", default="simple", choices=("simple", "ant"))
    parser.add_argument("--preset", default="deep", choices=sorted(ARCH_PRESETS))
    parser.add_argument("--profile", default="gpu", choices=sorted(BUDGET_PROFILES))
    parser.add_argument(
        "--num-envs",
        default="256,512,1024",
        help="comma-separated values to compare",
    )
    parser.add_argument("--epochs", type=int, default=PROBE_EPOCHS)
    args = parser.parse_args(argv)

    summary, accelerated = describe_devices()
    print(summary)
    if not accelerated:
        print("\nProbing anyway - the numbers below are CPU numbers.\n")

    reference = make_run_spec(args.maze, args.env, args.preset, profile=args.profile)
    results = []
    for value in [int(v) for v in args.num_envs.split(",")]:
        try:
            spec = reference.evolve(
                num_envs=value,
                num_evals=args.epochs,
                # Enough steps that each epoch is a real unit of work, but not
                # so many that the probe becomes a training run.
                total_env_steps=max(
                    value * reference.unroll_length * args.epochs * 2,
                    reference.min_replay_size * value * 2,
                ),
            )
        except ConfigError as exc:
            hint = suggest_num_envs(reference.episode_length, reference.batch_size, value)
            results.append(
                ProbeResult(
                    value, 0.0, 0, 0.0, 0.0, error=f"invalid ({exc.args[0].splitlines()[0]}); try {hint}"
                )
            )
            continue
        print(f"\n--- num_envs={value} ---", flush=True)
        results.append(probe_one(spec))

    print(format_report(results, reference))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
