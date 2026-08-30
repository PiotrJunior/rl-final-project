"""Run manifests.

Upstream checkpoints are a bare pickled 3-tuple with no record of the
architecture that produced them, and rebuilding the encoders needs
`repr_dim`, `h_dim`, `n_hidden`, `skip_connections`, `use_relu` and `use_ln`
(LLD section 2.7). Upstream's own `args.pkl` carries those, but it is a pickle
of a `flax.struct.dataclass`, so it embeds the defining module path and stops
loading the moment upstream moves a class. We write our own JSON instead and
treat a checkpoint without one as unusable.

The maze's ASCII grid is stored in the manifest too, so a run stays
interpretable even if the registry in `mazes/layouts.py` later changes: the
analysis reconstructs the exact maze that was trained on rather than the one
that currently answers to that name.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .presets import RunSpec

MANIFEST_NAME = "manifest.json"

# Bump when a field changes meaning. `load` refuses a manifest it does not
# understand rather than guessing.
SCHEMA_VERSION = 1


class ManifestError(RuntimeError):
    pass


def _git_sha(repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build(spec: RunSpec, run_dir: Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble the manifest for a run. Pure - does not touch the filesystem
    except to read git SHAs."""
    arch, env_spec, maze = spec.arch, spec.env_spec, spec.maze_spec
    root = _repo_root()
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": spec.run_id,
        "run_dir": str(run_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        # Enough to rebuild both encoders without any other file.
        "arch": {
            "preset": arch.name,
            "repr_dim": arch.repr_dim,
            "h_dim": arch.width,
            "n_hidden": arch.depth,
            "skip_connections": arch.skip_connections,
            "use_relu": arch.use_relu,
            "use_ln": arch.use_ln,
        },
        # Input widths of phi and psi, so the encoders can be initialised
        # without constructing the env.
        "dims": {
            "state_dim": env_spec.state_dim,
            "action_size": env_spec.action_size,
            "goal_size": env_spec.goal_size,
            "obs_size": env_spec.obs_size,
            "sa_encoder_input": env_spec.state_dim + env_spec.action_size,
            "g_encoder_input": env_spec.goal_size,
        },
        "crl": {
            "energy_fn": spec.energy_fn,
            "contrastive_loss_fn": spec.contrastive_loss_fn,
            "discounting": spec.discounting,
            "batch_size": spec.batch_size,
            "unroll_length": spec.unroll_length,
            "min_replay_size": spec.min_replay_size,
            "max_replay_size": spec.max_replay_size,
            "logsumexp_penalty_coeff": spec.logsumexp_penalty_coeff,
            "train_step_multiplier": spec.train_step_multiplier,
            "policy_lr": spec.policy_lr,
            "critic_lr": spec.critic_lr,
            "alpha_lr": spec.alpha_lr,
        },
        "run": {
            "env": spec.env,
            "backend": spec.resolved_backend,
            "seed": spec.seed,
            "total_env_steps": spec.total_env_steps,
            "effective_total_env_steps": spec.effective_total_env_steps,
            "actual_total_env_steps": spec.actual_total_env_steps,
            "episode_length": spec.episode_length,
            "num_envs": spec.num_envs,
            "num_evals": spec.num_evals,
            "num_eval_envs": spec.num_eval_envs,
            "action_repeat": spec.action_repeat,
            "eval_goal_region": spec.eval_goal_region,
        },
        "derived": {
            "env_steps_per_actor_step": spec.env_steps_per_actor_step,
            "num_prefill_env_steps": spec.num_prefill_env_steps,
            "prefill_env_steps_actual": spec.prefill_env_steps_actual,
            "num_training_steps_per_epoch": spec.num_training_steps_per_epoch,
            "env_steps_per_epoch": spec.env_steps_per_epoch,
            "replay_buffer_bytes": spec.replay_buffer_bytes,
            "utd_ratio": spec.utd_ratio,
        },
        # The maze as trained on, not as currently named.
        "maze": {
            "name": maze.name,
            "grid": list(maze.grid),
            "regions": list(maze.regions) if maze.regions else None,
            "scaling": maze.scaling,
            "shape": list(maze.shape),
            "n_free_cells": len(maze.free_cells()),
        },
        "provenance": {
            "latentmine_sha": _git_sha(root),
            "jaxgcrl_sha": _git_sha(root / "third_party" / "JaxGCRL"),
        },
        "spec": asdict(spec),
        **(extra or {}),
    }


def write(spec: RunSpec, run_dir: Path, extra: dict[str, Any] | None = None) -> Path:
    """Write `<run_dir>/manifest.json`."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / MANIFEST_NAME
    payload = build(spec, run_dir, extra)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return path


def load(run_dir: Path) -> dict[str, Any]:
    """Read a manifest, refusing an unknown schema version."""
    path = Path(run_dir)
    if path.is_dir():
        path = path / MANIFEST_NAME
    if not path.exists():
        raise ManifestError(
            f"no manifest at {path}. A checkpoint without its manifest cannot be loaded - "
            "the pickle records no architecture. See LLD section 2.7."
        )
    payload = json.loads(path.read_text())
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ManifestError(
            f"{path}: schema_version {version!r}, expected {SCHEMA_VERSION}. "
            "Refusing to guess at the layout of an older manifest."
        )
    return payload


def spec_from_manifest(payload: dict[str, Any]) -> RunSpec:
    """Rebuild the `RunSpec` a manifest was written from."""
    return RunSpec(**payload["spec"])


def config_hash(payload: dict[str, Any]) -> str:
    """Stable digest of the fields that must match for a resume to be valid.

    Excludes `created_at`, `run_dir` and provenance - a rebuilt checkout with
    a different SHA may still resume a run, but a changed architecture, maze or
    seed may not (LLD section 5.5).
    """
    import hashlib

    material = {
        "arch": payload["arch"],
        "dims": payload["dims"],
        "crl": payload["crl"],
        "run": payload["run"],
        "maze": payload["maze"],
        "schema_version": payload["schema_version"],
    }
    blob = json.dumps(material, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
