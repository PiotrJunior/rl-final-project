"""Construct maze envs directly, bypassing upstream's registry.

`jaxgcrl.utils.env.create_env` dispatches on substrings of the env name and
`RunConfig.env` is typed `Literal[legal_envs]`, evaluated at import time, so
neither can carry a name we invented (LLD section 4.2). We therefore install
the `make_maze` patch and instantiate `SimpleMaze` / `AntMaze` ourselves.

This module imports JAX (via brax) and is the boundary where that starts.
"""

from __future__ import annotations

import importlib
from typing import Any

from ..mazes import layouts, register
from .presets import EnvSpec, RunSpec


class EnvDimensionMismatch(RuntimeError):
    """A constructed env disagrees with the dimensions `presets.ENV_SPECS`
    records. Those numbers are relied on by `--dry-run`, by the manifest and
    by the decoder's per-group error split, so a mismatch is fatal rather
    than something to paper over."""


def _check_dims(env: Any, env_spec: EnvSpec) -> None:
    observed = {
        "state_dim": int(env.state_dim),
        "action_size": int(env.action_size),
        "goal_size": int(len(env.goal_indices)),
        "observation_size": int(env.observation_size),
    }
    expected = {
        "state_dim": env_spec.state_dim,
        "action_size": env_spec.action_size,
        "goal_size": env_spec.goal_size,
        "observation_size": env_spec.obs_size,
    }
    if observed != expected:
        raise EnvDimensionMismatch(
            f"{env_spec.cls} dimensions changed upstream.\n"
            f"  expected {expected}\n  observed {observed}\n"
            f"Update presets.ENV_SPECS['{env_spec.name}'] and re-check anything that "
            "groups observation dimensions (LLD section 2.4)."
        )


def build_env(spec: RunSpec, maze_name: str | None = None) -> Any:
    """Instantiate one maze env for `spec`, defaulting to its training maze."""
    register.install()
    env_spec = spec.env_spec
    maze = register.registered_specs()[maze_name or spec.maze]
    module = importlib.import_module(env_spec.module)
    cls = getattr(module, env_spec.cls)
    env = cls(
        backend=spec.resolved_backend,
        maze_layout_name=maze.name,
        maze_size_scaling=maze.scaling,
    )
    _check_dims(env, env_spec)
    return env


def build_envs(spec: RunSpec) -> tuple[Any, Any]:
    """`(train_env, eval_env)`.

    The eval env is the same maze unless `eval_goal_region` restricts its
    goals, mirroring upstream's `*_MAZE_EVAL` layouts. The restricted variant
    is derived from the region overlay and registered under a derived name, so
    the two mazes cannot drift apart geometrically - only their goal sets
    differ.
    """
    train_env = build_env(spec)
    if spec.eval_goal_region is None:
        return train_env, train_env

    eval_spec = layouts.get(spec.maze).eval_variant(spec.eval_goal_region)
    register.register_spec(eval_spec, overwrite=True)
    return train_env, build_env(spec, maze_name=eval_spec.name)
