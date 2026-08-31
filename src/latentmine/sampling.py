"""Sampling positions and states to embed.

`psi` takes only `(x, y)`, so the goal side needs no rollouts at all: raster
the free cells and embed. `phi` needs full states, and for `AntMaze` those
must be physically plausible - a 29-D state drawn from a Gaussian is
off-manifold, and the latent map of it would describe the encoder's
extrapolation rather than the maze (LLD section 7.3).
"""

from __future__ import annotations

import numpy as np

from .mazes import geometry as geo
from .mazes.layouts import MazeSpec


def goal_grid(
    spec: MazeSpec,
    subdivisions: int = 1,
    jitter: float = 0.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Raster the free cells into world positions.

    Returns `(world_xy (N, 2), cell_index (N,))`, where `cell_index` indexes
    `spec.free_cells()` so every sample can be traced back to its cell and to
    the geodesic distance matrix.

    `subdivisions=k` places a `k x k` lattice inside each cell, which is what
    makes the latent map a smooth field rather than one point per cell.
    `jitter` adds uniform noise in units of a cell, to check the encoder is
    not keyed to exact lattice positions.
    """
    if subdivisions < 1:
        raise ValueError(f"subdivisions must be >= 1, got {subdivisions}")
    cells = geo.free_cell_array(spec)
    if subdivisions == 1:
        offsets = np.zeros((1, 2))
    else:
        ticks = (np.arange(subdivisions) + 0.5) / subdivisions - 0.5
        offsets = np.stack(np.meshgrid(ticks, ticks, indexing="ij"), -1).reshape(-1, 2)

    # (cells, offsets, 2) in cell units, then scaled to world.
    grid = cells[:, None, :] + offsets[None, :, :]
    index = np.repeat(np.arange(len(cells)), len(offsets))
    grid = grid.reshape(-1, 2)

    if jitter > 0:
        rng = np.random.default_rng(seed)
        grid = grid + rng.uniform(-jitter, jitter, size=grid.shape)

    return grid * spec.scaling, index


def cell_centres(spec: MazeSpec) -> np.ndarray:
    """World `(x, y)` of every free cell, in canonical order."""
    return geo.cells_to_world(geo.free_cell_array(spec), spec.scaling)


def teleport_states(
    base_obs: np.ndarray,
    targets_xy: np.ndarray,
    goal_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Move observations to new `(x, y)` positions, keeping everything else.

    The observation is `concat(qpos, qvel, goal_xy)` with position at indices
    0 and 1 (both maze envs keep it - `exclude_current_positions_from_observation`
    is False, which is also why `goal_indices` is `[0, 1]`). Translating only
    those two leaves the pose, joint angles and velocities untouched, so an
    Ant state sampled from a real rollout stays on the manifold of states the
    physics can actually produce.

    `base_obs` broadcasts: pass one observation to place the same pose
    everywhere, or `N` of them to vary pose across positions.
    """
    targets = np.asarray(targets_xy, dtype=np.float32).reshape(-1, 2)
    base = np.asarray(base_obs, dtype=np.float32)
    if base.ndim == 1:
        base = np.broadcast_to(base, (len(targets), base.shape[0])).copy()
    elif len(base) != len(targets):
        raise ValueError(f"base_obs has {len(base)} rows but {len(targets)} targets were given")
    else:
        base = base.copy()

    base[:, 0:2] = targets
    if goal_xy is not None:
        base[:, -2:] = np.asarray(goal_xy, dtype=np.float32).reshape(-1, 2)
    return base


def split_state_and_goal(obs: np.ndarray, state_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Split an observation into the `phi` input and the `psi` input."""
    obs = np.asarray(obs)
    return obs[..., :state_dim], obs[..., state_dim:]


def pose_bank(obs: np.ndarray, n: int, seed: int | None = None) -> np.ndarray:
    """Sample `n` observations from rollout data, to use as poses.

    Sampling from real trajectories rather than synthesising a pose is what
    keeps `phi`'s inputs on-manifold for Ant.
    """
    obs = np.asarray(obs).reshape(-1, np.asarray(obs).shape[-1])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(obs), size=min(n, len(obs)), replace=False)
    return obs[idx]
