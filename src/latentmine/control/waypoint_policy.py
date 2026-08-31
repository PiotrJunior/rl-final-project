"""Condition the trained actor on waypoints.

Deliverable 3.3. The actor consumes `obs = concat(state, goal_xy)`, so
conditioning on a waypoint is just substituting the goal - no retraining, no
new network. Worth stating plainly because it is what makes the whole "exploit
the latent space" section cheap.

The evaluation protocol (LLD section 10.3) is stratified by geodesic distance.
The honest expectation is that waypoints help mainly on far pairs, and
stratifying is what shows that rather than washing it out in an average.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..mazes import geometry as geo
from ..mazes.layouts import MazeSpec

# Upstream's `goal_reach_thresh` for both maze envs.
GOAL_REACH_THRESH = 0.5


@dataclass
class Episode:
    """One waypoint-conditioned rollout."""

    reached: bool
    steps: int
    positions: np.ndarray
    waypoints_consumed: int
    final_distance: float


@dataclass
class ControlResult:
    """Success and cost for one waypoint source, over many start/goal pairs."""

    name: str
    success: np.ndarray
    steps: np.ndarray
    strata: np.ndarray
    episodes: list[Episode] = field(default_factory=list)

    def success_rate(self, stratum: str | None = None) -> float:
        mask = self.strata == stratum if stratum else np.ones(len(self.success), bool)
        return float(self.success[mask].mean()) if mask.any() else float("nan")

    def median_steps(self, stratum: str | None = None, successful_only: bool = True) -> float:
        mask = self.strata == stratum if stratum else np.ones(len(self.success), bool)
        if successful_only:
            mask = mask & (self.success > 0)
        return float(np.median(self.steps[mask])) if mask.any() else float("nan")

    def bootstrap_ci(self, alpha: float = 0.05, n: int = 2000, seed: int = 0):
        """Percentile bootstrap CI on the success rate.

        Reported alongside every rate: with a couple of hundred pairs the
        difference between two methods is easily inside the noise, and a bare
        percentage invites reading a win that is not there.
        """
        rng = np.random.default_rng(seed)
        draws = rng.choice(self.success, size=(n, len(self.success)), replace=True).mean(axis=1)
        return float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def stratify(spec: MazeSpec, pairs: list[tuple], n_strata: int = 3) -> np.ndarray:
    """Label start/goal pairs near / medium / far by geodesic distance."""
    index = {cell: k for k, cell in enumerate(spec.free_cells())}
    distances = np.array([geo.geodesic_from(spec, start)[index[goal]] for start, goal in pairs])
    names = ["near", "medium", "far"][:n_strata]
    edges = np.quantile(distances, np.linspace(0, 1, n_strata + 1)[1:-1])
    return np.array([names[int(np.searchsorted(edges, d))] for d in distances])


def sample_pairs(spec: MazeSpec, n: int = 200, seed: int = 0, min_cells: int = 3) -> list[tuple]:
    """Start/goal pairs, fixed across every method and seed being compared."""
    cells = list(spec.free_cells())
    index = {cell: k for k, cell in enumerate(cells)}
    rng = np.random.default_rng(seed)
    pairs = []
    guard = 0
    while len(pairs) < n and guard < 100 * n:
        guard += 1
        start, goal = (cells[i] for i in rng.choice(len(cells), 2, replace=False))
        if geo.geodesic_from(spec, start)[index[goal]] >= min_cells * spec.scaling:
            pairs.append((start, goal))
    return pairs


def rollout_with_waypoints(
    encoders,
    env,
    start_cell,
    goal_cell,
    waypoints: np.ndarray | None,
    max_steps: int = 500,
    max_steps_per_waypoint: int = 80,
    reach_thresh: float = GOAL_REACH_THRESH,
    seed: int = 0,
) -> Episode:
    """Drive the actor through `waypoints`, then to the true goal.

    The per-waypoint timeout is not optional. A decoded waypoint inside a wall
    is unreachable, and without a timeout the episode deadlocks there - the
    comparison against baselines would then be measuring timeouts rather than
    navigation.
    """
    import jax
    import jax.numpy as jnp

    reset = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    goal_xy = np.array(geo.cell_to_world(goal_cell, env_scaling(env, start_cell)), dtype=np.float32)
    state = reset(jax.random.PRNGKey(seed))
    state = _place(state, geo.cell_to_world(start_cell, env_scaling(env, start_cell)), goal_xy)

    route = [] if waypoints is None else [np.asarray(w, dtype=np.float32) for w in waypoints]
    route.append(goal_xy)

    positions, consumed, steps_on_current = [], 0, 0
    reached = False
    for _ in range(max_steps):
        obs = np.asarray(state.obs, dtype=np.float32).copy()
        position = obs[:2]
        positions.append(position.copy())

        target = route[min(consumed, len(route) - 1)]
        if np.linalg.norm(position - goal_xy) < reach_thresh:
            reached = True
            break
        if consumed < len(route) - 1 and (
            np.linalg.norm(position - target) < reach_thresh or steps_on_current >= max_steps_per_waypoint
        ):
            consumed += 1
            steps_on_current = 0
            target = route[consumed]

        obs[-2:] = target
        mean, _ = encoders.actor(obs[None, :])
        state = step_fn(state, jnp.tanh(mean)[0])
        steps_on_current += 1

    final = float(np.linalg.norm(np.asarray(state.obs)[:2] - goal_xy))
    return Episode(
        reached=reached,
        steps=len(positions),
        positions=np.array(positions),
        waypoints_consumed=consumed,
        final_distance=final,
    )


def env_scaling(env, _cell) -> float:
    """Maze scaling as the env was built with it."""
    return float(getattr(env, "maze_size_scaling", 4.0))


def _place(state, start_xy, goal_xy):
    """Move the agent to a chosen start and set its goal.

    Upstream samples both at random on reset, so a controlled comparison needs
    to override them - otherwise every method is evaluated on different pairs.
    """
    import jax.numpy as jnp

    q = state.pipeline_state.q
    q = q.at[0].set(start_xy[0]).at[1].set(start_xy[1])
    q = q.at[-2].set(goal_xy[0]).at[-1].set(goal_xy[1])
    pipeline_state = state.pipeline_state.replace(q=q)
    obs = jnp.asarray(state.obs).at[0].set(start_xy[0]).at[1].set(start_xy[1])
    obs = obs.at[-2].set(goal_xy[0]).at[-1].set(goal_xy[1])
    return state.replace(pipeline_state=pipeline_state, obs=obs)


def compare(results: list[ControlResult]) -> str:
    """A table of success rate and steps by stratum, for the write-up."""
    strata = ["near", "medium", "far"]
    header = f"{'method':<18s}" + "".join(f"{s:>22s}" for s in strata) + f"{'overall':>22s}"
    lines = [header, "-" * len(header)]
    for result in results:
        row = f"{result.name:<18s}"
        for stratum in [*strata, None]:
            rate = result.success_rate(stratum)
            steps = result.median_steps(stratum)
            row += f"{rate:>13.2f} ({steps:>4.0f}s)" if np.isfinite(rate) else f"{'-':>22s}"
        lines.append(row)
    return "\n".join(lines)
