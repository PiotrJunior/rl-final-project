"""Collect rollouts and their latents.

Deliverable 2.1 of the proposal, and the sanity check that gates everything
after it (LLD section 7.1): along a successful trajectory, the latent distance
from `phi(s_t, a_t)` to `psi(g)` must fall as the agent approaches the goal.
If it does not, the checkpoint is not worth analysing and no amount of
projection will fix it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .checkpoints import Encoders
from .embed import embed_goals, embed_state_actions, latent_distance


@dataclass
class Rollouts:
    """Trajectories with everything the analysis needs, all numpy."""

    obs: np.ndarray  # (episodes, steps, obs_size)
    actions: np.ndarray  # (episodes, steps, action_size)
    goal: np.ndarray  # (episodes, 2)
    success: np.ndarray  # (episodes,) whether the goal was reached
    dist: np.ndarray  # (episodes, steps) distance to goal in world units
    state_dim: int

    @property
    def n_episodes(self) -> int:
        return len(self.obs)

    @property
    def positions(self) -> np.ndarray:
        """`(episodes, steps, 2)` world xy - the first two observation dims."""
        return self.obs[..., :2]

    def states(self) -> np.ndarray:
        return self.obs[..., : self.state_dim]

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            obs=self.obs,
            actions=self.actions,
            goal=self.goal,
            success=self.success,
            dist=self.dist,
            state_dim=self.state_dim,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> Rollouts:
        with np.load(Path(path)) as data:
            return cls(
                obs=data["obs"],
                actions=data["actions"],
                goal=data["goal"],
                success=data["success"],
                dist=data["dist"],
                state_dim=int(data["state_dim"]),
            )


def collect(
    encoders: Encoders,
    env,
    n_episodes: int = 8,
    steps: int = 500,
    seed: int = 0,
    deterministic: bool = True,
) -> Rollouts:
    """Roll the trained actor out in `env`.

    `deterministic` takes `tanh(mean)`, which is what upstream's evaluator
    does; sampling instead is useful for coverage when gathering poses for
    `phi` (LLD section 7.3).
    """
    import jax
    import jax.numpy as jnp

    reset = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    all_obs, all_act, all_goal, all_success, all_dist = [], [], [], [], []
    for episode in range(n_episodes):
        key = jax.random.PRNGKey(seed + episode)
        state = reset(key)
        obs_seq, act_seq, dist_seq = [], [], []
        reached = False
        for _ in range(steps):
            mean, log_std = encoders.actor(state.obs[None, :])
            if deterministic:
                action = jnp.tanh(mean)[0]
            else:
                key, sub = jax.random.split(key)
                noise = jax.random.normal(sub, mean.shape) * jnp.exp(log_std)
                action = jnp.tanh(mean + noise)[0]

            obs_seq.append(np.asarray(state.obs))
            act_seq.append(np.asarray(action))
            dist_seq.append(float(np.linalg.norm(np.asarray(state.obs)[:2] - np.asarray(state.obs)[-2:])))
            reached = reached or bool(state.metrics.get("success", 0.0))
            state = step_fn(state, action)

        all_obs.append(np.stack(obs_seq))
        all_act.append(np.stack(act_seq))
        all_goal.append(np.asarray(obs_seq[0])[-2:])
        all_success.append(float(reached))
        all_dist.append(np.array(dist_seq))

    return Rollouts(
        obs=np.stack(all_obs),
        actions=np.stack(all_act),
        goal=np.stack(all_goal),
        success=np.array(all_success),
        dist=np.stack(all_dist),
        state_dim=encoders.manifest["dims"]["state_dim"],
    )


def latent_distance_to_goal(encoders: Encoders, rollouts: Rollouts) -> np.ndarray:
    """`(episodes, steps)` latent distance from `phi(s_t, a_t)` to `psi(g)`.

    This is the critic's own estimate of remaining time-to-goal. Plotted
    against the true distance it is the cheapest test that a checkpoint is
    sane.
    """
    episodes, steps = rollouts.obs.shape[:2]
    states = rollouts.states().reshape(episodes * steps, -1)
    actions = rollouts.actions.reshape(episodes * steps, -1)
    sa = embed_state_actions(encoders, states, actions)
    g = embed_goals(encoders, rollouts.goal)
    g = np.repeat(g, steps, axis=0)
    return latent_distance(sa, g, encoders.energy_fn).reshape(episodes, steps)


def approach_correlation(encoders: Encoders, rollouts: Rollouts, successful_only: bool = True):
    """Spearman correlation between latent distance to goal and true distance.

    The gate on LLD section 12 step 4: strongly positive means the critic has
    learned something about reaching goals, near zero means the checkpoint is
    not worth analysing.
    """
    from scipy.stats import spearmanr

    latent = latent_distance_to_goal(encoders, rollouts)
    true = rollouts.dist
    if successful_only and rollouts.success.any():
        keep = rollouts.success > 0
        latent, true = latent[keep], true[keep]
    return float(spearmanr(latent.ravel(), true.ravel()).statistic)
