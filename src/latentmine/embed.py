"""Latent-space embedding and distance.

The bridge between a trained critic and the analysis. Two things live here
that must not be reimplemented elsewhere:

**The energy function must match upstream's exactly**, or every latent
distance in the project is measured with a different ruler than the one the
critic was trained against. `energy` mirrors
`jaxgcrl.agents.crl.losses.energy_fn` term for term, and a test asserts the
two agree numerically.

**The right notion of distance depends on the energy function** (LLD section
2.3). Under `norm`/`l2` the latent space is Euclidean and `||u - v||` is
meaningful; under `dot`/`cosine` the geometry is spherical and the honest
distance is angular. `latent_distance` picks correctly from the energy name
rather than leaving it to the caller, because using a Euclidean distance on a
cosine-trained latent is a silent error that produces plausible-looking
nonsense.
"""

from __future__ import annotations

import numpy as np

from .checkpoints import Encoders

EUCLIDEAN_ENERGIES = ("norm", "l2")
SPHERICAL_ENERGIES = ("dot", "cosine")


def energy(name: str, x, y):
    """Mirror of upstream's `energy_fn`. Higher means "closer" - it is a
    critic value, not a distance."""
    import jax.numpy as jnp

    if name == "norm":
        return -jnp.sqrt(jnp.sum((x - y) ** 2, axis=-1) + 1e-6)
    if name == "l2":
        return -jnp.sum((x - y) ** 2, axis=-1)
    if name == "dot":
        return jnp.sum(x * y, axis=-1)
    if name == "cosine":
        return jnp.sum(x * y, axis=-1) / (jnp.linalg.norm(x) * jnp.linalg.norm(y) + 1e-6)
    raise ValueError(f"unknown energy function: {name}")


def is_euclidean(energy_fn: str) -> bool:
    """Whether linear interpolation and Euclidean distance are meaningful.

    `True` for `norm`/`l2`, where the critic is a (negated) distance and the
    latent space is a metric embedding of hitting time. `False` for
    `dot`/`cosine`, where interpolation must be spherical (slerp) and vectors
    should be normalised before projection.
    """
    if energy_fn in EUCLIDEAN_ENERGIES:
        return True
    if energy_fn in SPHERICAL_ENERGIES:
        return False
    raise ValueError(f"unknown energy function: {energy_fn}")


def pairwise_latent_distance(a: np.ndarray, b: np.ndarray, energy_fn: str) -> np.ndarray:
    """`(N, M)` distances between two sets of latents, under the geometry the
    critic was trained with.

    Euclidean energies give the L2 distance. Spherical energies give the
    angular distance in radians, which is a true metric on the sphere and so
    keeps downstream rank statistics meaningful.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if is_euclidean(energy_fn):
        diff = a[:, None, :] - b[None, :, :]
        return np.sqrt((diff**2).sum(-1))
    an = a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12, None)
    bn = b / np.clip(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12, None)
    return np.arccos(np.clip(an @ bn.T, -1.0, 1.0))


def latent_distance(a: np.ndarray, b: np.ndarray, energy_fn: str) -> np.ndarray:
    """Elementwise distance between paired latents. `a` and `b` broadcast."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if is_euclidean(energy_fn):
        return np.sqrt(((a - b) ** 2).sum(-1))
    an = a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12, None)
    bn = b / np.clip(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12, None)
    return np.arccos(np.clip((an * bn).sum(-1), -1.0, 1.0))


def _batched(fn, x: np.ndarray, batch_size: int, repr_dim: int) -> np.ndarray:
    if len(x) == 0:
        # Keep the second axis, so callers can index [:, k] without a special case.
        return np.zeros((0, repr_dim), dtype=np.float32)
    out = [np.asarray(fn(x[i : i + batch_size])) for i in range(0, len(x), batch_size)]
    return np.concatenate(out, axis=0)


def embed_goals(encoders: Encoders, goals_xy: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """`psi` over world `(x, y)` positions. `(N, 2) -> (N, repr_dim)`."""
    goals = np.asarray(goals_xy, dtype=np.float32).reshape(-1, 2)
    return _batched(encoders.psi, goals, batch_size, encoders.repr_dim)


def embed_state_actions(
    encoders: Encoders, states: np.ndarray, actions: np.ndarray, batch_size: int = 4096
) -> np.ndarray:
    """`phi` over state-action pairs. `(N, state_dim) x (N, action_size)`."""
    states = np.asarray(states, dtype=np.float32).reshape(len(states), -1)
    actions = np.asarray(actions, dtype=np.float32).reshape(len(actions), -1)
    if len(states) != len(actions):
        raise ValueError(f"states and actions disagree: {len(states)} vs {len(actions)}")
    return _batched(encoders.phi, np.concatenate([states, actions], axis=-1), batch_size, encoders.repr_dim)


def critic_values(
    encoders: Encoders, states: np.ndarray, actions: np.ndarray, goals_xy: np.ndarray
) -> np.ndarray:
    """`Q(s, a, g)` for paired inputs - the quantity the actor maximises."""
    sa = embed_state_actions(encoders, states, actions)
    g = embed_goals(encoders, goals_xy)
    return np.asarray(energy(encoders.energy_fn, sa, g))


def goal_distance_field(encoders: Encoders, goals_xy: np.ndarray, anchor_xy) -> np.ndarray:
    """Latent distance from one anchor goal to every goal in a grid.

    The visual form of the project's headline metric: overlay this on the maze
    and compare its contours with true geodesic ones (LLD section 7.2).
    """
    embedded = embed_goals(encoders, goals_xy)
    anchor = embed_goals(encoders, np.asarray(anchor_xy, dtype=np.float32).reshape(1, 2))
    return pairwise_latent_distance(anchor, embedded, encoders.energy_fn)[0]
