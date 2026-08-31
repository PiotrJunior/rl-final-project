"""Decoder networks, trained against a frozen CRL encoder.

Two decoders (LLD section 8.1):

* `D_g : R^d -> R^2` inverting `psi`. Small, and it should work - if it does
  not, something is wrong upstream of it.
* `D_sa : R^d -> R^{state_dim + action_size}` inverting `phi`. Genuinely
  ill-posed: `phi` is trained to be *invariant* to whatever does not affect
  future-goal occupancy, so perfect reconstruction is not expected. The
  per-group error profile is the result, not a failure - it says what the
  latent keeps and what it discards, which is the Ant question the proposal
  asks.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import variance_scaling


class Decoder(nn.Module):
    """MLP from a latent back to whatever was encoded.

    Same initialisation and activation as upstream's `Encoder`, so the two
    halves of the autoencoder are comparable and any difference in capacity is
    the depth and width we chose rather than an incidental one.
    """

    output_dim: int
    width: int = 512
    depth: int = 3
    use_ln: bool = True

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        x = z
        for _ in range(self.depth):
            x = nn.Dense(self.width, kernel_init=lecun_uniform, bias_init=nn.initializers.zeros)(x)
            if self.use_ln:
                x = nn.LayerNorm()(x)
            x = nn.swish(x)
        return nn.Dense(self.output_dim, kernel_init=lecun_uniform, bias_init=nn.initializers.zeros)(x)


# Observation-dimension groups for AntMaze's 29-D state, so reconstruction
# error can be reported per group rather than as one uninformative number
# (LLD section 6, metric E). Indices follow the MJCF: a free joint contributes
# 7 qpos and 6 qvel, then eight hinges.
ANT_STATE_GROUPS: dict[str, tuple[int, int]] = {
    "xy": (0, 2),
    "z": (2, 3),
    "orientation": (3, 7),
    "joint_angles": (7, 15),
    "linear_velocity": (15, 18),
    "angular_velocity": (18, 21),
    "joint_velocities": (21, 29),
}

SIMPLE_STATE_GROUPS: dict[str, tuple[int, int]] = {
    "xy": (0, 2),
    "velocity": (2, 4),
}


def state_groups(env: str) -> dict[str, tuple[int, int]]:
    """Dimension groups for an env's state vector."""
    if env == "ant":
        return dict(ANT_STATE_GROUPS)
    if env == "simple":
        return dict(SIMPLE_STATE_GROUPS)
    raise ValueError(f"unknown env {env!r}")


def action_group(env: str, state_dim: int, action_size: int) -> dict[str, tuple[int, int]]:
    """The action block appended after the state, for `D_sa` targets."""
    return {"action": (state_dim, state_dim + action_size)}
