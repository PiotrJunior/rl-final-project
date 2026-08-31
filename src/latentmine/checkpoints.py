"""Load a trained CRL critic back into callable encoders.

An upstream checkpoint is a bare pickled 3-tuple with **no record of the
architecture that produced it**, so rebuilding `Encoder` needs `repr_dim`,
`h_dim`, `n_hidden`, `skip_connections`, `use_relu` and `use_ln` from the
run's manifest. Nothing downstream should touch the pickle directly: go
through `load_encoders`, which refuses a checkpoint whose manifest is missing
(LLD section 2.7 and the convention in CLAUDE.md).

The returned object is the boundary between training and analysis. Everything
from LLD section 7 onward is written against `Encoders`, never against raw
parameters.
"""

from __future__ import annotations

import pickle
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .train import manifest as manifest_mod

CKPT_DIRNAME = "ckpt"
_STEP_RE = re.compile(r"step_(\d+)\.pkl$")


class CheckpointError(RuntimeError):
    pass


@dataclass(frozen=True)
class Encoders:
    """A trained critic, as two jitted functions plus what they mean.

    Attributes:
        phi: `(N, state_dim + action_size) -> (N, repr_dim)`, the sa_encoder.
        psi: `(N, goal_size) -> (N, repr_dim)`, the goal encoder. For every
            maze env `goal_size` is 2, so `psi` is a function of `(x, y)`
            alone and can be rastered densely over a maze with no rollouts -
            the cheapest and most informative probe in the project.
        actor: `(N, obs_size) -> (mean, log_std)`.
        energy_fn: the name of the critic's energy function. It decides both
            how latent distance is computed and whether interpolation may be
            linear (LLD section 2.3), so it travels with the encoders rather
            than being passed around separately.
        step: env-step count of the checkpoint these came from.
        manifest: the full run manifest, including the maze as trained on.
    """

    phi: Callable
    psi: Callable
    actor: Callable
    energy_fn: str
    repr_dim: int
    step: int
    manifest: dict[str, Any]
    run_dir: Path

    @property
    def maze_spec(self):
        """The maze this critic was trained on, rebuilt from the manifest
        rather than looked up by name - the registry may have changed since."""
        from .mazes.layouts import MazeSpec

        m = self.manifest["maze"]
        return MazeSpec(
            name=m["name"],
            grid=tuple(m["grid"]),
            regions=tuple(m["regions"]) if m["regions"] else None,
            scaling=m["scaling"],
            notes="rebuilt from manifest",
        )

    @property
    def run_id(self) -> str:
        return self.manifest["run_id"]


def list_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """`(step, path)` for every checkpoint in a run, oldest first."""
    ckpt_dir = Path(run_dir) / CKPT_DIRNAME
    if not ckpt_dir.is_dir():
        return []
    found = []
    for path in ckpt_dir.glob("step_*.pkl"):
        match = _STEP_RE.search(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def resolve_checkpoint(run_dir: Path, step: int | str = "last") -> tuple[int, Path]:
    """Pick a checkpoint by step number, or `"last"` / `"first"`."""
    available = list_checkpoints(run_dir)
    if not available:
        raise CheckpointError(f"no checkpoints under {Path(run_dir) / CKPT_DIRNAME}")
    if step == "last":
        return available[-1]
    if step == "first":
        return available[0]
    for found_step, path in available:
        if found_step == int(step):
            return found_step, path
    raise CheckpointError(
        f"no checkpoint at step {step} in {run_dir}; available: {[s for s, _ in available]}"
    )


def load_params(path: Path) -> tuple[Any, Any, Any]:
    """The raw `(alpha, actor, critic)` triple. Prefer `load_encoders`."""
    with Path(path).open("rb") as fh:
        payload = pickle.load(fh)
    if not (isinstance(payload, tuple) and len(payload) == 3):
        raise CheckpointError(f"{path}: expected a 3-tuple, got {type(payload).__name__}")
    critic = payload[2]
    missing = {"sa_encoder", "g_encoder"} - set(critic)
    if missing:
        raise CheckpointError(f"{path}: critic params missing {sorted(missing)}")
    return payload


def load_encoders(run_dir: Path, step: int | str = "last", jit: bool = True) -> Encoders:
    """Rebuild `phi`, `psi` and the actor from a run directory.

    Requires `manifest.json` beside the checkpoints; the pickle alone is not
    enough to know what shape of network to instantiate.
    """
    import jax
    from jaxgcrl.agents.crl.networks import Actor, Encoder

    run_dir = Path(run_dir)
    payload = manifest_mod.load(run_dir)
    arch, dims = payload["arch"], payload["dims"]

    resolved_step, path = resolve_checkpoint(run_dir, step)
    _, actor_params, critic_params = load_params(path)

    def make_encoder() -> Encoder:
        return Encoder(
            repr_dim=arch["repr_dim"],
            network_width=arch["h_dim"],  # upstream: h_dim is width
            network_depth=arch["n_hidden"],  # upstream: n_hidden is depth
            skip_connections=arch["skip_connections"],
            use_relu=arch["use_relu"],
            use_ln=arch["use_ln"],
        )

    sa_encoder, g_encoder = make_encoder(), make_encoder()
    actor = Actor(
        action_size=dims["action_size"],
        network_width=arch["h_dim"],
        network_depth=arch["n_hidden"],
        skip_connections=arch["skip_connections"],
        use_relu=arch["use_relu"],
        # Upstream builds the Actor without use_ln even when the encoders have
        # it, so --use_ln deepens and normalises only the critic. Mirrored here
        # deliberately: a checkpoint loaded with LayerNorm in the actor would
        # not match its saved parameters.
    )

    def phi(x):
        return sa_encoder.apply(critic_params["sa_encoder"], x)

    def psi(g):
        return g_encoder.apply(critic_params["g_encoder"], g)

    def actor_apply(obs):
        return actor.apply(actor_params, obs)

    if jit:
        phi, psi, actor_apply = jax.jit(phi), jax.jit(psi), jax.jit(actor_apply)

    return Encoders(
        phi=phi,
        psi=psi,
        actor=actor_apply,
        energy_fn=payload["crl"]["energy_fn"],
        repr_dim=arch["repr_dim"],
        step=resolved_step,
        manifest=payload,
        run_dir=run_dir,
    )


def load_encoder_series(run_dir: Path, every: int = 1) -> list[Encoders]:
    """Encoders for every `every`-th checkpoint, oldest first.

    For watching the latent space organise itself over training (LLD 5.4) -
    the clearest qualitative evidence that structure emerges rather than being
    an artefact of initialisation.
    """
    steps = [s for s, _ in list_checkpoints(run_dir)][::every]
    return [load_encoders(run_dir, step=s) for s in steps]
