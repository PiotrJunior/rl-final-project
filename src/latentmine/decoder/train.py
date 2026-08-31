"""Train a decoder against a frozen CRL encoder.

The encoder never receives a gradient: this is an autoencoder whose first half
is fixed, and the question is how much of the input the second half can
recover. Reporting is per dimension group and per split (LLD sections 6E and
8.2) - a single scalar would hide exactly the structure we are after.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .data import Split, standardise
from .model import Decoder


@dataclass
class DecoderFit:
    """A trained decoder and everything needed to judge it."""

    params: Any
    apply_fn: Any
    scaler: tuple[np.ndarray, np.ndarray]
    history: list[dict[str, float]] = field(default_factory=list)
    errors: dict[str, dict[str, float]] = field(default_factory=dict)
    groups: dict[str, tuple[int, int]] = field(default_factory=dict)

    def decode(self, latents: np.ndarray) -> np.ndarray:
        """Latents -> reconstructed targets, in original units."""
        mean, std = self.scaler
        out = np.asarray(self.apply_fn(self.params, np.asarray(latents, dtype=np.float32)))
        return out * std + mean

    def summary(self) -> str:
        lines = []
        for split_name, per_group in self.errors.items():
            parts = "  ".join(f"{g}={v:.3f}" for g, v in per_group.items())
            lines.append(f"  {split_name:<22s} {parts}")
        return "\n".join(lines)


def fit(
    latents: np.ndarray,
    targets: np.ndarray,
    split: Split,
    groups: dict[str, tuple[int, int]],
    width: int = 512,
    depth: int = 3,
    steps: int = 4000,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    seed: int = 0,
    patience: int = 10,
    log_every: int = 250,
) -> DecoderFit:
    """Supervised training with early stopping on the same-region validation set.

    Early stopping deliberately watches `val`, not `test`: `test` is the
    held-out region, and selecting on it would leak exactly the generalisation
    we are trying to measure.
    """
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training.train_state import TrainState

    latents = np.asarray(latents, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)

    (y_train, y_val, y_test), scaler = standardise(
        targets[split.train], targets[split.val], targets[split.test]
    )
    x_train, x_val = latents[split.train], latents[split.val]
    x_test = latents[split.test]

    model = Decoder(output_dim=targets.shape[1], width=width, depth=depth)
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    state = TrainState.create(
        apply_fn=model.apply,
        params=model.init(init_key, jnp.zeros((1, latents.shape[1]))),
        tx=optax.adam(optax.cosine_decay_schedule(learning_rate, steps)),
    )

    def loss_fn(params, x, y):
        return jnp.mean((model.apply(params, x) - y) ** 2)

    @jax.jit
    def update(state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(state.params, x, y)
        return state.apply_gradients(grads=grads), loss

    @jax.jit
    def evaluate(params, x, y):
        return loss_fn(params, x, y)

    history: list[dict[str, float]] = []
    best = (float("inf"), state.params)
    stale = 0
    for step in range(steps):
        key, batch_key = jax.random.split(key)
        idx = jax.random.choice(batch_key, len(x_train), (min(batch_size, len(x_train)),), replace=False)
        state, loss = update(state, x_train[np.asarray(idx)], y_train[np.asarray(idx)])

        if (step + 1) % log_every == 0 or step == steps - 1:
            val_loss = float(evaluate(state.params, x_val, y_val)) if len(x_val) else float("nan")
            history.append({"step": step + 1, "train_loss": float(loss), "val_loss": val_loss})
            if np.isfinite(val_loss) and val_loss < best[0] - 1e-6:
                best, stale = (val_loss, state.params), 0
            else:
                stale += 1
                if stale >= patience:
                    break

    params = best[1] if np.isfinite(best[0]) else state.params
    errors = {}
    for name, (x, y) in (
        ("train", (x_train, y_train)),
        ("val (same regions)", (x_val, y_val)),
        ("test (held-out regions)", (x_test, y_test)),
    ):
        errors[name] = _group_errors(model, params, x, y, groups)

    return DecoderFit(
        params=params,
        apply_fn=model.apply,
        scaler=scaler,
        history=history,
        errors=errors,
        groups=groups,
    )


def _group_errors(model, params, x, y, groups) -> dict[str, float]:
    """Normalised RMSE per dimension group, on standardised targets."""
    if len(x) == 0:
        return {name: float("nan") for name in ("all", *groups)}
    predicted = np.asarray(model.apply(params, x))
    out = {"all": float(np.sqrt(np.mean((predicted - y) ** 2)))}
    for name, (start, stop) in groups.items():
        stop = min(stop, y.shape[1])
        if start >= stop:
            continue
        out[name] = float(np.sqrt(np.mean((predicted[:, start:stop] - y[:, start:stop]) ** 2)))
    return out


def save(fit: DecoderFit, path: Path) -> Path:
    """Persist a fitted decoder as msgpack plus a JSON sidecar."""
    import json

    from flax import serialization

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialization.to_bytes(fit.params))
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "errors": fit.errors,
                "groups": {k: list(v) for k, v in fit.groups.items()},
                "history": fit.history,
                "scaler_mean": fit.scaler[0].tolist(),
                "scaler_std": fit.scaler[1].tolist(),
            },
            indent=2,
        )
        + "\n"
    )
    return path
