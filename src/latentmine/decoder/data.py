"""Datasets for the frozen-encoder decoders, and the split that makes them mean
something.

**Spatial holdout is the design decision that matters here** (LLD section 8.2).
A random split lets the decoder memorise a lookup table over sampled cells, and
reconstruction error then measures nothing. Holding out whole *regions* - three
rooms to train, the fourth to test - is what makes the number a statement about
whether the latent space is smoothly organised, which is also what the
interpolation experiment depends on, since interpolants necessarily fall
between training points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..mazes import geometry as geo
from ..mazes.layouts import MazeSpec


@dataclass
class Split:
    """Indices into a sample array, and what the split means."""

    train: np.ndarray
    val: np.ndarray  # held-out samples from training regions
    test: np.ndarray  # held-out regions entirely
    kind: str
    held_out: tuple[str, ...] = ()

    def describe(self) -> str:
        return (
            f"{self.kind} split: {len(self.train)} train, {len(self.val)} val "
            f"(same regions), {len(self.test)} test (held-out {', '.join(self.held_out) or 'none'})"
        )


def spatial_split(
    spec: MazeSpec,
    cell_index: np.ndarray,
    hold_out: str | tuple[str, ...] | None = None,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> Split:
    """Hold out whole regions, plus a random validation slice of the rest.

    Three numbers come out of this and they must always be reported together:
    training error, held-out-samples error (same regions), and held-out-region
    error.

    Read the third carefully. A held-out region is outside the coordinate
    range the decoder saw, so it measures **extrapolation**, not smoothness -
    see `scattered_split`, which measures the interpolation question instead.
    """
    if spec.regions is None:
        return geodesic_split(spec, cell_index, val_fraction=val_fraction, seed=seed)

    labels = spec.region_labels()
    if hold_out is None:
        hold_out = (labels[-1],)
    elif isinstance(hold_out, str):
        hold_out = (hold_out,)
    unknown = set(hold_out) - set(labels)
    if unknown:
        raise ValueError(f"{spec.name}: unknown region(s) {sorted(unknown)}; known {list(labels)}")

    cells = geo.free_cell_array(spec)
    sample_region = np.array([spec.regions[i][j] for (i, j) in cells[cell_index]])
    in_test = np.isin(sample_region, hold_out)

    rest = np.flatnonzero(~in_test)
    rng = np.random.default_rng(seed)
    rng.shuffle(rest)
    n_val = int(round(val_fraction * len(rest)))
    return Split(
        train=np.sort(rest[n_val:]),
        val=np.sort(rest[:n_val]),
        test=np.flatnonzero(in_test),
        kind="region",
        held_out=tuple(hold_out),
    )


def geodesic_split(
    spec: MazeSpec,
    cell_index: np.ndarray,
    test_fraction: float = 0.25,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> Split:
    """Hold out a contiguous arc, for mazes with no rooms.

    `spiral` and `loop` have no region overlay, so "hold out a room" is
    undefined. Holding out the cells furthest along the corridor from the
    start is the same idea - a contiguous chunk of space the decoder never
    sees - expressed in the only structure those mazes have.
    """
    cells = geo.free_cell_array(spec)
    order = geo.geodesic_from(spec, spec.start_cells()[0])
    cutoff = np.quantile(order, 1.0 - test_fraction)
    in_test = order[cell_index] >= cutoff

    rest = np.flatnonzero(~in_test)
    rng = np.random.default_rng(seed)
    rng.shuffle(rest)
    n_val = int(round(val_fraction * len(rest)))
    assert len(cells) > 0
    return Split(
        train=np.sort(rest[n_val:]),
        val=np.sort(rest[:n_val]),
        test=np.flatnonzero(in_test),
        kind="geodesic arc",
        held_out=(f"furthest {test_fraction:.0%} by geodesic distance",),
    )


def scattered_split(
    spec: MazeSpec,
    cell_index: np.ndarray,
    test_fraction: float = 0.2,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> Split:
    """Hold out individual cells scattered through the maze.

    The complement of `spatial_split`, and both are needed to read the result
    (LLD section 8.2). A held-out *region* lies outside the coordinate range
    the decoder ever saw, so recovering it is **extrapolation**, and an MLP
    does not extrapolate: on a latent that is an exact linear map of position -
    the easiest possible case - the region-holdout error is 1.62 against 0.01
    on training cells. That number says the decoder cannot extrapolate; it
    says nothing about whether the latent space is smoothly organised.

    Scattered cells are inside the range but unseen, so they measure
    interpolation, which is the question "is the latent smooth?" actually
    asks. Report the two together: high scattered error means the latent is
    not smoothly organised, while low scattered error with high region error
    means it is smooth but the decoder was asked to extrapolate.
    """
    n = len(cell_index)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test = int(round(test_fraction * n))
    n_val = int(round(val_fraction * n))
    return Split(
        train=np.sort(order[n_test + n_val :]),
        val=np.sort(order[n_test : n_test + n_val]),
        test=np.sort(order[:n_test]),
        kind="scattered",
        held_out=(f"{test_fraction:.0%} of cells, scattered",),
    )


def standardise(train: np.ndarray, *others: np.ndarray):
    """Per-dimension standardisation fitted on the training split only.

    Without it, velocity dimensions dominate the MSE and the position
    dimensions - the ones the project is about - are effectively ignored.
    Fitting on the training split only keeps the held-out regions honest.
    """
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    scaled = [(train - mean) / std] + [(other - mean) / std for other in others]
    return scaled, (mean, std)
