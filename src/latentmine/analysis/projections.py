"""2-D projections of the latent space.

Deliberately explicit about two things that decide whether a picture means
anything:

**Seeds.** t-SNE and UMAP are stochastic. An unseeded projection is not
reproducible and cannot be compared across mazes or checkpoints, so the seed
is a required argument with a default, never global state.

**Geometry.** Under `dot`/`cosine` energies the latent space is spherical, so
vectors are normalised before projection - Euclidean PCA on unnormalised
spherical latents mostly recovers the magnitude, which carries no information
about the maze (LLD section 2.3).

PCA is implemented here rather than imported so that explained variance comes
back with the projection: if two components carry most of the variance, `psi`
is effectively a 2-D map and the picture is honest; if they carry little, the
projection is discarding the structure and `trustworthiness` should say so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..embed import is_euclidean


@dataclass
class Projection:
    """A 2-D embedding and enough context to judge it."""

    coords: np.ndarray  # (N, 2)
    method: str
    explained_variance_ratio: np.ndarray | None = None
    seed: int = 0

    @property
    def explained(self) -> float:
        """Fraction of variance the two components carry (PCA only)."""
        if self.explained_variance_ratio is None:
            return float("nan")
        return float(self.explained_variance_ratio[:2].sum())


def prepare(latents: np.ndarray, energy_fn: str) -> np.ndarray:
    """Centre, and normalise first if the geometry is spherical."""
    x = np.asarray(latents, dtype=np.float64)
    if not is_euclidean(energy_fn):
        x = x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)
    return x - x.mean(axis=0, keepdims=True)


def pca(latents: np.ndarray, energy_fn: str = "norm", n_components: int = 2) -> Projection:
    """Principal components via SVD, with the variance spectrum retained."""
    x = prepare(latents, energy_fn)
    _, singular, vt = np.linalg.svd(x, full_matrices=False)
    variance = singular**2
    ratio = variance / variance.sum() if variance.sum() > 0 else variance
    return Projection(
        coords=x @ vt[:n_components].T,
        method="pca",
        explained_variance_ratio=ratio,
    )


def tsne(
    latents: np.ndarray,
    energy_fn: str = "norm",
    seed: int = 0,
    perplexity: float | None = None,
) -> Projection:
    """t-SNE. Perplexity defaults to a sane fraction of the sample count,
    because the sklearn default of 30 is invalid for the small grids here
    (a maze has 24-68 free cells)."""
    from sklearn.manifold import TSNE

    x = prepare(latents, energy_fn)
    if perplexity is None:
        perplexity = max(5.0, min(30.0, (len(x) - 1) / 3.0))
    coords = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        random_state=seed,
    ).fit_transform(x)
    return Projection(coords=np.asarray(coords), method="tsne", seed=seed)


def umap(
    latents: np.ndarray,
    energy_fn: str = "norm",
    seed: int = 0,
    n_neighbors: int | None = None,
    min_dist: float = 0.1,
) -> Projection:
    """UMAP. The method most likely to show `loop`'s cycle, which has no
    faithful linear 2-D embedding."""
    import umap as umap_lib

    x = prepare(latents, energy_fn)
    if n_neighbors is None:
        n_neighbors = max(2, min(15, len(x) // 4))
    coords = umap_lib.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    ).fit_transform(x)
    return Projection(coords=np.asarray(coords), method="umap", seed=seed)


METHODS = {"pca": pca, "tsne": tsne, "umap": umap}


def project(latents: np.ndarray, method: str = "pca", energy_fn: str = "norm", seed: int = 0):
    """Dispatch by name, so callers can loop over methods."""
    if method not in METHODS:
        raise ValueError(f"unknown projection {method!r}; known: {sorted(METHODS)}")
    if method == "pca":
        return pca(latents, energy_fn)
    return METHODS[method](latents, energy_fn, seed=seed)


def procrustes_error(coords: np.ndarray, targets: np.ndarray) -> float:
    """Residual after the best rotation, scaling and translation onto `targets`.

    Applied to a projection against true maze coordinates, this asks how close
    the latent map is to being the maze itself up to a rigid transform. Zero
    means the picture *is* the floor plan.
    """
    a = np.asarray(coords, dtype=float)
    b = np.asarray(targets, dtype=float)
    a = a - a.mean(0)
    b = b - b.mean(0)
    a_norm, b_norm = np.linalg.norm(a), np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return float("nan")
    a, b = a / a_norm, b / b_norm
    u, s, vt = np.linalg.svd(a.T @ b)
    return float(1.0 - s.sum() ** 2)
