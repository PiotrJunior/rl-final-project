"""Embedding and latent-distance tests.

The energy function must match upstream's exactly - it is the ruler every
latent measurement in the project is taken with - and the notion of distance
must follow from it, since using a Euclidean distance on a cosine-trained
latent produces plausible-looking nonsense rather than an error.
"""

import numpy as np
import pytest

from latentmine import embed


class TestEnergyMatchesUpstream:
    @pytest.mark.slow
    @pytest.mark.parametrize("name", ["norm", "l2", "dot", "cosine"])
    def test_numerically_identical_to_upstreams(self, name):
        pytest.importorskip("jaxgcrl")
        import jax.numpy as jnp
        from jaxgcrl.agents.crl.losses import energy_fn as upstream_energy

        rng = np.random.default_rng(0)
        x = jnp.asarray(rng.normal(size=(6, 8)), dtype=jnp.float32)
        y = jnp.asarray(rng.normal(size=(6, 8)), dtype=jnp.float32)
        np.testing.assert_allclose(
            np.asarray(embed.energy(name, x, y)),
            np.asarray(upstream_energy(name, x, y)),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_unknown_energy_is_rejected(self):
        with pytest.raises(ValueError, match="unknown energy function"):
            embed.energy("euclidean", np.zeros(3), np.zeros(3))


class TestGeometrySelection:
    def test_norm_and_l2_are_euclidean(self):
        assert embed.is_euclidean("norm")
        assert embed.is_euclidean("l2")

    def test_dot_and_cosine_are_not(self):
        # Under these, interpolation must be spherical and vectors normalised
        # before projection (LLD 2.3).
        assert not embed.is_euclidean("dot")
        assert not embed.is_euclidean("cosine")

    def test_unknown_energy_is_rejected(self):
        with pytest.raises(ValueError, match="unknown energy"):
            embed.is_euclidean("triplet")


class TestDistances:
    def test_euclidean_distance_is_the_l2_norm(self):
        a = np.array([[0.0, 0.0], [3.0, 4.0]])
        b = np.array([[3.0, 4.0]])
        d = embed.pairwise_latent_distance(a, b, "norm")
        np.testing.assert_allclose(d[:, 0], [5.0, 0.0])

    def test_spherical_distance_is_the_angle(self):
        a = np.array([[1.0, 0.0], [0.0, 2.0]])  # magnitude must not matter
        b = np.array([[1.0, 0.0]])
        d = embed.pairwise_latent_distance(a, b, "cosine")
        np.testing.assert_allclose(d[:, 0], [0.0, np.pi / 2], atol=1e-7)

    def test_spherical_distance_ignores_magnitude(self):
        a = np.array([[1.0, 1.0]])
        d1 = embed.pairwise_latent_distance(a, np.array([[2.0, 2.0]]), "dot")
        np.testing.assert_allclose(d1, [[0.0]], atol=1e-7)

    def test_paired_distance_agrees_with_the_matrix_diagonal(self):
        rng = np.random.default_rng(1)
        a, b = rng.normal(size=(5, 4)), rng.normal(size=(5, 4))
        for fn in ("norm", "cosine"):
            np.testing.assert_allclose(
                embed.latent_distance(a, b, fn),
                np.diag(embed.pairwise_latent_distance(a, b, fn)),
                atol=1e-9,
            )

    def test_distance_is_symmetric_and_zero_on_the_diagonal(self):
        rng = np.random.default_rng(2)
        a = rng.normal(size=(7, 5))
        for fn in ("norm", "l2", "dot", "cosine"):
            d = embed.pairwise_latent_distance(a, a, fn)
            np.testing.assert_allclose(d, d.T, atol=1e-9)
            np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-7)

    def test_a_euclidean_ruler_on_spherical_latents_disagrees(self):
        # Documents why the geometry is chosen from the energy name rather
        # than left to the caller: the two rulers rank pairs differently.
        a = np.array([[1.0, 0.0], [10.0, 0.0]])
        b = np.array([[0.0, 1.0]])
        euclid = embed.pairwise_latent_distance(a, b, "norm")[:, 0]
        angular = embed.pairwise_latent_distance(a, b, "cosine")[:, 0]
        assert euclid[0] < euclid[1]  # magnitude dominates
        assert angular[0] == pytest.approx(angular[1])  # direction is identical
