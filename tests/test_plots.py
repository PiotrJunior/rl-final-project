"""Figure tests.

Plots are checked for the things that can silently go wrong - a transposed
maze, a colour ramp that invents structure, a missing bar that reads as zero -
rather than for pixel appearance.
"""

import numpy as np
import pytest

from latentmine.analysis import metrics as M
from latentmine.analysis import plots, projections
from latentmine.mazes import geometry as geo
from latentmine.mazes import layouts as L


def geodesic_latent(spec, dims=8):
    d = geo.geodesic_matrix(spec)
    n = len(d)
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (d**2) @ centering
    values, vectors = np.linalg.eigh(gram)
    top = np.argsort(values)[::-1][:dims]
    return vectors[:, top] * np.sqrt(np.clip(values[top], 0, None))


@pytest.fixture
def spec():
    return L.get("two_rooms")


class TestDistanceField:
    def test_writes_a_figure(self, spec, tmp_path):
        anchor = spec.free_cells()[0]
        out = plots.latent_distance_field(spec, geo.geodesic_from(spec, anchor), anchor, tmp_path / "f.png")
        assert out.exists() and out.stat().st_size > 5000

    def test_creates_missing_directories(self, spec, tmp_path):
        anchor = spec.free_cells()[0]
        out = plots.latent_distance_field(
            spec, geo.geodesic_from(spec, anchor), anchor, tmp_path / "a" / "b" / "f.png"
        )
        assert out.exists()


class TestLatentMap:
    def test_writes_a_figure(self, spec, tmp_path):
        latent = geodesic_latent(spec)
        proj = projections.pca(latent)
        out = plots.latent_map(
            spec, proj.coords, np.arange(len(latent)), tmp_path / "m.png", explained=proj.explained
        )
        assert out.exists()

    def test_handles_subdivided_grids(self, spec, tmp_path):
        from latentmine import sampling

        _, index = sampling.goal_grid(spec, subdivisions=2)
        rng = np.random.default_rng(0)
        coords = rng.normal(size=(len(index), 2))
        assert plots.latent_map(spec, coords, index, tmp_path / "m.png").exists()


class TestColourHelpers:
    def test_position_colours_are_in_gamut(self, spec):
        world = geo.cells_to_world(geo.free_cell_array(spec), spec.scaling)
        rgb = plots._position_to_rgb(world, spec)
        assert rgb.shape == (len(world), 3)
        assert (rgb >= 0).all() and (rgb <= 1).all()

    def test_neighbouring_cells_get_neighbouring_colours(self, spec):
        # What makes the projection panel readable: a colour jump would look
        # like a discontinuity in the latent space that is not there.
        world = geo.cells_to_world(geo.free_cell_array(spec), spec.scaling)
        rgb = plots._position_to_rgb(world, spec)
        cells = geo.free_cell_array(spec)
        index = {tuple(c): k for k, c in enumerate(cells)}
        for (i, j), k in index.items():
            if (i + 1, j) in index:
                assert np.abs(rgb[k] - rgb[index[(i + 1, j)]]).max() < 0.2

    def test_coordinate_colours_survive_a_degenerate_axis(self):
        # A projection that collapses to a line must not divide by zero.
        coords = np.stack([np.linspace(0, 1, 9), np.zeros(9)], axis=1)
        rgb = plots._coords_to_rgb(coords)
        assert np.isfinite(rgb).all()


class TestRolloutPanel:
    def test_writes_a_figure(self, spec, tmp_path):
        steps = 40
        rng = np.random.default_rng(0)
        positions = rng.uniform(4, 28, size=(steps, 2))
        out = plots.rollout_panel(
            spec,
            positions,
            latent_to_goal=np.linspace(10, 1, steps),
            true_to_goal=np.linspace(30, 2, steps),
            goal_xy=np.array([20.0, 30.0]),
            out=tmp_path / "r.png",
        )
        assert out.exists()


class TestMetricSummary:
    def _summaries(self):
        out = []
        for name in L.names():
            spec = L.get(name)
            latent = geodesic_latent(spec)
            d = np.sqrt(((latent[:, None, :] - latent[None, :, :]) ** 2).sum(-1))
            out.append(M.aggregate([M.evaluate(spec, d) for _ in range(3)]))
        return out

    def test_writes_a_figure_for_the_whole_maze_set(self, tmp_path):
        assert plots.metric_summary(self._summaries(), tmp_path / "s.png").exists()

    def test_undefined_metrics_do_not_become_zero_bars(self, tmp_path):
        # open_room and loop have no wall-facing pairs, so the ratio is
        # undefined; the figure labels that rather than drawing a zero.
        summaries = self._summaries()
        undefined = [s["maze"] for s in summaries if not np.isfinite(s["wall_crossing_ratio"]["mean"])]
        assert set(undefined) == {"open_room", "loop"}
        assert plots.metric_summary(summaries, tmp_path / "s.png").exists()

    def test_empty_input_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="no summaries"):
            plots.metric_summary([], tmp_path / "s.png")
