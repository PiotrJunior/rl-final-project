"""Decoder tests.

The split logic carries most of the weight here: a random split lets the
decoder memorise a lookup table and makes reconstruction error meaningless, so
these check that whole regions really are held out and that the scaler is
fitted on training data only.
"""

import numpy as np
import pytest

from latentmine.decoder import data, model
from latentmine.mazes import geometry as geo
from latentmine.mazes import layouts as L


class TestStateGroups:
    def test_ant_groups_tile_the_29_dimensions(self):
        groups = model.state_groups("ant")
        covered = sorted(groups.values())
        assert covered[0][0] == 0 and covered[-1][1] == 29
        for (_, end), (start, _) in zip(covered[:-1], covered[1:], strict=True):
            assert end == start, "groups must tile without gaps or overlap"

    def test_simple_groups_tile_the_4_dimensions(self):
        covered = sorted(model.state_groups("simple").values())
        assert covered[0][0] == 0 and covered[-1][1] == 4

    def test_xy_is_the_first_two_dimensions_in_both(self):
        # The dims the goal encoder also sees; the decoder's error on them is
        # directly comparable with psi's own inversion.
        for env in ("ant", "simple"):
            assert model.state_groups(env)["xy"] == (0, 2)

    def test_unknown_env_is_rejected(self):
        with pytest.raises(ValueError, match="unknown env"):
            model.state_groups("humanoid")

    def test_the_action_block_follows_the_state(self):
        assert model.action_group("ant", 29, 8) == {"action": (29, 37)}


class TestSpatialSplit:
    def test_a_whole_region_is_held_out(self):
        spec = L.get("four_rooms")
        index = np.arange(len(spec.free_cells()))
        split = data.spatial_split(spec, index, hold_out="d")
        cells = geo.free_cell_array(spec)
        test_regions = {spec.regions[i][j] for (i, j) in cells[split.test]}
        assert test_regions == {"d"}

    def test_the_held_out_region_appears_nowhere_in_training(self):
        spec = L.get("four_rooms")
        index = np.arange(len(spec.free_cells()))
        split = data.spatial_split(spec, index, hold_out="d")
        cells = geo.free_cell_array(spec)
        train_regions = {spec.regions[i][j] for (i, j) in cells[split.train]}
        assert "d" not in train_regions

    def test_splits_are_disjoint_and_complete(self):
        spec = L.get("two_rooms")
        index = np.arange(len(spec.free_cells()))
        split = data.spatial_split(spec, index)
        joined = np.concatenate([split.train, split.val, split.test])
        assert len(joined) == len(index)
        assert len(set(joined.tolist())) == len(index)

    def test_validation_comes_from_training_regions(self):
        # Early stopping watches val; if it came from the held-out region it
        # would leak exactly the generalisation being measured.
        spec = L.get("four_rooms")
        index = np.arange(len(spec.free_cells()))
        split = data.spatial_split(spec, index, hold_out="d")
        cells = geo.free_cell_array(spec)
        assert "d" not in {spec.regions[i][j] for (i, j) in cells[split.val]}

    def test_unknown_region_is_rejected(self):
        spec = L.get("two_rooms")
        with pytest.raises(ValueError, match="unknown region"):
            data.spatial_split(spec, np.arange(len(spec.free_cells())), hold_out="z")

    def test_mazes_without_rooms_fall_back_to_a_geodesic_arc(self):
        spec = L.get("spiral")
        split = data.spatial_split(spec, np.arange(len(spec.free_cells())))
        assert split.kind == "geodesic arc"
        assert len(split.test) > 0

    def test_the_geodesic_arc_is_the_far_end_of_the_corridor(self):
        spec = L.get("spiral")
        split = data.geodesic_split(spec, np.arange(len(spec.free_cells())), test_fraction=0.25)
        order = geo.geodesic_from(spec, spec.start_cells()[0])
        assert order[split.test].min() >= order[split.train].max() - 1e-9

    def test_the_split_describes_itself(self):
        spec = L.get("four_rooms")
        split = data.spatial_split(spec, np.arange(len(spec.free_cells())), hold_out="d")
        assert "held-out d" in split.describe()

    def test_subdivided_grids_split_by_region_too(self):
        from latentmine import sampling

        spec = L.get("two_rooms")
        _, index = sampling.goal_grid(spec, subdivisions=2)
        split = data.spatial_split(spec, index, hold_out="b")
        cells = geo.free_cell_array(spec)
        assert {spec.regions[i][j] for (i, j) in cells[index[split.test]]} == {"b"}


class TestStandardise:
    def test_training_data_becomes_zero_mean_unit_variance(self):
        rng = np.random.default_rng(0)
        train = rng.normal(loc=5, scale=3, size=(200, 4))
        (scaled_train,), _ = data.standardise(train)
        np.testing.assert_allclose(scaled_train.mean(0), 0, atol=1e-9)
        np.testing.assert_allclose(scaled_train.std(0), 1, atol=1e-9)

    def test_the_scaler_is_fitted_on_training_data_only(self):
        rng = np.random.default_rng(1)
        train = rng.normal(size=(100, 3))
        test = rng.normal(loc=50, size=(20, 3))
        (_, scaled_test), (mean, std) = data.standardise(train, test)
        np.testing.assert_allclose(scaled_test, (test - mean) / std)
        assert scaled_test.mean() > 5, "held-out data must not be re-centred"

    def test_constant_dimensions_do_not_divide_by_zero(self):
        train = np.concatenate([np.ones((10, 1)), np.arange(10)[:, None]], axis=1)
        (scaled,), _ = data.standardise(train)
        assert np.isfinite(scaled).all()


@pytest.mark.slow
class TestFit:
    """Train a decoder on a synthetic latent whose inverse is known."""

    def test_it_learns_an_invertible_map(self):
        pytest.importorskip("jaxgcrl")
        from latentmine.decoder.train import fit

        spec = L.get("two_rooms")
        world = geo.cells_to_world(geo.free_cell_array(spec), spec.scaling)
        # A latent that is an invertible linear map of position: the decoder
        # must be able to recover xy from it almost exactly.
        rng = np.random.default_rng(0)
        projection = rng.normal(size=(2, 16))
        latents = world @ projection

        split = data.spatial_split(spec, np.arange(len(world)), hold_out="b")
        result = fit(latents, world, split, {"xy": (0, 2)}, steps=1500, log_every=250)

        assert result.errors["train"]["xy"] < 0.2
        # Held-out region: the real test. A linear map generalises, so this
        # should also be low - if it is not, the training loop is broken.
        assert result.errors["test (held-out regions)"]["xy"] < 0.6
        assert result.decode(latents[:5]).shape == (5, 2)

    def test_all_three_error_numbers_are_reported(self):
        pytest.importorskip("jaxgcrl")
        from latentmine.decoder.train import fit

        spec = L.get("two_rooms")
        world = geo.cells_to_world(geo.free_cell_array(spec), spec.scaling)
        rng = np.random.default_rng(0)
        latents = world @ rng.normal(size=(2, 16))
        split = data.spatial_split(spec, np.arange(len(world)), hold_out="b")
        result = fit(latents, world, split, {"xy": (0, 2)}, steps=400, log_every=200)
        assert set(result.errors) == {"train", "val (same regions)", "test (held-out regions)"}


@pytest.mark.slow
class TestExploitPipeline:
    """The exploitation half end to end, on a real (tiny) checkpoint."""

    @pytest.fixture(scope="class")
    def trained(self, tmp_path_factory):
        pytest.importorskip("jaxgcrl.envs.simple_maze")
        from latentmine.train.presets import make_run_spec
        from latentmine.train.run_crl import train

        spec = make_run_spec("two_rooms", "simple", profile="smoke", num_evals=2, total_env_steps=60_000)
        return train(
            spec,
            tmp_path_factory.mktemp("runs"),
            wandb_enabled=False,
            wandb_mode="offline",
            wandb_project="test",
        )

    def test_the_goal_decoder_reports_all_three_errors(self, trained):
        from latentmine import checkpoints
        from latentmine.exploit import fit_goal_decoder

        encoders = checkpoints.load_encoders(trained)
        fit, _, _, _, split = fit_goal_decoder(encoders, subdivisions=2, steps=600)
        assert set(fit.errors) == {"train", "val (same regions)", "test (held-out regions)"}
        assert split.kind == "region"
        # The whole point of the spatial holdout: the held-out region is
        # harder than held-out samples from training regions. If these were
        # comparable, the split would not be testing generalisation.
        assert fit.errors["test (held-out regions)"]["xy"] > fit.errors["val (same regions)"]["xy"]

    def test_all_four_candidate_paths_are_produced_and_scored(self, trained):
        from latentmine import checkpoints, embed, sampling
        from latentmine.exploit import fit_goal_decoder, waypoint_paths

        encoders = checkpoints.load_encoders(trained)
        fit, xy, latents, _, _ = fit_goal_decoder(encoders, subdivisions=2, steps=400)
        paths = waypoint_paths(encoders, fit, xy, latents, (1, 1), (7, 9))

        assert [p.name for p in paths] == ["straight", "geodesic", "latent_linear", "latent_graph"]
        for path in paths:
            assert 0.0 <= path.valid_fraction <= 1.0
            assert np.isfinite(path.length)
        # The oracle is the calibration point: whatever the latent paths do,
        # the geodesic route is legal by construction.
        oracle = next(p for p in paths if p.name == "geodesic")
        assert oracle.valid_fraction == 1.0
        assert len(sampling.cell_centres(encoders.maze_spec)) == len(
            embed.embed_goals(encoders, sampling.cell_centres(encoders.maze_spec))
        )

    def test_bottleneck_waypoints_land_in_free_space(self, trained):
        from latentmine import checkpoints, embed, sampling
        from latentmine.exploit import bottleneck_waypoints
        from latentmine.mazes import geometry as geo

        encoders = checkpoints.load_encoders(trained)
        spec = encoders.maze_spec
        latents = embed.embed_goals(encoders, sampling.cell_centres(spec))
        for point in bottleneck_waypoints(encoders, latents, top=3):
            assert spec.is_free(*geo.world_to_cell(tuple(point), spec.scaling))
