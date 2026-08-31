"""Rollout tests.

The important one is `TestSanityGate`: LLD section 12 makes "latent distance
to the goal falls as the agent approaches it" the gate on the whole analysis
pipeline. A checkpoint that fails it is broken, and no projection or metric
downstream will mean anything.
"""

import numpy as np
import pytest

from latentmine.rollouts import Rollouts


def _fake(episodes=3, steps=10, obs_size=6, action_size=2, state_dim=4):
    rng = np.random.default_rng(0)
    return Rollouts(
        obs=rng.normal(size=(episodes, steps, obs_size)).astype(np.float32),
        actions=rng.normal(size=(episodes, steps, action_size)).astype(np.float32),
        goal=rng.normal(size=(episodes, 2)).astype(np.float32),
        success=np.array([1.0, 0.0, 1.0][:episodes]),
        dist=rng.uniform(size=(episodes, steps)),
        state_dim=state_dim,
    )


class TestContainer:
    def test_shapes_and_accessors(self):
        r = _fake()
        assert r.n_episodes == 3
        assert r.positions.shape == (3, 10, 2)
        assert r.states().shape == (3, 10, 4)

    def test_positions_are_the_first_two_observation_dims(self):
        r = _fake()
        np.testing.assert_allclose(r.positions, r.obs[..., :2])

    def test_round_trips_through_npz(self, tmp_path):
        r = _fake()
        loaded = Rollouts.load(r.save(tmp_path / "roll.npz"))
        np.testing.assert_allclose(loaded.obs, r.obs)
        np.testing.assert_allclose(loaded.goal, r.goal)
        assert loaded.state_dim == r.state_dim

    def test_save_creates_missing_directories(self, tmp_path):
        path = _fake().save(tmp_path / "nested" / "deep" / "roll.npz")
        assert path.exists()


@pytest.mark.slow
class TestSanityGate:
    """A real rollout from a real checkpoint."""

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

    @pytest.fixture(scope="class")
    def collected(self, trained):
        from latentmine import checkpoints, rollouts
        from latentmine.train.envs import build_env
        from latentmine.train.manifest import spec_from_manifest

        enc = checkpoints.load_encoders(trained)
        env = build_env(spec_from_manifest(enc.manifest))
        return enc, rollouts.collect(enc, env, n_episodes=2, steps=60, seed=0)

    def test_rollouts_have_the_expected_shape(self, collected):
        enc, roll = collected
        dims = enc.manifest["dims"]
        assert roll.obs.shape == (2, 60, dims["obs_size"])
        assert roll.actions.shape == (2, 60, dims["action_size"])
        assert roll.goal.shape == (2, 2)

    def test_the_agent_stays_inside_the_maze(self, collected):
        from latentmine.mazes import geometry as geo

        enc, roll = collected
        spec = enc.maze_spec
        outside = 0
        for point in roll.positions.reshape(-1, 2):
            cell = geo.world_to_cell(tuple(float(v) for v in point), spec.scaling)
            inside_grid = 0 <= cell[0] < spec.n_rows and 0 <= cell[1] < spec.n_cols
            if not (inside_grid and spec.is_free(*cell)):
                outside += 1
        # Contact dynamics let it clip a wall briefly; a large fraction outside
        # would mean the maze geometry never reached the physics.
        assert outside / roll.positions.reshape(-1, 2).shape[0] < 0.25

    def test_latent_distance_to_goal_is_well_formed(self, collected):
        from latentmine.rollouts import latent_distance_to_goal

        enc, roll = collected
        d = latent_distance_to_goal(enc, roll)
        assert d.shape == (2, 60)
        assert np.isfinite(d).all()
        assert (d >= 0).all()

    def test_actions_are_within_the_tanh_range(self, collected):
        _, roll = collected
        assert np.abs(roll.actions).max() <= 1.0 + 1e-6

    def test_approach_correlation_runs(self, collected):
        """On a 60k-step checkpoint the critic has barely learned anything, so
        this asserts the statistic is computable and finite, not that it is
        large. Section 12 step 4 requires it to be strongly positive on a real
        run before any analysis is trusted."""
        from latentmine.rollouts import approach_correlation

        enc, roll = collected
        rho = approach_correlation(enc, roll, successful_only=False)
        assert np.isfinite(rho)
        assert -1.0 <= rho <= 1.0
