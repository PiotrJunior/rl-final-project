"""Integration tests for env construction and a real (tiny) training run.

All marked `slow`: they need the pinned upstream installed.

    pip install -e third_party/JaxGCRL
    pytest tests -q -m slow

These are what turn the hardcoded numbers in `presets.ENV_SPECS` from an
assumption into a checked fact, and what verify the checkpoint payload the
whole analysis pipeline depends on.
"""

import pickle

import pytest

from latentmine.mazes import layouts
from latentmine.train.presets import ENV_SPECS, make_run_spec

pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def _upstream():
    pytest.importorskip("jaxgcrl.envs.simple_maze")
    from latentmine.mazes import register

    yield
    register.uninstall()


class TestEnvConstruction:
    @pytest.mark.parametrize("maze", list(layouts.names()))
    def test_every_maze_builds_a_simple_env(self, maze):
        from latentmine.train.envs import build_env

        env = build_env(make_run_spec(maze, "simple"))
        assert env.state_dim == 4

    def test_ant_env_builds(self):
        from latentmine.train.envs import build_env

        env = build_env(make_run_spec("two_rooms", "ant"))
        assert env.state_dim == 29

    @pytest.mark.parametrize("env_name", sorted(ENV_SPECS))
    def test_hardcoded_dimensions_match_the_real_env(self, env_name):
        """`presets.ENV_SPECS` is what makes --dry-run and the manifest work
        without JAX. This is the test that keeps it honest."""
        from latentmine.train.envs import build_env

        spec = ENV_SPECS[env_name]
        env = build_env(make_run_spec("two_rooms", env_name))
        assert env.state_dim == spec.state_dim
        assert env.action_size == spec.action_size
        assert len(env.goal_indices) == spec.goal_size
        assert env.observation_size == spec.obs_size

    def test_goal_is_the_first_two_state_dimensions(self):
        # What makes psi a function of (x, y) alone, and the whole dense
        # latent-map approach possible (LLD 2.4).
        from latentmine.train.envs import build_env

        env = build_env(make_run_spec("two_rooms", "simple"))
        assert list(env.goal_indices) == [0, 1]

    def test_wall_geometry_reaches_the_physics_model(self):
        from latentmine.train.envs import build_env

        spec = make_run_spec("two_rooms", "simple")
        env = build_env(spec)
        maze = spec.maze_spec
        n_walls = maze.n_rows * maze.n_cols - len(maze.free_cells())
        # One geom per wall cell, plus whatever the base asset contributes.
        assert env.sys.ngeom >= n_walls

    def test_start_and_goal_sets_come_from_our_spec(self):
        from latentmine.train.envs import build_env

        spec = make_run_spec("four_rooms", "simple")
        env = build_env(spec)
        assert len(env.possible_starts) == len(spec.maze_spec.start_cells())
        assert len(env.possible_goals) == len(spec.maze_spec.goal_cells())

    def test_eval_variant_restricts_goals_but_not_geometry(self):
        from latentmine.train.envs import build_envs

        spec = make_run_spec("four_rooms", "simple", eval_goal_region="d")
        train_env, eval_env = build_envs(spec)
        assert train_env is not eval_env
        assert len(eval_env.possible_goals) < len(train_env.possible_goals)
        assert eval_env.sys.ngeom == train_env.sys.ngeom

    def test_no_eval_region_reuses_the_training_env(self):
        from latentmine.train.envs import build_envs

        train_env, eval_env = build_envs(make_run_spec("two_rooms", "simple"))
        assert train_env is eval_env


class TestEnvStepping:
    def test_reset_and_step_produce_the_advertised_shapes(self):
        import jax

        from latentmine.train.envs import build_env

        spec = make_run_spec("two_rooms", "simple")
        env = build_env(spec)
        state = jax.jit(env.reset)(jax.random.PRNGKey(0))
        assert state.obs.shape == (spec.env_spec.obs_size,)

        action = jax.numpy.zeros(spec.env_spec.action_size)
        assert jax.jit(env.step)(state, action).obs.shape == state.obs.shape

    def test_the_agent_starts_inside_the_maze(self):
        import jax

        from latentmine.mazes import geometry as geo
        from latentmine.train.envs import build_env

        spec = make_run_spec("two_rooms", "simple")
        env = build_env(spec)
        state = jax.jit(env.reset)(jax.random.PRNGKey(0))
        cell = geo.world_to_cell(tuple(float(v) for v in state.obs[:2]), spec.maze_spec.scaling)
        assert spec.maze_spec.is_free(*cell), f"reset put the agent at {cell}, a wall"


class TestEndToEnd:
    """A real but tiny run. Slow even on a GPU; the point is that every piece
    fits together, not that anything is learned."""

    @pytest.fixture
    def tiny(self):
        return make_run_spec(
            "two_rooms",
            "simple",
            total_env_steps=50_000,
            episode_length=101,
            num_envs=64,
            num_evals=1,
            num_eval_envs=8,
        ).evolve(min_replay_size=100, max_replay_size=1000)

    def test_a_run_completes_and_leaves_a_loadable_artifact_set(self, tiny, tmp_path):
        from latentmine.train import manifest
        from latentmine.train.run_crl import train

        run_dir = train(tiny, tmp_path, wandb_enabled=False, wandb_mode="offline", wandb_project="test")

        assert manifest.load(run_dir)["run_id"] == tiny.run_id
        assert (run_dir / "metrics.jsonl").read_text().strip(), "no metrics recorded"

        checkpoints = sorted((run_dir / "ckpt").glob("step_*.pkl"))
        assert checkpoints, "no checkpoint written"

        # The payload the analysis pipeline depends on (LLD 2.7).
        with checkpoints[-1].open("rb") as fh:
            alpha, actor, critic = pickle.load(fh)
        assert set(critic) == {"sa_encoder", "g_encoder"}
        assert "log_alpha" in alpha
        assert actor

    def test_checkpoint_cadence_follows_num_evals(self, tiny, tmp_path):
        from latentmine.train.run_crl import train

        spec = tiny.evolve(num_evals=2, total_env_steps=60_000)
        run_dir = train(spec, tmp_path, wandb_enabled=False, wandb_mode="offline", wandb_project="test")
        assert len(list((run_dir / "ckpt").glob("step_*.pkl"))) == spec.num_evals
