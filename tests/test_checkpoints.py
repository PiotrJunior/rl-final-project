"""Checkpoint-loading tests.

The pickle records no architecture, so the manifest is what makes a checkpoint
loadable at all. These cover the discovery and refusal logic without JAX, and
the real rebuild against a trained run under `slow`.
"""

import pickle

import pytest

from latentmine import checkpoints
from latentmine.checkpoints import CheckpointError
from latentmine.train import manifest
from latentmine.train.presets import make_run_spec


@pytest.fixture
def run_dir(tmp_path):
    spec = make_run_spec("two_rooms", "simple", profile="smoke")
    d = tmp_path / spec.run_id
    manifest.write(spec, d)
    (d / "ckpt").mkdir()
    for step in (1000, 5000, 20000):
        payload = ({"log_alpha": 0.0}, {"params": {}}, {"sa_encoder": {}, "g_encoder": {}})
        (d / "ckpt" / f"step_{step}.pkl").write_bytes(pickle.dumps(payload))
    return d


class TestDiscovery:
    def test_lists_checkpoints_oldest_first(self, run_dir):
        assert [s for s, _ in checkpoints.list_checkpoints(run_dir)] == [1000, 5000, 20000]

    def test_no_ckpt_directory_is_empty_not_an_error(self, tmp_path):
        assert checkpoints.list_checkpoints(tmp_path) == []

    def test_resolves_last_first_and_exact(self, run_dir):
        assert checkpoints.resolve_checkpoint(run_dir, "last")[0] == 20000
        assert checkpoints.resolve_checkpoint(run_dir, "first")[0] == 1000
        assert checkpoints.resolve_checkpoint(run_dir, 5000)[0] == 5000

    def test_a_missing_step_lists_what_is_available(self, run_dir):
        with pytest.raises(CheckpointError, match=r"available: \[1000, 5000, 20000\]"):
            checkpoints.resolve_checkpoint(run_dir, 1234)

    def test_no_checkpoints_at_all_is_an_error(self, tmp_path):
        with pytest.raises(CheckpointError, match="no checkpoints"):
            checkpoints.resolve_checkpoint(tmp_path)

    def test_non_checkpoint_files_are_ignored(self, run_dir):
        (run_dir / "ckpt" / "notes.txt").write_text("hi")
        (run_dir / "ckpt" / "final").write_bytes(b"x")
        assert len(checkpoints.list_checkpoints(run_dir)) == 3


class TestPayloadValidation:
    def test_rejects_a_non_triple(self, tmp_path):
        path = tmp_path / "bad.pkl"
        path.write_bytes(pickle.dumps({"not": "a tuple"}))
        with pytest.raises(CheckpointError, match="expected a 3-tuple"):
            checkpoints.load_params(path)

    def test_rejects_a_critic_without_both_encoders(self, tmp_path):
        path = tmp_path / "bad.pkl"
        path.write_bytes(pickle.dumps(({}, {}, {"sa_encoder": {}})))
        with pytest.raises(CheckpointError, match=r"missing \['g_encoder'\]"):
            checkpoints.load_params(path)

    def test_accepts_the_upstream_shape(self, run_dir):
        alpha, actor, critic = checkpoints.load_params(run_dir / "ckpt" / "step_1000.pkl")
        assert set(critic) == {"sa_encoder", "g_encoder"}


class TestManifestRequirement:
    def test_loading_without_a_manifest_refuses(self, run_dir):
        (run_dir / "manifest.json").unlink()
        with pytest.raises(Exception, match="records no architecture"):
            checkpoints.load_encoders(run_dir)


@pytest.mark.slow
class TestAgainstARealRun:
    """Rebuild encoders from an actual trained checkpoint."""

    @pytest.fixture(scope="class")
    def trained(self, tmp_path_factory):
        pytest.importorskip("jaxgcrl.envs.simple_maze")
        from latentmine.train.run_crl import train

        spec = make_run_spec("two_rooms", "simple", profile="smoke", num_evals=2, total_env_steps=60_000)
        return train(
            spec,
            tmp_path_factory.mktemp("runs"),
            wandb_enabled=False,
            wandb_mode="offline",
            wandb_project="test",
        )

    def test_encoders_rebuild_and_have_the_advertised_shapes(self, trained):
        import numpy as np

        enc = checkpoints.load_encoders(trained)
        dims = enc.manifest["dims"]

        goals = np.zeros((7, dims["goal_size"]), dtype=np.float32)
        assert np.asarray(enc.psi(goals)).shape == (7, enc.repr_dim)

        sa = np.zeros((7, dims["sa_encoder_input"]), dtype=np.float32)
        assert np.asarray(enc.phi(sa)).shape == (7, enc.repr_dim)

        obs = np.zeros((7, dims["obs_size"]), dtype=np.float32)
        mean, log_std = enc.actor(obs)
        assert np.asarray(mean).shape == (7, dims["action_size"])

    def test_psi_takes_only_xy(self, trained):
        # The fact the whole dense-latent-map approach rests on.
        enc = checkpoints.load_encoders(trained)
        assert enc.manifest["dims"]["g_encoder_input"] == 2

    def test_psi_is_not_constant(self, trained):
        # A collapsed goal encoder makes every downstream metric noise.
        import numpy as np

        enc = checkpoints.load_encoders(trained)
        from latentmine import sampling

        xy, _ = sampling.goal_grid(enc.maze_spec)
        latents = np.asarray(enc.psi(xy.astype(np.float32)))
        assert latents.std() > 1e-4, "psi collapsed to a point"

    def test_the_maze_is_rebuilt_from_the_manifest(self, trained):
        enc = checkpoints.load_encoders(trained)
        assert enc.maze_spec.name == "two_rooms"
        assert len(enc.maze_spec.free_cells()) == 57

    def test_selecting_an_earlier_checkpoint_gives_different_weights(self, trained):
        import numpy as np

        steps = [s for s, _ in checkpoints.list_checkpoints(trained)]
        assert len(steps) >= 2
        first = checkpoints.load_encoders(trained, step="first")
        last = checkpoints.load_encoders(trained, step="last")
        assert first.step < last.step
        probe = np.zeros((4, 2), dtype=np.float32)
        assert not np.allclose(np.asarray(first.psi(probe)), np.asarray(last.psi(probe)))

    def test_the_series_loader_walks_every_checkpoint(self, trained):
        series = checkpoints.load_encoder_series(trained)
        assert [e.step for e in series] == [s for s, _ in checkpoints.list_checkpoints(trained)]
