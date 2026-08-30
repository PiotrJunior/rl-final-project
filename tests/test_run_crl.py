"""Entrypoint tests: argument handling, the dry-run report, and the promise
that `--dry-run` imports no JAX."""

import json
import subprocess
import sys

import pytest

from latentmine.train import run_crl
from latentmine.train.presets import make_run_spec


class TestArguments:
    def test_maze_is_required(self):
        with pytest.raises(SystemExit):
            run_crl.build_parser().parse_args(["--env", "simple"])

    def test_unknown_maze_is_rejected_by_the_parser(self):
        with pytest.raises(SystemExit):
            run_crl.build_parser().parse_args(["--maze", "labyrinth"])

    def test_defaults_resolve_to_the_provisional_budget(self):
        args = run_crl.build_parser().parse_args(["--maze", "two_rooms"])
        spec = run_crl.spec_from_args(args)
        assert (spec.env, spec.preset, spec.seed) == ("simple", "deep", 1)
        assert spec == make_run_spec("two_rooms", "simple")

    def test_overrides_are_applied(self):
        args = run_crl.build_parser().parse_args(
            [
                "--maze",
                "spiral",
                "--env",
                "ant",
                "--preset",
                "deeper",
                "--seed",
                "3",
                "--num-envs",
                "64",
                "--energy-fn",
                "dot",
            ]
        )
        spec = run_crl.spec_from_args(args)
        assert (spec.num_envs, spec.energy_fn, spec.preset, spec.seed) == (64, "dot", "deeper", 3)

    def test_unset_options_do_not_override_the_budget(self):
        args = run_crl.build_parser().parse_args(["--maze", "two_rooms"])
        assert run_crl.spec_from_args(args).num_envs == make_run_spec("two_rooms", "simple").num_envs


class TestDescribe:
    def test_reports_the_facts_a_launch_decision_needs(self):
        text = run_crl.describe(make_run_spec("two_rooms", "simple"))
        for expected in ("run_id", "replay buffer", "utd ratio", "prefill", "per epoch", "crash costs"):
            assert expected in text

    def test_warns_when_updates_per_step_fall_well_below_upstream(self):
        # Halving episode_length halves the gradient updates per env step.
        quiet = make_run_spec("two_rooms", "simple", episode_length=1001, num_envs=128)
        loud = quiet.evolve(episode_length=251)
        assert "train-step-multiplier" not in run_crl.describe(quiet)
        assert "train-step-multiplier" in run_crl.describe(loud)

    def test_states_the_divisibility_constraint_it_checked(self):
        assert "% 256 == 0 OK" in run_crl.describe(make_run_spec("two_rooms", "simple"))


class TestDryRun:
    def test_exits_clean_and_writes_nothing(self, tmp_path, capsys):
        assert run_crl.main(["--maze", "two_rooms", "--runs-dir", str(tmp_path), "--dry-run"]) == 0
        assert list(tmp_path.iterdir()) == []
        assert "dry run" in capsys.readouterr().out

    def test_json_mode_emits_a_manifest(self, tmp_path, capsys):
        run_crl.main(["--maze", "spiral", "--runs-dir", str(tmp_path), "--dry-run", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["maze"]["name"] == "spiral"
        assert payload["arch"]["n_hidden"] == 4

    def test_configuration_error_exits_two_with_a_message(self, capsys):
        code = run_crl.main(["--maze", "two_rooms", "--num-envs", "100", "--dry-run"])
        assert code == 2
        assert "divisible by batch_size" in capsys.readouterr().err

    def test_imports_no_jax(self, tmp_path):
        """The whole point of --dry-run on a laptop: a one-second check that a
        multi-hour run is configured correctly, with no training stack."""
        code = (
            "import sys, latentmine.train.run_crl as r;"
            f" r.main(['--maze','two_rooms','--runs-dir',{str(tmp_path)!r},'--dry-run']);"
            " print('LOADED' if {'jax','brax','mujoco'} & set(sys.modules) else 'CLEAN')"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert out.stdout.strip().endswith("CLEAN"), out.stdout


class TestProgressCallback:
    def test_appends_scalar_metrics_and_skips_arrays(self, tmp_path):
        spec = make_run_spec("two_rooms", "simple")
        run_dir = tmp_path / spec.run_id
        run_dir.mkdir(parents=True)
        progress = run_crl.progress_printer(spec, run_dir, wandb_run=None)

        progress(1000, {"training/critic_loss": 0.5, "eval/episode_success": 0.25})
        progress(2000, {"training/critic_loss": 0.4, "training/entropy": [1, 2, 3]})

        records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
        assert [r["step"] for r in records] == [1000, 2000]
        assert records[0]["eval/episode_success"] == 0.25
        assert "training/entropy" not in records[1], "non-scalar metrics must be dropped"

    def test_tolerates_the_full_upstream_signature(self, tmp_path):
        # train_fn calls progress_fn(step, metrics, make_policy, params, env, do_render=...)
        spec = make_run_spec("two_rooms", "simple")
        run_dir = tmp_path / "r"
        run_dir.mkdir()
        progress = run_crl.progress_printer(spec, run_dir, wandb_run=None)
        progress(10, {"training/sps": 12.0}, lambda p: None, {}, object(), do_render=True)
        assert (run_dir / "metrics.jsonl").exists()
