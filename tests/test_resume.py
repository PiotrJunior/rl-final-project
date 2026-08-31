"""Resume tests.

Split three ways: the checkpoint store on its own (fast, no JAX), the drift
guard on the vendored `train_fn` copy, and a real kill-and-resume cycle.

The last of these is the one that matters. LLD section 3.5 makes it the
milestone for this step: a run is only resumable if it has actually been
killed and resumed, and the training curve shows no discontinuity at the seam.
"""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from latentmine.train import resume
from latentmine.train.resume import ResumeError, ResumePoint


def _point(value, epoch=1, config_hash="abc"):
    import numpy as np

    return ResumePoint(
        training_state={"w": np.full((4,), float(value), dtype=np.float32)},
        env_state={"q": np.arange(3, dtype=np.float32)},
        buffer_state={"data": np.zeros((2, 2), dtype=np.float32)},
        key=np.array([0, value], dtype=np.uint32),
        next_epoch=epoch,
        training_walltime=1.5 * value,
        config_hash=config_hash,
    )


def _template():
    return _point(0).to_payload()


class TestStore:
    def test_round_trip(self, tmp_path):
        resume.save(tmp_path, _point(7, epoch=3))
        got = resume.load(tmp_path, _template())
        assert got.next_epoch == 3
        assert got.training_walltime == pytest.approx(10.5)
        assert got.training_state["w"].tolist() == [7.0] * 4

    def test_missing_checkpoint_is_none_not_an_error(self, tmp_path):
        assert resume.load(tmp_path, _template()) is None

    def test_slots_alternate_so_the_previous_state_survives(self, tmp_path):
        resume.save(tmp_path, _point(1))
        first = resume.read_pointer(tmp_path)["slot"]
        resume.save(tmp_path, _point(2))
        second = resume.read_pointer(tmp_path)["slot"]
        assert first != second
        assert resume._slot_path(tmp_path, first).exists()
        assert resume._slot_path(tmp_path, second).exists()

    def test_a_torn_write_falls_back_to_the_other_slot(self, tmp_path):
        resume.save(tmp_path, _point(1, epoch=1))
        resume.save(tmp_path, _point(2, epoch=2))
        current = resume.read_pointer(tmp_path)["slot"]
        # Simulate a crash partway through writing the newest slot.
        resume._slot_path(tmp_path, current).write_bytes(b"truncated garbage")
        recovered = resume.load(tmp_path, _template())
        assert recovered.next_epoch == 1, "should fall back to the older slot"

    def test_both_slots_corrupt_is_an_error_not_a_silent_restart(self, tmp_path):
        resume.save(tmp_path, _point(1))
        resume.save(tmp_path, _point(2))
        for slot in resume.SLOTS:
            resume._slot_path(tmp_path, slot).write_bytes(b"garbage")
        with pytest.raises(ResumeError, match="no readable resume checkpoint"):
            resume.load(tmp_path, _template())

    def test_config_mismatch_refuses_rather_than_restarting(self, tmp_path):
        # Silently starting a multi-hour run from zero is worse than failing.
        resume.save(tmp_path, _point(1, config_hash="aaaa"))
        with pytest.raises(ResumeError, match="refusing to resume"):
            resume.load(tmp_path, _template(), config_hash="bbbb")

    def test_matching_config_hash_is_accepted(self, tmp_path):
        resume.save(tmp_path, _point(1, config_hash="aaaa"))
        assert resume.load(tmp_path, _template(), config_hash="aaaa") is not None

    def test_unknown_format_version_is_refused(self, tmp_path):
        resume.save(tmp_path, _point(1))
        path = resume._pointer_path(tmp_path)
        pointer = json.loads(path.read_text())
        pointer["format_version"] = 99
        path.write_text(json.dumps(pointer))
        with pytest.raises(ResumeError, match="format_version"):
            resume.load(tmp_path, _template())

    def test_no_temporary_files_are_left_behind(self, tmp_path):
        resume.save(tmp_path, _point(1))
        assert not list(tmp_path.glob("*.tmp"))

    def test_clear_removes_everything(self, tmp_path):
        resume.save(tmp_path, _point(1))
        resume.save(tmp_path, _point(2))
        resume.clear(tmp_path)
        assert resume.load(tmp_path, _template()) is None
        assert not list(tmp_path.glob("state_*"))

    def test_clear_on_an_empty_directory_is_harmless(self, tmp_path):
        resume.clear(tmp_path)


@pytest.mark.slow
class TestVendoredCopy:
    def test_upstream_has_not_drifted(self):
        """If this fails, upstream's crl.py changed and the vendored copy in
        crl_resumable.py must be re-derived from it - not silently trusted."""
        pytest.importorskip("jaxgcrl")
        from latentmine.train import crl_resumable

        assert crl_resumable.upstream_crl_sha256() == crl_resumable.UPSTREAM_CRL_SHA256, (
            "third_party/JaxGCRL/jaxgcrl/agents/crl/crl.py has changed. Re-derive "
            "src/latentmine/train/crl_resumable.py from it and update UPSTREAM_CRL_SHA256."
        )

    def test_it_is_a_crl_so_hyperparameters_carry_over(self):
        pytest.importorskip("jaxgcrl")
        from jaxgcrl.agents.crl import CRL

        from latentmine.train.crl_resumable import ResumableCRL

        assert issubclass(ResumableCRL, CRL)

    def test_the_resume_hooks_are_present(self):
        import inspect

        pytest.importorskip("jaxgcrl")
        from latentmine.train.crl_resumable import ResumableCRL

        source = inspect.getsource(ResumableCRL.train_fn)
        for hook in ("resume hook 1", "resume hook 2", "resume hook 3"):
            assert hook in source, f"{hook} missing from the vendored copy"


# A short run driven through the CLI, so the child process can be killed the
# way a real interruption would kill it. Sized so a full run takes well under
# a minute: each test launches two of these.
_EPOCHS = 4
_SMOKE = [
    "--maze",
    "two_rooms",
    "--env",
    "simple",
    "--profile",
    "smoke",
    "--num-evals",
    str(_EPOCHS),
    "--total-env-steps",
    "60000",
]


def _steps(output: str) -> list[int]:
    """Env-step counts from the progress lines of a run's stdout."""
    return [
        int(line.split("step")[1].split()[0].replace(",", ""))
        for line in output.splitlines()
        if line.startswith("[")
    ]


def _run_cli(runs_dir: Path, extra=(), kill_after_epochs: int | None = None):
    """Launch training as a subprocess; optionally SIGKILL it mid-run."""
    env = {**os.environ, "PYTHONPATH": "src", "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "latentmine.train.run_crl", *_SMOKE, "--runs-dir", str(runs_dir), *extra],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    lines = []
    epochs = 0
    try:
        for line in proc.stdout:
            lines.append(line)
            if line.startswith("["):
                epochs += 1
                if kill_after_epochs is not None and epochs >= kill_after_epochs:
                    # SIGKILL, not SIGTERM: no cleanup handler runs, which is
                    # the case the atomic writes exist for.
                    proc.kill()
                    break
        proc.wait(timeout=180)
    finally:
        # Never leak a training subprocess. If pytest itself is interrupted,
        # an orphaned run keeps burning CPU and slows every later test.
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)
        proc.stdout.close()
    return proc.returncode, "".join(lines)


@pytest.mark.slow
class TestKillAndResume:
    """The milestone of LLD section 3.5: kill a run and continue it."""

    @pytest.fixture(autouse=True)
    def _upstream(self):
        pytest.importorskip("jaxgcrl.envs.simple_maze")

    def test_a_killed_run_resumes_where_it_stopped(self, tmp_path):
        code, out = _run_cli(tmp_path, kill_after_epochs=2)
        assert code != 0, "the run should have been killed, not finished"

        run_dir = next(tmp_path.iterdir())
        pointer = resume.read_pointer(run_dir / resume.RESUME_DIRNAME)
        assert pointer is not None, "no resume checkpoint was written"
        killed_at = pointer["next_epoch"]
        assert 0 < killed_at < _EPOCHS

        steps_before = _steps(out)
        one_epoch = steps_before[1] - steps_before[0] if len(steps_before) > 1 else 0

        code2, out2 = _run_cli(tmp_path)
        assert code2 == 0, out2[-3000:]
        assert "resuming from epoch" in out2

        steps_after = _steps(out2)
        # The resumed process must continue the counter, not restart it. It
        # may repeat the epoch it was killed during - the checkpoint is
        # written after that epoch's progress line, so a kill in between costs
        # that epoch - but it must never go backwards further than that.
        assert steps_after[0] >= steps_before[-1] - one_epoch, (
            f"resumed at {steps_after[0]} having already reached {steps_before[-1]}"
        )
        # And it runs exactly the epochs the checkpoint says are left.
        assert len(steps_after) == _EPOCHS - killed_at

        # Finishing clears the resume state, so re-running is a fresh run.
        assert resume.read_pointer(run_dir / resume.RESUME_DIRNAME) is None

    def test_the_seam_never_skips_work(self, tmp_path):
        """The guarantee is that a crash costs *at most* one epoch and skips
        none. A resumed run may repeat the epoch it died during, because the
        resume checkpoint is written after that epoch's progress line - so a
        repeated step count is expected. A step count that jumps forward by
        more than one epoch is not: that would be lost training, and the
        curve would be a lie."""
        code, out = _run_cli(tmp_path, kill_after_epochs=2)
        assert code != 0
        code2, out2 = _run_cli(tmp_path)
        assert code2 == 0

        steps = _steps(out) + _steps(out2)
        deltas = [b - a for a, b in zip(steps, steps[1:], strict=False)]
        epoch = max(deltas)
        assert all(d in (0, epoch) for d in deltas), (
            f"step deltas across the seam were {deltas}; expected only 0 (a repeated "
            f"epoch) or {epoch} (a normal one)"
        )
        assert deltas.count(0) <= 1, f"more than one epoch was repeated: {deltas}"
        # The run still reaches the end of its schedule.
        assert len(_steps(out2)) + len(_steps(out)) >= _EPOCHS

    def test_resume_never_starts_from_scratch(self, tmp_path):
        code, _ = _run_cli(tmp_path, kill_after_epochs=2)
        assert code != 0
        code2, out2 = _run_cli(tmp_path, extra=["--resume", "never"])
        assert code2 == 0
        assert "resuming from epoch" not in out2
        steps = [line for line in out2.splitlines() if line.startswith("[")]
        assert len(steps) == _EPOCHS, "with --resume never the whole schedule reruns"

    def test_a_changed_configuration_refuses_to_resume(self, tmp_path):
        code, _ = _run_cli(tmp_path, kill_after_epochs=2)
        assert code != 0
        run_dir = next(tmp_path.iterdir())
        pointer_path = resume._pointer_path(run_dir / resume.RESUME_DIRNAME)
        pointer = json.loads(pointer_path.read_text())
        pointer["config_hash"] = "0" * 16
        pointer_path.write_text(json.dumps(pointer))

        code2, out2 = _run_cli(tmp_path)
        assert code2 != 0
        assert "refusing to resume" in out2


@pytest.mark.slow
def test_sigkill_during_a_write_cannot_corrupt_the_store(tmp_path):
    """Atomicity, exercised directly: kill a process mid-write and confirm the
    previous checkpoint is still loadable."""
    script = f"""
import os, signal, sys
sys.path.insert(0, {str(Path("src").resolve())!r})
import numpy as np
from latentmine.train import resume
from tests.test_resume import _point
d = {str(tmp_path)!r}
resume.save(d, _point(1, epoch=1))
real = resume._atomic_write
def slow(path, data):
    if path.name.startswith("state_"):
        os.kill(os.getpid(), signal.SIGKILL)   # die mid-write of the new slot
    real(path, data)
resume._atomic_write = slow
resume.save(d, _point(2, epoch=2))
"""
    env = {**os.environ, "PYTHONPATH": f"src:{Path.cwd()}"}
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, env=env)
    assert proc.returncode == -signal.SIGKILL

    got = resume.load(tmp_path, _template())
    assert got is not None and got.next_epoch == 1
    assert not list(Path(tmp_path).glob("*.tmp"))


@pytest.mark.slow
class TestCheckpointingDisabled:
    """The timing probe runs without a checkpoint directory. Upstream binds
    `params` only inside the branch that writes one, so returning from such a
    run raises UnboundLocalError - hit for real by the probe."""

    def test_a_run_without_a_checkpoint_dir_returns_normally(self, tmp_path):
        pytest.importorskip("jaxgcrl.envs.simple_maze")
        from latentmine.train.envs import build_envs
        from latentmine.train.presets import make_run_spec
        from latentmine.train.run_crl import build_agent, build_config

        spec = make_run_spec("two_rooms", "simple", profile="smoke", num_evals=1, total_env_steps=40_000)
        train_env, eval_env = build_envs(spec)
        agent = build_agent(spec)
        _, params, _ = agent.train_fn(
            config=build_config(spec, checkpoint_dir=None),
            train_env=train_env,
            eval_env=eval_env,
        )
        alpha, actor, critic = params
        assert set(critic) == {"sa_encoder", "g_encoder"}
        assert not list(tmp_path.iterdir()), "nothing should have been written"
