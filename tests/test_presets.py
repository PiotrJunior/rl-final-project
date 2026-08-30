"""Run-configuration tests.

Everything here runs without JAX. The point of the config layer is that a
misconfigured run fails in a second with an actionable message, rather than
after JAX has compiled and then tripped a bare `assert` inside `train_fn`.
"""

import pytest

from latentmine.mazes import layouts
from latentmine.train.presets import (
    ARCH_PRESETS,
    ENV_SPECS,
    PROVISIONAL_BUDGETS,
    ConfigError,
    make_run_spec,
    suggest_num_envs,
)


class TestArchPresets:
    def test_depth_and_width_are_named_the_right_way_round(self):
        # Upstream calls depth `n_hidden` and width `h_dim`. Getting this
        # backwards silently trains a 256-layer network 4 units wide.
        deep = ARCH_PRESETS["deep"]
        assert (deep.depth, deep.width) == (4, 256)

    def test_deep_is_deeper_than_shallow(self):
        assert ARCH_PRESETS["deep"].depth > ARCH_PRESETS["shallow"].depth
        assert ARCH_PRESETS["deeper"].depth > ARCH_PRESETS["deep"].depth

    def test_a_preset_isolates_depth_from_layernorm(self):
        # deep vs shallow changes both depth and LayerNorm, so it cannot
        # support a claim about depth alone.
        deep, arm = ARCH_PRESETS["deep"], ARCH_PRESETS["deep_noln"]
        assert (deep.depth, deep.width) == (arm.depth, arm.width)
        assert deep.use_ln and not arm.use_ln

    def test_every_preset_records_why_it_exists(self):
        assert all(p.notes.strip() for p in ARCH_PRESETS.values())


class TestEnvSpecs:
    def test_dimensions_match_the_upstream_envs(self):
        # Cross-checked against the MJCF in LLD 2.4; asserted against the real
        # envs by the slow test in test_train_envs.py.
        assert (ENV_SPECS["simple"].state_dim, ENV_SPECS["simple"].action_size) == (4, 2)
        assert (ENV_SPECS["ant"].state_dim, ENV_SPECS["ant"].action_size) == (29, 8)

    def test_obs_size_is_state_plus_goal(self):
        for spec in ENV_SPECS.values():
            assert spec.obs_size == spec.state_dim + spec.goal_size

    def test_transition_row_width(self):
        # observation + action + reward + discount + truncation + traj_id
        assert ENV_SPECS["simple"].transition_floats == 6 + 2 + 4
        assert ENV_SPECS["ant"].transition_floats == 31 + 8 + 4


class TestDerivedQuantities:
    @pytest.fixture
    def spec(self):
        return make_run_spec("two_rooms", "simple", num_envs=128, episode_length=501)

    def test_they_mirror_train_fns_arithmetic(self, spec):
        assert spec.env_steps_per_actor_step == spec.num_envs * spec.unroll_length
        assert spec.num_prefill_env_steps == spec.min_replay_size * spec.num_envs
        assert spec.num_training_steps_per_epoch == (
            (spec.total_env_steps - spec.num_prefill_env_steps)
            // (spec.num_evals * spec.env_steps_per_actor_step)
        )

    def test_actual_total_never_exceeds_the_request(self, spec):
        # Integer division truncates, so the run is a little short of the ask.
        assert spec.actual_total_env_steps <= spec.total_env_steps

    def test_buffer_size_is_the_documented_product(self, spec):
        assert spec.replay_buffer_bytes == (
            spec.max_replay_size * spec.num_envs * spec.env_spec.transition_floats * 4
        )

    def test_ant_buffer_is_larger_than_simple(self):
        simple = make_run_spec("two_rooms", "simple", num_envs=128)
        ant = make_run_spec("two_rooms", "ant", num_envs=128)
        assert ant.replay_buffer_bytes > simple.replay_buffer_bytes

    def test_run_id_is_self_describing_and_stable(self, spec):
        assert spec.run_id == "simple_two_rooms_d4_w256_r64_norm_s1"
        assert spec.evolve(seed=7).run_id.endswith("_s7")
        assert "d2_w256" in spec.evolve(preset="shallow").run_id


class TestValidation:
    def test_batch_divisibility_is_checked_before_launch(self):
        # CRL.check_config asserts this; we want the failure at parse time.
        with pytest.raises(ConfigError, match="divisible by batch_size"):
            make_run_spec("two_rooms", "simple", num_envs=100, episode_length=501)

    def test_the_divisibility_error_suggests_a_fix(self):
        with pytest.raises(ConfigError, match=r"nearest valid num_envs: \[128"):
            make_run_spec("two_rooms", "simple", num_envs=100, episode_length=501)

    def test_suggestions_are_actually_valid(self):
        for n in suggest_num_envs(episode_length=501, batch_size=256, near=100):
            assert n * 500 % 256 == 0

    def test_empty_epoch_is_caught_with_a_remedy(self):
        with pytest.raises(ConfigError, match="num_training_steps_per_epoch would be 0"):
            make_run_spec("two_rooms", "simple", total_env_steps=200_000, num_evals=100)

    def test_unknown_names_are_rejected(self):
        with pytest.raises(ConfigError, match="unknown maze"):
            make_run_spec("labyrinth", "simple")
        with pytest.raises(ConfigError, match="unknown env"):
            make_run_spec("two_rooms", "quadruped")
        with pytest.raises(ConfigError, match="unknown preset"):
            make_run_spec("two_rooms", "simple", preset="enormous")

    def test_unknown_energy_and_loss_functions_are_rejected(self):
        with pytest.raises(ConfigError, match="unknown energy_fn"):
            make_run_spec("two_rooms", "simple", energy_fn="euclidean")
        with pytest.raises(ConfigError, match="unknown contrastive_loss_fn"):
            make_run_spec("two_rooms", "simple", contrastive_loss_fn="triplet")

    def test_nonpositive_sizes_are_rejected(self):
        with pytest.raises(ConfigError, match="num_envs must be positive"):
            make_run_spec("two_rooms", "simple", num_envs=0)

    def test_eval_region_must_exist_on_that_maze(self):
        with pytest.raises(ConfigError, match="has no regions overlay"):
            make_run_spec("spiral", "simple", eval_goal_region="a")
        with pytest.raises(ConfigError, match="unknown region"):
            make_run_spec("four_rooms", "simple", eval_goal_region="z")
        make_run_spec("four_rooms", "simple", eval_goal_region="d")  # valid

    def test_evolve_revalidates(self):
        spec = make_run_spec("two_rooms", "simple")
        with pytest.raises(ConfigError, match="divisible by batch_size"):
            spec.evolve(num_envs=100)


class TestProvisionalBudgets:
    @pytest.mark.parametrize("env", sorted(PROVISIONAL_BUDGETS))
    @pytest.mark.parametrize("maze", list(layouts.names()))
    @pytest.mark.parametrize("preset", sorted(ARCH_PRESETS))
    def test_every_default_combination_is_valid(self, env, maze, preset):
        # A shipped default that fails validation is a trap for whoever runs
        # it first.
        spec = make_run_spec(maze, env, preset)
        assert spec.num_training_steps_per_epoch > 0

    @pytest.mark.parametrize("env", sorted(PROVISIONAL_BUDGETS))
    def test_defaults_keep_the_buffer_under_a_gigabyte(self, env):
        # 32 GB laptop, and the buffer dominates a resume checkpoint.
        assert make_run_spec("two_rooms", env).replay_buffer_bytes < 1e9

    def test_a_crash_costs_at_most_a_few_percent(self):
        spec = make_run_spec("two_rooms", "simple")
        assert spec.env_steps_per_epoch / spec.actual_total_env_steps < 0.05


class TestModuleIsJaxFree:
    def test_importing_presets_pulls_in_no_jax(self):
        import subprocess
        import sys

        out = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import latentmine.train.presets;"
                " print('jax' in sys.modules or 'brax' in sys.modules)",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert out.stdout.strip() == "False", out.stdout
