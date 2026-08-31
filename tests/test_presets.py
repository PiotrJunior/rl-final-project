"""Run-configuration tests.

Everything here runs without JAX. The point of the config layer is that a
misconfigured run fails in a second with an actionable message, rather than
after JAX has compiled and then tripped a bare `assert` inside `train_fn`.
"""

import pytest

from latentmine.mazes import layouts
from latentmine.train.presets import (
    ARCH_PRESETS,
    BUDGET_PROFILES,
    ENV_SPECS,
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
        return make_run_spec("two_rooms", "simple", profile="laptop", num_envs=128, episode_length=501)

    def test_they_mirror_train_fns_arithmetic(self, spec):
        assert spec.env_steps_per_actor_step == spec.num_envs * spec.unroll_length
        assert spec.num_prefill_env_steps == spec.min_replay_size * spec.num_envs
        # We round the schedule up, then hand train_fn a total that
        # floor-divides back to it - so its arithmetic and ours agree on the
        # value that matters.
        assert spec._steps_per_epoch_for(spec.effective_total_env_steps) == (
            spec.num_training_steps_per_epoch
        )

    def test_actual_total_covers_the_request(self, spec):
        # Rounded up, so the run is a little over the ask rather than under it.
        assert spec.actual_total_env_steps >= spec.total_env_steps
        assert spec.covers_requested_budget

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
    @pytest.mark.parametrize("profile", sorted(BUDGET_PROFILES))
    @pytest.mark.parametrize("env", ["simple", "ant"])
    @pytest.mark.parametrize("maze", list(layouts.names()))
    @pytest.mark.parametrize("preset", sorted(ARCH_PRESETS))
    def test_every_default_combination_is_valid(self, profile, env, maze, preset):
        # A shipped default that fails validation is a trap for whoever runs
        # it first.
        spec = make_run_spec(maze, env, preset, profile=profile)
        assert spec.num_training_steps_per_epoch > 0
        assert spec.satisfies_upstream_final_assert
        assert spec.covers_requested_budget

    @pytest.mark.parametrize("env", ["simple", "ant"])
    def test_defaults_keep_the_buffer_under_a_gigabyte(self, env):
        # The buffer dominates a resume checkpoint and sits in GPU memory
        # alongside the model.
        assert make_run_spec("two_rooms", env).replay_buffer_bytes < 1e9

    def test_a_crash_costs_at_most_a_few_percent(self):
        spec = make_run_spec("two_rooms", "simple", profile="gpu")
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


class TestPrefillExhaustsBudget:
    def test_error_says_trimming_evals_cannot_help(self):
        # When prefill alone exceeds total_env_steps, suggesting a lower
        # num_evals is useless - the advice has to point at prefill.
        with pytest.raises(ConfigError, match="prefill alone"):
            make_run_spec(
                "two_rooms",
                "simple",
                total_env_steps=50_000,
                episode_length=101,
                num_envs=64,
                num_evals=1,
                min_replay_size=1000,
            )


class TestUpstreamFinalAssert:
    """`train_fn` ends with `assert total_steps >= config.total_env_steps`, and
    its own schedule can fall short of the total it was handed - a run trains
    for hours and then dies while returning. These pin the workaround."""

    def test_prefill_accounting_has_two_distinct_meanings(self):
        # Upstream sizes the budget with min_replay_size * num_envs but the
        # prefill loop runs ceil(min_replay/unroll) whole actor steps.
        spec = make_run_spec("two_rooms", "simple", num_envs=128)
        assert spec.num_prefill_env_steps == 1000 * 128
        assert spec.prefill_env_steps_actual == 17 * 128 * 62
        assert spec.prefill_env_steps_actual > spec.num_prefill_env_steps

    def test_prefill_accounting_agrees_when_unroll_divides_min_replay(self):
        spec = make_run_spec("two_rooms", "simple", num_envs=128, min_replay_size=62 * 8)
        assert spec.prefill_env_steps_actual == spec.num_prefill_env_steps

    def test_effective_total_is_what_the_run_reaches(self):
        spec = make_run_spec("two_rooms", "simple")
        assert spec.effective_total_env_steps == spec.actual_total_env_steps

    def test_the_budget_is_rounded_up_not_down(self):
        # Floor division discards up to num_evals * env_steps_per_actor_step,
        # which at GPU-sized num_envs is millions of steps.
        spec = make_run_spec("two_rooms", "simple", profile="gpu")
        assert spec.covers_requested_budget
        assert spec.actual_total_env_steps >= spec.total_env_steps

    def test_overshoot_is_bounded_by_one_epoch(self):
        for profile in BUDGET_PROFILES:
            for env in ("simple", "ant"):
                spec = make_run_spec("two_rooms", env, profile=profile)
                overshoot = spec.actual_total_env_steps - spec.total_env_steps
                assert 0 <= overshoot < spec.num_evals * spec.env_steps_per_actor_step

    def test_the_passed_total_reproduces_our_schedule(self):
        # train_fn recomputes steps-per-epoch from what we hand it; if that
        # disagreed with our number it would run a different schedule.
        for profile in BUDGET_PROFILES:
            spec = make_run_spec("two_rooms", "simple", profile=profile)
            assert spec._steps_per_epoch_for(spec.effective_total_env_steps) == (
                spec.num_training_steps_per_epoch
            )

    def test_profiles_are_distinct_where_it_matters(self):
        gpu = make_run_spec("two_rooms", "simple", profile="gpu")
        laptop = make_run_spec("two_rooms", "simple", profile="laptop")
        assert gpu.num_envs > laptop.num_envs
        assert gpu.episode_length > laptop.episode_length
        # Shortening episodes is what costs the laptop profile its utd ratio.
        assert gpu.utd_ratio > laptop.utd_ratio

    def test_upstreams_own_defaults_would_have_failed(self):
        # Documents the bug rather than just working around it silently:
        # 50M requested, ~47.9M reached.
        spec = make_run_spec(
            "two_rooms",
            "ant",
            total_env_steps=50_000_000,
            episode_length=1001,
            num_envs=256,
            num_evals=200,
        )
        assert spec._reachable_steps_for(50_000_000) < 50_000_000
        assert spec.satisfies_upstream_final_assert  # ...but ours does

    @pytest.mark.parametrize("env", ["simple", "ant"])
    @pytest.mark.parametrize("maze", list(layouts.names()))
    def test_every_shipped_default_satisfies_it(self, env, maze):
        assert make_run_spec(maze, env).satisfies_upstream_final_assert

    @pytest.mark.parametrize("num_evals", [1, 7, 50, 100, 199])
    @pytest.mark.parametrize("num_envs", [64, 128, 256])
    @pytest.mark.parametrize("total", [3_000_000, 7_500_000])
    def test_any_spec_that_constructs_satisfies_it(self, num_evals, num_envs, total):
        """The real invariant: a config either fails validation with a remedy,
        or it reaches the total it hands upstream. Never the third thing -
        constructing fine and then dying on the assert hours later."""
        try:
            spec = make_run_spec(
                "two_rooms",
                "simple",
                num_envs=num_envs,
                num_evals=num_evals,
                total_env_steps=total,
            )
        except ConfigError:
            return  # rejected up front, which is the other acceptable outcome
        assert spec.num_training_steps_per_epoch > 0
        assert spec.satisfies_upstream_final_assert
