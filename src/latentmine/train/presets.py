"""Training configuration: architecture presets, env facts, and the run spec.

Deliberately free of JAX, brax and mujoco imports. Every constraint upstream
enforces with a bare `assert` deep inside `train_fn` is checked here instead,
so a misconfigured run fails in a second with an actionable message rather
than after JAX has finished compiling. It is also what lets `run_crl
--dry-run` report the whole resolved configuration, its derived quantities and
its memory footprint without a training stack installed.

See LLD sections 5.2 (presets), 5.3 (hyperparameters) and 5.6 (why the budget
defaults here are provisional).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from ..mazes import layouts

EnergyFn = Literal["norm", "l2", "dot", "cosine"]
ContrastiveLossFn = Literal["fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"]

ENERGY_FNS = ("norm", "l2", "dot", "cosine")
CONTRASTIVE_LOSS_FNS = ("fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce")


class ConfigError(ValueError):
    """A run configuration violates a constraint upstream would only catch later."""


# ---------------------------------------------------------------------------
# architecture presets (LLD section 5.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchPreset:
    """Encoder/actor architecture.

    Upstream's field names invert the usual convention: `n_hidden` is network
    **depth** and `h_dim` is network **width**. Named plainly here and mapped
    back when the `CRL` dataclass is constructed.
    """

    name: str
    depth: int  # -> CRL.n_hidden
    width: int  # -> CRL.h_dim
    skip_connections: int
    use_ln: bool
    repr_dim: int = 64
    use_relu: bool = False
    notes: str = ""


ARCH_PRESETS: dict[str, ArchPreset] = {
    "shallow": ArchPreset(
        name="shallow",
        depth=2,
        width=256,
        skip_connections=0,
        use_ln=False,
        notes="Upstream's default. The baseline the depth claim is measured against.",
    ),
    "deep": ArchPreset(
        name="deep",
        depth=4,
        width=256,
        skip_connections=4,
        use_ln=True,
        notes="Project default. Depth and LayerNorm move together - see the confound note below.",
    ),
    "deeper": ArchPreset(
        name="deeper",
        depth=8,
        width=512,
        skip_connections=4,
        use_ln=True,
        notes="First thing to cut if the compute budget bites.",
    ),
    "deep_noln": ArchPreset(
        name="deep_noln",
        depth=4,
        width=256,
        skip_connections=4,
        use_ln=False,
        notes=(
            "Third arm that isolates depth from LayerNorm. Without it, 'deep beats "
            "shallow' is a claim about a configuration, not about depth, since the "
            "two presets differ in both."
        ),
    ),
}


# ---------------------------------------------------------------------------
# environment facts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvSpec:
    """Static dimensions of an upstream maze env family.

    Hardcoded so the config layer needs no JAX. `train.envs.build_envs`
    asserts every field against the constructed env, so these are checked
    invariants rather than assumptions - a `state_dim` that drifts upstream
    fails loudly at env construction, not silently in the analysis.
    """

    name: str
    module: str
    cls: str
    state_dim: int
    action_size: int
    goal_size: int = 2
    backend: str = "spring"
    notes: str = ""

    @property
    def obs_size(self) -> int:
        """What upstream asserts equals `env.observation_size`."""
        return self.state_dim + self.goal_size

    @property
    def transition_floats(self) -> int:
        """Width of one flattened replay-buffer row: observation, action,
        reward, discount, truncation, traj_id."""
        return self.obs_size + self.action_size + 4


ENV_SPECS: dict[str, EnvSpec] = {
    "simple": EnvSpec(
        name="simple",
        module="jaxgcrl.envs.simple_maze",
        cls="SimpleMaze",
        state_dim=4,
        action_size=2,
        notes="Planar point mass: (x, y, vx, vy). The cheapest env in the repo.",
    ),
    "ant": EnvSpec(
        name="ant",
        module="jaxgcrl.envs.ant_maze",
        cls="AntMaze",
        state_dim=29,
        action_size=8,
        notes="15 qpos + 14 qvel. The latent must encode gait and orientation, not just position.",
    ),
}


# Provisional budgets. LLD section 5.6: these are placeholders until the
# timing probe measures throughput on the target laptop, and every one of them
# should be replaced by a measured value before a long run.
PROVISIONAL_BUDGETS: dict[str, dict[str, int]] = {
    "simple": {
        "total_env_steps": 5_000_000,
        "episode_length": 501,
        "num_envs": 128,
        "num_evals": 100,
        "num_eval_envs": 32,
    },
    "ant": {
        "total_env_steps": 10_000_000,
        "episode_length": 1001,
        "num_envs": 128,
        "num_evals": 100,
        "num_eval_envs": 32,
    },
}


# ---------------------------------------------------------------------------
# the run spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSpec:
    """Everything that defines one training run."""

    maze: str
    env: str
    preset: str
    seed: int

    total_env_steps: int
    episode_length: int
    num_envs: int
    num_evals: int
    num_eval_envs: int

    # CRL hyperparameters. Upstream defaults except `contrastive_loss_fn`:
    # sym_infonce constrains both encoders, which matters because we later
    # interpolate in psi-space and decode it (LLD section 5.3).
    batch_size: int = 256
    unroll_length: int = 62
    min_replay_size: int = 1000
    max_replay_size: int = 10000
    discounting: float = 0.99
    energy_fn: EnergyFn = "norm"
    contrastive_loss_fn: ContrastiveLossFn = "sym_infonce"
    logsumexp_penalty_coeff: float = 0.1
    train_step_multiplier: int = 1
    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4

    action_repeat: int = 1
    backend: str | None = None
    visualization_interval: int = 10
    eval_goal_region: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    # -- lookups -----------------------------------------------------------

    @property
    def arch(self) -> ArchPreset:
        return ARCH_PRESETS[self.preset]

    @property
    def env_spec(self) -> EnvSpec:
        return ENV_SPECS[self.env]

    @property
    def maze_spec(self) -> layouts.MazeSpec:
        return layouts.get(self.maze)

    @property
    def resolved_backend(self) -> str:
        return self.backend or self.env_spec.backend

    @property
    def run_id(self) -> str:
        a = self.arch
        return f"{self.env}_{self.maze}_d{a.depth}_w{a.width}_r{a.repr_dim}_{self.energy_fn}_s{self.seed}"

    # -- derived quantities ------------------------------------------------
    #
    # These mirror the arithmetic inside `CRL.train_fn` exactly. Recomputing
    # them here is what makes the constraints checkable before launch.

    @property
    def env_steps_per_actor_step(self) -> int:
        return self.num_envs * self.unroll_length

    @property
    def num_prefill_env_steps(self) -> int:
        return self.min_replay_size * self.num_envs

    @property
    def num_training_steps_per_epoch(self) -> int:
        return (self.total_env_steps - self.num_prefill_env_steps) // (
            self.num_evals * self.env_steps_per_actor_step
        )

    @property
    def env_steps_per_epoch(self) -> int:
        """Upper bound on what a crash costs: one epoch of work (LLD 5.5)."""
        return self.num_training_steps_per_epoch * self.env_steps_per_actor_step

    @property
    def actual_total_env_steps(self) -> int:
        """What the run really executes - integer division makes this differ
        from the requested `total_env_steps`."""
        return self.num_prefill_env_steps + self.num_evals * self.env_steps_per_epoch

    @property
    def replay_buffer_bytes(self) -> int:
        """`buffer_state.data` is (max_replay_size, num_envs, row) float32.
        The dominant term in a resume checkpoint (LLD 5.5)."""
        return self.max_replay_size * self.num_envs * self.env_spec.transition_floats * 4

    @property
    def utd_ratio(self) -> float:
        """Update-to-data ratio, as `run.py` reports it."""
        numerator = self.num_envs * self.episode_length * self.train_step_multiplier / self.batch_size
        return numerator / (self.num_envs * self.unroll_length)

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        if self.maze not in layouts.MAZES:
            raise ConfigError(f"unknown maze {self.maze!r}; known: {list(layouts.names())}")
        if self.env not in ENV_SPECS:
            raise ConfigError(f"unknown env {self.env!r}; known: {sorted(ENV_SPECS)}")
        if self.preset not in ARCH_PRESETS:
            raise ConfigError(f"unknown preset {self.preset!r}; known: {sorted(ARCH_PRESETS)}")
        if self.energy_fn not in ENERGY_FNS:
            raise ConfigError(f"unknown energy_fn {self.energy_fn!r}; known: {list(ENERGY_FNS)}")
        if self.contrastive_loss_fn not in CONTRASTIVE_LOSS_FNS:
            raise ConfigError(
                f"unknown contrastive_loss_fn {self.contrastive_loss_fn!r}; "
                f"known: {list(CONTRASTIVE_LOSS_FNS)}"
            )
        for field, value in (
            ("seed", self.seed),
            ("num_envs", self.num_envs),
            ("num_evals", self.num_evals),
            ("num_eval_envs", self.num_eval_envs),
            ("episode_length", self.episode_length),
            ("total_env_steps", self.total_env_steps),
            ("batch_size", self.batch_size),
            ("unroll_length", self.unroll_length),
        ):
            if value < (0 if field == "seed" else 1):
                raise ConfigError(f"{field} must be positive, got {value}")

        if self.eval_goal_region is not None:
            spec = self.maze_spec
            if spec.regions is None:
                raise ConfigError(f"--eval-goal-region given but maze {self.maze!r} has no regions overlay")
            if self.eval_goal_region not in spec.region_labels():
                raise ConfigError(
                    f"unknown region {self.eval_goal_region!r} for maze {self.maze!r}; "
                    f"known: {list(spec.region_labels())}"
                )

        self._check_batch_divisibility()
        self._check_epoch_is_nonempty()

    def _check_batch_divisibility(self) -> None:
        """`CRL.check_config`'s constraint, checked before launch."""
        if self.num_envs * (self.episode_length - 1) % self.batch_size == 0:
            return
        raise ConfigError(
            f"num_envs * (episode_length - 1) must be divisible by batch_size: "
            f"{self.num_envs} * {self.episode_length - 1} = "
            f"{self.num_envs * (self.episode_length - 1)} is not divisible by {self.batch_size}.\n"
            f"  nearest valid num_envs: "
            f"{suggest_num_envs(self.episode_length, self.batch_size, self.num_envs)}"
        )

    def _check_epoch_is_nonempty(self) -> None:
        """Upstream asserts `num_training_steps_per_epoch > 0` after setup."""
        if self.num_training_steps_per_epoch > 0:
            return
        budget = self.total_env_steps - self.num_prefill_env_steps
        min_total = self.num_prefill_env_steps + self.num_evals * self.env_steps_per_actor_step
        raise ConfigError(
            f"num_training_steps_per_epoch would be 0: {self.num_evals} evals of "
            f"{self.env_steps_per_actor_step} env steps each do not fit in "
            f"{self.total_env_steps} steps after {self.num_prefill_env_steps} of prefill.\n"
            f"  raise total_env_steps to at least {min_total}, or lower num_evals to at most "
            f"{max(1, budget // self.env_steps_per_actor_step)}."
        )

    def evolve(self, **changes) -> RunSpec:
        """A copy with fields replaced, revalidated."""
        return replace(self, **changes)


def suggest_num_envs(episode_length: int, batch_size: int, near: int, span: int = 4096) -> list[int]:
    """`num_envs` values near `near` that satisfy the divisibility constraint."""
    valid = [n for n in range(1, span + 1) if n * (episode_length - 1) % batch_size == 0]
    return sorted(valid, key=lambda n: (abs(n - near), n))[:4]


def make_run_spec(maze: str, env: str, preset: str = "deep", seed: int = 1, **overrides) -> RunSpec:
    """A run spec with the provisional laptop budget for `env`, plus overrides."""
    if env not in PROVISIONAL_BUDGETS:
        raise ConfigError(f"unknown env {env!r}; known: {sorted(PROVISIONAL_BUDGETS)}")
    budget = dict(PROVISIONAL_BUDGETS[env])
    budget.update({k: v for k, v in overrides.items() if v is not None})
    return RunSpec(maze=maze, env=env, preset=preset, seed=seed, **budget)
