# CLAUDE.md

Orientation for agent sessions working in this repository. Read
`docs/LOW_LEVEL_DESIGN.md` before writing code — it is the contract this
project follows and it records facts about upstream that are easy to get
wrong.

## What this project is

RL final project, *Proposal 4 — Contrastive RL latent mining*. We train CRL
(Contrastive RL) agents on a designed set of mazes and then investigate what
the critic's latent space encodes: whether latent distance tracks geodesic
rather than Euclidean distance, whether the maze is visible in a 2-D
projection, whether a frozen-encoder decoder can invert the latent, and
whether decoded latent interpolants are usable as navigation waypoints.

A negative result is an acceptable outcome; an *unmeasured* result is not.
Section 6 of the LLD defines the metrics, and every claim in the write-up
should cite one.

## Status

Build steps 1-3 of LLD §12 are done: the package scaffold, the five-maze set,
grid/world geometry, registration with upstream's env classes, and the
training entrypoint (presets, run-spec validation, manifests, env
construction, CLI).

161 fast tests pass. 23 `slow` tests need `jaxgcrl` installed and have **not**
been run — they cover the upstream monkey-patch, the env dimension table, and
a tiny end-to-end training run. Run them before the first long run.

Next is step 3.5: the resumable `train_fn` derivative (§5.5) and the timing
probe (§5.6). Do not start a long run before the resume path is tested by a
deliberate `SIGKILL`.

## Layout (target — most of this is not built yet)

```
docs/LOW_LEVEL_DESIGN.md   the design contract
third_party/JaxGCRL/       git submodule, pinned to 7c53a074      [done]
src/latentmine/            all logic lives here
  mazes/                   MazeSpec registry, geometry, register  [done]
  train/                   presets, manifest, envs, run_crl        [done]
  checkpoints.py           load params -> jitted phi/psi
  embed.py, sampling.py, rollouts.py
  analysis/                projections, metrics, plots, bottleneck, reconstruct
  decoder/                 frozen-encoder decoder + latent interpolation
  control/                 waypoint-conditioned rollouts
notebooks/                 exploration only, nothing load-bearing
tests/
```

## Conventions

- **All logic goes in `src/latentmine/`.** Notebooks may import and plot; if a
  notebook cell grows a function, move it into the package. The pipeline must
  be reproducible from the CLI.
- **One coordinate convention, one place.** Upstream maps maze grid row `i` to
  world `x` and column `j` to world `y`. `mazes/geometry.py` owns that
  conversion; nothing else may reimplement it. A transposition here silently
  inverts every figure in the project.
- **Never load a checkpoint without its `manifest.json`.** The pickle is a
  bare 3-tuple with no architecture record; rebuilding the encoders needs
  `repr_dim`, `h_dim`, `n_hidden`, `skip_connections`, `use_relu`, `use_ln`.
  Go through `checkpoints.load_encoders`, not through raw `pickle`.
- **Upstream checkpoints cannot resume training.** They hold params only — no
  optimiser state, step count, replay buffer or RNG. Crash recovery goes
  through our own resume checkpoints (LLD §5.5); never assume a
  `ckpt/step_*.pkl` is enough to continue a run.
- Seeds are explicit arguments, never global state. Every figure and metric is
  reported over ≥ 3 seeds.
- Artifacts are derived data and are gitignored: `runs/`, `artifacts/`,
  `figures/`, `wandb/`. Commit code and configs, not outputs.
- Python 3.10, ruff (line length 110) matching upstream's config.

## Upstream facts worth memorising

Pinned: `MichalBortkiewicz/JaxGCRL` @ `7c53a074`. LLD §2 has the full detail;
the traps:

- **`n_hidden` is network depth, `h_dim` is width.** Not what the names
  suggest.
- In every maze env the goal is just `(x, y)`, so `psi : R^2 -> R^d` can be
  rastered densely over the maze with no rollouts at all. This is the cheapest
  and most informative visualisation in the project — reach for it first.
- `energy_fn="norm"` means the critic is `-||phi - psi||_2`, so latent
  distance is a hitting-time estimate and linear interpolation is meaningful.
  Under `dot`/`cosine` the geometry is spherical and interpolation must be
  slerp. Do not mix these up.
- Checkpoints are `(alpha_params, actor_params, critic_params)` where
  `critic_params` has keys `sa_encoder` and `g_encoder`.
- `CRL.check_config` enforces `num_envs * (episode_length - 1) % batch_size == 0`.
- Upstream's `notebooks/visualize.ipynb` is **stale** — it calls
  `make_crl_networks`, `get_env_config` and `args.env_name`, none of which
  exist at the pinned commit. Don't copy it.
- We deliberately bypass `jaxgcrl.utils.env.create_env` and `run.py`'s tyro
  CLI, because `legal_envs` is baked into a `Literal` type at import time and
  cannot be extended cleanly. Envs are constructed directly and `train_fn` is
  called programmatically. See LLD §4.2.

## Commands

Nothing is implemented yet; these are the intended entry points and should be
kept accurate as they land.

Target hardware is an M1 Pro MacBook: **CPU-only JAX, no CUDA**. Budgets come
from the timing probe in LLD §5.6, not from upstream's single-GPU throughput
claims. Long runs go under `caffeinate -i`.

```bash
# setup
git submodule update --init --recursive
pip install -e third_party/JaxGCRL -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -e .

# fast checks
pytest tests -q -m "not slow"
ruff check src tests && ruff format --check src tests

# render the maze set (first milestone; catches the axis convention)
python -m latentmine.mazes.render --out figures/mazes

# check a configuration in one second, importing no JAX
python -m latentmine.train.run_crl --maze two_rooms --env simple --dry-run

# train
caffeinate -i python -m latentmine.train.run_crl \
    --maze two_rooms --env simple --preset deep --seed 1

# throughput probe before setting any budget
python -m latentmine.train.probe --env simple --num-envs 64,128,256
```

## Git

- Work on `claude/contrastive-rl-latent-mining-n6xr0n`.
- Commits are authored as the repository owner. Do not add Claude as an author
  or co-author, and do not put model identifiers in commit messages, code
  comments, or anything else pushed to the repo.
- Don't open a pull request unless asked.
