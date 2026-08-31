# Low Level Design — Contrastive RL Latent Mining

Project: *Proposal 4 — CRL latent mining.* Understand what a Contrastive RL
(CRL) critic encodes in its latent space, whether that information is legible
to a human, and whether it can be fed back into the algorithm as usable
structure (waypoints, subgoals, a reconstructed map).

Status: **design only.** No implementation has landed yet; this document is
the contract that the implementation should follow. Section 12 tracks which
parts are built.

---

## 1. Problem statement, made concrete

CRL trains two encoders with an InfoNCE-style objective over
(state-action, future-goal) pairs:

- `phi(s, a) -> R^d` — the state-action encoder (`sa_encoder`)
- `psi(g)    -> R^d` — the goal encoder (`g_encoder`)

and uses `Q(s, a, g) ~ energy(phi(s,a), psi(g))` as the critic. With
`energy_fn = "norm"` the critic is `-||phi - psi||_2`, i.e. **latent distance
is (up to monotone transform) a discounted hitting-time estimate**. The
research question of this project follows directly:

> If latent distance approximates *time-to-reach*, and time-to-reach in a maze
> is the geodesic distance around walls, then the latent space should be a
> metric embedding of the maze's *navigation graph*, not of its Euclidean
> coordinates.

Everything below is designed to test that single sentence, and then to exploit
it. Concretely we want falsifiable answers to:

- **Q1 (geometry).** Does latent distance track *geodesic* distance better
  than *Euclidean* distance? By how much, and where does it fail?
- **Q2 (legibility).** Does a 2D projection of the latent space show the maze?
  Do rooms become clusters, corridors become necks?
- **Q3 (dynamics).** In Ant mazes, the goal is still only `(x, y)` but
  `phi` sees the full 29-D state. What does `phi` encode *beyond* position —
  gait phase, body orientation, velocity? Decoder reconstruction error per
  observation group is the measurement.
- **Q4 (invertibility).** Can a frozen-encoder decoder recover the state from
  the latent, and does it *generalise spatially* (train on 3 rooms, test on
  the 4th)?
- **Q5 (utility).** Do decoded latent interpolants form a navigable path, and
  does conditioning the actor on them beat direct goal-conditioning?

A negative result on Q1/Q2 is an acceptable outcome, but only if it is
*measured* rather than eyeballed. Section 6 defines the metrics that make a
negative result publishable rather than inconclusive.

---

## 2. Upstream: what JaxGCRL actually gives us

Pinned upstream: `MichalBortkiewicz/JaxGCRL` @ `7c53a074` (2026-06-06).
Vendored as a git submodule at `third_party/JaxGCRL`. Facts below were read
off that commit and the implementation depends on them; if the submodule is
bumped, re-verify this section.

### 2.1 Encoder architecture (`jaxgcrl/agents/crl/networks.py`)

```python
class Encoder(nn.Module):
    repr_dim: int = 64
    network_width: int = 256      # <- CRL config field `h_dim`
    network_depth: int = 4        # <- CRL config field `n_hidden`
    skip_connections: int = 0     # 0 = off; else "add skip every k layers"
    use_relu: bool = False        # False => swish
    use_ln: bool = False          # LayerNorm before activation
```

`network_depth` hidden layers of `network_width`, then a linear head to
`repr_dim`. Skip connections are additive with period `skip_connections`.

### 2.2 CRL agent config (`jaxgcrl/agents/crl/crl.py`, `@dataclass class CRL`)

Fields we care about: `repr_dim=64`, `h_dim=256`, `n_hidden=2`,
`skip_connections=4`, `use_relu=False`, `use_ln=False`,
`contrastive_loss_fn ∈ {fwd_infonce, sym_infonce, bwd_infonce, binary_nce}`,
`energy_fn ∈ {norm, l2, dot, cosine}`, `logsumexp_penalty_coeff=0.1`,
`discounting=0.99`, `batch_size=256`, `unroll_length=62`,
`min/max_replay_size`, `train_step_multiplier`.

Note the naming trap: **`n_hidden` is depth, `h_dim` is width.** Both the
actor and both encoders share them. Also, the `Actor` is constructed *without*
`use_ln` while the encoders get it — so `--use_ln` deepens/normalises only the
critic. That asymmetry is fine for us (we care about the critic) but must be
stated in the report.

Hard constraint enforced by `CRL.check_config`:
`num_envs * (episode_length - 1) % batch_size == 0`.

### 2.3 Losses (`jaxgcrl/agents/crl/losses.py`)

```python
energy_fn("norm",   x, y) = -sqrt(sum((x-y)^2) + 1e-6)   # negative L2
energy_fn("l2",     x, y) = -sum((x-y)^2)                # negative squared L2
energy_fn("dot",    x, y) =  sum(x*y)
energy_fn("cosine", x, y) =  normalised dot
```

Critic loss builds `logits[i, j] = energy(phi_i, psi_j)` over the batch, then
InfoNCE plus `logsumexp_penalty_coeff * mean(logsumexp(logits, axis=1)^2)`.

**Design consequence:** the choice of `energy_fn` dictates the geometry of the
latent space and therefore how we may interpolate in it (Section 8.2). `norm`
and `l2` give a Euclidean-ish space (linear interpolation is meaningful);
`dot`/`cosine` give a spherical/conic space (use slerp, and normalise before
PCA). Our default is `energy_fn=norm`, which is also the setting under which
"latent distance = hitting time" is best motivated.

### 2.4 Observation and goal layout for maze envs

- `SimpleMaze`: `state_dim = 4` (x, y, vx, vy), `goal_indices = [0, 1]`,
  `goal_reach_thresh = 0.5`, action size 2.
- `AntMaze`: `state_dim = 29`, `goal_indices = [0, 1]`,
  `goal_reach_thresh = 0.5`, action size 8.
- `env.observation_size == state_dim + len(goal_indices)`; the observation is
  `concat(qpos[:-2], qvel[:-2], target_xy)`.

The `[:-2]` slices drop the `target` body's two slide joints, which every maze
XML appends. So `state_dim` is the *agent's* DOF count:

| env | qpos | qvel | `state_dim` |
|---|---|---|---|
| `SimpleMaze` | 2 (slide x, slide y) | 2 | 4 |
| `AntMaze` | 15 (3 pos + 4 quat + 8 hinges) | 14 (3 lin + 3 ang + 8 hinges) | 29 |

`SimpleMaze`'s torso has its free joint commented out in `simple_maze.xml` and
replaced by two slide joints, so it is a planar point mass: `(x, y, vx, vy)`.

Both maze envs set `exclude_current_positions_from_observation=False` (vanilla
Brax `Ant` defaults it to `True`). That is why `goal_indices = [0, 1]` works at
all — `x` and `y` are retained at the front of the observation, so the goal is
literally a slice of the state, and `psi` and the first two state dims live in
the same coordinate frame.

Index layout for AntMaze's 29 dims, which metric E (Section 6) groups over:

| slice | contents |
|---|---|
| `0:2`   | `x`, `y` — the goal dims |
| `2:3`   | `z` (torso height) |
| `3:7`   | root quaternion (orientation) |
| `7:15`  | 8 joint angles (hip/ankle × 4) |
| `15:18` | root linear velocity |
| `18:21` | root angular velocity |
| `21:29` | 8 joint velocities |

The velocity ordering follows the MuJoCo/Brax free-joint convention; assert it
against a real `env.reset()` in the tests rather than trusting this table, since
a mis-grouped metric E would misattribute what `phi` encodes.

So `psi` takes a **2-D input** (the goal `xy`) in every maze env. This is the
single most useful fact in the project: `psi : R^2 -> R^d` can be rastered
over a dense grid, giving a per-pixel latent map of the maze with no rollouts
at all. `phi` is the interesting, high-dimensional side.

### 2.5 Maze construction (`jaxgcrl/envs/simple_maze.py`, `ant_maze.py`)

Layouts are Python lists of lists. Cell values: `1` = wall block,
`"r"` = possible reset/start cell, `"g"` = possible goal cell, `0` = free but
neither. World coordinates are `(i * scaling, j * scaling)` with
`maze_size_scaling = 4.0`; `make_maze()` emits one MuJoCo box geom per wall
cell into `assets/simple_maze.xml` / `ant_maze.xml`. Layout selection is a
hardcoded `if/elif` chain on `maze_layout_name` inside `make_maze`.

Note the axis convention: the **row index `i` maps to world `x`** and the
**column index `j` maps to world `y`**. Every plot in this project must use
the same convention or the visualisations will silently be transposed.
`src/latentmine/mazes/geometry.py` owns this conversion and nothing else may
duplicate it.

### 2.6 Env registry (`jaxgcrl/utils/env.py`)

`legal_envs` is a tuple of strings used as `Literal[legal_envs]` in
`RunConfig`, and `create_env` dispatches by substring (`"ant" in env_name`,
then `env_name[4:]` as the layout name; `env_name[7:]` for `simple_`). This is
why we do **not** go through `create_env` — see Section 4.2.

### 2.7 Checkpoints (`crl.py: save_params`, `run.py`)

Per-eval checkpoints are written to
`runs/run_{exp_name}_s_{seed}/ckpt/step_{env_steps}.pkl` when
`checkpoint_logdir` is set, and the final one to `.../ckpt/final`. The payload
is a **3-tuple**, pickled:

```python
(alpha_state.params,           # {"log_alpha": ...}
 actor_state.params,           # actor FrozenDict
 critic_state.params)          # {"sa_encoder": ..., "g_encoder": ...}
```

`run.py` also dumps `runs/.../args.pkl` = `vars(config)` with keys
`agent` and `run`. **The checkpoint does not record architecture**, so
rebuilding `Encoder` requires reading `args.pkl` for `repr_dim`, `h_dim`,
`n_hidden`, `skip_connections`, `use_relu`, `use_ln`. Our loader (Section 5.1)
treats a checkpoint without a sibling `args.pkl` as unusable, and additionally
writes its own `manifest.json` so we never depend on unpickling a
`flax.struct.dataclass` whose definition may have moved upstream.

### 2.8 Known-stale upstream code

`notebooks/visualize.ipynb` calls `networks.make_crl_networks`,
`networks.make_inference_fn`, `get_env_config` and `args.env_name` — none of
which exist at the pinned commit. Do not copy it. It is still useful as a
sketch of the intended flow (load params → rebuild encoders → roll out →
PCA of `phi` coloured by timestep), and Section 7.1 is its corrected successor.

---

## 3. Repository layout

```
rl-final-project/
├── CLAUDE.md                    # orientation for agent sessions
├── README.md
├── pyproject.toml               # package `latentmine`, deps, ruff config
├── docs/
│   ├── LOW_LEVEL_DESIGN.md      # this file
│   └── results/                 # written up as experiments land
├── third_party/JaxGCRL/         # git submodule, pinned
├── configs/
│   ├── mazes/                   # one YAML per maze spec (ASCII art + meta)
│   └── train/                   # training presets: depth sweep, seeds
├── src/latentmine/
│   ├── mazes/
│   │   ├── layouts.py           # MazeSpec dataclass + ASCII registry
│   │   ├── geometry.py          # grid<->world, occupancy, BFS geodesics
│   │   └── register.py          # patch upstream make_maze with our registry
│   ├── train/
│   │   ├── run_crl.py           # programmatic entrypoint (bypasses tyro)
│   │   └── presets.py           # depth/width/seed grids
│   ├── checkpoints.py           # manifest, param loading, encoder rebuild
│   ├── embed.py                 # batched phi/psi, energy fns, latent distance
│   ├── sampling.py              # grid goal sampling, state teleporting
│   ├── rollouts.py              # policy & random rollout collection -> npz
│   ├── analysis/
│   │   ├── projections.py       # PCA / t-SNE / UMAP with fixed seeds
│   │   ├── metrics.py           # Q1..Q5 metrics (Section 6)
│   │   ├── plots.py             # every figure in the report
│   │   ├── bottleneck.py        # advanced: bottleneck identification
│   │   └── reconstruct.py       # advanced: maze reconstruction from latents
│   ├── decoder/
│   │   ├── model.py             # MLP decoder definitions
│   │   ├── data.py              # dataset assembly + spatial holdout splits
│   │   ├── train.py             # supervised training loop
│   │   └── interpolate.py       # latent interpolation -> decoded waypoints
│   └── control/
│       └── waypoint_policy.py   # waypoint-conditioned rollout + baselines
├── scripts/                     # thin shell wrappers for cluster submission
├── notebooks/                   # exploration only; nothing load-bearing
└── tests/
```

Rule: **`src/latentmine/` holds all logic, notebooks hold none.** A notebook
may import from `latentmine` and plot; if a notebook cell grows a function, it
moves into the package. This keeps the pipeline reproducible from the CLI.

---

## 4. Mazes

### 4.1 `MazeSpec` and the ASCII format

```python
@dataclass(frozen=True)
class MazeSpec:
    name: str
    grid: tuple[str, ...]        # ASCII rows; '#'=wall '.'=free 'S'=start 'G'=goal-only
    regions: tuple[str, ...] | None = None   # optional ASCII overlay, same shape
    scaling: float = 4.0
    notes: str = ""              # what hypothesis this maze tests
```

ASCII is the source of truth because it is diffable and reviewable. Characters:

| char | meaning                                        | upstream cell |
|------|------------------------------------------------|---------------|
| `#`  | wall block                                     | `1`           |
| `S`  | start (also a valid goal)                      | `"r"`         |
| `.`  | free, valid goal                               | `"g"`         |
| `G`  | free, valid goal, but never a start            | `"g"`         |
| `-`  | free, **not** a goal (eval layouts only)       | `0`           |

**`regions` — hand-authored room labels.** Upstream has no concept of a room,
a region, or any grouping of cells: a layout is a flat grid of wall / start /
goal codes and nothing more. But metric C (room purity, Section 6) and the
decoder's spatial holdout (Section 8.2) both need to know which room a cell
belongs to, so `MazeSpec` carries an optional second ASCII overlay of the same
shape, one character per cell naming its region:

```
grid            regions
#########       #########
#S..#...#       #aaa#bbb#
#...#...#       #aaa#bbb#
#...+...#       #aaa+bbb#      ('+' = doorway, its own region)
#########       #########
```

These labels are **authored by hand, not derived.** It is tempting to recover
rooms automatically (spectral clustering of the free-cell graph, say), but the
hypothesis under test is precisely *whether latent structure recovers rooms* —
so a ground truth produced by a clustering algorithm would make metric C a
comparison of two clusterings rather than a measurement against truth. We
designed these mazes, so we know the answer; write it down explicitly and keep
it independent of anything the analysis does.

`regions` is `None` for mazes that genuinely have no rooms (`spiral`, `loop` —
a single corridor has no meaningful partition). Metric C is therefore reported
only for `open_room`, `two_rooms` and `four_rooms`, and
`metrics.py` must skip rather than fabricate it elsewhere. Doorway cells get
their own region label so they can be excluded from purity scoring — a cell in
a doorway has genuinely ambiguous membership, and counting it as a failure of
either room would understate the metric.

`MazeSpec.to_upstream_layout()` returns the list-of-lists upstream wants, by a
literal one-character-to-one-code mapping with no train/eval transformation
hidden inside it. Eval layouts are produced instead by
`MazeSpec.eval_variant(region)`, which restricts goals to the named room and
turns the rest into free-but-not-a-goal - derived from the overlay rather than
authored a second time, so a wall edit cannot desynchronise a maze from its
eval twin.

One upstream constraint discovered while implementing this: upstream cells are
**single-valued**, `"r"` or `"g"` and never both, and `find_goals` collects only
`"g"`. So a start cell is *not* sampled as a goal. `S` therefore means "start,
and not a goal"; with one start per maze that costs a single cell of goal
coverage, which is cheaper than diverging from upstream's semantics.

The ASCII rows are indexed `grid[i][j]`, so row `i` → world `x`, column `j` →
world `y`, matching Section 2.5. When plotted with `imshow` this means
`origin="lower"` and a transpose, which `plots.maze_background()` handles once.

### 4.2 Registration without forking

Upstream `make_maze` is an `if/elif` chain and `legal_envs` is baked into a
`Literal` type at import time. Rather than fork, we do two things:

1. `mazes/register.py` monkey-patches
   `jaxgcrl.envs.simple_maze.make_maze` and `jaxgcrl.envs.ant_maze.make_maze`
   with a registry-backed version that looks the name up in our `MazeSpec`
   registry and **falls back to the original function** for upstream names.
   The patch must be applied before any env is constructed; `register.py`
   exposes `install()` and `latentmine.__init__` calls it.
2. We never call `jaxgcrl.utils.env.create_env`, and never go through
   `run.py`'s tyro CLI. `train/run_crl.py` constructs `SimpleMaze(...)` /
   `AntMaze(...)` directly with `maze_layout_name=<our name>`, builds
   `CRL(...)` and `RunConfig(...)` in Python, and calls
   `agent.train_fn(train_env=..., eval_env=..., config=..., progress_fn=...)`.
   This sidesteps the `Literal[legal_envs]` restriction entirely.

Rationale: a fork would have to be rebased every time upstream moves, and the
`Literal` type makes a clean CLI extension impossible anyway. Two patched
functions is a smaller surface than a maintained fork. The patch is covered by
a test that asserts upstream names still resolve (Section 11).

### 4.3 The maze set

**None of these are upstream's.** JaxGCRL ships exactly three maze layouts
(`u_maze`, `big_maze`, `hardest_maze`, plus two `*_EVAL` goal-restricted
variants), shared by `SimpleMaze`, `AntMaze` and `HumanoidMaze`. They are
general-purpose navigation benchmarks — they were not built to isolate any
particular property of a representation, and none of them is a clean control.
`hardest_maze` in particular confounds everything at once: it has rooms,
corridors, dead ends and cycles simultaneously, so a result on it cannot be
attributed to any single structural feature.

We therefore author our own set, and keep upstream's three reachable through
the `register.py` fallback (Section 4.2) for two purposes only: the setup
sanity run required by the proposal ("training sample runs on Simple maze and
Ant maze"), and as a comparison point if a reviewer asks how our mazes relate
to the benchmark's.

**Five layouts, now implemented** in `src/latentmine/mazes/layouts.py`. Four
carry the "≥ 4 mazes" requirement; the fifth is a control that doubles as an
ablation partner. The M0-M5 labels an earlier draft used are gone - the mazes
are referred to by name, which also stops `M1` meaning both a maze and a
MacBook.

Measured properties of the implemented set (`detour = d_geo / d_euc`,
`peak/mean` is the betweenness contrast of Section 9):

| maze | grid | free cells | max detour | mean detour | betweenness peak/mean | cut vertices |
|---|---|---|---|---|---|---|
| `open_room`  | 9x11  | 63 | 1.08 | 1.04 | 2.51 | 0 |
| `two_rooms`  | 9x11  | 57 | 4.00 | 1.18 | 8.09 | 3 |
| `four_rooms` | 11x11 | 68 | 3.83 | 1.23 | 3.71 | 0 |
| `spiral`     | 11x11 | 49 | 15.00 | 3.68 | 1.53 | 47 |
| `loop`       | 9x9   | 24 | 2.00 | 1.29 | 1.00 | 0 |

**`open_room` — control, and the ablation partner of `two_rooms`.**
It is exactly `two_rooms` with the dividing wall removed: same extent, same
start, same region labels, differing in precisely the six cells of column 5.
The two roles are served by one maze deliberately - an earlier draft had a
separate `two_rooms_open`, but a second all-open room is the same experiment
twice, and merging them saves a full training run across seeds and envs, which
is worth having on a laptop budget.

As a control: with no interior wall the measured max detour is 1.08, so
geodesic and Euclidean distance agree to within the octile discretisation
error. Whatever structure appears in this maze's latent projection is imposed
by the method, not by the maze, and every claim about the other four is stated
relative to it. Without this baseline "we see clusters" is unfalsifiable.
As an ablation: the a/b split is a bisection with no wall behind it, so room
purity here is the null that purity on `two_rooms` is read against. Column 5
keeps the doorway label in both mazes so purity is scored over exactly the same
56 cells - the delta is then attributable to the wall and nothing else.

**`two_rooms` — one dividing wall, one doorway.**
Tests the cleanest version of Q1: cells straddling the wall are one step apart
in space and far apart through the maze (max detour 4.0). Also the sharpest
bottleneck signal in the set, at 8.1x the mean betweenness.

One correction from implementing it: the passage is **three** cells, not one.
The wall occupies column 5 only, so crossing means traversing
`(4,4) -> (4,5) -> (4,6)`, and all three are cut vertices. Detection scoring
must expect a three-cell answer here, not a single cell.

**`four_rooms` — four halls, four doorways.**
Tests whether the latent space clusters *by room*, giving room purity a
meaningful denominator, and gives the bottleneck detectors four targets so
precision is measurable rather than only recall.

**`spiral` — one winding corridor, width 1, no branches.**
Maximal decoupling of geodesic from Euclidean distance: measured max detour is
15.0 and the mean is 3.68, far above every other maze. The two corridor
endpoints are 48 steps apart through the maze and about 5.7 cells apart in
space. If the latent space is a hitting-time embedding the spiral should
*unroll* into a 1-D curve under PCA, which makes this the most diagnostic maze
in the set. It also happens to have 49 free cells against `open_room`'s 63, so
the extreme case and the control are within a factor of 1.3 on sample size.

Its two endpoints are the set's only true dead ends - no shortest path between
any other pair passes through them - which is the dead-end probe an earlier
draft proposed bolting onto `four_rooms`. It comes free here, so no stubs are
carved. Two cells is a thin sample; a dedicated dead-end maze stays an optional
extra rather than something the argument rests on.

**`loop` — a ring corridor around a solid centre.**
The only maze with a cycle, so opposite points are joined by two equally good
routes. Tests something PCA cannot express: a ring has no faithful linear 2-D
embedding, so this is where t-SNE and UMAP should visibly beat PCA. Its
betweenness is perfectly uniform (peak/mean exactly 1.00, a useful sanity check
on the implementation), so it is also the maze where any bottleneck detector
should report nothing at all.

Each maze is instantiated for **both** `SimpleMaze` (2-D point mass, fast,
used for debugging and for all pipeline development) and `AntMaze` (29-D,
where the latent must encode gait and orientation on top of position).

---

## 5. Training

### 5.1 Run identity and artifacts

Every run gets `run_id = f"{env}_{maze}_d{n_hidden}_w{h_dim}_r{repr_dim}_{energy_fn}_s{seed}"`.
`train/run_crl.py` writes, alongside upstream's `ckpt/` and `args.pkl`:

```
runs/<run_id>/manifest.json    # our own record: architecture, maze name,
                               # upstream commit, git SHA of this repo, seed,
                               # env/goal dims, energy_fn, wandb run id
```

`checkpoints.py:load_encoders(run_dir, step="final")` reads `manifest.json`
(never `args.pkl` — see 2.7), rebuilds two `Encoder` modules with the recorded
hyperparameters, unpickles the 3-tuple, and returns

```python
Encoders(phi: Callable[[Array], Array],   # (N, state_dim+action_dim) -> (N, d)
         psi: Callable[[Array], Array],   # (N, 2)                    -> (N, d)
         actor_apply, energy_fn, meta)
```

both `jax.jit`-compiled and batched. Nothing downstream touches raw params.

### 5.2 Depth, as the proposal requires

The proposal calls for depth, citing *Scaling CRL*. Default and sweep:

| preset      | `n_hidden` | `h_dim` | `skip_connections` | `use_ln` | `repr_dim` |
|-------------|-----------|---------|--------------------|----------|------------|
| `shallow`   | 2         | 256     | 0                  | False    | 64         |
| `deep`      | 4         | 256     | 4                  | True     | 64         |
| `deeper`    | 8         | 512     | 4                  | True     | 64         |

`deep` is the project default; `shallow` is run on `two_rooms` and `spiral`
only, as evidence
for the "depth improves representation quality" claim, measured with our Q1
metric rather than with return. Note that depth and LayerNorm move together in
upstream's config space (both encoders take `use_ln`, and residual blocks
without normalisation are unstable at depth 8), so the `shallow`→`deep`
comparison is a comparison of *configurations*, not of depth alone. Say so in
the report rather than over-claiming.

`repr_dim` stays 64 throughout. A `repr_dim ∈ {8, 16, 64}` sweep on `spiral` is a
cheap optional extra: low `repr_dim` should force the maze structure to be more
explicitly laid out, and 8 dimensions may be directly plottable.

### 5.3 Hyperparameters

Following upstream defaults except where noted:
`contrastive_loss_fn=sym_infonce` (symmetric InfoNCE constrains both encoders,
which matters when we later interpolate in `psi`-space and decode),
`energy_fn=norm`, `discounting=0.99`, `batch_size=256`, `unroll_length=62`,
`logsumexp_penalty_coeff=0.1`, `episode_length=1001`, `num_envs=512`
(satisfies `512*1000 % 256 == 0`), `action_repeat=1`.

Budget: SimpleMaze 10M steps, AntMaze 50M, `num_envs=512`,
`episode_length=1001` — the `gpu` profile of Section 5.6. Seeds `{1, 2, 3}`.
`visualization_interval=10`, `checkpoint_logdir` always set.

An `energy_fn ∈ {norm, dot}` comparison on `two_rooms` is a deliberate secondary
experiment: the "Why is CRL latent space nice" paper's claims are
energy-dependent, and having both lets us say which geometry is more legible.

### 5.5 Checkpointing and resume

#### What upstream saves, and why it cannot resume

At each of the `num_evals` epoch boundaries `crl.py` writes a pickled 3-tuple
of **parameters only** (Section 2.7). Everything else the loop carries is
discarded:

| carried at the epoch boundary | in upstream's checkpoint? |
|---|---|
| `actor/critic/alpha_state.params` | yes |
| `actor/critic/alpha_state.opt_state` (Adam moments) | **no** |
| `training_state.env_steps`, `gradient_steps` | no |
| `env_state` — the `num_envs` in-flight episodes | no |
| `buffer_state` — replay data, insert/sample positions, buffer RNG | no |
| the outer `key` | no |
| loop counter `ne`, `training_walltime` | no |

`progress_fn` cannot fix this from outside: it is handed
`training_state.actor_state.params` and nothing else. So upstream checkpoints
are **restart-for-analysis artifacts, not resume artifacts**, and crash
recovery has to be added.

#### Two tiers, deliberately different

- **Analysis checkpoints** — `runs/<run_id>/ckpt/step_*.pkl`. Upstream's exact
  format and cadence, params only, immutable, one per epoch, all retained.
  These feed the latent snapshots (Section 5.4) and every downstream analysis.
  Keeping the format byte-compatible means `checkpoints.load_encoders` also
  works on runs produced by an unmodified upstream, which is worth more than
  any tidying we might do.
- **Resume checkpoints** — `runs/<run_id>/resume/`. Everything in the table
  above, rewritten every epoch, only the last two retained. Never read by the
  analysis code.

Splitting them matters because their retention policies are opposite: analysis
wants every step kept forever and is small; resume wants only the newest and is
large.

#### Size

`buffer_state.data` has shape `(max_replay_size, num_envs, data_size)` in
float32, where `data_size = obs_size + action_size + 4` (reward, discount,
truncation, traj_id):

| env | `data_size` | buffer @ `num_envs=512` | @ `num_envs=128` |
|---|---|---|---|
| `SimpleMaze` | 6 + 2 + 4 = 12 | 245 MB | 61 MB |
| `AntMaze` | 31 + 8 + 4 = 43 | 880 MB | 220 MB |

Params plus Adam moments for the `deep` preset are only single-digit MB, and
`env_state` is tens of MB. So the buffer is essentially the whole checkpoint,
and at laptop-scale `num_envs` it is small enough that **checkpointing it is
clearly worth it**: the alternative is re-paying prefill on every resume, which
is `min_replay_size * num_envs` env steps (512k at upstream defaults — around
5% of a 10M-step run) and also throws away the buffer's contents.

#### Atomicity — the part that actually saves the run

The failure that destroys a run is not a crash between writes, it is a crash
*during* one. So:

1. Serialize to `resume/state_<slot>.msgpack.tmp` in the same directory as the
   target (a rename is only atomic within a filesystem).
2. `flush()` then `os.fsync()` the file descriptor.
3. `os.replace()` onto `resume/state_<slot>.msgpack` — atomic on APFS and any
   POSIX filesystem.
4. Write a new `resume/latest.json` (pointing at the slot, with the step count
   and a SHA-256 of the payload) through the same tmp-fsync-replace dance.

Slots alternate `a`/`b`, so a torn write can never damage the previous good
checkpoint. On load, verify the checksum from `latest.json`; on mismatch, fall
back to the other slot and log loudly. Nothing is ever written in place.

**Serialization is `flax.serialization.to_bytes` (msgpack), not `pickle`.**
Pickling a `flax.struct.dataclass` embeds the defining module path, so a
submodule bump that moves a class silently makes every checkpoint unloadable —
this is exactly why we already refuse to read upstream's `args.pkl`
(Section 2.7). msgpack stores arrays only; the pytree template is rebuilt from
`manifest.json` before deserializing.

#### Where the hook goes

The epoch loop is plain Python (`for ne in range(config.num_evals)`), so the
boundary is natural — but it sits inside `CRL.train_fn`, and no callback there
receives the training state. Monkey-patching cannot reach into the middle of a
function, so we vendor a derivative:

`src/latentmine/train/crl_resumable.py` — a copy of `train_fn` taken at the
pinned commit, with three changes: load resume state before the loop; start
the loop at `ne_start` instead of 0; save resume state after each epoch. The
`assert total_steps >= config.total_env_steps` at the end has to become
resume-aware.

Drift guard: a test asserts the SHA-256 of upstream's `crl.py` equals the value
recorded when the copy was taken. Bumping the submodule then fails loudly and
someone re-derives the copy, instead of the fork silently diverging.

*Rejected alternative — segmented training.* Chain several short runs, each
loading the previous one's params. It needs no code changes at all, which is
genuinely appealing. But it resets the Adam moments and refills the replay
buffer at every boundary, putting a transient in every training curve at a
known location, and those artifacts land in the same plots we intend to draw
conclusions from. Kept only as an emergency fallback if the vendored copy
proves unmaintainable.

#### What "at most one epoch" actually means

Implementing this exposed a detail worth stating precisely. The order inside
the loop is: run the epoch, evaluate, call `progress_fn`, write the analysis
checkpoint, then write the resume checkpoint. A process killed between the
progress line and the resume write has *completed* that epoch's work but not
recorded it, so the resumed run redoes it.

That is the guarantee working, not a bug: a crash costs at most one epoch and
skips none. But it means the step counter across a resume seam can **repeat** a
value, and a test asserting a perfectly uniform sequence of steps fails on a
correct implementation. The test therefore asserts what is actually promised -
every step delta is either zero (one repeated epoch) or one epoch, never more,
and at most one repeat. Reordering the writes would move the window, not close
it, so it is left as is and documented instead.

#### Granularity is a dial

One epoch is
`(total_env_steps - min_replay_size * num_envs) / (num_evals * num_envs * unroll_length)`
training steps, and a crash costs at most one epoch. So `num_evals` *is* the
"how much work can I afford to lose" setting. Raising it is not free: each eval
runs `num_eval_envs` complete episodes, which on CPU is a real cost — so lower
`num_eval_envs` (32 is plenty for a health check) when raising `num_evals`.
Note the guard `assert num_training_steps_per_epoch > 0`: pushing `num_evals`
too high for a given `total_env_steps` fails at startup.

`train/run_crl.py` takes `--resume auto|never|<path>`, defaulting to `auto`:
resume if `resume/latest.json` exists and its config hash matches the requested
config, otherwise start fresh. A config mismatch is a hard error, never a
silent fresh start — silently restarting a 6-hour run from zero is worse than
crashing.

### 5.6 Hardware: train on GPU, analyse anywhere

Training runs on a CUDA machine. That is a change from an earlier draft of
this document, which assumed an M1 Pro laptop and cut the experiment plan
accordingly; those cuts are reverted. The laptop is still where analysis and
development happen, and that split is a comfortable one because the two halves
have very different costs.

**Why the split works.** The expensive half is rollout collection and gradient
steps. The analysis half is not: the central object, `psi`, takes only `(x, y)`
(Section 2.4), so the dense latent map over a 68-cell maze is 68 forward passes
through a 4x256 MLP. Projections, metrics and the decoder all run on arrays of
a few thousand rows. None of that needs a GPU, and none of it needs the
training stack to be fast.

So: train on the GPU box, copy `runs/<run_id>/` back, and do everything from
Section 7 onward locally. The manifest makes those directories self-describing
(Section 5.1), including the maze grid, so the analysis machine needs no
knowledge of how the run was launched.

**Budget profiles** (`train/presets.py: BUDGET_PROFILES`):

| profile | env | `num_envs` | `episode_length` | requested steps | utd |
|---|---|---|---|---|---|
| `gpu` | simple | 512 | 1001 | 10M | 0.0631 |
| `gpu` | ant | 512 | 1001 | 50M | 0.0631 |
| `laptop` | simple | 128 | 501 | 5M | 0.0316 |
| `laptop` | ant | 128 | 1001 | 10M | 0.0631 |
| `smoke` | either | 64 | 101 | 200k | 0.0064 |

`gpu` is the default and what the reported runs use. `laptop` exists for
working on the pipeline without a GPU to hand, and `smoke` for a minute-scale
end-to-end check. Note what the laptop profile costs: shortening
`episode_length` to 501 halves the update-to-data ratio, because fewer
gradient updates are taken per env step. `describe` prints ours next to
upstream's and suggests `--train-step-multiplier 2` when it drops. Episode
length is not a free knob in any case - `flatten_batch` samples the future
goal within an episode, so it sets the horizon the critic is trained over, and
it must be identical across every maze and seed in a comparison.

**Installation on the CUDA box.** Upstream pins `jaxlib==0.4.25+cuda12.cudnn89`
on Linux and requires CUDA >= 12.3:

```bash
pip install -e third_party/JaxGCRL -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -e .
```

Upstream's package `__init__` imports `wandb_osh` even for env-only use, so
`wandb` and `wandb-osh` are required whether or not logging is enabled.
Confirm the GPU is actually in use before starting a long run - JAX falls back
to CPU silently, and a run that would take twenty minutes takes a day:
`python -c "import jax; print(jax.devices())"` must show a `cuda` device.

**The experiment plan, restored.** All five mazes x both envs x three seeds on
the `deep` preset, plus `shallow` on `two_rooms` and `spiral` for the depth
comparison, plus `deep_noln` on the same two to isolate depth from LayerNorm
(Section 5.2). At upstream's stated single-GPU throughput this is hours, not
days, so the earlier plan to reduce Ant to one or two mazes is unnecessary.

**The timing probe still earns its place**, but its job has changed: not to
decide whether the plan is feasible, but to confirm the GPU is being used and
to pick `num_envs` for the specific card. Run it once on the target machine
before queueing the grid.

## 6. Metrics — how a negative result gets stated

The proposal explicitly permits a negative result. It is only worth anything
if it is quantitative. All of these are implemented in
`analysis/metrics.py` and computed for every (maze, env, preset, seed).

Let `F` be the set of free cells, `d_geo(u, v)` the BFS distance on the
4-connected free-cell graph, `d_euc(u, v)` the Euclidean distance in world
coordinates, and `d_lat(u, v) = ||psi(u) - psi(v)||_2`.

**A. Geodesic vs Euclidean correlation (Q1, headline).**
Spearman `rho(d_lat, d_geo)` and `rho(d_lat, d_euc)` over all pairs in `F`.
Report both plus the gap. On `open_room` the two are near-identical by
construction — that is the control that makes the gap on the other four meaningful.
Also report partial correlation of `d_lat` with `d_geo` controlling for
`d_euc`, which is the statistic that actually isolates "knows about walls".

**B. Wall-crossing ratio (Q1, sharp version).**
Take pairs `(u, v)` with `d_euc` below one cell but `d_geo` above a threshold —
i.e. cells that are adjacent through a wall. Compute
`median(d_lat over wall pairs) / median(d_lat over non-wall pairs at equal d_euc)`.
A ratio near 1 means the encoder ignores walls; large means it respects them.
This is a single number per maze and is the cleanest headline result.

**C. Room purity (Q2).**
For mazes carrying a `regions` overlay (Section 4.1) — `open_room`,
`two_rooms`, `four_rooms` —
fraction of each cell's `k=10` latent nearest neighbours sharing its region
label, excluding doorway cells. Compared against the same statistic computed on
raw `(x, y)`, which is the "no information beyond position" null. Not defined
for `spiral`/`loop`, which have no rooms.

**D. Projection faithfulness (Q2).**
Trustworthiness and continuity of the 2-D PCA/t-SNE/UMAP embedding w.r.t. the
latent space, so we can distinguish "the latent space has no maze structure"
from "the projection destroyed it". Without this, a bad t-SNE plot is
uninterpretable. Also report PCA explained-variance ratio: if two components
carry most of the variance, `psi` is effectively a 2-D map and the projection
is honest.

**E. Decoder reconstruction error, per observation group (Q3, Q4).**
For AntMaze, group the 29 state dims into `xy`, `z`/orientation quaternion,
joint angles, and velocities; report normalised RMSE per group on a held-out
*spatial* split. This directly answers "what beyond position is in there".

**F. Waypoint path validity (Q5).**
Fraction of decoded interpolants that land in free space; path length ratio
against the BFS geodesic; monotonicity of progress along the geodesic.

**G. Downstream success (Q5).**
Success rate and steps-to-goal for waypoint-conditioned rollouts vs baselines
(Section 10.3), over ≥ 200 start/goal pairs, with bootstrap CIs.

Every metric is reported as mean ± std over seeds. A finding that does not
survive across three seeds is reported as not surviving.

---

## 7. Latent space exploration

### 7.1 Rollouts and their latents (deliverable 2.1)

`rollouts.py:collect(run_dir, n_episodes, policy="actor"|"random", seed)` →
`artifacts/<run_id>/rollouts.npz` with arrays `obs (E, T, obs_dim)`,
`act (E, T, act_dim)`, `goal (E, 2)`, `qpos`, `done`, `success`. Rendering to
HTML uses `brax.io.html.render(env.sys.tree_replace({'opt.timestep': env.dt}),
pipeline_states)`, as upstream's notebook does.

Paired figure, one per rollout: maze floorplan with the trajectory on the
left, PCA of `phi(s_t, a_t)` along the same trajectory on the right, both
coloured by timestep, with the goal's `psi(g)` marked in the latent panel.
A correct picture shows latent distance to `psi(g)` decreasing monotonically
as the agent approaches — this is a *sanity check on the critic itself* and
should be run before any of the fancier analysis. If it fails, the checkpoint
is broken and nothing downstream is meaningful.

### 7.2 Dense latent maps (deliverable 2.2)

Because `psi` takes only `(x, y)` (Section 2.4), we do not need rollouts to
map the goal latent space: raster the free cells at sub-cell resolution (e.g.
5×5 samples per cell, with jitter), embed, project. Products:

- **Latent map:** 2-D projection of `psi(F)`, each point coloured by its true
  maze `(x, y)` under a 2-D colourmap. If the maze structure is present, the
  colour field in latent space is continuous within rooms and discontinuous
  across walls.
- **Inverse map:** the maze floorplan, each cell coloured by its projected
  latent coordinate. This is the more legible direction and the figure most
  likely to make the report — it shows the maze *segmenting itself*.
- **Distance field:** pick an anchor cell; colour every cell by
  `d_lat(anchor, ·)`; overlay BFS iso-contours. If the latent respects walls
  the two contour families agree. This single figure is the visual form of
  metric A and is the most convincing one-panel result.

For `phi` we need full states, which is what `sampling.py` is for
(Section 7.3), and the same three products are generated for `phi` at a fixed
canonical pose and action.

### 7.3 State sampling for `phi`

`sampling.py:grid_states(env, cells, pose="canonical"|"from_rollout", key)`.
For `SimpleMaze` this is trivial: set `q[:2]` to the cell centre, `qd` to zero
or to a sampled velocity. For `AntMaze` a physically valid 29-D state is
required, so we take poses from real rollout data and translate their `xy` to
the target cell, keeping joint angles and velocities intact. Sampling a pose
from a Gaussian would produce off-manifold states and the resulting latent map
would say more about extrapolation than about the maze.

Two sweeps fall out of this and answer Q3 directly:
- **fixed pose, vary `xy`** → spatial structure of `phi`;
- **fixed `xy`, vary pose/velocity** → the dynamics manifold at one location.
The relative spread of the two tells us how `phi`'s capacity is divided
between "where am I" and "how am I moving", which is exactly the Ant-specific
question the proposal asks.

### 7.4 Structure ablation (deliverable 2.3)

`two_rooms` vs `open_room` (wall present vs removed), identical otherwise.
Report the delta in
metrics A, B, C and the two latent maps side by side. Secondary ablation:
train on `four_rooms` and evaluate that encoder on states from `open_room` —
i.e. does the latent space encode *this* maze or mazes in general.

---

## 8. Decoder (deliverable 3.1)

### 8.1 Models and training

Two decoders, both trained with the CRL encoders **frozen**:

- `D_g : R^d -> R^2`, inverting `psi`. Small (3×256 MLP). This one should work;
  if it does not, something is wrong with the pipeline.
- `D_sa : R^d -> R^{state_dim + action_dim}`, inverting `phi`. Larger
  (4×512, LayerNorm, swish). This one is genuinely ill-posed: `phi` is trained
  to be *invariant* to whatever does not affect future-goal occupancy, so
  perfect reconstruction is not expected and the **per-group error profile is
  the result**, not a failure. Report it as such (metric E).

Training: Adam `3e-4`, cosine decay, batch 1024, MSE on per-dimension
standardised targets (otherwise velocity dims dominate the loss and the
position dims — the ones we care about — are ignored). Early stopping on the
held-out split.

### 8.2 Spatial holdout — the design decision that matters most here

Splitting the decoder's data randomly would let it memorise a lookup table
over sampled cells, and reconstruction error would be meaninglessly low.
Instead: **hold out whole regions.** For `four_rooms`, train on three rooms and
test on the fourth, using the `regions` overlay (Section 4.1) as the split; for
`spiral`,
which has no rooms, hold out a contiguous arc of the spiral by geodesic index
from the corridor's start. Report train,
in-distribution-held-out-samples, and held-out-region errors separately. Only
the third number says anything about whether the latent space is smoothly
organised. This is also what makes the interpolation experiment (Section 8.3)
credible, since interpolants necessarily land between training points.

### 8.3 Interpolation (deliverable 3.2)

Given start `s` and goal `g`: embed `z_0 = psi(g_s)`, `z_1 = psi(g)`, take
`z_t` for `t ∈ [0, 1]`, decode `ĝ_t = D_g(z_t)`, plot on the floorplan.

Interpolation scheme follows the energy function (Section 2.3): **linear** for
`energy_fn ∈ {norm, l2}`, **slerp on the unit sphere** for `{dot, cosine}`.
Using linear interpolation under a cosine energy is a real bug we are
pre-empting, not a hypothetical.

Baselines to plot alongside, because "the path bends around the wall" only
means something in contrast:
1. straight line in raw `(x, y)` — cuts through walls by construction;
2. BFS geodesic — the oracle;
3. latent linear interpolation — the thing under test;
4. **latent graph geodesic** — build a kNN graph over `psi(F)`, take the
   shortest path from `z_0` to `z_1` in latent space, decode its nodes.

(4) is the designed fallback for a likely failure mode: even if the latent
space encodes the maze correctly, the *straight line* between two latents may
leave the manifold of realisable goals and decode to garbage. If (3) fails and
(4) succeeds, the conclusion is "the structure is there but the latent space
is not convex", which is a much more interesting finding than "interpolation
doesn't work" — and it is only available if we build (4). Metric F scores all
four.

---

## 9. Bottlenecks (advanced)

**Ground truth is betweenness centrality, not articulation points.** The
earlier draft of this section named articulation points, which is exact,
cheap, and wrong for most of this maze set - a correction that fell out of
implementing `mazes/geometry.py`:

- `four_rooms` has **no articulation points at all**. Four doorways mean there
  is always a second route, so no single cell disconnects the maze - yet the
  four doorways are plainly the bottlenecks, and this is precisely the maze
  meant to make detector *precision* measurable.
- `spiral` has 47 of 49 free cells as articulation points, since every interior
  cell of a corridor is a cut vertex. Labelling 96% of a maze "bottleneck" is
  no more useful than labelling none of it.

`geometry.betweenness_centrality` (Brandes on the weighted maze graph) degrades
gracefully in both cases, and its peak-to-mean contrast separates the set
cleanly: 8.09 on `two_rooms`, 3.71 on `four_rooms`, 2.51 on `open_room`, and
exactly 1.00 on `loop`, whose ring is uniform by symmetry. `articulation_points`
is kept for the cases where it *is* the right question (it returns the correct
three-cell passage for `two_rooms`).

One scoring caveat, measured: in `four_rooms` all four doorways score an
identical 0.246, but their eight flanking cells score 0.251 and so outrank
them, because a flanking cell carries the doorway's traffic *plus* intra-room
traffic. A detector that points at a cell adjacent to a doorway has not failed,
so detection must be scored with a **one-cell tolerance** rather than by exact
cell match.

`analysis/bottleneck.py`, three independent detectors over `psi(F)`, scored
against that ground truth:

1. **Spectral.** kNN graph on latent distances, Fiedler vector of the
   normalised Laplacian; cells adjacent to a sign change are candidate
   bottlenecks. Natural fit for `two_rooms` (a single cut).
2. **Betweenness.** Betweenness centrality on the latent kNN graph. Handles
   `four_rooms`' four doorways, where a single Fiedler cut is the wrong model.
3. **Latent stretch.** Finite-difference estimate of `||∂psi/∂(x,y)||` per
   cell. A doorway forces all traffic through a small spatial region, so if
   latent distance is hitting time the map should be locally *stretched* there.
   This detector uses no graph construction at all, so it fails independently
   of the other two — worth having for that reason alone.

Report precision/recall against ground truth per maze. `four_rooms` is the
discriminating case; `open_room` is the false-positive test — a detector
that "finds" bottlenecks in an empty room is measuring its own hyperparameters.

**Bottlenecks as subgoals.** Feed detected bottlenecks as intermediate goals
for start/goal pairs whose geodesic crosses them, and score with metric G
against the same baselines as Section 10.3.

## 9b. Maze reconstruction (advanced)

`analysis/reconstruct.py`: for adjacent cell pairs in a candidate grid,
declare an edge open iff `d_lat` is below a threshold calibrated on known-open
pairs; the resulting edge set gives an occupancy grid. Score with edge-F1 and
occupancy IoU against ground truth. The interesting question is not whether
this works on the training maze — it is whether the threshold calibrated on
`two_rooms` transfers to `four_rooms` without recalibration.

---

## 10. Control (deliverable 3.3)

### 10.1 Waypoint conditioning

The actor consumes `obs = concat(state, goal_xy)`, so conditioning on a
waypoint is just substituting the goal — no retraining required. This is worth
stating explicitly because it is what makes the whole "exploit the latent
space" section cheap.

`control/waypoint_policy.py:rollout_with_waypoints(encoders, env, waypoints,
switch_rule)` where `switch_rule` advances to the next waypoint when
`||xy - w_i|| < goal_reach_thresh` (0.5) **or** after `max_steps_per_waypoint`,
whichever comes first. The timeout is not optional: an unreachable decoded
waypoint (inside a wall) would otherwise deadlock the episode and the
comparison against baselines would measure timeouts rather than navigation.

### 10.2 Waypoint sources

Decoded latent interpolants (§8.3), decoded latent-graph geodesics,
detected bottlenecks (§9), BFS oracle waypoints, straight-line `xy`
waypoints, and no waypoints (direct goal).

### 10.3 Evaluation protocol

≥ 200 start/goal pairs per maze, stratified by geodesic distance (near /
medium / far), fixed across all methods and seeds. Report success rate and
steps-to-goal with bootstrap CIs. The honest expectation: waypoints help
mainly in the far stratum, and stratifying is what will show that rather than
washing it out in an average.

---

## 11. Testing

Fast tests, no GPU, run in CI:

- `geometry`: grid↔world round-trip; BFS distances against hand-computed
  values on a 5×5 maze; the row→x / col→y convention asserted explicitly
  (this is the bug most likely to silently invert every figure).
- `layouts`: every `MazeSpec` parses, is rectangular, is fully walled at the
  border, has ≥ 1 start, and has all free cells connected (a disconnected
  maze would make BFS distances infinite and every metric NaN).
- `register`: after `install()`, upstream names (`u_maze`, `big_maze`,
  `hardest_maze`) still resolve, and ours resolve too.
- `checkpoints`: save a tiny fake 3-tuple + manifest, reload, assert
  `phi`/`psi` output shapes.
- `metrics`: on a synthetic latent space constructed as an exact isometric
  embedding of BFS distance, metric A must return `rho ≈ 1`; on random
  latents, `rho ≈ 0`. Metrics that cannot detect a known-good and a known-bad
  case are not evidence.

Slow tests (marked, not in CI): 10k-step SimpleMaze training smoke run,
end-to-end through embedding and one figure.

---

## 12. Build order

Each step ends with something inspectable; nothing is built before the thing
that validates it.

1. ~~Scaffolding: `pyproject.toml`, submodule, package skeleton, tests.~~ **Done.**
2. ~~`mazes/` (layouts, geometry, register) + tests.~~ **Done** - five mazes
   render to `figures/mazes/`, 65 fast tests pass, 6 integration tests for the
   upstream monkey-patch are written and marked `slow`, awaiting a machine with
   `jaxgcrl` installed. The axis convention is asserted directly rather than
   through a round trip, which a transposition would pass.
3. ~~`train/run_crl.py` + `manifest.json` + wandb.~~ **Done** - `presets.py`
   (architecture presets, env dimension table, `RunSpec` with every upstream
   constraint checked before launch), `manifest.py`, `envs.py`, and the CLI.
   `--dry-run` resolves and reports a whole configuration without importing
   JAX. The 10k-step milestone run is written as a `slow` test
   (`TestEndToEnd`) and still needs a machine with `jaxgcrl` installed.
3.5 ~~`train/crl_resumable.py` (Section 5.5) and the timing probe.~~ **Done**,
   milestone included: a run is SIGKILLed mid-training and resumed, and the
   test asserts it skips no work. One correction from doing it - see below.
4. ~~`checkpoints.py`, `embed.py`.~~ **Done.** Milestone: load a checkpoint, embed a grid,
   confirm `d_lat` to the goal decreases along a successful rollout** (§7.1).
   Do not proceed past this until it holds.
5. Full training across all five mazes × 3 seeds, `deep` preset — needs the
   GPU box.
6. ~~`analysis/` — projections, metrics A–D, plots.~~ **Done.** Metrics
   validated against three synthetic latents whose answers are known in
   advance: a classical-MDS embedding of the maze's own geodesic matrix, the
   Euclidean distance matrix (position but no walls), and noise. The
   position-only case is the one that matters — it scores rho 0.93 against
   geodesic distance and partial rho 0.00, which is exactly why the partial
   statistic exists.
7. ~~AntMaze training path.~~ Built; the env dimension table is asserted
   against the real env rather than assumed.
8. ~~`decoder/` — `D_g`, `D_sa`, spatial holdout, metric E.~~ **Done**, with
   the holdout separating as designed: 0.02 on held-out samples from training
   regions against 1.07 on a held-out region.
9. ~~`decoder/interpolate.py` + metric F, all four path baselines.~~ **Done.**
   The latent graph path already beats linear interpolation on a real
   checkpoint (0.92 of waypoints in free space against 0.76) — the
   non-convexity failure mode baseline 4 exists to catch.
10. ~~`control/` + metric G.~~ **Done** — waypoint-conditioned rollouts,
    stratified pairs, bootstrap intervals.
11. ~~Advanced: `bottleneck.py`, then `reconstruct.py`.~~ **Done**, with two
    corrections recorded in Section 9.
12. Write-up into `docs/results/` — **the remaining step**, and it needs the
    GPU runs.

Everything except the write-up is implemented and tested; what remains is
running the grid on the GPU box and interpreting the output. If time runs
short, cutting `deeper`, the `energy_fn` comparison and the advanced
extensions is preferred to cutting seeds — one seed of everything is worth
less than three seeds of the core.

---

## 13. Risks

| Risk | Why it bites | Mitigation |
|---|---|---|
| Ant never learns to navigate the harder mazes | Latents from a failed policy encode nothing about the maze | Validate with `eval/episode_success` before analysis; fall back to SimpleMaze for those layouts and say so |
| `psi` collapses (all goals map near one point) | InfoNCE degenerate solution; every metric goes to noise | Monitor `categorical_accuracy` and `logits_neg` during training; treat a collapsed run as a failed run, not as a negative result |
| Row/col ↔ x/y transposition | Every figure silently wrong, conclusions inverted | Single conversion in `geometry.py`, asserted in tests, no duplication |
| Decoder memorises instead of generalising | Reconstruction looks great, means nothing | Spatial holdout (§8.2), three error numbers always reported together |
| Linear latent interpolation leaves the manifold | Decoded waypoints are garbage; looks like a negative result but is not | Latent graph geodesic (§8.3 baseline 4) built up front |
| Upstream submodule bump breaks the monkey-patch | Silent fallback to wrong layouts | `register` test asserts both our and upstream names resolve; pin the commit |
| `deep` vs `shallow` confounded with LayerNorm | Over-claiming "depth helps" | State the confound; optionally run `n_hidden=4, use_ln=False` as a third arm |
| JAX silently falls back to CPU on the GPU box | A twenty-minute run takes a day and nobody notices until it doesn't finish | Probe asserts a `cuda` device before any long run (§5.6) |
| Runs happen on a machine we do not own | Interruptions are outside our control, and a lost run is someone else's inconvenience too | Resume checkpoints (§5.5), tested by deliberate `SIGKILL` before the grid is queued |
| Crash or interruption loses hours of training | Upstream checkpoints cannot resume — params only, no optimiser state | Resume checkpoints (§5.5), resume path tested by deliberate `SIGKILL` before any long run |
| Torn checkpoint write during a crash | The one artifact you need is the one that is corrupt | tmp + fsync + atomic rename, two-slot rotation, checksum verified on load (§5.5) |
| Vendored `train_fn` copy drifts from upstream | Silent divergence from the pinned implementation | SHA-256 drift test on upstream `crl.py` |
