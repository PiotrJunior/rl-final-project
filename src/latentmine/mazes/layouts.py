"""Maze specifications: the ASCII source of truth for every maze we train on.

Upstream JaxGCRL ships three general-purpose benchmark layouts and no notion of
a room. Neither fits this project: we need mazes designed to isolate one
structural property each, plus controls, plus per-cell region labels for the
room-purity metric and the decoder's spatial holdout. See LLD sections 4.1-4.3.

This module is deliberately free of JAX, brax and mujoco imports so that the
maze set and its geometry can be inspected and tested without a training stack
installed.
"""

from __future__ import annotations

from dataclasses import dataclass

# ASCII vocabulary for `MazeSpec.grid`.
WALL = "#"
START = "S"  # free, a valid start; NOT sampled as a goal (see GOAL_CHARS)
FREE = "."  # free, a valid goal
GOAL_ONLY = "G"  # free, a valid goal, never a start
NO_GOAL = "-"  # free, but never sampled as a goal (eval layouts)

FREE_CHARS = frozenset({START, FREE, GOAL_ONLY, NO_GOAL})
# Upstream cells are single-valued - `"r"` or `"g"`, never both - so a start
# cell is not sampled as a goal. With one start per maze that costs a single
# cell of goal coverage, and matching upstream's semantics exactly is worth
# more than recovering it.
GOAL_CHARS = frozenset({FREE, GOAL_ONLY})
START_CHARS = frozenset({START})

# In a `regions` overlay this label marks a doorway: a free cell whose room
# membership is genuinely ambiguous. Excluded from room-purity scoring rather
# than being counted as a failure of whichever room it is not assigned to.
DOORWAY = "+"

# Upstream cell codes (jaxgcrl.envs.simple_maze / ant_maze).
_UPSTREAM_WALL = 1
_UPSTREAM_RESET = "r"
_UPSTREAM_GOAL = "g"
_UPSTREAM_FREE = 0


class MazeSpecError(ValueError):
    """A maze specification is malformed."""


@dataclass(frozen=True)
class MazeSpec:
    """One maze layout.

    Attributes:
        name: registry key, and the `maze_layout_name` passed to upstream envs.
        grid: ASCII rows. `grid[i][j]` is the cell at grid row `i`, column `j`.
        regions: optional overlay of the same shape assigning each free cell a
            room label. Hand-authored, never derived - see LLD section 4.1 for
            why. `None` for mazes with no meaningful room partition.
        scaling: world units per cell. Must match the env's
            `maze_size_scaling`; upstream's default is 4.0.
        notes: the hypothesis this maze exists to test.
    """

    name: str
    grid: tuple[str, ...]
    regions: tuple[str, ...] | None = None
    scaling: float = 4.0
    notes: str = ""

    def __post_init__(self) -> None:
        self._validate()

    # -- validation --------------------------------------------------------

    def _validate(self) -> None:
        if not self.grid:
            raise MazeSpecError(f"{self.name}: empty grid")

        widths = {len(row) for row in self.grid}
        if len(widths) != 1:
            raise MazeSpecError(f"{self.name}: rows have differing widths {sorted(widths)}")

        allowed = FREE_CHARS | {WALL}
        for i, row in enumerate(self.grid):
            bad = set(row) - allowed
            if bad:
                raise MazeSpecError(f"{self.name}: row {i} has unknown characters {sorted(bad)}")

        # A non-walled border would let the agent escape the maze, and would
        # make the geodesic graph meaningless.
        if any(c != WALL for c in self.grid[0]) or any(c != WALL for c in self.grid[-1]):
            raise MazeSpecError(f"{self.name}: top/bottom border is not solid wall")
        if any(row[0] != WALL or row[-1] != WALL for row in self.grid):
            raise MazeSpecError(f"{self.name}: left/right border is not solid wall")

        if not self.start_cells():
            raise MazeSpecError(f"{self.name}: no start cell ('{START}')")

        if self.scaling <= 0:
            raise MazeSpecError(f"{self.name}: scaling must be positive, got {self.scaling}")

        if self.regions is not None:
            self._validate_regions()

    def _validate_regions(self) -> None:
        assert self.regions is not None
        if len(self.regions) != self.n_rows or any(len(r) != self.n_cols for r in self.regions):
            raise MazeSpecError(f"{self.name}: regions overlay shape does not match grid")
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                wall = self.grid[i][j] == WALL
                labelled_wall = self.regions[i][j] == WALL
                if wall != labelled_wall:
                    raise MazeSpecError(
                        f"{self.name}: regions overlay disagrees with grid at ({i}, {j}): "
                        f"grid={self.grid[i][j]!r} regions={self.regions[i][j]!r}"
                    )

    # -- shape -------------------------------------------------------------

    @property
    def n_rows(self) -> int:
        return len(self.grid)

    @property
    def n_cols(self) -> int:
        return len(self.grid[0])

    @property
    def shape(self) -> tuple[int, int]:
        return self.n_rows, self.n_cols

    # -- cell queries ------------------------------------------------------

    def char(self, i: int, j: int) -> str:
        return self.grid[i][j]

    def is_wall(self, i: int, j: int) -> bool:
        return self.grid[i][j] == WALL

    def is_free(self, i: int, j: int) -> bool:
        return self.grid[i][j] in FREE_CHARS

    def _cells_matching(self, chars: frozenset[str]) -> tuple[tuple[int, int], ...]:
        return tuple(
            (i, j) for i in range(self.n_rows) for j in range(self.n_cols) if self.grid[i][j] in chars
        )

    def free_cells(self) -> tuple[tuple[int, int], ...]:
        """All traversable cells, in row-major order. This ordering is the
        canonical index used by every distance matrix and embedding array."""
        return self._cells_matching(FREE_CHARS)

    def goal_cells(self) -> tuple[tuple[int, int], ...]:
        return self._cells_matching(GOAL_CHARS)

    def start_cells(self) -> tuple[tuple[int, int], ...]:
        return self._cells_matching(START_CHARS)

    # -- regions -----------------------------------------------------------

    def region_of(self, i: int, j: int) -> str | None:
        """Room label of a cell, or None if this maze carries no overlay."""
        if self.regions is None:
            return None
        return self.regions[i][j]

    def region_labels(self) -> tuple[str, ...]:
        """Distinct room labels, doorways excluded, sorted."""
        if self.regions is None:
            return ()
        return tuple(sorted({self.regions[i][j] for (i, j) in self.free_cells()} - {DOORWAY}))

    def cells_in_region(self, label: str) -> tuple[tuple[int, int], ...]:
        if self.regions is None:
            raise MazeSpecError(f"{self.name}: no regions overlay")
        return tuple((i, j) for (i, j) in self.free_cells() if self.regions[i][j] == label)

    def scorable_cells(self) -> tuple[tuple[int, int], ...]:
        """Free cells eligible for room-purity scoring: doorways excluded."""
        if self.regions is None:
            raise MazeSpecError(f"{self.name}: no regions overlay, room purity is undefined")
        return tuple((i, j) for (i, j) in self.free_cells() if self.regions[i][j] != DOORWAY)

    # -- conversion --------------------------------------------------------

    def to_upstream_layout(self) -> list[list]:
        """The list-of-lists form `jaxgcrl.envs.*_maze.make_maze` consumes.

        Mapping is literal, one character to one cell code - no train/eval
        transformation happens here. Use `eval_variant` to restrict goals.
        """
        mapping = {
            WALL: _UPSTREAM_WALL,
            START: _UPSTREAM_RESET,
            FREE: _UPSTREAM_GOAL,
            GOAL_ONLY: _UPSTREAM_GOAL,
            NO_GOAL: _UPSTREAM_FREE,
        }
        return [[mapping[c] for c in row] for row in self.grid]

    def eval_variant(self, goal_regions: str | tuple[str, ...], name: str | None = None) -> MazeSpec:
        """A copy whose goals are restricted to the given room label(s).

        Mirrors upstream's `*_MAZE_EVAL` layouts: start cells are unchanged,
        cells inside `goal_regions` stay samplable as goals, everything else
        becomes free-but-not-a-goal. Derived from the overlay rather than
        authored a second time, so a wall edit cannot desynchronise the two.
        """
        if self.regions is None:
            raise MazeSpecError(f"{self.name}: eval_variant needs a regions overlay")
        wanted = (goal_regions,) if isinstance(goal_regions, str) else tuple(goal_regions)
        unknown = set(wanted) - set(self.region_labels())
        if unknown:
            raise MazeSpecError(f"{self.name}: unknown region label(s) {sorted(unknown)}")

        rows = []
        for i in range(self.n_rows):
            row = []
            for j in range(self.n_cols):
                c = self.grid[i][j]
                if c == WALL or c == START:
                    row.append(c)
                elif self.regions[i][j] in wanted:
                    row.append(GOAL_ONLY)
                else:
                    row.append(NO_GOAL)
            rows.append("".join(row))
        return MazeSpec(
            name=name or f"{self.name}_eval",
            grid=tuple(rows),
            regions=self.regions,
            scaling=self.scaling,
            notes=f"eval variant of {self.name}; goals restricted to region(s) {wanted}",
        )


# ---------------------------------------------------------------------------
# The maze set (LLD section 4.3)
#
# None of these are upstream's. Each isolates one structural property; the two
# controls are what make the other four interpretable. Grid row `i` maps to
# world x and column `j` to world y - see geometry.py, which owns that
# conversion.
# ---------------------------------------------------------------------------

OPEN_ROOM = MazeSpec(
    name="open_room",
    grid=(
        "###########",
        "#S........#",
        "#.........#",
        "#.........#",
        "#.........#",
        "#.........#",
        "#.........#",
        "#.........#",
        "###########",
    ),
    regions=(
        "###########",
        "#aaaa+bbbb#",
        "#aaaa+bbbb#",
        "#aaaa+bbbb#",
        "#aaaa+bbbb#",
        "#aaaa+bbbb#",
        "#aaaa+bbbb#",
        "#aaaa+bbbb#",
        "###########",
    ),
    notes=(
        "Control, and the ablation partner of two_rooms - it is exactly two_rooms "
        "with the dividing wall removed, so the pair differs by one wall and nothing "
        "else: same extent, same start, same region labels, same hyperparameters. "
        "Serving both roles with one maze saves a training run, which is worth "
        "having when the budget is a laptop CPU.\n\n"
        "As a control: with no interior wall, geodesic and Euclidean distance agree "
        "up to grid anisotropy, so any structure visible in this maze's latent "
        "projection is imposed by the method rather than by the maze. As an "
        "ablation: the a/b split is a bisection with no wall behind it, so room "
        "purity here is the null value that purity on two_rooms is read against. "
        "Column 5 keeps the doorway label so both mazes score purity over exactly "
        "the same 56 cells."
    ),
)

TWO_ROOMS = MazeSpec(
    name="two_rooms",
    grid=(
        "###########",
        "#S...#....#",
        "#....#....#",
        "#....#....#",
        "#.........#",
        "#....#....#",
        "#....#....#",
        "#....#....#",
        "###########",
    ),
    regions=(
        "###########",
        "#aaaa#bbbb#",
        "#aaaa#bbbb#",
        "#aaaa#bbbb#",
        "#aaaa+bbbb#",
        "#aaaa#bbbb#",
        "#aaaa#bbbb#",
        "#aaaa#bbbb#",
        "###########",
    ),
    notes=(
        "One dividing wall pierced by a single doorway at (4, 5). The cleanest test "
        "of whether latent distance follows geodesic rather than Euclidean distance: "
        "cells straddling the wall are one step apart in space and far apart through "
        "the maze. Also the ground truth for bottleneck detection - exactly one "
        "articulation point, and we know where it is. Paired with open_room, which "
        "is this maze minus the wall."
    ),
)

FOUR_ROOMS = MazeSpec(
    name="four_rooms",
    grid=(
        "###########",
        "#S...#....#",
        "#.........#",
        "#....#....#",
        "#....#....#",
        "##.#####.##",
        "#....#....#",
        "#....#....#",
        "#.........#",
        "#....#....#",
        "###########",
    ),
    regions=(
        "###########",
        "#aaaa#bbbb#",
        "#aaaa+bbbb#",
        "#aaaa#bbbb#",
        "#aaaa#bbbb#",
        "##+#####+##",
        "#cccc#dddd#",
        "#cccc#dddd#",
        "#cccc+dddd#",
        "#cccc#dddd#",
        "###########",
    ),
    notes=(
        "Four halls joined by four doorways. Tests whether the latent space clusters "
        "by room, giving room purity a meaningful denominator, and gives the "
        "bottleneck detectors four targets instead of one so precision is measurable "
        "and not just recall."
    ),
)

# Generated by the spiral carve in LLD section 4.3 and then frozen here: a
# width-1 corridor, 49 free cells, no junctions, endpoints at (1, 1) and
# (5, 5). Those two endpoints are the maze set's only true dead ends - no
# shortest path between any other pair of cells passes through them.
SPIRAL = MazeSpec(
    name="spiral",
    grid=(
        "###########",
        "#S........#",
        "#########.#",
        "#.......#.#",
        "#.#####.#.#",
        "#.#...#.#.#",
        "#.#.###.#.#",
        "#.#.....#.#",
        "#.#######.#",
        "#.........#",
        "###########",
    ),
    notes=(
        "One long winding corridor. Maximally decouples geodesic from Euclidean "
        "distance: the two endpoints are 48 steps apart through the maze and about "
        "5.7 cells apart in space. If latent distance is a hitting-time estimate the "
        "spiral should unroll into a 1-D curve under PCA, which makes this the most "
        "diagnostic maze in the set. No rooms, so it carries no regions overlay and "
        "room purity is undefined here."
    ),
)

LOOP = MazeSpec(
    name="loop",
    grid=(
        "#########",
        "#S......#",
        "#.#####.#",
        "#.#####.#",
        "#.#####.#",
        "#.#####.#",
        "#.#####.#",
        "#.......#",
        "#########",
    ),
    notes=(
        "A ring corridor around a solid centre - the only maze here with a cycle, so "
        "opposite points are joined by two equally good routes. A ring has no "
        "faithful linear 2-D embedding, so this is where t-SNE and UMAP should "
        "visibly beat PCA, and where we learn whether latent distance handles two "
        "competing shortest paths. No rooms, so no regions overlay."
    ),
)


MAZES: dict[str, MazeSpec] = {spec.name: spec for spec in (OPEN_ROOM, TWO_ROOMS, FOUR_ROOMS, SPIRAL, LOOP)}


def get(name: str) -> MazeSpec:
    """Look up a maze by name."""
    try:
        return MAZES[name]
    except KeyError:
        raise KeyError(f"unknown maze {name!r}; known mazes: {sorted(MAZES)}") from None


def names() -> tuple[str, ...]:
    return tuple(sorted(MAZES))
