"""Maze specification tests: the validator, the registry, and the invariants
every maze in the set must satisfy."""

import pytest

from latentmine.mazes import geometry as geo
from latentmine.mazes import layouts as L
from latentmine.mazes.layouts import MazeSpec, MazeSpecError


class TestValidation:
    def test_rows_must_be_rectangular(self):
        with pytest.raises(MazeSpecError, match="differing widths"):
            MazeSpec(name="bad", grid=("####", "#S.#", "###"))

    def test_border_must_be_solid(self):
        with pytest.raises(MazeSpecError, match="border"):
            MazeSpec(name="bad", grid=("####", "#S..", "####"))
        with pytest.raises(MazeSpecError, match="border"):
            MazeSpec(name="bad", grid=("#.##", "#S.#", "####"))

    def test_unknown_characters_are_rejected(self):
        with pytest.raises(MazeSpecError, match="unknown characters"):
            MazeSpec(name="bad", grid=("####", "#Sx#", "####"))

    def test_a_start_is_required(self):
        with pytest.raises(MazeSpecError, match="no start cell"):
            MazeSpec(name="bad", grid=("####", "#..#", "####"))

    def test_scaling_must_be_positive(self):
        with pytest.raises(MazeSpecError, match="scaling"):
            MazeSpec(name="bad", grid=("####", "#S.#", "####"), scaling=0.0)

    def test_regions_overlay_must_match_the_grid(self):
        with pytest.raises(MazeSpecError, match="shape does not match"):
            MazeSpec(name="bad", grid=("####", "#S.#", "####"), regions=("####", "#aa#"))
        with pytest.raises(MazeSpecError, match="disagrees with grid"):
            # A label where the grid has a wall.
            MazeSpec(name="bad", grid=("####", "#S.#", "####"), regions=("####", "#aaa", "####"))


class TestCellQueries:
    def test_char_classes(self):
        spec = MazeSpec(name="t", grid=("#####", "#S.G#", "#---#", "#####"))
        assert spec.start_cells() == ((1, 1),)
        # The start is not among the goals - upstream cells are single-valued.
        assert set(spec.goal_cells()) == {(1, 2), (1, 3)}
        assert len(spec.free_cells()) == 6  # S . G and three '-'
        assert spec.is_free(2, 1) and not spec.is_wall(2, 1)

    def test_free_cells_are_row_major(self):
        spec = L.get("two_rooms")
        assert list(spec.free_cells()) == sorted(spec.free_cells())


class TestRegions:
    def test_labels_exclude_the_doorway_marker(self):
        assert L.get("two_rooms").region_labels() == ("a", "b")
        assert L.get("four_rooms").region_labels() == ("a", "b", "c", "d")

    def test_mazes_without_rooms_carry_no_overlay(self):
        for name in ("spiral", "loop"):
            spec = L.get(name)
            assert spec.regions is None
            assert spec.region_labels() == ()
            with pytest.raises(MazeSpecError, match="room purity is undefined"):
                spec.scorable_cells()

    def test_ablation_pair_scores_over_identical_cell_sets(self):
        # open_room is two_rooms minus the dividing wall. Purity must be
        # computed over the same cells in both, or the delta is not
        # attributable to the wall.
        walled, opened = L.get("two_rooms"), L.get("open_room")
        assert set(walled.scorable_cells()) == set(opened.scorable_cells())
        assert len(walled.scorable_cells()) == 56
        for label in ("a", "b"):
            assert set(walled.cells_in_region(label)) == set(opened.cells_in_region(label))

    def test_ablation_pair_differs_only_by_the_wall(self):
        walled, opened = L.get("two_rooms"), L.get("open_room")
        assert walled.shape == opened.shape
        assert walled.start_cells() == opened.start_cells()
        differing = [
            (i, j)
            for i in range(walled.n_rows)
            for j in range(walled.n_cols)
            if walled.char(i, j) != opened.char(i, j)
        ]
        # Exactly the six cells of column 5 that the wall occupies.
        assert differing == [(i, 5) for i in (1, 2, 3, 5, 6, 7)]


class TestUpstreamConversion:
    def test_characters_map_to_upstream_cell_codes(self):
        spec = MazeSpec(name="t", grid=("#####", "#S.G#", "#---#", "#####"))
        assert spec.to_upstream_layout() == [
            [1, 1, 1, 1, 1],
            [1, "r", "g", "g", 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]

    def test_every_maze_converts(self):
        for name in L.names():
            layout = L.get(name).to_upstream_layout()
            spec = L.get(name)
            assert len(layout) == spec.n_rows
            assert all(len(row) == spec.n_cols for row in layout)


class TestEvalVariant:
    def test_goals_are_restricted_to_the_named_region(self):
        spec = L.get("four_rooms")
        ev = spec.eval_variant("d")
        assert set(ev.goal_cells()) == set(spec.cells_in_region("d")) - set(spec.start_cells())
        assert ev.start_cells() == spec.start_cells()
        # Walls are untouched, so the two mazes stay geometrically identical.
        assert geo.occupancy(ev).tolist() == geo.occupancy(spec).tolist()

    def test_unknown_region_is_rejected(self):
        with pytest.raises(MazeSpecError, match="unknown region"):
            L.get("four_rooms").eval_variant("z")

    def test_needs_an_overlay(self):
        with pytest.raises(MazeSpecError, match="needs a regions overlay"):
            L.get("spiral").eval_variant("a")


class TestTheMazeSet:
    def test_registry_is_keyed_by_name(self):
        for name in L.names():
            assert L.get(name).name == name

    def test_unknown_maze_raises_with_the_known_names(self):
        with pytest.raises(KeyError, match="unknown maze"):
            L.get("labyrinth")

    def test_the_set_meets_the_project_requirement(self):
        # Four designed mazes plus a control that doubles as the ablation
        # partner of two_rooms.
        assert len(L.names()) >= 5
        assert "open_room" in L.names()

    @pytest.mark.parametrize("name", L.names())
    def test_maze_invariants(self, name):
        spec = L.get(name)
        assert geo.is_connected(spec), "all free cells must be mutually reachable"
        assert spec.start_cells(), "needs a start"
        assert spec.notes.strip(), "every maze must record the hypothesis it tests"
        assert len(spec.free_cells()) >= 20, "too small to embed meaningfully"

    @pytest.mark.parametrize("name", L.names())
    def test_starts_are_disjoint_from_goals(self, name):
        # Mirrors upstream: `find_starts` collects "r" cells and `find_goals`
        # collects "g" cells, and no cell is both.
        spec = L.get(name)
        assert not (set(spec.start_cells()) & set(spec.goal_cells()))
        assert set(spec.start_cells()) | set(spec.goal_cells()) <= set(spec.free_cells())
