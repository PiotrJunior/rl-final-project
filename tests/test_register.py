"""Tests for registering our maze specs with upstream.

Split in two. The XML builder is pure stdlib+numpy and is checked here against
upstream's documented geom formula, so it runs in the fast suite with no
training stack installed. The monkey-patch itself needs `jaxgcrl` and is marked
`slow`; those tests are what guard against a submodule bump silently breaking
the fallback to upstream's own layouts.
"""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from latentmine.mazes import layouts as L
from latentmine.mazes import register as R

MINIMAL_ASSET = """<mujoco model="test">
  <worldbody>
    <body name="torso" pos="0 0 0.75"/>
  </worldbody>
</mujoco>
"""

# Upstream's U_MAZE, verbatim from jaxgcrl/envs/simple_maze.py at the pinned
# commit. Used to check our builder against a layout we did not author.
U_MAZE = [
    [1, 1, 1, 1, 1],
    [1, "r", "g", "g", 1],
    [1, 1, 1, "g", 1],
    [1, "g", "g", "g", 1],
    [1, 1, 1, 1, 1],
]


@pytest.fixture
def asset(tmp_path):
    path = tmp_path / "simple_maze.xml"
    path.write_text(MINIMAL_ASSET)
    return str(path)


class TestBuildMazeXml:
    def test_one_box_geom_per_wall_cell(self, asset):
        root = ET.fromstring(R.build_maze_xml(asset, U_MAZE, 4.0))
        blocks = [g for g in root.iter("geom") if (g.get("name") or "").startswith("block_")]
        n_walls = sum(c == 1 for row in U_MAZE for c in row)
        assert len(blocks) == n_walls == 18
        assert all(g.get("type") == "box" for g in blocks)

    def test_geom_position_and_size_follow_upstreams_formula(self, asset):
        scaling, height = 4.0, R.MAZE_HEIGHT
        root = ET.fromstring(R.build_maze_xml(asset, U_MAZE, scaling))
        by_name = {g.get("name"): g for g in root.iter("geom") if g.get("name")}
        for i, row in enumerate(U_MAZE):
            for j, cell in enumerate(row):
                if cell != 1:
                    continue
                g = by_name[f"block_{i}_{j}"]
                pos = [float(v) for v in g.get("pos").split()]
                size = [float(v) for v in g.get("size").split()]
                # Row indexes x, column indexes y - the project's one convention.
                assert pos == pytest.approx([i * scaling, j * scaling, height / 2 * scaling])
                assert size == pytest.approx([0.5 * scaling, 0.5 * scaling, height / 2 * scaling])

    def test_free_cells_get_no_geom(self, asset):
        root = ET.fromstring(R.build_maze_xml(asset, U_MAZE, 4.0))
        names = {g.get("name") for g in root.iter("geom")}
        assert "block_1_1" not in names  # the reset cell
        assert "block_1_2" not in names  # a goal cell

    def test_existing_asset_content_is_preserved(self, asset):
        root = ET.fromstring(R.build_maze_xml(asset, U_MAZE, 4.0))
        assert root.find(".//body[@name='torso']") is not None

    def test_missing_worldbody_is_an_error(self, tmp_path):
        path = tmp_path / "bad.xml"
        path.write_text("<mujoco model='test'/>")
        with pytest.raises(ValueError, match="no <worldbody>"):
            R.build_maze_xml(str(path), U_MAZE, 4.0)

    @pytest.mark.parametrize("name", L.names())
    def test_every_maze_in_the_set_builds(self, asset, name):
        spec = L.get(name)
        root = ET.fromstring(R.build_maze_xml(asset, spec.to_upstream_layout(), spec.scaling))
        blocks = [g for g in root.iter("geom") if (g.get("name") or "").startswith("block_")]
        n_walls = spec.n_rows * spec.n_cols - len(spec.free_cells())
        assert len(blocks) == n_walls


class TestCellsOf:
    def test_finds_starts_and_goals_in_grid_order(self):
        starts = R.cells_of(U_MAZE, "r")
        np.testing.assert_array_equal(starts, [[1, 1]])
        goals = R.cells_of(U_MAZE, "g")
        np.testing.assert_array_equal(goals, [[1, 2], [1, 3], [2, 3], [3, 1], [3, 2], [3, 3]])

    def test_matches_the_spec_it_came_from(self):
        spec = L.get("four_rooms")
        layout = spec.to_upstream_layout()
        np.testing.assert_array_equal(R.cells_of(layout, "r"), np.array(spec.start_cells(), float))
        np.testing.assert_array_equal(R.cells_of(layout, "g"), np.array(spec.goal_cells(), float))


@pytest.mark.slow
class TestInstall:
    """Needs the pinned upstream installed (`pip install -e third_party/JaxGCRL`)."""

    @pytest.fixture(autouse=True)
    def _upstream(self):
        pytest.importorskip("jaxgcrl.envs.simple_maze")
        yield
        R.uninstall()

    def test_install_is_idempotent(self):
        assert R.install() == ("simple_maze", "ant_maze")
        assert R.install() == ()
        assert R.is_installed()

    def test_upstream_layouts_still_resolve(self):
        import jaxgcrl.envs.simple_maze as sm

        R.install()
        for name in ("u_maze", "big_maze", "hardest_maze"):
            xml, starts, goals = sm.make_maze(name, 4.0)
            assert len(xml) > 0 and len(starts) > 0 and len(goals) > 0

    def test_our_layouts_resolve(self):
        import jaxgcrl.envs.simple_maze as sm

        R.install()
        for name in L.names():
            spec = L.get(name)
            xml, starts, goals = sm.make_maze(name, spec.scaling)
            root = ET.fromstring(xml)
            blocks = [g for g in root.iter("geom") if (g.get("name") or "").startswith("block_")]
            assert len(blocks) == spec.n_rows * spec.n_cols - len(spec.free_cells())
            assert len(starts) == len(spec.start_cells())
            assert len(goals) == len(spec.goal_cells())

    def test_our_builder_agrees_with_upstream_on_an_upstream_layout(self):
        """The equivalence check: fed upstream's own U_MAZE, our XML builder
        must produce exactly what upstream's `make_maze` does. This is what
        catches a drift in the geom formula after a submodule bump."""
        import os

        import jaxgcrl.envs.simple_maze as sm

        theirs, _, _ = sm.make_maze("u_maze", 4.0)  # before install: the original
        asset = os.path.join(os.path.dirname(os.path.realpath(sm.__file__)), "assets", "simple_maze.xml")
        ours = R.build_maze_xml(asset, U_MAZE, 4.0, sm.MAZE_HEIGHT)
        assert ours == theirs

    def test_unknown_name_still_raises_upstreams_error(self):
        import jaxgcrl.envs.simple_maze as sm

        R.install()
        with pytest.raises(ValueError, match="Unknown maze layout"):
            sm.make_maze("no_such_maze", 4.0)

    def test_uninstall_restores_the_original(self):
        import jaxgcrl.envs.simple_maze as sm

        original = sm.make_maze
        R.install()
        assert sm.make_maze is not original
        R.uninstall()
        assert sm.make_maze is original
