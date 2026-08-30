"""Make our maze specs available to upstream's env classes.

Upstream selects a layout with a hardcoded ``if/elif`` chain inside
``make_maze``, and bakes the list of environment names into a ``Literal`` type
at import time (``jaxgcrl.utils.env.legal_envs``), so neither can be extended
cleanly from outside. Rather than fork, we do two things - see LLD section 4.2:

1. ``install()`` replaces ``make_maze`` in the two maze env modules with a
   registry-backed version that falls back to the original for upstream's own
   layout names.
2. We never call ``jaxgcrl.utils.env.create_env`` and never go through
   ``run.py``'s tyro CLI, so the ``Literal`` never sees our names at all.

``build_maze_xml`` is kept free of JAX, brax and mujoco imports so it can be
tested without a training stack; only ``install`` needs upstream present.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from collections.abc import Callable
from typing import Any

import numpy as np

from .layouts import MAZES, MazeSpec

# Upstream's wall block height, in units of `maze_size_scaling`
# (`jaxgcrl.envs.simple_maze.MAZE_HEIGHT`). Repeated here because
# `build_maze_xml` must stay importable without upstream; `install` checks the
# two agree and refuses to proceed if upstream has changed it.
MAZE_HEIGHT = 0.5

_WALL_RGBA = "0.7 0.5 0.3 1.0"

_ORIGINALS: dict[str, Callable] = {}


def build_maze_xml(
    asset_xml_path: str,
    layout: list[list],
    scaling: float,
    maze_height: float = MAZE_HEIGHT,
) -> bytes:
    """Emit the MuJoCo XML for a maze layout, matching upstream `make_maze`.

    One box geom per wall cell, named `block_<i>_<j>`, centred at
    `(i * scaling, j * scaling)` - the same row->x / column->y convention
    `geometry` documents.
    """
    tree = ET.parse(asset_xml_path)
    worldbody = tree.find(".//worldbody")
    if worldbody is None:
        raise ValueError(f"{asset_xml_path}: no <worldbody> to attach maze blocks to")

    for i, row in enumerate(layout):
        for j, cell in enumerate(row):
            if cell != 1:
                continue
            ET.SubElement(
                worldbody,
                "geom",
                # `:d` / `:f` reproduce upstream's "%d" / "%f" byte for byte
                # (both render six decimal places), which is what lets
                # test_our_builder_agrees_with_upstream_on_an_upstream_layout
                # compare the two XML strings directly.
                name=f"block_{i:d}_{j:d}",
                pos=f"{i * scaling:f} {j * scaling:f} {maze_height / 2 * scaling:f}",
                size=f"{0.5 * scaling:f} {0.5 * scaling:f} {maze_height / 2 * scaling:f}",
                type="box",
                material="",
                contype="1",
                conaffinity="1",
                rgba=_WALL_RGBA,
            )
    return ET.tostring(tree.getroot())


def cells_of(layout: list[list], code) -> np.ndarray:
    """World positions of every cell equal to `code`, as upstream's
    `find_starts` / `find_goals` compute them - except returning numpy so this
    stays importable without JAX. `install` converts to `jnp`."""
    out = [[i, j] for i, row in enumerate(layout) for j, cell in enumerate(row) if cell == code]
    return np.array(out, dtype=float)


def _make_registry_maze_fn(module: Any, original: Callable) -> Callable:
    """Wrap upstream's `make_maze` so our specs resolve and its own still do."""

    def make_maze(maze_layout_name: str, maze_size_scaling: float):
        spec: MazeSpec | None = MAZES.get(maze_layout_name)
        if spec is None:
            return original(maze_layout_name, maze_size_scaling)

        from jax import numpy as jnp

        layout = spec.to_upstream_layout()
        asset = os.path.join(
            os.path.dirname(os.path.realpath(module.__file__)), "assets", _asset_name(module)
        )
        xml_string = build_maze_xml(asset, layout, maze_size_scaling, module.MAZE_HEIGHT)
        starts = jnp.array(cells_of(layout, module.RESET) * maze_size_scaling)
        goals = jnp.array(cells_of(layout, module.GOAL) * maze_size_scaling)
        return xml_string, starts, goals

    return make_maze


def _asset_name(module: Any) -> str:
    """Which asset XML a maze env module builds on."""
    name = module.__name__.rsplit(".", 1)[-1]
    return f"{name}.xml"


def install(modules: tuple[str, ...] = ("simple_maze", "ant_maze")) -> tuple[str, ...]:
    """Patch upstream's `make_maze` in place. Idempotent; returns what it patched.

    Must run before any maze env is constructed. `latentmine/__init__.py`
    calls it, so importing anything from this package is enough.
    """
    import importlib

    patched = []
    for short in modules:
        module = importlib.import_module(f"jaxgcrl.envs.{short}")
        if short in _ORIGINALS:
            continue  # already installed
        if module.MAZE_HEIGHT != MAZE_HEIGHT:
            raise RuntimeError(
                f"jaxgcrl.envs.{short}.MAZE_HEIGHT is {module.MAZE_HEIGHT}, expected "
                f"{MAZE_HEIGHT}. Upstream changed the wall geometry; re-check register.py "
                "against the pinned commit before trusting any maze it builds."
            )
        _ORIGINALS[short] = module.make_maze
        module.make_maze = _make_registry_maze_fn(module, _ORIGINALS[short])
        patched.append(short)
    return tuple(patched)


def uninstall() -> None:
    """Restore upstream's `make_maze`. For tests; not used in normal operation."""
    import importlib

    for short, original in list(_ORIGINALS.items()):
        importlib.import_module(f"jaxgcrl.envs.{short}").make_maze = original
        del _ORIGINALS[short]


def is_installed() -> bool:
    return bool(_ORIGINALS)
