"""Analyse a trained run: embed, measure, plot.

    python -m latentmine.analysis.run runs/simple_two_rooms_d4_w256_r64_norm_s1
    python -m latentmine.analysis.run runs/* --out artifacts --figures figures

Reads only a run directory - the manifest makes it self-describing, including
the maze it was trained on - so this runs on a laptop while training happens
elsewhere (LLD section 5.6).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .. import checkpoints, embed, sampling
from ..mazes import geometry as geo
from . import metrics as M
from . import plots, projections


def analyse_run(
    run_dir: Path,
    out_dir: Path,
    figure_dir: Path,
    step: int | str = "last",
    subdivisions: int = 1,
    projection_methods: tuple[str, ...] = ("pca",),
    seed: int = 0,
) -> dict:
    """Embed a run's goal grid, compute every metric, and write the figures."""
    encoders = checkpoints.load_encoders(run_dir, step=step)
    spec = encoders.maze_spec

    # psi takes only (x, y), so the whole latent map costs one forward pass
    # per cell and needs no rollouts at all.
    cell_xy = sampling.cell_centres(spec)
    latents = embed.embed_goals(encoders, cell_xy)
    d_lat = embed.pairwise_latent_distance(latents, latents, encoders.energy_fn)

    report = M.evaluate(spec, d_lat, run_id=encoders.run_id, step=encoders.step)

    figure_dir = Path(figure_dir) / encoders.run_id
    anchor = spec.start_cells()[0]
    anchor_index = list(spec.free_cells()).index(anchor)
    plots.latent_distance_field(
        spec,
        d_lat[anchor_index],
        anchor,
        figure_dir / "distance_field.png",
        title=f"{encoders.run_id} @ step {encoders.step:,}",
    )

    dense_xy, dense_index = sampling.goal_grid(spec, subdivisions=subdivisions)
    dense = embed.embed_goals(encoders, dense_xy)
    d_dense = embed.pairwise_latent_distance(dense, dense, encoders.energy_fn)

    for method in projection_methods:
        projection = projections.project(dense, method, encoders.energy_fn, seed=seed)
        plots.latent_map(
            spec,
            projection.coords,
            dense_index,
            figure_dir / f"latent_map_{method}.png",
            title=f"{encoders.run_id}: {method.upper()}",
            explained=projection.explained if method == "pca" else None,
        )
        low = np.linalg.norm(projection.coords[:, None, :] - projection.coords[None, :, :], axis=-1)
        report.projection[f"{method}_trustworthiness"] = M.trustworthiness(d_dense, low)
        report.projection[f"{method}_continuity"] = M.continuity(d_dense, low)
        if method == "pca":
            report.projection["pca_explained"] = projection.explained
            report.projection["pca_procrustes_error"] = projections.procrustes_error(
                projection.coords, geo.cells_to_world(geo.free_cell_array(spec), spec.scaling)[dense_index]
            )

    out_dir = Path(out_dir) / encoders.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "latents.npz", cell_xy=cell_xy, latents=latents, d_lat=d_lat)
    payload = report.as_dict()
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="analysis.run", description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="run directories")
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument("--step", default="last")
    parser.add_argument("--subdivisions", type=int, default=3)
    parser.add_argument("--projections", default="pca", help="comma-separated: pca,tsne,umap")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    reports = []
    for run in args.runs:
        payload = analyse_run(
            run,
            args.out,
            args.figures,
            step=args.step,
            subdivisions=args.subdivisions,
            projection_methods=tuple(args.projections.split(",")),
            seed=args.seed,
        )
        reports.append(payload)
        geometry = payload["geometry"]
        wall = payload["wall_crossing"]
        print(
            f"{payload['run_id']:<44s} step {payload['step']:>10,}  "
            f"rho_geo={geometry['rho_geodesic']:+.3f}  "
            f"partial={geometry['partial_rho_geodesic']:+.3f}  "
            f"wall={wall['ratio']:.2f}"
        )

    # Group by maze so seeds aggregate together, as every reported figure must.
    by_maze: dict[str, list] = {}
    for payload in reports:
        by_maze.setdefault(payload["maze"], []).append(payload)
    if any(len(v) > 1 for v in by_maze.values()):
        print("\nper maze, mean +/- sd over seeds:")
        for maze, group in sorted(by_maze.items()):
            values = [g["geometry"]["partial_rho_geodesic"] for g in group]
            print(
                f"  {maze:<14s} partial rho = {np.mean(values):+.3f} +/- {np.std(values):.3f}"
                f"  (n={len(values)})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
