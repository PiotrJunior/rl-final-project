# Contrastive RL Latent Mining

RL final project (Proposal 4). We train [CRL](https://arxiv.org/pdf/2206.07568)
agents with [JaxGCRL](https://github.com/MichalBortkiewicz/JaxGCRL) on a set of
purpose-designed mazes, and then ask what the critic's latent space actually
encodes:

- Does latent distance track the **geodesic** distance through the maze, or
  just Euclidean distance? (Walls, or no walls?)
- Is the maze **visible** in a 2-D projection of the latent space?
- Can a **decoder** trained against a frozen encoder invert the latent back
  into states — and what does it fail to recover?
- Do **latent interpolants**, decoded, form a navigable path? Are they useful
  as waypoints for the policy?

Extensions: identifying maze bottlenecks from latent structure and using them
as subgoals, and reconstructing the maze layout from the latent space alone.

## Status

The pipeline is implemented and tested; what remains is running it on a GPU and
writing up the results. See [`docs/LOW_LEVEL_DESIGN.md`](docs/LOW_LEVEL_DESIGN.md)
for the design contract — the maze set and the hypothesis each maze tests, the
metrics, and the upstream traps the implementation works around — and
[`CLAUDE.md`](CLAUDE.md) for a short orientation.

## Quick start

```bash
git submodule update --init --recursive
pip install -e third_party/JaxGCRL -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -e . && pip install wandb wandb-osh

pytest tests -q -m "not slow"          # fast checks, no JAX needed
python -m latentmine.mazes.render      # the maze set

python -m latentmine.train.probe                       # confirm the GPU is used
python -m latentmine.train.run_crl --maze two_rooms --dry-run   # check a config in a second
python -m latentmine.train.run_crl --maze two_rooms --seed 1    # train (resumes after a crash)

python -m latentmine.analysis.run runs/*               # metrics and figures
python -m latentmine.exploit runs/<run_id>             # decoder, interpolation, waypoints
```

## The maze set

Five layouts, each isolating one structural property. `open_room` is exactly
`two_rooms` minus the dividing wall, so it serves as both the global control
and the ablation partner.

| maze | tests | max detour | betweenness peak/mean |
|---|---|---|---|
| `open_room` | control: geodesic ≡ Euclidean | 1.08 | 2.51 |
| `two_rooms` | one wall, one doorway | 4.00 | 8.09 |
| `four_rooms` | clustering by room; four doorways | 3.83 | 3.71 |
| `spiral` | geodesic ≫ Euclidean | 15.00 | 1.53 |
| `loop` | a cycle; no faithful linear 2-D embedding | 2.00 | 1.00 |

## References

- [Contrastive Reinforcement Learning](https://arxiv.org/pdf/2206.07568)
- [JaxGCRL](https://github.com/MichalBortkiewicz/JaxGCRL)
- [Why is CRL latent space "nice"](https://arxiv.org/pdf/2403.04082)
- [Scaling CRL](https://wang-kevin3290.github.io/scaling-crl/)
