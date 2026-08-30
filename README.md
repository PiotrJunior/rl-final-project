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

Design stage — see [`docs/LOW_LEVEL_DESIGN.md`](docs/LOW_LEVEL_DESIGN.md) for
the full plan, the maze set and the rationale behind each one, the metrics, and
the build order. [`CLAUDE.md`](CLAUDE.md) is the short orientation for anyone
(or anything) picking the repo up.

## References

- [Contrastive Reinforcement Learning](https://arxiv.org/pdf/2206.07568)
- [JaxGCRL](https://github.com/MichalBortkiewicz/JaxGCRL)
- [Why is CRL latent space "nice"](https://arxiv.org/pdf/2403.04082)
- [Scaling CRL](https://wang-kevin3290.github.io/scaling-crl/)
