"""Contrastive RL latent mining.

Importing this package does **not** patch upstream. Registration touches
`jaxgcrl` module globals, so it is an explicit step rather than an import side
effect - call `latentmine.mazes.register.install()` before constructing a maze
env. `latentmine.mazes` on its own stays free of JAX, brax and mujoco, so the
maze set and its geometry can be inspected and tested without a training stack.
"""

__version__ = "0.1.0"
