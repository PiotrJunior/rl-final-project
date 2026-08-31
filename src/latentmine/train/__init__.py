"""Training: configuration, env construction, manifests and the CRL entrypoint.

`presets` and `manifest` import no JAX, so configuration can be resolved,
validated and reported without a training stack. `envs` and the `train`
function in `run_crl` are where JAX starts.
"""

from .presets import (
    ARCH_PRESETS,
    BUDGET_PROFILES,
    ENV_SPECS,
    ArchPreset,
    ConfigError,
    EnvSpec,
    RunSpec,
    make_run_spec,
)

__all__ = [
    "ARCH_PRESETS",
    "BUDGET_PROFILES",
    "ENV_SPECS",
    "ArchPreset",
    "ConfigError",
    "EnvSpec",
    "RunSpec",
    "make_run_spec",
]
