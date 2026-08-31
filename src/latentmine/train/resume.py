"""Crash-resumable training checkpoints.

Upstream writes parameters only - no optimiser state, step count, replay
buffer, env state or RNG - so its checkpoints restore a model for analysis but
cannot continue a run (LLD section 5.5). This module stores everything the
epoch loop carries, so a killed run resumes where it stopped.

Two properties matter more than anything else here:

**Atomicity.** The failure that destroys a run is not a crash between writes,
it is a crash *during* one. Every write goes to a temporary file in the target
directory, is flushed and `fsync`ed, then `os.replace`d into place - atomic on
any POSIX filesystem. Two slots alternate, so a torn write can never damage
the previous good checkpoint, and the pointer file naming the current slot is
updated by the same dance.

**Format.** `flax.serialization` msgpack, not `pickle`. Pickling a
`flax.struct.dataclass` embeds the defining module path, so a submodule bump
that moves a class silently makes every checkpoint unloadable - the same trap
that makes upstream's `args.pkl` untrustworthy (LLD section 2.7). msgpack
stores arrays only, and the pytree structure comes from a live template built
by the training setup itself.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RESUME_DIRNAME = "resume"
POINTER_NAME = "latest.json"
SLOTS = ("a", "b")

# Bump when the payload layout changes.
FORMAT_VERSION = 1


class ResumeError(RuntimeError):
    pass


@dataclass
class ResumePoint:
    """Everything the epoch loop carries across an iteration."""

    training_state: Any  # params AND optimiser state AND env/gradient step counts
    env_state: Any  # the num_envs in-flight episodes
    buffer_state: Any  # replay data, insert/sample positions, buffer RNG
    key: Any  # the outer RNG
    next_epoch: int  # loop index to resume at
    training_walltime: float
    config_hash: str

    def to_payload(self) -> dict[str, Any]:
        import numpy as np

        return {
            "training_state": self.training_state,
            "env_state": self.env_state,
            "buffer_state": self.buffer_state,
            "key": self.key,
            "meta": {
                "next_epoch": np.asarray(self.next_epoch, dtype=np.int32),
                "training_walltime": np.asarray(self.training_walltime, dtype=np.float64),
            },
        }


def _atomic_write(path: Path, data: bytes) -> None:
    """Write `data` to `path` so a crash leaves either the old file or the new
    one, never a partial one."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)  # make the rename itself durable
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _slot_path(resume_dir: Path, slot: str) -> Path:
    return Path(resume_dir) / f"state_{slot}.msgpack"


def _pointer_path(resume_dir: Path) -> Path:
    return Path(resume_dir) / POINTER_NAME


def read_pointer(resume_dir: Path) -> dict[str, Any] | None:
    path = _pointer_path(resume_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def save(resume_dir: Path, point: ResumePoint) -> Path:
    """Write a resume checkpoint into the next slot and repoint at it."""
    from flax import serialization

    resume_dir = Path(resume_dir)
    resume_dir.mkdir(parents=True, exist_ok=True)

    pointer = read_pointer(resume_dir)
    used = pointer.get("slot") if pointer else None
    slot = SLOTS[0] if used != SLOTS[0] else SLOTS[1]

    blob = serialization.to_bytes(point.to_payload())
    _atomic_write(_slot_path(resume_dir, slot), blob)
    _atomic_write(
        _pointer_path(resume_dir),
        json.dumps(
            {
                "format_version": FORMAT_VERSION,
                "slot": slot,
                "sha256": hashlib.sha256(blob).hexdigest(),
                "next_epoch": int(point.next_epoch),
                "training_walltime": float(point.training_walltime),
                "config_hash": point.config_hash,
                "bytes": len(blob),
            },
            indent=2,
        ).encode(),
    )
    return _slot_path(resume_dir, slot)


def _load_slot(resume_dir: Path, slot: str, template: dict[str, Any], sha: str | None):
    from flax import serialization

    path = _slot_path(resume_dir, slot)
    if not path.exists():
        raise ResumeError(f"{path} does not exist")
    blob = path.read_bytes()
    if sha is not None and hashlib.sha256(blob).hexdigest() != sha:
        raise ResumeError(f"{path} failed its checksum")
    return serialization.from_bytes(template, blob)


def load(resume_dir: Path, template: dict[str, Any], config_hash: str | None = None) -> ResumePoint | None:
    """Restore the newest good checkpoint, or `None` if there is none.

    `template` is a pytree of the same structure as the payload, normally built
    by the training setup that is about to be resumed - msgpack carries arrays,
    not structure.

    A checksum failure on the current slot falls back to the other slot rather
    than aborting: that is what the rotation is for.
    """
    resume_dir = Path(resume_dir)
    pointer = read_pointer(resume_dir)
    if pointer is None:
        return None
    if pointer.get("format_version") != FORMAT_VERSION:
        raise ResumeError(
            f"{_pointer_path(resume_dir)}: format_version {pointer.get('format_version')!r}, "
            f"expected {FORMAT_VERSION}"
        )
    if config_hash is not None and pointer.get("config_hash") != config_hash:
        raise ResumeError(
            f"refusing to resume: the checkpoint in {resume_dir} was written for configuration "
            f"{pointer.get('config_hash')!r}, but this run is {config_hash!r}. Point --runs-dir "
            "elsewhere, or delete the resume directory to start over."
        )

    primary = pointer.get("slot", SLOTS[0])
    order = [(primary, pointer.get("sha256"))] + [(s, None) for s in SLOTS if s != primary]
    errors = []
    for slot, sha in order:
        try:
            restored = _load_slot(resume_dir, slot, template, sha)
        except (ResumeError, OSError, ValueError) as exc:
            errors.append(f"slot {slot}: {exc}")
            continue
        meta = restored["meta"]
        return ResumePoint(
            training_state=restored["training_state"],
            env_state=restored["env_state"],
            buffer_state=restored["buffer_state"],
            key=restored["key"],
            next_epoch=int(meta["next_epoch"]),
            training_walltime=float(meta["training_walltime"]),
            config_hash=pointer.get("config_hash", ""),
        )
    raise ResumeError("no readable resume checkpoint: " + "; ".join(errors))


def clear(resume_dir: Path) -> None:
    """Remove all resume state. Used when a run finishes normally."""
    for slot in SLOTS:
        _slot_path(resume_dir, slot).unlink(missing_ok=True)
    _pointer_path(resume_dir).unlink(missing_ok=True)
