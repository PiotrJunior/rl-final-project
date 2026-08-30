"""Manifest tests.

A checkpoint without a manifest cannot be loaded - the pickle records no
architecture - so the manifest is load-bearing and gets tested like it.
"""

import json

import pytest

from latentmine.train import manifest
from latentmine.train.manifest import ManifestError
from latentmine.train.presets import make_run_spec


@pytest.fixture
def spec():
    return make_run_spec("two_rooms", "simple")


class TestBuild:
    def test_carries_everything_needed_to_rebuild_the_encoders(self, spec, tmp_path):
        arch = manifest.build(spec, tmp_path)["arch"]
        # Exactly the fields Encoder() needs; the pickle has none of them.
        assert set(arch) >= {"repr_dim", "h_dim", "n_hidden", "skip_connections", "use_relu", "use_ln"}
        assert (arch["n_hidden"], arch["h_dim"]) == (spec.arch.depth, spec.arch.width)

    def test_records_encoder_input_widths(self, spec, tmp_path):
        dims = manifest.build(spec, tmp_path)["dims"]
        assert dims["sa_encoder_input"] == dims["state_dim"] + dims["action_size"]
        assert dims["g_encoder_input"] == dims["goal_size"] == 2

    def test_embeds_the_maze_as_trained_on(self, spec, tmp_path):
        maze = manifest.build(spec, tmp_path)["maze"]
        assert maze["grid"] == list(spec.maze_spec.grid)
        assert maze["regions"] == list(spec.maze_spec.regions)
        assert maze["n_free_cells"] == len(spec.maze_spec.free_cells())

    def test_maze_without_regions_serialises_as_null(self, tmp_path):
        built = manifest.build(make_run_spec("spiral", "simple"), tmp_path)
        assert built["maze"]["regions"] is None

    def test_records_both_repository_shas(self, spec, tmp_path):
        prov = manifest.build(spec, tmp_path)["provenance"]
        assert set(prov) == {"latentmine_sha", "jaxgcrl_sha"}

    def test_is_json_serialisable(self, spec, tmp_path):
        json.dumps(manifest.build(spec, tmp_path))


class TestRoundTrip:
    def test_write_then_load(self, spec, tmp_path):
        path = manifest.write(spec, tmp_path / spec.run_id)
        assert path.name == manifest.MANIFEST_NAME
        assert manifest.load(path.parent)["run_id"] == spec.run_id

    def test_load_accepts_a_directory_or_a_file(self, spec, tmp_path):
        path = manifest.write(spec, tmp_path / spec.run_id)
        assert manifest.load(path) == manifest.load(path.parent)

    def test_spec_survives_the_round_trip(self, spec, tmp_path):
        manifest.write(spec, tmp_path / "r")
        assert manifest.spec_from_manifest(manifest.load(tmp_path / "r")) == spec


class TestRefusals:
    def test_missing_manifest_says_why_it_matters(self, tmp_path):
        with pytest.raises(ManifestError, match="records no architecture"):
            manifest.load(tmp_path)

    def test_unknown_schema_version_is_refused_not_guessed(self, spec, tmp_path):
        path = manifest.write(spec, tmp_path / "r")
        payload = json.loads(path.read_text())
        payload["schema_version"] = 999
        path.write_text(json.dumps(payload))
        with pytest.raises(ManifestError, match="Refusing to guess"):
            manifest.load(path)


class TestConfigHash:
    def test_is_stable_across_writes(self, spec, tmp_path):
        a = manifest.config_hash(manifest.build(spec, tmp_path / "a"))
        b = manifest.config_hash(manifest.build(spec, tmp_path / "b"))
        assert a == b, "run_dir and timestamp must not affect the hash"

    @pytest.mark.parametrize(
        "change", [{"seed": 2}, {"preset": "shallow"}, {"maze": "spiral"}, {"energy_fn": "dot"}]
    )
    def test_changes_when_resuming_would_be_invalid(self, spec, tmp_path, change):
        # These are the differences that must block a resume.
        other = spec.evolve(**change)
        assert manifest.config_hash(manifest.build(spec, tmp_path)) != manifest.config_hash(
            manifest.build(other, tmp_path)
        )

    def test_ignores_provenance(self, spec, tmp_path):
        built = manifest.build(spec, tmp_path)
        before = manifest.config_hash(built)
        built["provenance"]["latentmine_sha"] = "0" * 40
        assert manifest.config_hash(built) == before
