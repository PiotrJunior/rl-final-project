"""Control-evaluation tests.

Covers the protocol - stratified pairs, bootstrap intervals - without needing
a trained policy. The rollout itself is exercised under `slow`.
"""

import numpy as np

from latentmine.control import waypoint_policy as wp
from latentmine.mazes import geometry as geo
from latentmine.mazes import layouts as L


class TestPairSampling:
    def test_pairs_are_reproducible(self):
        spec = L.get("two_rooms")
        assert wp.sample_pairs(spec, 20, seed=3) == wp.sample_pairs(spec, 20, seed=3)

    def test_pairs_are_far_enough_apart_to_be_interesting(self):
        spec = L.get("two_rooms")
        index = {c: k for k, c in enumerate(spec.free_cells())}
        for start, goal in wp.sample_pairs(spec, 40, seed=0, min_cells=3):
            assert geo.geodesic_from(spec, start)[index[goal]] >= 3 * spec.scaling

    def test_endpoints_are_free_cells(self):
        spec = L.get("four_rooms")
        for start, goal in wp.sample_pairs(spec, 30, seed=1):
            assert spec.is_free(*start) and spec.is_free(*goal)
            assert start != goal


class TestStratification:
    def test_pairs_are_split_into_three_bands(self):
        spec = L.get("four_rooms")
        pairs = wp.sample_pairs(spec, 90, seed=0)
        strata = wp.stratify(spec, pairs)
        assert set(strata) == {"near", "medium", "far"}

    def test_far_pairs_really_are_further(self):
        spec = L.get("four_rooms")
        pairs = wp.sample_pairs(spec, 90, seed=0)
        strata = wp.stratify(spec, pairs)
        index = {c: k for k, c in enumerate(spec.free_cells())}
        distances = np.array([geo.geodesic_from(spec, s)[index[g]] for s, g in pairs])
        assert distances[strata == "far"].mean() > distances[strata == "near"].mean()

    def test_bands_are_roughly_balanced(self):
        spec = L.get("two_rooms")
        pairs = wp.sample_pairs(spec, 120, seed=2)
        counts = np.unique(wp.stratify(spec, pairs), return_counts=True)[1]
        assert counts.max() / counts.min() < 2.0


class TestResultSummaries:
    def _result(self, name, success, steps, strata):
        return wp.ControlResult(
            name=name,
            success=np.array(success, dtype=float),
            steps=np.array(steps, dtype=float),
            strata=np.array(strata),
        )

    def test_success_rate_overall_and_by_stratum(self):
        result = self._result("x", [1, 1, 0, 0], [10, 20, 30, 40], ["near", "near", "far", "far"])
        assert result.success_rate() == 0.5
        assert result.success_rate("near") == 1.0
        assert result.success_rate("far") == 0.0

    def test_median_steps_counts_only_successes_by_default(self):
        result = self._result("x", [1, 0, 1], [10, 999, 20], ["near"] * 3)
        assert result.median_steps("near") == 15.0
        assert result.median_steps("near", successful_only=False) == 20.0

    def test_bootstrap_interval_brackets_the_estimate(self):
        rng = np.random.default_rng(0)
        success = (rng.random(200) < 0.6).astype(float)
        result = self._result("x", success, np.full(200, 10.0), np.array(["near"] * 200))
        low, high = result.bootstrap_ci()
        assert low < result.success_rate() < high
        assert high - low < 0.25

    def test_an_empty_stratum_is_nan_not_zero(self):
        result = self._result("x", [1], [5], ["near"])
        assert np.isnan(result.success_rate("far"))

    def test_compare_renders_every_method(self):
        results = [
            self._result("direct", [1, 0], [10, 20], ["near", "far"]),
            self._result("waypoints", [1, 1], [12, 30], ["near", "far"]),
        ]
        table = wp.compare(results)
        assert "direct" in table and "waypoints" in table
        assert "near" in table and "far" in table
