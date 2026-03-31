"""Tests for profile-driven checkpoint cost model."""

import json
import tempfile
from pathlib import Path

import pytest

from vllm.v1.core.checkpoint_cost_model import CheckpointCostModel
from vllm.v1.core.checkpoint_controller import CheckpointController
from vllm.v1.core.sched.benders.cost_tables import CostTableBuilder


def _create_test_profile(is_real: bool) -> str:
    """Create a test profile JSON file.

    Args:
        is_real: If True, marks as real measurement; if False, as placeholder

    Returns:
        Path to temporary JSON file
    """
    profile = {
        "meta": {
            "model": "test-model",
            "dtype": "float16",
            "device": "cuda",
            "block_size_tokens": 16,
            "generated_at": "2026-03-30T00:00:00",
            "is_real_measurement": is_real,
            "notes": "REAL measurements" if is_real else "PLACEHOLDER DATA FOR TESTING",
        },
        "prefill_ms_by_tokens": {
            "16": 1.0,
            "32": 1.5,
            "64": 2.5,
            "128": 4.0,
            "256": 7.0,
            "512": 14.0,
        },
        "load_ms_by_bytes": {
            "1024": 0.1,
            "2048": 0.2,
            "4096": 0.35,
            "8192": 0.65,
            "16384": 1.2,
        },
        "checkpoint_ms_by_bytes": {
            "1024": 0.05,
            "2048": 0.1,
            "4096": 0.18,
            "8192": 0.35,
            "16384": 0.68,
        },
        "publication_overhead_ms": 1.0,
    }

    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(profile, f)
    f.close()
    return f.name


class TestCheckpointCostModel:
    """Test CheckpointCostModel loading and cost evaluation."""

    def test_load_real_profile(self):
        """Test loading a real measurement profile."""
        profile_path = _create_test_profile(is_real=True)
        try:
            model = CheckpointCostModel(profile_path)
            assert model is not None
            # Verify c0 is loaded
            assert model._c0 == 1.0
        finally:
            Path(profile_path).unlink()

    def test_reject_placeholder_profile(self):
        """Test that placeholder profiles are rejected with clear error."""
        profile_path = _create_test_profile(is_real=False)
        try:
            with pytest.raises(ValueError) as exc_info:
                CheckpointCostModel(profile_path)

            error_msg = str(exc_info.value)
            assert "placeholder" in error_msg.lower()
            assert "profile_checkpoint_costs.py" in error_msg
        finally:
            Path(profile_path).unlink()

    def test_interpolation_t_prefill(self):
        """Test T_prefill interpolation at measurement and interpolated points."""
        profile_path = _create_test_profile(is_real=True)
        try:
            model = CheckpointCostModel(profile_path)

            # Exact measurement points
            assert model.t_prefill(16) == 1.0
            assert model.t_prefill(32) == 1.5
            assert model.t_prefill(64) == 2.5

            # Interpolated points
            t_48 = model.t_prefill(48)
            assert 1.5 < t_48 < 2.5  # Should be between 32 and 64

            # Anchor: t_prefill(0) = 0 (added automatically)
            assert model.t_prefill(0) == 0.0

            # Interpolation between anchor (0, 0.0) and first data point (16, 1.0)
            t_8 = model.t_prefill(8)
            assert t_8 == 0.5  # Midpoint interpolation

            # Clamping at upper boundary
            t_1024 = model.t_prefill(1024)
            assert t_1024 == 14.0  # Clamp to last point
        finally:
            Path(profile_path).unlink()

    def test_interpolation_t_load(self):
        """Test T_load interpolation."""
        profile_path = _create_test_profile(is_real=True)
        try:
            model = CheckpointCostModel(profile_path)

            # Exact points
            assert model.t_load(1024) == 0.1
            assert model.t_load(4096) == 0.35

            # Interpolated
            t_2560 = model.t_load(2560)
            assert 0.2 < t_2560 < 0.35
        finally:
            Path(profile_path).unlink()

    def test_should_publish_decision(self):
        """Test the should_publish decision rule."""
        profile_path = _create_test_profile(is_real=True)
        try:
            model = CheckpointCostModel(profile_path)

            # Test a scenario where replay cost is high (many unpublished tokens)
            # and checkpoint cost is low (small delta_S)
            # This should publish
            should_publish = model.should_publish(
                published_tokens=64,
                unpublished_tokens=256,  # Large u = high replay cost
                published_bytes=1024,
                delta_bytes=512,  # Small delta_S = low checkpoint cost
                lambda_=1.0,
            )
            assert should_publish is True

            # Test a scenario where replay cost is low and checkpoint cost is high
            # This should NOT publish
            should_not_publish = model.should_publish(
                published_tokens=256,
                unpublished_tokens=16,  # Small u = low replay cost
                published_bytes=8192,
                delta_bytes=8192,  # Large delta_S = high checkpoint cost
                lambda_=1.0,
            )
            assert should_not_publish is False
        finally:
            Path(profile_path).unlink()


class TestCheckpointControllerWithProfile:
    """Test CheckpointController integration with profile-driven cost model."""

    def test_controller_with_real_profile(self):
        """Test that CheckpointController accepts real profile."""
        profile_path = _create_test_profile(is_real=True)
        try:
            controller = CheckpointController(
                block_size=16,
                cost_profile_path=profile_path,
                # Note: no linear model inputs (zero values)
                replay_throughput_tokens_per_sec=0.0,
                load_bandwidth_bytes_per_sec=0.0,
                checkpoint_bandwidth_bytes_per_sec=0.0,
            )

            # Verify profile was loaded
            assert controller._cost_model is not None
            # Verify economic policy is available (via profile, not linear inputs)
            assert controller._economic_policy_available is True
        finally:
            Path(profile_path).unlink()

    def test_controller_rejects_placeholder_profile(self):
        """Test that CheckpointController rejects placeholder profiles early."""
        profile_path = _create_test_profile(is_real=False)
        try:
            with pytest.raises(ValueError) as exc_info:
                CheckpointController(
                    block_size=16,
                    cost_profile_path=profile_path,
                )

            error_msg = str(exc_info.value)
            assert "placeholder" in error_msg.lower()
        finally:
            Path(profile_path).unlink()

    def test_controller_fallback_to_linear(self):
        """Test that CheckpointController falls back to linear model when no profile."""
        controller = CheckpointController(
            block_size=16,
            cost_profile_path="",  # Empty = no profile
            replay_throughput_tokens_per_sec=100.0,
            load_bandwidth_bytes_per_sec=1000.0,
            checkpoint_bandwidth_bytes_per_sec=1000.0,
        )

        # Verify profile was not loaded
        assert controller._cost_model is None
        # Verify economic policy is still available (via linear inputs)
        assert controller._economic_policy_available is True

    def test_controller_no_policy_without_profile_or_linear(self):
        """Test that economic policy is unavailable without profile or linear inputs."""
        controller = CheckpointController(
            block_size=16,
            cost_profile_path="",  # No profile
            # No linear model inputs (all zero/default)
            replay_throughput_tokens_per_sec=0.0,
            load_bandwidth_bytes_per_sec=0.0,
            checkpoint_bandwidth_bytes_per_sec=0.0,
        )

        assert controller._cost_model is None
        assert controller._economic_policy_available is False

    def test_controller_prefers_profile_over_linear(self):
        """Test that profile-driven takes precedence when both are available."""
        profile_path = _create_test_profile(is_real=True)
        try:
            controller = CheckpointController(
                block_size=16,
                cost_profile_path=profile_path,
                # Also provide linear inputs
                replay_throughput_tokens_per_sec=100.0,
                load_bandwidth_bytes_per_sec=1000.0,
                checkpoint_bandwidth_bytes_per_sec=1000.0,
            )

            # Both should be available
            assert controller._cost_model is not None
            assert controller._economic_policy_available is True

            # When _should_checkpoint_by_economic_policy is called,
            # it should use the profile model (checked in integration tests)
        finally:
            Path(profile_path).unlink()


class TestCostTableBuilderWithProfile:
    """Test CostTableBuilder integration with profile-driven cost model."""

    def _make_builder(self, cost_model=None):
        return CostTableBuilder(
            planning_horizon=1.0,
            prefill_throughput=1000.0,
            decode_throughput=500.0,
            load_bandwidth=1e9,
            checkpoint_bandwidth=1e9,
            replay_throughput=1000.0,
            detection_time_sec=0.1,
            block_size=16,
            kv_bytes_per_token=8192,
            checkpoint_lambda=1.0,
            cost_model=cost_model,
        )

    def test_profile_driven_replay_cost(self):
        """Test that profile-driven replay cost uses interpolation instead of linear."""
        profile_path = _create_test_profile(is_real=True)
        try:
            model = CheckpointCostModel(profile_path)

            builder_profile = self._make_builder(cost_model=model)
            builder_linear = self._make_builder(cost_model=None)

            # Active request with 128 computed tokens, 64 checkpointed
            costs_profile = builder_profile._compute_costs_raw(
                request_id="r1",
                prompt_len=128,
                generation_len=200,
                num_computed_tokens=128,
                num_output_tokens=0,
                num_checkpointed_tokens=64,
                checkpoint_size_bytes=64 * 8192,
                is_active=True,
                assigned_replica_id=0,
                ttft_slo_ms=None,
                tpot_slo_ms=None,
                failure_gap_slo_ms=None,
            )

            costs_linear = builder_linear._compute_costs_raw(
                request_id="r1",
                prompt_len=128,
                generation_len=200,
                num_computed_tokens=128,
                num_output_tokens=0,
                num_checkpointed_tokens=64,
                checkpoint_size_bytes=64 * 8192,
                is_active=True,
                assigned_replica_id=0,
                ttft_slo_ms=None,
                tpot_slo_ms=None,
                failure_gap_slo_ms=None,
            )

            # Both should have non-negative replay time
            assert costs_profile.replay_time_sec >= 0
            assert costs_linear.replay_time_sec >= 0

            # Profile replay cost should be based on T_prefill(128) - T_prefill(64)
            # = 4.0 - 2.5 = 1.5 ms = 0.0015 sec
            expected_replay_ms = model.t_prefill(128) - model.t_prefill(64)
            assert abs(costs_profile.replay_time_sec - expected_replay_ms / 1000.0) < 1e-6

            # Linear replay cost = 64 / 1000.0 = 0.064 sec
            assert abs(costs_linear.replay_time_sec - 64 / 1000.0) < 1e-6

            # They should differ (profile is non-linear)
            assert costs_profile.replay_time_sec != costs_linear.replay_time_sec
        finally:
            Path(profile_path).unlink()

    def test_profile_driven_load_cost(self):
        """Test that profile-driven load cost uses interpolation."""
        profile_path = _create_test_profile(is_real=True)
        try:
            model = CheckpointCostModel(profile_path)
            builder = self._make_builder(cost_model=model)

            # Active request with known checkpoint size
            costs = builder._compute_costs_raw(
                request_id="r1",
                prompt_len=128,
                generation_len=200,
                num_computed_tokens=128,
                num_output_tokens=0,
                num_checkpointed_tokens=64,
                checkpoint_size_bytes=4096,
                is_active=True,
                assigned_replica_id=0,
                ttft_slo_ms=None,
                tpot_slo_ms=None,
                failure_gap_slo_ms=None,
            )

            # Profile: T_load(4096) = 0.35 ms = 0.00035 sec
            expected = model.t_load(4096) / 1000.0
            assert abs(costs.restore_time_sec - expected) < 1e-6
        finally:
            Path(profile_path).unlink()

    def test_no_profile_fallback_unchanged(self):
        """Test that without profile, costs are computed exactly as before."""
        builder = self._make_builder(cost_model=None)

        costs = builder._compute_costs_raw(
            request_id="r1",
            prompt_len=100,
            generation_len=200,
            num_computed_tokens=0,
            num_output_tokens=0,
            num_checkpointed_tokens=0,
            checkpoint_size_bytes=0,
            is_active=False,
            assigned_replica_id=None,
            ttft_slo_ms=None,
            tpot_slo_ms=None,
            failure_gap_slo_ms=None,
        )

        # Pending request: replay = prompt_len / replay_throughput = 100 / 1000 = 0.1 sec
        assert abs(costs.replay_time_sec - 0.1) < 1e-6
        # No checkpoint → restore_time = 0
        assert costs.restore_time_sec == 0.0

    def test_pending_request_with_profile(self):
        """Test that pending requests use T_prefill(prompt_len) for replay."""
        profile_path = _create_test_profile(is_real=True)
        try:
            model = CheckpointCostModel(profile_path)
            builder = self._make_builder(cost_model=model)

            costs = builder._compute_costs_raw(
                request_id="r1",
                prompt_len=128,
                generation_len=200,
                num_computed_tokens=0,
                num_output_tokens=0,
                num_checkpointed_tokens=0,
                checkpoint_size_bytes=0,
                is_active=False,
                assigned_replica_id=None,
                ttft_slo_ms=None,
                tpot_slo_ms=None,
                failure_gap_slo_ms=None,
            )

            # Pending: published_tokens=0, recovery_tokens=128
            # replay = T_prefill(128) - T_prefill(0) = 4.0 - 0.0 = 4.0 ms
            # (T_prefill(0) = 0.0 thanks to the (0, 0.0) anchor)
            expected = model.t_prefill(128) / 1000.0  # 4.0 ms → 0.004 sec
            assert abs(costs.replay_time_sec - expected) < 1e-6
        finally:
            Path(profile_path).unlink()
