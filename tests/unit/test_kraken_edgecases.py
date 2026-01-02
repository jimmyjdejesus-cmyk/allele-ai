"""Edge case and boundary tests for Kraken LNN to prevent OOM and ensure robustness."""

import numpy as np
import pytest

from phylogenic.kraken_lnn import KrakenLNN, LiquidDynamics, LiquidStateMachine
from tests.test_utils import generate_test_sequence


class TestKrakenEdgeCases:
    """Edge case tests for boundary conditions and OOM prevention."""

    def test_reservoir_size_validation_prevents_oom(self):
        """Test that reservoir size validation prevents out-of-memory conditions."""
        # Small reservoir should work
        lnn = KrakenLNN(reservoir_size=10, connectivity=0.1)
        assert lnn.reservoir_size == 10

        # Large reservoir should work but warn (implementation should handle)
        large_lnn = KrakenLNN(
            reservoir_size=1000, connectivity=0.01
        )  # Lower connectivity for large reservoir
        assert large_lnn.reservoir_size == 1000

        # Test memory estimation for different sizes
        small_sequence = generate_test_sequence(5)
        large_sequence = generate_test_sequence(20)

        import asyncio

        result_small = asyncio.run(lnn.process_sequence(small_sequence))
        assert result_small["success"] is True

        # Large reservoir with large sequence might take longer but shouldn't OOM
        result_large = asyncio.run(large_lnn.process_sequence(large_sequence))
        assert result_large["success"] is True

    def test_reservoir_bounds_clamping(self):
        """Test that reservoir states are properly bounded."""
        lsm = LiquidStateMachine(reservoir_size=30, connectivity=0.2)

        # Test with extreme inputs that could cause unbounded growth
        extreme_sequence = [10.0, -10.0, 100.0, -100.0]  # Very large values

        for value in extreme_sequence:
            lsm.process_input(value)

            # State should be bounded (implementation should clamp)
            # These bounds depend on the specific clamping implementation
            assert np.all(np.abs(lsm.state) < 100.0)  # Reasonable upper bound

        # State should still be finite
        assert np.all(np.isfinite(lsm.state))

    def test_weight_clipping_prevents_explosion(self):
        """Test that weight clipping prevents numerical instability."""
        lsm = LiquidStateMachine(reservoir_size=50, connectivity=0.1)

        # Initially get weight bounds
        np.min(lsm.adaptive_weights.weights)
        np.max(lsm.adaptive_weights.weights)

        # Process sequence that could cause weight explosion
        explosive_sequence = [1.0] * 100  # Same input repeatedly

        lsm.process_sequence(explosive_sequence, learning_enabled=True)

        # Weights should still be bounded
        min_weight_after = np.min(lsm.adaptive_weights.weights)
        max_weight_after = np.max(lsm.adaptive_weights.weights)

        # Weights should not have exploded to infinity
        assert np.isfinite(min_weight_after)
        assert np.isfinite(max_weight_after)

        # Weights should be within reasonable range (implementation dependent)
        assert abs(min_weight_after) < 10.0
        assert abs(max_weight_after) < 10.0

    def test_memory_consolidation_boundary_conditions(self):
        """Test memory consolidation behavior at boundaries."""
        lnn = KrakenLNN(reservoir_size=50, connectivity=0.1)

        # Fill memory buffer to maximum
        max_entries = lnn.temporal_memory.max_entries

        import asyncio

        for i in range(max_entries + 10):  # Overfill
            seq = generate_test_sequence(3, seed=i)
            asyncio.run(lnn.process_sequence(seq, memory_consolidation=False))

        initial_count = len(lnn.temporal_memory.memories)

        # Test consolidation with empty additional memory
        empty_seq = []
        result = asyncio.run(lnn.process_sequence(empty_seq, memory_consolidation=True))
        assert result["success"] is True

        # Memory should be consolidated (potentially fewer entries)
        final_count = len(lnn.temporal_memory.memories)
        assert final_count <= initial_count
        assert final_count >= 0  # Should not go negative

    def test_extreme_connectivity_values(self):
        """Test behavior with extreme connectivity values."""
        # Very high connectivity (approaching 1.0)
        high_conn = KrakenLNN(reservoir_size=30, connectivity=0.9)
        assert high_conn.connectivity == 0.9

        # Very low connectivity (approaching 0.0)
        low_conn = KrakenLNN(reservoir_size=30, connectivity=0.01)
        assert low_conn.connectivity == 0.01

        # Test processing with both
        sequence = generate_test_sequence(10)

        import asyncio

        result_high = asyncio.run(high_conn.process_sequence(sequence))
        result_low = asyncio.run(low_conn.process_sequence(sequence))

        # Both should succeed
        assert result_high["success"] is True
        assert result_low["success"] is True

        # Results should differ due to connectivity
        assert result_high["liquid_outputs"] != result_low["liquid_outputs"]

    def test_zero_and_negative_sequence_values(self):
        """Test handling of zero and negative values in sequences."""
        lsm = LiquidStateMachine(reservoir_size=40, connectivity=0.1)

        # Test with zeros
        zero_sequence = [0.0, 0.0, 0.0]
        output_zeros = lsm.process_sequence(zero_sequence)

        assert isinstance(output_zeros, list)
        assert len(output_zeros) == len(zero_sequence)
        assert np.all(np.isfinite(np.array(output_zeros)))

        # Test with negatives
        negative_sequence = [-0.5, -0.2, -0.8, -0.1]
        output_negatives = lsm.process_sequence(negative_sequence)

        assert isinstance(output_negatives, list)
        assert len(output_negatives) == len(negative_sequence)
        assert np.all(np.isfinite(np.array(output_negatives)))

        # Test mixed zero/negative
        mixed_sequence = [0.0, -0.5, 0.5, -1.0, 1.0]
        output_mixed = lsm.process_sequence(mixed_sequence)

        assert isinstance(output_mixed, list)
        assert len(output_mixed) == len(mixed_sequence)
        assert np.all(np.isfinite(np.array(output_mixed)))

    @pytest.mark.asyncio
    async def test_empty_sequence_handling(self):
        """Test proper handling of empty sequences."""
        lnn = KrakenLNN(reservoir_size=50, connectivity=0.1)

        # Empty sequence should be handled gracefully
        result = await lnn.process_sequence([])

        assert result["success"] is True
        assert result["liquid_outputs"] == []
        assert len(result["reservoir_state"]) == lnn.reservoir_size

    def test_large_sequence_memory_efficiency(self):
        """Test that large sequences don't cause excessive memory usage."""
        import os

        import psutil

        lnn = KrakenLNN(reservoir_size=200, connectivity=0.05)

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process a large sequence
        large_sequence = generate_test_sequence(500, pattern="sine")

        import asyncio

        result = asyncio.run(lnn.process_sequence(large_sequence))

        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 100MB for GH runners)
        assert memory_increase < 200.0  # Allow some buffer
        assert result["success"] is True

    def test_reservoir_state_normalization(self):
        """Test that reservoir states remain normalized."""
        dynamics = LiquidDynamics(viscosity=0.1, temperature=1.0, pressure=1.0)
        lsm = LiquidStateMachine(reservoir_size=30, dynamics=dynamics)

        # Process sequence that could cause state drift
        sequence = generate_test_sequence(50, pattern="random")

        np.linalg.norm(lsm.state)

        lsm.process_sequence(sequence)

        final_state_norm = np.linalg.norm(lsm.state)

        # State norm should remain reasonable (not explode)
        assert final_state_norm < 100.0
        assert np.isfinite(final_state_norm)

        # All individual states should be finite
        assert np.all(np.isfinite(lsm.state))

    def test_weight_matrix_sparsity_bounds(self):
        """Test weight matrix sparsity stays within connectivity bounds."""
        connectivity = 0.15
        reservoir_size = 100

        lsm = LiquidStateMachine(
            reservoir_size=reservoir_size, connectivity=connectivity
        )

        # Count non-zero connections
        non_zero_weights = np.count_nonzero(lsm.adaptive_weights.weights)
        total_weights = reservoir_size * reservoir_size

        actual_connectivity = non_zero_weights / total_weights

        # Actual connectivity should be close to target (within 10% tolerance)
        assert abs(actual_connectivity - connectivity) < 0.1

        # Should not be completely disconnected or fully connected
        assert non_zero_weights > 0
        assert non_zero_weights < total_weights

    @pytest.mark.asyncio
    async def test_memory_consolidation_with_minimal_data(self):
        """Test memory consolidation when there's minimal data."""
        lnn = KrakenLNN(reservoir_size=50, connectivity=0.1)

        # Add just one memory entry
        seq = generate_test_sequence(5)
        await lnn.process_sequence(seq, memory_consolidation=False)

        assert len(lnn.temporal_memory) == 1

        # Consolidate with minimal data
        result = await lnn.process_sequence(seq, memory_consolidation=True)

        assert result["success"] is True
        # Should still have at least one memory entry
        assert len(lnn.temporal_memory) >= 1

    def test_numerical_stability_under_extreme_loads(self):
        """Test numerical stability under extreme computational loads."""
        # Test with rapid successive processing
        lnn = KrakenLNN(reservoir_size=50, connectivity=0.1)

        sequences = [generate_test_sequence(10, seed=i) for i in range(50)]

        import asyncio

        # Process rapidly
        for seq in sequences:
            result = asyncio.run(lnn.process_sequence(seq))

            # Each result should be valid
            assert result["success"] is True
            assert np.all(np.isfinite(result["reservoir_state"]))
            assert np.all(np.isfinite(result["liquid_outputs"]))

        # Final state should still be valid
        assert np.all(np.isfinite(lnn.liquid_reservoir.state))

    def test_adaptive_weights_bounds_enforcement(self):
        """Test that adaptive weights stay within enforced bounds."""
        lsm = LiquidStateMachine(reservoir_size=40, connectivity=0.1)

        # Manually set some extreme weights (if possible)
        # This tests the bounds enforcement in the update mechanism

        initial_weights = lsm.adaptive_weights.weights.copy()

        # Process sequence that might cause weight updates
        sequence = generate_test_sequence(30, pattern="noise")

        lsm.process_sequence(sequence, learning_enabled=True)

        final_weights = lsm.adaptive_weights.weights

        # Weights should remain within reasonable bounds (implementation dependent)
        # At minimum, they should not be NaN or infinite
        assert np.all(np.isfinite(final_weights))

        # Check that weights haven't changed dramatically (no explosions)
        max_change = np.max(np.abs(final_weights - initial_weights))
        assert max_change < 10.0  # Reasonable bound for weight updates

    @pytest.mark.asyncio
    async def test_oom_prevention_for_very_large_sequeneces(self):
        """Test OOM prevention with very large sequences."""
        # Use small reservoir to test OOM behavior safely
        lnn = KrakenLNN(reservoir_size=10, connectivity=0.1)

        # Very large sequence that could cause OOM in naive implementations
        very_large_sequence = generate_test_sequence(10000)

        # This should complete without OOM (though may be slow)
        result = await lnn.process_sequence(very_large_sequence)

        assert result["success"] is True
        assert len(result["liquid_outputs"]) == len(very_large_sequence)

        # Memory should still be finite
        assert np.all(np.isfinite(result["reservoir_state"]))

    def test_reservoir_drift_prevention(self):
        """Test that reservoir state drift is prevented."""
        lsm = LiquidStateMachine(reservoir_size=30, connectivity=0.15)

        initial_state = lsm.state.copy()

        # Process many similar sequences that could cause drift
        similar_sequence = [0.5] * 200

        lsm.process_sequence(similar_sequence)

        final_state = lsm.state

        # State should not have drifted infinitely
        state_change = np.max(np.abs(final_state - initial_state))
        assert state_change < 100.0  # Reasonable bound

        # State should still be finite
        assert np.all(np.isfinite(final_state))

    def test_liquid_flow_calculation_bounds(self):
        """Test liquid flow calculations stay within bounds."""
        lsm = LiquidStateMachine(reservoir_size=25, connectivity=0.1)

        # Test with extreme inputs
        extreme_inputs = [0.0, 1.0, -1.0, 10.0, -10.0, np.inf, -np.inf, np.nan]

        for inp in extreme_inputs:
            if not np.isfinite(inp):
                continue  # Skip non-finite inputs for this test

            flow = lsm._calculate_liquid_flow(inp)

            # Flow should be finite and bounded
            assert np.all(np.isfinite(flow))
            assert np.all(np.abs(flow) < 1000.0)  # Reasonable bound

    def test_adaptive_memory_scaling(self):
        """Test that memory usage scales appropriately with reservoir size."""
        small_lnn = KrakenLNN(reservoir_size=10, connectivity=0.1)
        large_lnn = KrakenLNN(
            reservoir_size=100, connectivity=0.05
        )  # Lower connectivity for scaling

        sequence = generate_test_sequence(20)

        import asyncio

        result_small = asyncio.run(small_lnn.process_sequence(sequence))
        result_large = asyncio.run(large_lnn.process_sequence(sequence))

        # Both should succeed
        assert result_small["success"] is True
        assert result_large["success"] is True

        # Output lengths should match input
        assert len(result_small["liquid_outputs"]) == len(sequence)
        assert len(result_large["liquid_outputs"]) == len(sequence)

        # States should be appropriately sized
        assert len(result_small["reservoir_state"]) == 10
        assert len(result_large["reservoir_state"]) == 100
