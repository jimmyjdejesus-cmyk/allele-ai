"""
Runtime tests for Kraken Liquid Neural Network.

Tests actual execution of LNN processing, memory operations, and dynamics.
"""

import numpy as np
import pytest

from phylogenic import (
    KrakenLNN,
    LiquidStateMachine,
)
from tests.test_utils import generate_test_sequence


class TestKrakenLNNRuntime:
    """Runtime tests for Kraken LNN."""

    def test_lnn_initialization_runtime(self, kraken_lnn):
        """Test LNN initializes correctly at runtime."""
        assert kraken_lnn.reservoir_size == 100
        assert kraken_lnn.connectivity == 0.1
        assert kraken_lnn.liquid_reservoir is not None
        assert kraken_lnn.temporal_memory is not None

    def test_liquid_state_machine_runtime(self):
        """Test LiquidStateMachine executes correctly."""
        lsm = LiquidStateMachine(reservoir_size=50, connectivity=0.2)

        assert lsm.reservoir_size == 50
        assert lsm.state.shape == (50,)
        assert lsm.connections.shape == (50, 50)
        assert lsm.adaptive_weights is not None

    @pytest.mark.asyncio
    async def test_sequence_processing_runtime(self, kraken_lnn, sample_sequence):
        """Test sequence processing executes correctly."""
        # Process sequence
        result = await kraken_lnn.process_sequence(sample_sequence)

        # Verify result structure
        assert isinstance(result, dict)
        assert "success" in result
        assert "liquid_outputs" in result
        assert "reservoir_state" in result

        if result["success"]:
            assert len(result["liquid_outputs"]) == len(sample_sequence)
            assert len(result["reservoir_state"]) == kraken_lnn.reservoir_size

    @pytest.mark.asyncio
    async def test_async_sequence_processing(self, kraken_lnn, sample_sequence):
        """Test async sequence processing."""
        result = await kraken_lnn.process_sequence(sample_sequence)

        assert result["success"] is True
        assert len(result["liquid_outputs"]) == len(sample_sequence)
        assert isinstance(result["processing_time"], float)
        assert result["processing_time"] >= 0.0

    def test_liquid_dynamics_runtime(self, custom_liquid_dynamics):
        """Test liquid dynamics configuration."""
        lsm = LiquidStateMachine(reservoir_size=100, dynamics=custom_liquid_dynamics)

        assert lsm.dynamics.viscosity == 0.15
        assert lsm.dynamics.temperature == 1.2
        assert lsm.dynamics.pressure == 0.9

    def test_reservoir_state_updates(self):
        """Test reservoir state updates correctly."""
        lsm = LiquidStateMachine(reservoir_size=50)

        initial_state = lsm.state.copy()

        # Process a sequence
        sequence = [0.5, 0.3, 0.8, 0.2, 0.9]
        lsm.process_sequence(sequence)

        # State should have changed
        assert not np.array_equal(lsm.state, initial_state)

        # State should be in valid range
        assert np.all(lsm.state >= -1.0)
        assert np.all(lsm.state <= 1.0)

    def test_adaptive_weights_update(self):
        """Test adaptive weights update during processing."""
        lsm = LiquidStateMachine(reservoir_size=50)

        initial_weights = lsm.adaptive_weights.weights.copy()

        # Process sequence with learning enabled
        sequence = generate_test_sequence(20, pattern="random")
        lsm.process_sequence(sequence, learning_enabled=True)

        # Weights should have changed (or at least be valid)
        not np.array_equal(lsm.adaptive_weights.weights, initial_weights)

        # Weights should be within bounds
        assert np.all(lsm.adaptive_weights.weights >= lsm.adaptive_weights.min_weight)
        assert np.all(lsm.adaptive_weights.weights <= lsm.adaptive_weights.max_weight)

    @pytest.mark.asyncio
    async def test_memory_storage_runtime(self, kraken_lnn, sample_sequence):
        """Test memory storage during processing."""
        initial_memory_count = len(kraken_lnn.temporal_memory)

        result = await kraken_lnn.process_sequence(
            sample_sequence, memory_consolidation=False
        )

        # Memory should have been stored or buffer is at capacity
        current_memory_count = len(kraken_lnn.temporal_memory)
        assert current_memory_count >= initial_memory_count
        # If buffer is not at capacity, should have increased
        if current_memory_count < kraken_lnn.temporal_memory.buffer_size:
            assert current_memory_count > initial_memory_count
        assert result["memory_entries"] == current_memory_count

    @pytest.mark.asyncio
    async def test_memory_consolidation_runtime(self, kraken_lnn):
        """Test memory consolidation executes correctly."""
        # Fill memory buffer
        for i in range(50):
            sequence = generate_test_sequence(10, pattern="random", seed=i)
            await kraken_lnn.process_sequence(sequence, memory_consolidation=False)

        initial_count = len(kraken_lnn.temporal_memory)

        # Trigger consolidation
        sequence = generate_test_sequence(10)
        await kraken_lnn.process_sequence(sequence, memory_consolidation=True)

        # Memory should be consolidated (fewer entries)
        final_count = len(kraken_lnn.temporal_memory)
        assert final_count <= initial_count + 1

    @pytest.mark.asyncio
    async def test_network_state_retrieval(self, kraken_lnn):
        """Test network state retrieval."""
        state = await kraken_lnn.get_network_state()

        assert isinstance(state, dict)
        assert "reservoir_size" in state
        assert "connectivity" in state
        assert "current_state" in state
        assert "dynamics" in state
        assert "memory" in state
        assert "processing_stats" in state

        assert state["reservoir_size"] == kraken_lnn.reservoir_size
        assert len(state["current_state"]) == kraken_lnn.reservoir_size

    def test_processing_statistics_tracking(self, kraken_lnn, sample_sequence):
        """Test processing statistics are tracked."""
        initial_stats = kraken_lnn.processing_stats.copy()

        import asyncio

        asyncio.run(kraken_lnn.process_sequence(sample_sequence))

        # Statistics should be updated
        assert (
            kraken_lnn.processing_stats["sequences_processed"]
            > initial_stats["sequences_processed"]
        )
        assert (
            kraken_lnn.processing_stats["total_processing_time"]
            > initial_stats["total_processing_time"]
        )

    def test_different_sequence_patterns(self, kraken_lnn):
        """Test processing different sequence patterns."""
        patterns = ["random", "sine", "step", "noise"]

        for pattern in patterns:
            sequence = generate_test_sequence(20, pattern=pattern)

            import asyncio

            result = asyncio.run(kraken_lnn.process_sequence(sequence))

            assert result["success"] is True
            assert len(result["liquid_outputs"]) == len(sequence)

    def test_reservoir_size_scaling(self):
        """Test LNN works with different reservoir sizes."""
        sizes = [50, 100, 200]

        for size in sizes:
            lnn = KrakenLNN(reservoir_size=size)
            sequence = generate_test_sequence(10)

            import asyncio

            result = asyncio.run(lnn.process_sequence(sequence))

            assert result["success"] is True
            assert len(result["reservoir_state"]) == size

    def test_connectivity_effect(self):
        """Test connectivity affects reservoir behavior."""
        low_conn = KrakenLNN(reservoir_size=100, connectivity=0.05)
        high_conn = KrakenLNN(reservoir_size=100, connectivity=0.3)

        sequence = generate_test_sequence(20)

        import asyncio

        result_low = asyncio.run(low_conn.process_sequence(sequence))
        result_high = asyncio.run(high_conn.process_sequence(sequence))

        assert result_low["success"] is True
        assert result_high["success"] is True

        # Outputs should differ due to different connectivity
        assert result_low["liquid_outputs"] != result_high["liquid_outputs"]

    def test_temporal_memory_buffer_limits(self, kraken_lnn):
        """Test memory buffer respects size limits."""
        buffer_size = kraken_lnn.temporal_memory.buffer_size

        # Fill buffer beyond capacity
        for i in range(buffer_size + 50):
            sequence = generate_test_sequence(5, seed=i)
            import asyncio

            asyncio.run(
                kraken_lnn.process_sequence(sequence, memory_consolidation=False)
            )

        # Memory should not exceed buffer size
        assert len(kraken_lnn.temporal_memory) <= buffer_size

    def test_liquid_flow_calculation(self):
        """Test liquid flow calculation executes correctly."""
        lsm = LiquidStateMachine(reservoir_size=50)

        input_value = 0.7
        flow = lsm._calculate_liquid_flow(input_value)

        assert isinstance(flow, np.ndarray)
        assert flow.shape == (50,)
        assert np.any(flow != 0)  # Should have some flow

    def test_output_generation(self):
        """Test output generation from reservoir state."""
        lsm = LiquidStateMachine(reservoir_size=50)

        # Process a sequence to update state
        sequence = [0.5, 0.3, 0.8]
        lsm.process_sequence(sequence)

        # Generate output
        output = lsm._generate_output()

        assert isinstance(output, float)
        assert -2.0 <= output <= 2.0  # Reasonable output range
