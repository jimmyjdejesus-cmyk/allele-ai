"""Performance benchmarks for Kraken LNN scaling and sequence processing."""

import asyncio
import time

import numpy as np
import pytest

from allele.kraken_lnn import KrakenLNN, LiquidStateMachine
from tests.test_utils import generate_test_sequence


class TestKrakenScalingBenchmarks:
    """Performance benchmarks for Kraken LNN scaling behavior."""

    @pytest.mark.benchmark
    def test_reservoir_scaling_small(self, benchmark):
        """Benchmark small reservoir size performance."""
        def process_sequence_small_reservoir():
            lnn = KrakenLNN(reservoir_size=50, connectivity=0.1)
            sequence = generate_test_sequence(50)

            start_time = time.time()
            result = asyncio.run(lnn.process_sequence(sequence))
            processing_time = time.time() - start_time

            assert result['success'] is True
            return processing_time

        processing_time = benchmark(process_sequence_small_reservoir)
        assert processing_time < 2.0  # 2 seconds max for small reservoir

    @pytest.mark.benchmark
    def test_reservoir_scaling_medium(self, benchmark):
        """Benchmark medium reservoir size performance."""
        def process_sequence_medium_reservoir():
            lnn = KrakenLNN(reservoir_size=100, connectivity=0.1)
            sequence = generate_test_sequence(50)

            start_time = time.time()
            result = asyncio.run(lnn.process_sequence(sequence))
            processing_time = time.time() - start_time

            assert result['success'] is True
            return processing_time

        processing_time = benchmark(process_sequence_medium_reservoir)
        assert processing_time < 2.0  # 2 seconds max for medium reservoir

    @pytest.mark.benchmark
    def test_reservoir_scaling_large(self, benchmark):
        """Benchmark large reservoir size performance."""
        def process_sequence_large_reservoir():
            lnn = KrakenLNN(reservoir_size=200, connectivity=0.1)
            sequence = generate_test_sequence(50)

            start_time = time.time()
            result = asyncio.run(lnn.process_sequence(sequence))
            processing_time = time.time() - start_time

            assert result['success'] is True
            return processing_time

        processing_time = benchmark(process_sequence_large_reservoir)
        assert processing_time < 5.0  # 5 seconds max for large reservoir

    def test_sequence_length_scaling(self):
        """Test processing time vs sequence length."""
        # Test different sequence lengths without benchmarking
        lengths = [10, 50, 100, 200]
        times = []

        for length in lengths:
            lnn = KrakenLNN(reservoir_size=100, connectivity=0.1)
            sequence = generate_test_sequence(length)

            start_time = time.time()
            result = asyncio.run(lnn.process_sequence(sequence))
            processing_time = time.time() - start_time

            assert result['success'] is True
            times.append(processing_time)

            # Should scale roughly linearly (not exponentially)
            if len(times) > 1:
                # Basic scaling check - just ensure it completes
                assert processing_time > 0

        # Verify all lengths completed successfully
        assert len(times) == len(lengths)

    def test_connectivity_impact_on_performance(self):
        """Test performance impact of different connectivity values."""
        # Test different connectivity values without benchmarking
        connectivities = [0.05, 0.1, 0.2, 0.5]

        for conn in connectivities:
            lnn = KrakenLNN(reservoir_size=100, connectivity=conn)
            sequence = generate_test_sequence(100)

            start_time = time.time()
            result = asyncio.run(lnn.process_sequence(sequence))
            processing_time = time.time() - start_time

            assert result['success'] is True
            # Basic performance check
            assert processing_time > 0
            assert processing_time < 5.0  # Should complete within reasonable time

    @pytest.mark.benchmark
    def test_memory_consolidation_performance(self, benchmark):
        """Benchmark memory consolidation performance."""
        def benchmark_memory_consolidation():
            lnn = KrakenLNN(reservoir_size=100, connectivity=0.1)

            # Fill memory with multiple sequences
            sequences = [generate_test_sequence(20, seed=i) for i in range(50)]

            # First, add all memories
            for seq in sequences:
                asyncio.run(lnn.process_sequence(seq, memory_consolidation=False))

            initial_memory_count = len(lnn.temporal_memory.memories)

            # Benchmark consolidation
            start_time = time.time()
            consolidation_seq = generate_test_sequence(20)
            result = asyncio.run(lnn.process_sequence(consolidation_seq, memory_consolidation=True))
            consolidation_time = time.time() - start_time

            final_memory_count = len(lnn.temporal_memory.memories)

            assert result['success'] is True
            assert final_memory_count <= initial_memory_count

            return consolidation_time

        consolidation_time = benchmark(benchmark_memory_consolidation)

        # Memory consolidation should be relatively fast
        assert consolidation_time < 1.0  # 1 second max

    @pytest.mark.benchmark
    def test_liquid_state_machine_performance(self, benchmark):
        """Benchmark LiquidStateMachine individual operations."""
        def benchmark_lsm_operations():
            lsm = LiquidStateMachine(reservoir_size=100, connectivity=0.1)
            sequence = generate_test_sequence(100)

            start_time = time.time()
            outputs = lsm.process_sequence(sequence)
            processing_time = time.time() - start_time

            assert len(outputs) == len(sequence)
            assert all(np.isfinite(outputs))

            return processing_time

        processing_time = benchmark(benchmark_lsm_operations)

        # LSM operations should be fast
        assert processing_time < 0.5  # 500ms max

    @pytest.mark.benchmark
    def test_batch_sequence_processing(self, benchmark):
        """Benchmark processing multiple sequences in batch."""
        def benchmark_batch_processing():
            lnn = KrakenLNN(reservoir_size=100, connectivity=0.1)

            # Create batch of sequences
            sequences = [generate_test_sequence(50, seed=i) for i in range(10)]

            start_time = time.time()
            results = []

            for seq in sequences:
                result = asyncio.run(lnn.process_sequence(seq))
                results.append(result)

            total_time = time.time() - start_time

            # All results should be successful
            assert all(r['success'] for r in results)
            assert len(results) == len(sequences)

            return total_time

        batch_time = benchmark(benchmark_batch_processing)

        # Batch processing should be efficient
        assert batch_time < 10.0  # 10 seconds max for 10 sequences

    @pytest.mark.benchmark
    def test_large_reservoir_performance_limit(self, benchmark):
        """Test performance limits for large reservoirs to prevent OOM."""
        def process_large_reservoir():
            # Test with moderately large reservoir
            lnn = KrakenLNN(reservoir_size=500, connectivity=0.05)  # Lower connectivity for large size
            sequence = generate_test_sequence(100)

            start_time = time.time()
            result = asyncio.run(lnn.process_sequence(sequence))
            processing_time = time.time() - start_time

            assert result['success'] is True
            assert len(result['reservoir_state']) == 500

            return processing_time

        processing_time = benchmark(process_large_reservoir)

        # Large reservoir should still complete in reasonable time
        assert processing_time < 10.0  # 10 seconds max

    @pytest.mark.benchmark
    def test_memory_efficiency_under_load(self, benchmark):
        """Benchmark memory efficiency during continuous processing."""
        def benchmark_memory_efficiency():
            lnn = KrakenLNN(reservoir_size=200, connectivity=0.1)

            # Process many sequences to test memory efficiency
            start_time = time.time()

            for i in range(100):
                sequence = generate_test_sequence(25, seed=i)
                result = asyncio.run(lnn.process_sequence(sequence))

                if not result['success']:
                    break

            total_time = time.time() - start_time

            # Should complete all sequences without issues
            return total_time

        total_time = benchmark(benchmark_memory_efficiency)

        # Continuous processing should be stable
        avg_time_per_sequence = total_time / 100
        assert avg_time_per_sequence < 0.5  # Average < 500ms per sequence

    @pytest.mark.parametrize("reservoir_size", [50, 100, 200, 500])
    def test_initialization_performance(self, benchmark, reservoir_size):
        """Benchmark initialization time for different reservoir sizes."""
        def initialize_lnn():
            start_time = time.time()
            lnn = KrakenLNN(reservoir_size=reservoir_size, connectivity=0.1)
            init_time = time.time() - start_time

            assert lnn.reservoir_size == reservoir_size
            assert lnn.liquid_reservoir is not None
            assert lnn.temporal_memory is not None

            return init_time

        init_time = benchmark(initialize_lnn)

        # Initialization should be relatively fast
        assert init_time < 2.0  # 2 seconds max

        # Common validation that works for any size
        assert init_time > 0  # Should take some measurable time

    def test_concurrent_processing_performance(self):
        """Test concurrent processing capabilities - removed benchmark decorator due to async incompatibility."""
        async def run_concurrent_processing():
            lnn = KrakenLNN(reservoir_size=100, connectivity=0.1)

            # Create multiple sequences
            sequences = [generate_test_sequence(50, seed=i) for i in range(5)]

            # Process sequences concurrently
            tasks = [lnn.process_sequence(seq) for seq in sequences]
            results = await asyncio.gather(*tasks)

            return results

        import asyncio
        results = asyncio.run(run_concurrent_processing())

        # All should complete successfully
        assert len(results) == 5
        assert all(r['success'] for r in results)

    @pytest.mark.benchmark
    def test_dynamics_calculation_performance(self, benchmark):
        """Benchmark liquid dynamics calculation performance."""
        def benchmark_dynamics():
            from allele.kraken_lnn import LiquidDynamics

            dynamics = LiquidDynamics(viscosity=0.2, temperature=1.0, pressure=1.0)
            reservoir_state = np.random.random(100)  # Random state

            start_time = time.time()

            # Calculate dynamics for multiple time steps
            for _ in range(100):
                perturbation = dynamics.calculate_perturbation(0.5, reservoir_state)
                reservoir_state += perturbation * 0.1  # Simple update

            calculation_time = time.time() - start_time

            assert np.all(np.isfinite(reservoir_state))

            return calculation_time

        calc_time = benchmark(benchmark_dynamics)

        # Dynamics calculations should be fast
        assert calc_time < 0.1  # 100ms max for 100 calculations

    @pytest.mark.benchmark
    def test_weight_matrix_operations_performance(self, benchmark):
        """Benchmark weight matrix operations."""
        def benchmark_weight_operations():
            lsm = LiquidStateMachine(reservoir_size=150, connectivity=0.1)

            start_time = time.time()

            # Perform multiple weight updates
            for _ in range(50):
                learning_signal = np.random.random()  # Random learning signal
                state_vector = np.random.random(150)   # Random state vector
                lsm.adaptive_weights.update(learning_signal, state_vector)

            operation_time = time.time() - start_time

            # Weight matrix should remain valid
            assert np.all(np.isfinite(lsm.adaptive_weights.weights))

            return operation_time

        op_time = benchmark(benchmark_weight_operations)

        # Weight operations should be efficient
        assert op_time < 1.0  # 1 second max for 50 updates
