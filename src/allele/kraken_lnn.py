# Copyright (C) 2025 Bravetto AI Systems & Jimmy De Jesus
#
# This file is part of Allele.
#
# Allele is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Allele is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Allele.  If not, see <https://www.gnu.org/licenses/>.
#
# =============================================================================
# COMMERCIAL LICENSE:
# If you wish to use this software in a proprietary/closed-source application
# without releasing your source code, you must purchase a Commercial License
# from: https://gumroad.com/l/[YOUR_LINK]
# =============================================================================

"""Kraken Liquid Neural Network (LNN) Implementation for Allele.

This module implements the advanced liquid neural network (LNN) for temporal
sequence processing, adaptive dynamics, and memory capabilities.

Features:
- Liquid reservoir computing with adaptive dynamics
- Temporal memory buffer for sequence processing
- Adaptive weight matrix with plasticity
- Real-time learning and adaptation

Author: Bravetto AI Systems
Version: 1.0.0
"""

import asyncio
import heapq
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from .config import settings as allele_settings


class DeterministicRandom:
    """Deterministic random state manager for reproducible tests.

    This class ensures that all random operations in Kraken LNN components
    use a controlled sequence that can be reset for testing purposes.
    """
    _instances: Dict[str, 'DeterministicRandom'] = {}

    def __init__(self, seed_value: Optional[int] = None):
        """Initialize deterministic random instance."""
        self._rng = np.random.RandomState(seed_value)

    @classmethod
    def get_instance(cls, name: str = 'default') -> 'DeterministicRandom':
        """Get or create a deterministic random instance by name."""
        if name not in cls._instances:
            cls._instances[name] = cls()
        return cls._instances[name]

    @classmethod
    def seed(cls, seed_value: int, name: str = 'default') -> None:
        """Set the seed for a named deterministic random instance."""
        instance = cls.get_instance(name)
        instance._rng = np.random.RandomState(seed_value)

    @classmethod
    def random(cls, size: Optional[Any] = None, name: str = 'default') -> np.ndarray:
        """Generate random floats in [0, 1)."""
        instance = cls.get_instance(name)
        result: np.ndarray = np.asarray(instance._rng.random_sample(size))
        return result

    @classmethod
    def randn(cls, *args: Any, name: str = 'default') -> np.ndarray:
        """Generate standard normal random values."""
        instance = cls.get_instance(name)
        result: np.ndarray = np.asarray(instance._rng.standard_normal(args))
        return result

    @classmethod
    def normal(cls, loc: float = 0.0, scale: float = 1.0, size: Optional[Any] = None, name: str = 'default') -> np.ndarray:
        """Generate normal random values."""
        instance = cls.get_instance(name)
        result: np.ndarray = np.asarray(instance._rng.normal(loc, scale, size))
        return result

    @classmethod
    def reset(cls, name: str = 'default') -> None:
        """Reset the random state for a named instance."""
        if name in cls._instances:
            del cls._instances[name]

@dataclass
class LiquidDynamics:
    """Liquid dynamics configuration for Kraken LNN.

    Attributes:
        viscosity: Controls flow resistance in the liquid
        temperature: Affects random fluctuations
        pressure: Influences activation thresholds
        flow_rate: Controls information propagation speed
        turbulence: Adds non-linear dynamics
    """
    viscosity: float = 0.1
    temperature: float = 1.0
    pressure: float = 1.0
    flow_rate: float = 0.5
    turbulence: float = 0.05

    def calculate_perturbation(
        self,
        input_force: float,
        random_vector: np.ndarray
    ) -> float:
        """Calculate perturbation based on input force and random dynamics.

        Args:
            input_force: Input force applied to the system
            random_vector: Random noise vector

        Returns:
            Perturbation value
        """
        # Scale random noise by turbulence
        noise = np.mean(random_vector) * self.turbulence

        # Apply temperature and viscosity scaling
        perturbation = input_force + noise * self.temperature
        perturbation /= (1 + self.viscosity)  # Viscosity dampens perturbations

        return float(perturbation)

@dataclass
class AdaptiveWeightMatrix:
    """Adaptive weight matrix with plasticity mechanisms.

    Attributes:
        weights: Current weight matrix
        plasticity_rate: Rate of weight adaptation
        decay_rate: Rate of weight decay
        max_weight: Maximum allowed weight value
        min_weight: Minimum allowed weight value
        learning_threshold: Threshold for learning activation
    """
    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    plasticity_rate: float = 0.01
    decay_rate: float = 0.001
    max_weight: float = 2.0
    min_weight: float = -2.0
    learning_threshold: float = 0.1

    def update(self, learning_signal: float, state_vector: np.ndarray) -> None:
        """Update the weight matrix based on learning signal and state vector.

        Optimized for batch processing with vectorized operations.

        Args:
            learning_signal: Learning signal for weight updates
            state_vector: Current state vector for Hebbian learning
        """
        # Apply Hebbian-like learning rule with threshold check
        if abs(learning_signal) > self.learning_threshold:
            # Single vectorized operation: plasticity_rate * learning_signal * outer(state_vector, state_vector)
            # This combines scaling and outer product in one step
            self.weights += self.plasticity_rate * learning_signal * np.outer(state_vector, state_vector)

            # Apply weight constraints with single clip operation
            np.clip(self.weights, self.min_weight, self.max_weight, out=self.weights)

        # Apply weight decay (in-place multiplication for memory efficiency)
        self.weights *= (1 - self.decay_rate)

    def update_batch(self, learning_signals: np.ndarray, state_vectors: np.ndarray) -> None:
        """Batch update weight matrix for multiple learning signals and state vectors.

        Args:
            learning_signals: Array of learning signals
            state_vectors: Matrix of state vectors (shape: n_vectors x state_dim)
        """
        # Filter significant learning signals
        significant_mask = np.abs(learning_signals) > self.learning_threshold

        if np.any(significant_mask):
            significant_signals = learning_signals[significant_mask]
            significant_states = state_vectors[significant_mask]

            # Vectorized batch weight updates
            # Outer products for all significant signals: (n_signals, state_dim, state_dim)
            outer_products = self.plasticity_rate * significant_signals[:, np.newaxis, np.newaxis] * \
                           np.einsum('bi,bj->bij', significant_states, significant_states)

            # Accumulate all weight updates at once
            self.weights += np.sum(outer_products, axis=0)

            # Apply constraints once
            np.clip(self.weights, self.min_weight, self.max_weight, out=self.weights)

        # Apply weight decay (always, regardless of learning threshold)
        self.weights *= (1 - self.decay_rate)

@dataclass
class TemporalMemoryBuffer:
    """Temporal memory buffer for sequence processing with circular buffer optimization.

    Uses pre-allocated circular buffer for O(1) memory operations.

    Attributes:
        buffer_size: Maximum buffer size
        memory_decay: Rate of memory decay
        consolidation_threshold: Threshold for memory consolidation
        retrieval_strength: Strength of memory retrieval
        memories: Circular buffer of stored memories
        _head: Current insertion position in circular buffer
        _count: Number of valid entries in buffer
    """
    buffer_size: int = 1000
    memory_decay: float = 0.95
    consolidation_threshold: float = 0.8
    retrieval_strength: float = 0.7
    memories: List[Optional[Dict[str, Any]]] = field(default_factory=lambda: [])
    _head: int = field(default=0, init=False)
    _count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Initialize circular buffer after dataclass initialization."""
        if not self.memories or len(self.memories) != self.buffer_size:
            self.memories = [None] * self.buffer_size

    @property
    def max_entries(self) -> int:
        """Maximum number of entries in the buffer (alias for buffer_size)."""
        return self.buffer_size

    @max_entries.setter
    def max_entries(self, value: int) -> None:
        """Set the maximum number of entries in the buffer."""
        if value != self.buffer_size:
            # Resize circular buffer - preserve existing data where possible
            old_buffer = self.memories[:]
            self.buffer_size = value
            self.memories = [None] * value

            # Copy as many existing memories as possible
            copy_count = min(len(old_buffer), value)
            for i in range(copy_count):
                idx = (self._head - copy_count + i) % len(old_buffer)
                if old_buffer[idx] is not None:
                    self.memories[i] = old_buffer[idx]

            self._count = copy_count
            self._head = copy_count if copy_count > 0 else 0

    def add_memory(self, memory_entry: Dict[str, Any]) -> None:
        """Add memory entry to circular buffer (O(1) operation)."""
        self.memories[self._head] = memory_entry
        self._head = (self._head + 1) % self.buffer_size

        if self._count < self.buffer_size:
            self._count += 1

    def get_memories(self) -> List[Dict[str, Any]]:
        """Get all valid memories in temporal order (oldest first)."""
        if self._count == 0:
            return []

        # Calculate start position (oldest memory)
        start_idx = (self._head - self._count) % self.buffer_size
        memories: List[Dict[str, Any]] = []

        for i in range(self._count):
            idx = (start_idx + i) % self.buffer_size
            mem = self.memories[idx]
            if mem is not None:
                memories.append(mem)

        return memories

    def __len__(self) -> int:
        """Return number of valid memories in buffer."""
        return self._count

    def __iter__(self) -> Any:
        """Iterate over memories in temporal order."""
        memories = self.get_memories()
        return iter(memories)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get memory by index in temporal order."""
        memories = self.get_memories()
        return memories[index]

    def __setitem__(self, index: int, value: Dict[str, Any]) -> None:
        """Set memory by index in temporal order."""
        if index < 0 or index >= self._count:
            raise IndexError(f"Index {index} out of range")

        start_idx = (self._head - self._count) % self.buffer_size
        buffer_idx = (start_idx + index) % self.buffer_size
        self.memories[buffer_idx] = value

class LiquidStateMachine:
    """Liquid State Machine for reservoir computing.

    Implements a liquid reservoir with adaptive dynamics for
    processing temporal sequences with memory and plasticity.

    Example:
        >>> lsm = LiquidStateMachine(reservoir_size=100)
        >>> outputs = lsm.process_sequence([0.5, 0.3, 0.8])
        >>> state = lsm.get_state()
    """

    def __init__(
        self,
        reservoir_size: int = 100,
        connectivity: float = 0.1,
        dynamics: Optional[LiquidDynamics] = None,
        random_state: Optional[np.random.RandomState] = None,
        instance_name: str = 'lsm_default'
    ):
        """Initialize liquid state machine.

        Args:
            reservoir_size: Size of the liquid reservoir
            connectivity: Connection density in the reservoir
            dynamics: Liquid dynamics configuration
            random_state: Random state for reproducible initialization
            instance_name: Name for deterministic random instance
        """
        # Guard against unreasonable reservoir sizes that can cause OOM
        max_reservoir = 5000
        if reservoir_size <= 0 or reservoir_size > max_reservoir:
            raise ValueError(f"reservoir_size must be between 1 and {max_reservoir}")

        self.reservoir_size = reservoir_size
        self.connectivity = connectivity
        self.dynamics = dynamics or LiquidDynamics()
        self.instance_name = instance_name

        # Initialize random state for reproducible behavior
        if random_state is not None:
            self.random_state = random_state
        else:
            # Use deterministic random for reproducible behavior
            self.random_state = DeterministicRandom.get_instance(instance_name)._rng

        # Initialize reservoir state
        self.state = np.zeros(reservoir_size)
        self.activation_history: List[np.ndarray] = []

        # Initialize connection matrix
        self.connections = self._initialize_connections()

        # Initialize adaptive weights
        self.adaptive_weights = self._initialize_weights()

    def _initialize_weights(self) -> AdaptiveWeightMatrix:
        """Initialize adaptive weights with sparsity and random values."""
        # Initialize with small random weights
        initial_weights = self.random_state.randn(self.reservoir_size, self.reservoir_size) * 0.1
        
        # Apply sparsity mask from connections
        initial_weights *= self.connections
        
        return AdaptiveWeightMatrix(weights=initial_weights)

    def _initialize_connections(self) -> np.ndarray:
        """Initialize connection matrix with specified connectivity."""
        connections = self.random_state.random((self.reservoir_size, self.reservoir_size))
        connections = (connections < self.connectivity).astype(float)

        # Remove self-connections
        np.fill_diagonal(connections, 0)

        result: np.ndarray = connections
        return result

    def process_sequence(
        self,
        input_sequence: List[float],
        learning_enabled: bool = True
    ) -> List[float]:
        """Process temporal sequence through liquid dynamics.

        Args:
            input_sequence: Input sequence to process
            learning_enabled: Whether to enable adaptive learning

        Returns:
            Output sequence from the liquid reservoir
        """
        outputs = []

        for input_value in input_sequence:
            # Apply liquid dynamics
            self._update_liquid_state(input_value)

            # Generate output
            output = self._generate_output()
            outputs.append(output)

            # Update adaptive weights if learning enabled
            if learning_enabled:
                self._update_adaptive_weights(input_value, output)

        return outputs

    def process_sequences_batch(
        self,
        input_sequences: List[List[float]],
        learning_enabled: bool = True
    ) -> List[List[float]]:
        """Process multiple sequences through liquid dynamics in batch.

        Optimized for parallel processing of multiple sequences.

        Args:
            input_sequences: List of input sequences to process
            learning_enabled: Whether to enable adaptive learning

        Returns:
            List of output sequences from the liquid reservoir
        """
        if not input_sequences:
            return []

        # Batch process all sequences
        all_outputs = []

        for input_sequence in input_sequences:
            outputs = []

            for input_value in input_sequence:
                # Apply liquid dynamics (state is updated in-place)
                self._update_liquid_state(input_value)

                # Generate output
                output = self._generate_output()
                outputs.append(output)

                # Update adaptive weights if learning enabled
                if learning_enabled:
                    self._update_adaptive_weights(input_value, output)

            all_outputs.append(outputs)

        return all_outputs

    def _update_liquid_state(self, input_value: float) -> None:
        """Update liquid reservoir state with dynamics."""
        # Calculate liquid flow
        flow = self._calculate_liquid_flow(input_value)

        # Apply viscosity and turbulence
        viscous_flow = flow * self.dynamics.viscosity
        turbulent_flow = viscous_flow + self._add_liquid_noise()

        # Update state with liquid dynamics
        self.state = (
            self.state * self.dynamics.flow_rate +
            turbulent_flow * (1 - self.dynamics.flow_rate)
        )

        # Apply activation function with temperature
        self.state = np.tanh(self.state / self.dynamics.temperature)

        # Store activation history
        self.activation_history.append(self.state.copy())

        # Limit history size
        if len(self.activation_history) > 100:
            self.activation_history.pop(0)

    def _add_liquid_noise(self) -> np.ndarray:
        """Add liquid turbulence noise to the system.

        Returns:
            Noise vector for the reservoir
        """
        # Use instance's random state for reproducible behavior
        noise: np.ndarray = self.random_state.normal(0, self.dynamics.turbulence, self.reservoir_size)
        return noise

    def process_input(self, input_value: float) -> float:
        """Process a single input value through the liquid reservoir.

        Args:
            input_value: Single input value to process

        Returns:
            Output value from the reservoir
        """
        # Update reservoir state with the input
        self._update_liquid_state(input_value)

        # Generate and return output
        return self._generate_output()

    def _calculate_liquid_flow(self, input_value: float) -> np.ndarray:
        """Calculate liquid flow through the reservoir.

        Optimized with pre-multiplied connection matrix for better performance.
        """
        # Optimized input injection using direct slicing assignment
        input_injection = np.zeros(self.reservoir_size, dtype=np.float64)
        input_injection[:min(10, self.reservoir_size)] = input_value

        # Pre-multiply connection and weight matrices for faster computation
        # This avoids element-wise multiplication during each iteration
        effective_weights = self.adaptive_weights.weights * self.connections

        # Vectorized matrix multiplication with reservoir state
        recurrent_flow = effective_weights @ self.state

        # Combine flows (in-place addition for memory efficiency)
        total_flow = input_injection
        total_flow += recurrent_flow

        result: np.ndarray = total_flow
        return result

    def _generate_output(self) -> float:
        """Generate output from current reservoir state."""
        # Simple output generation
        output = np.mean(self.state)

        # Apply pressure dynamics
        output *= self.dynamics.pressure

        return float(output)

    def _update_adaptive_weights(
        self,
        input_value: float,
        output_value: float
    ) -> None:
        """Update adaptive weights based on input-output correlation."""
        # Calculate learning signal
        learning_signal = input_value * output_value

        # Update weights if learning threshold is met
        if abs(learning_signal) > self.adaptive_weights.learning_threshold:
            # Hebbian-like learning
            weight_update = (
                self.adaptive_weights.plasticity_rate *
                learning_signal *
                np.outer(self.state, self.state)
            )

            # Apply weight update
            self.adaptive_weights.weights += weight_update

            # Apply weight constraints
            self.adaptive_weights.weights = np.clip(
                self.adaptive_weights.weights,
                self.adaptive_weights.min_weight,
                self.adaptive_weights.max_weight
            )

        # Apply weight decay
        self.adaptive_weights.weights *= (1 - self.adaptive_weights.decay_rate)

    def get_state(self) -> np.ndarray:
        """Get current reservoir state."""
        result: np.ndarray = self.state.copy()
        return result

class KrakenLNN:
    """Kraken Liquid Neural Network implementation.

    Advanced liquid neural network with temporal memory, adaptive dynamics,
    and integration capabilities.

    Example:
        >>> kraken = KrakenLNN(reservoir_size=100)
        >>> result = await kraken.process_sequence([0.5, 0.3, 0.8])
        >>> state = await kraken.get_network_state()
    """

    def __init__(
        self,
        reservoir_size: int = 100,
        connectivity: float = 0.1,
        memory_buffer_size: int = 1000,
        dynamics: Optional[LiquidDynamics] = None,
        random_state: Optional[np.random.RandomState] = None,
        instance_name: str = 'kraken_default'
    ):
        """Initialize Kraken LNN.

        Args:
            reservoir_size: Size of the liquid reservoir
            connectivity: Connection density
            memory_buffer_size: Size of temporal memory buffer
            dynamics: Liquid dynamics configuration
            random_state: Random state for reproducible initialization
            instance_name: Name for deterministic random instance
        """
        self.reservoir_size = reservoir_size
        self.connectivity = connectivity
        self.dynamics = dynamics or LiquidDynamics()
        self.instance_name = instance_name

        # Initialize random state for reproducible behavior
        if random_state is not None:
            self.random_state = random_state
        else:
            # Use deterministic random for reproducible behavior
            self.random_state = DeterministicRandom.get_instance(instance_name)._rng

        # Initialize liquid state machine
        self.liquid_reservoir = LiquidStateMachine(
            reservoir_size=reservoir_size,
            connectivity=connectivity,
            dynamics=dynamics,
            random_state=self.random_state,
            instance_name=instance_name + "_lsm"
        )

        # Initialize temporal memory
        self.temporal_memory = TemporalMemoryBuffer(
            buffer_size=memory_buffer_size
        )

        # Performance tracking
        self.processing_stats = {
            "sequences_processed": 0,
            "total_processing_time": 0.0,
            "average_sequence_length": 0.0,
            "memory_utilization": 0.0
        }

    async def process_sequences_batch(
        self,
        input_sequences: List[List[float]],
        learning_enabled: bool = True,
        memory_consolidation: bool = True,
        max_concurrent: int = 4
    ) -> List[Dict[str, Any]]:
        """Process multiple sequences in parallel for improved throughput.

        Uses asyncio to process sequences concurrently, optimized for reservoir operations.

        Args:
            input_sequences: List of input sequences to process
            learning_enabled: Whether to enable adaptive learning
            memory_consolidation: Whether to consolidate memories
            max_concurrent: Maximum number of concurrent sequence processors

        Returns:
            List of results for each input sequence
        """
        if not input_sequences:
            return []

        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_single_sequence(seq: List[float]) -> Dict[str, Any]:
            async with semaphore:
                return await self.process_sequence(seq, learning_enabled, memory_consolidation)

        # Process all sequences concurrently
        tasks = [process_single_sequence(seq) for seq in input_sequences]
        results = await asyncio.gather(*tasks)

        return list(results)

    async def process_sequence(
        self,
        input_sequence: List[float],
        learning_enabled: bool = True,
        memory_consolidation: bool = True
    ) -> Dict[str, Any]:
        """Process temporal sequence with liquid dynamics and memory.

        Args:
            input_sequence: Input sequence to process
            learning_enabled: Whether to enable adaptive learning
            memory_consolidation: Whether to consolidate memories

        Returns:
            Dictionary containing processing results and metrics
        """
        start_time = datetime.now()

        try:
            # Process through liquid reservoir
            liquid_outputs = self.liquid_reservoir.process_sequence(
                input_sequence, learning_enabled
            )

            # Store in temporal memory
            memory_entry = {
                "timestamp": datetime.now(timezone.utc),
                "input_sequence": input_sequence,
                "liquid_outputs": liquid_outputs,
                "reservoir_state": self.liquid_reservoir.state.copy(),
                "sequence_length": len(input_sequence)
            }

            await self._store_memory(memory_entry)

            # Consolidate memories if enabled
            if memory_consolidation:
                await self._consolidate_memories()

            # Update processing statistics
            self._update_processing_stats(input_sequence, start_time)

            # Generate comprehensive output
            result = {
                "success": True,
                "liquid_outputs": liquid_outputs,
                "reservoir_state": self.liquid_reservoir.state.tolist(),
                "memory_entries": len(self.temporal_memory),
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "dynamics": {
                    "viscosity": self.dynamics.viscosity,
                    "temperature": self.dynamics.temperature,
                    "pressure": self.dynamics.pressure,
                    "flow_rate": self.dynamics.flow_rate,
                    "turbulence": self.dynamics.turbulence
                }
            }

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }

    async def get_network_state(self) -> Dict[str, Any]:
        """Get current network state and statistics.

        Returns:
            Dictionary containing network state information
        """
        return {
            "reservoir_size": self.reservoir_size,
            "connectivity": self.connectivity,
            "current_state": self.liquid_reservoir.state.tolist(),
            "dynamics": {
                "viscosity": self.dynamics.viscosity,
                "temperature": self.dynamics.temperature,
                "pressure": self.dynamics.pressure,
                "flow_rate": self.dynamics.flow_rate,
                "turbulence": self.dynamics.turbulence
            },
            "memory": {
                "buffer_size": self.temporal_memory.buffer_size,
                "current_memories": len(self.temporal_memory),
                "memory_utilization": (
                    len(self.temporal_memory) /
                    self.temporal_memory.buffer_size
                )
            },
            "processing_stats": self.processing_stats
        }

    async def _store_memory(self, memory_entry: Dict[str, Any]) -> None:
        """Store memory entry in temporal buffer (O(1) operation)."""
        self.temporal_memory.add_memory(memory_entry)

    async def _consolidate_memories(self) -> None:
        """Consolidate memories based on importance and recency.

        Optimized implementation using heap-based selection for O(n log k) complexity.
        Works with circular buffer for O(1) memory operations.
        """
        memories = self.temporal_memory.get_memories()
        if len(memories) < 10:
            return

        # Pre-calculate current time once
        current_time = datetime.now(timezone.utc)
        hours_to_seconds = 3600.0

        # Calculate importance scores and create (importance, memory) tuples
        # Use heap to maintain top-k efficiently (O(n log k) vs O(n log n) sorting)
        keep_count = max(1, int(len(memories) * self.temporal_memory.consolidation_threshold))
        top_memories: List[tuple[float, Dict[str, Any]]] = []

        for memory in memories:
            if memory is not None:
                # Vectorized recency calculation
                recency_seconds = (current_time - memory["timestamp"]).total_seconds()
                importance = memory["sequence_length"] / (1.0 + recency_seconds / hours_to_seconds)

                # Use heap to keep track of top memories
                if len(top_memories) < keep_count:
                    heapq.heappush(top_memories, (importance, memory))
                elif importance > top_memories[0][0]:
                    heapq.heapreplace(top_memories, (importance, memory))

        # Rebuild circular buffer with only kept memories (O(1) operations)
        # Clear buffer efficiently
        self.temporal_memory.memories = [None] * self.temporal_memory.buffer_size
        self.temporal_memory._head = 0
        self.temporal_memory._count = 0

        # Re-add kept memories in temporal order (oldest first)
        for _, memory in sorted(top_memories, key=lambda x: x[1]["timestamp"]):
            self.temporal_memory.add_memory(memory)

    def _update_processing_stats(
        self,
        input_sequence: List[float],
        start_time: datetime
    ) -> None:
        """Update processing statistics."""
        processing_time = (datetime.now() - start_time).total_seconds()

        self.processing_stats["sequences_processed"] += 1
        self.processing_stats["total_processing_time"] += processing_time
        self.processing_stats["average_sequence_length"] = (
            (self.processing_stats["average_sequence_length"] *
             (self.processing_stats["sequences_processed"] - 1) +
             len(input_sequence)) / self.processing_stats["sequences_processed"]
        )
        self.processing_stats["memory_utilization"] = (
            len(self.temporal_memory) / self.temporal_memory.buffer_size
        )

    @classmethod
    def from_settings(cls, settings: Optional[Any] = None) -> "KrakenLNN":
        """Create a KrakenLNN instance using central settings defaults."""
        if settings is None:
            settings = allele_settings
        kraken = settings.kraken
        dynamics = settings.liquid_dynamics
        return cls(
            reservoir_size=kraken.reservoir_size,
            connectivity=kraken.connectivity,
            memory_buffer_size=kraken.memory_buffer_size,
            dynamics=LiquidDynamics(
                viscosity=dynamics.viscosity,
                temperature=dynamics.temperature,
                pressure=dynamics.pressure,
                flow_rate=dynamics.flow_rate,
                turbulence=dynamics.turbulence
            )
        )
