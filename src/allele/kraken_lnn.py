"""
Kraken Liquid Neural Network (LNN) Implementation for Allele.

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

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import numpy as np
import asyncio

from .exceptions import AbeNLPError


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


@dataclass
class TemporalMemoryBuffer:
    """Temporal memory buffer for sequence processing.

    Attributes:
        buffer_size: Maximum buffer size
        memory_decay: Rate of memory decay
        consolidation_threshold: Threshold for memory consolidation
        retrieval_strength: Strength of memory retrieval
        memories: List of stored memories
    """
    buffer_size: int = 1000
    memory_decay: float = 0.95
    consolidation_threshold: float = 0.8
    retrieval_strength: float = 0.7
    memories: List[Dict[str, Any]] = field(default_factory=list)


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
        dynamics: Optional[LiquidDynamics] = None
    ):
        """Initialize liquid state machine.

        Args:
            reservoir_size: Size of the liquid reservoir
            connectivity: Connection density in the reservoir
            dynamics: Liquid dynamics configuration
        """
        self.reservoir_size = reservoir_size
        self.connectivity = connectivity
        self.dynamics = dynamics or LiquidDynamics()

        # Initialize reservoir state
        self.state = np.zeros(reservoir_size)
        self.activation_history = []

        # Initialize connection matrix
        self.connections = self._initialize_connections()

        # Initialize adaptive weights
        self.adaptive_weights = AdaptiveWeightMatrix(
            weights=np.random.randn(reservoir_size, reservoir_size) * 0.1
        )

    def _initialize_connections(self) -> np.ndarray:
        """Initialize connection matrix with specified connectivity."""
        connections = np.random.random((self.reservoir_size, self.reservoir_size))
        connections = (connections < self.connectivity).astype(float)

        # Remove self-connections
        np.fill_diagonal(connections, 0)

        return connections

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

    def _update_liquid_state(self, input_value: float) -> None:
        """Update liquid reservoir state with dynamics."""
        # Calculate liquid flow
        flow = self._calculate_liquid_flow(input_value)

        # Apply viscosity and turbulence
        viscous_flow = flow * self.dynamics.viscosity
        turbulent_flow = viscous_flow + np.random.normal(
            0, self.dynamics.turbulence, self.reservoir_size
        )

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

    def _calculate_liquid_flow(self, input_value: float) -> np.ndarray:
        """Calculate liquid flow through the reservoir."""
        # Input injection
        input_injection = np.zeros(self.reservoir_size)
        input_injection[:min(10, self.reservoir_size)] = input_value

        # Recurrent connections with adaptive weights
        recurrent_flow = np.dot(
            self.adaptive_weights.weights * self.connections,
            self.state
        )

        # Combine flows
        total_flow = input_injection + recurrent_flow

        return total_flow

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
        return self.state.copy()


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
        dynamics: Optional[LiquidDynamics] = None
    ):
        """Initialize Kraken LNN.

        Args:
            reservoir_size: Size of the liquid reservoir
            connectivity: Connection density
            memory_buffer_size: Size of temporal memory buffer
            dynamics: Liquid dynamics configuration
        """
        self.reservoir_size = reservoir_size
        self.connectivity = connectivity
        self.dynamics = dynamics or LiquidDynamics()

        # Initialize liquid state machine
        self.liquid_reservoir = LiquidStateMachine(
            reservoir_size=reservoir_size,
            connectivity=connectivity,
            dynamics=dynamics
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
                "memory_entries": len(self.temporal_memory.memories),
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
                "current_memories": len(self.temporal_memory.memories),
                "memory_utilization": (
                    len(self.temporal_memory.memories) /
                    self.temporal_memory.buffer_size
                )
            },
            "processing_stats": self.processing_stats
        }

    async def _store_memory(self, memory_entry: Dict[str, Any]) -> None:
        """Store memory entry in temporal buffer."""
        self.temporal_memory.memories.append(memory_entry)

        # Remove old memories if buffer is full
        if len(self.temporal_memory.memories) > self.temporal_memory.buffer_size:
            self.temporal_memory.memories.pop(0)

    async def _consolidate_memories(self) -> None:
        """Consolidate memories based on importance and recency."""
        if len(self.temporal_memory.memories) < 10:
            return

        # Calculate memory importance scores
        importance_scores = []
        for memory in self.temporal_memory.memories:
            # Importance based on sequence length and recency
            recency = (
                datetime.now(timezone.utc) - memory["timestamp"]
            ).total_seconds()
            importance = memory["sequence_length"] / (1 + recency / 3600)  # Hours
            importance_scores.append(importance)

        # Keep top memories
        sorted_indices = np.argsort(importance_scores)[::-1]
        keep_count = int(
            len(self.temporal_memory.memories) *
            self.temporal_memory.consolidation_threshold
        )

        self.temporal_memory.memories = [
            self.temporal_memory.memories[i]
            for i in sorted_indices[:keep_count]
        ]

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
            len(self.temporal_memory.memories) / self.temporal_memory.buffer_size
        )

