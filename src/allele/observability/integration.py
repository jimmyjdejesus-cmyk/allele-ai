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

"""Observability integration for Allele components.

This module provides observability integration for existing Allele components
including EvolutionEngine, KrakenLNN, and NLPAgent.

Author: Bravetto AI Systems
Version: 1.0.0
"""

from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
import asyncio
import uuid
import psutil
import gc

from ..evolution import EvolutionEngine, EvolutionConfig
from ..kraken_lnn import KrakenLNN
from ..agent import NLPAgent, AgentConfig
from ..genome import ConversationalGenome
from .types import ComponentType, MetricType
from .collector import ComponentMetricsCollector
from .engine import get_observability_engine

import structlog
logger = structlog.get_logger(__name__)


class ObservableEvolutionEngine(EvolutionEngine):
    """Evolution engine with integrated observability and metrics collection."""
    
    def __init__(self, config: EvolutionConfig, component_id: Optional[str] = None):
        """Initialize observable evolution engine.
        
        Args:
            config: Evolution configuration
            component_id: Component ID for observability (auto-generated if None)
        """
        super().__init__(config)
        
        # Generate component ID if not provided
        self.component_id = component_id or f"evolution_engine_{uuid.uuid4().hex[:12]}"
        
        # Initialize observability
        self.observability_engine = get_observability_engine()
        self.metrics_collector = self.observability_engine.register_evolution_engine(self.component_id)
        
        # Set correlation ID for tracing
        correlation_id = f"evolution_{uuid.uuid4().hex[:12]}"
        self.metrics_collector.set_correlation_id(correlation_id)
        
        # Evolution-specific metrics
        self._generation_times: List[float] = []
        self._fitness_history: List[float] = []
        self._diversity_history: List[float] = []
        self._population_sizes: List[int] = []
        
        # Resource tracking
        self._start_memory = 0
        self._peak_memory = 0
        self._cpu_samples: List[float] = []
        
        logger.info(f"Observable evolution engine initialized: {self.component_id}",
                   config=config.__dict__)
    
    async def evolve(
        self,
        population: List[ConversationalGenome],
        fitness_function: Callable[[ConversationalGenome], float],
        generations: Optional[int] = None
    ) -> ConversationalGenome:
        """Evolve population with comprehensive observability."""
        
        # Record start metrics
        process = psutil.Process()
        self._start_memory = process.memory_info().rss / (1024 * 1024)  # MB
        self._peak_memory = self._start_memory
        
        correlation_id = self.metrics_collector.performance_metrics_correlation_id
        logger.info(f"Starting evolution: {len(population)} genomes, {generations or self.config.generations} generations",
                   correlation_id=correlation_id)
        
        try:
            # Record initial population metrics
            await self._record_population_metrics(population, generation=0, is_initial=True)
            
            # Run evolution with observability
            best_genome = await super().evolve(population, fitness_function, generations)
            
            # Record final evolution summary
            await self._record_evolution_summary(best_genome)
            
            return best_genome
            
        except Exception as e:
            # Record failure metrics
            self.metrics_collector.record_success(False, "evolution_run")
            logger.error(f"Evolution failed: {e}", correlation_id=correlation_id, exc_info=True)
            raise
        finally:
            # Cleanup and final metrics
            await self._cleanup_evolution_metrics()
    
    async def _record_population_metrics(self, 
                                       population: List[ConversationalGenome], 
                                       generation: int,
                                       is_initial: bool = False) -> None:
        """Record comprehensive population metrics."""
        
        # Calculate population statistics
        fitness_scores = [g.fitness_score for g in population if hasattr(g, 'fitness_score')]
        best_fitness = max(fitness_scores) if fitness_scores else 0.0
        avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0.0
        min_fitness = min(fitness_scores) if fitness_scores else 0.0
        
        # Calculate diversity
        diversity = self._calculate_diversity(population)
        
        # Resource usage
        process = psutil.Process()
        current_memory = process.memory_info().rss / (1024 * 1024)  # MB
        cpu_percent = process.cpu_percent()
        
        # Update peak memory
        if current_memory > self._peak_memory:
            self._peak_memory = current_memory
        
        # Record metrics
        if not is_initial:
            # Generation timing
            generation_time = time.time() - getattr(self, '_generation_start_time', time.time())
            self._generation_times.append(generation_time)
            self.metrics_collector.record_latency(generation_time * 1000, "generation")
        
        # Population metrics
        self.metrics_collector.record_custom_metric(
            name="generation_number",
            value=generation,
            unit="count"
        )
        
        self.metrics_collector.record_custom_metric(
            name="best_fitness",
            value=best_fitness,
            unit="score"
        )
        
        self.metrics_collector.record_custom_metric(
            name="average_fitness",
            value=avg_fitness,
            unit="score"
        )
        
        self.metrics_collector.record_custom_metric(
            name="fitness_range",
            value=best_fitness - min_fitness,
            unit="score"
        )
        
        self.metrics_collector.record_custom_metric(
            name="population_diversity",
            value=diversity,
            unit="score"
        )
        
        self.metrics_collector.record_custom_metric(
            name="population_size",
            value=len(population),
            unit="count"
        )
        
        # Resource metrics
        self.metrics_collector.record_resource_usage(
            memory_mb=current_memory,
            cpu_percent=cpu_percent
        )
        
        # Evolution-specific metrics
        if generation > 0:
            # Fitness improvement
            if len(self._fitness_history) > 0:
                previous_best = self._fitness_history[-1]
                improvement = best_fitness - previous_best
                self.metrics_collector.record_custom_metric(
                    name="fitness_improvement",
                    value=improvement,
                    unit="score"
                )
            
            # Convergence rate (if fitness plateaus)
            if len(self._fitness_history) >= 5:
                recent_improvements = []
                for i in range(max(0, len(self._fitness_history) - 5), len(self._fitness_history)):
                    if i > 0:
                        improvement = self._fitness_history[i] - self._fitness_history[i-1]
                        recent_improvements.append(improvement)
                
                if recent_improvements:
                    convergence_rate = sum(recent_improvements) / len(recent_improvements)
                    self.metrics_collector.record_custom_metric(
                        name="convergence_rate",
                        value=convergence_rate,
                        unit="score_per_generation"
                    )
        
        # Update histories
        self._fitness_history.append(best_fitness)
        self._diversity_history.append(diversity)
        self._population_sizes.append(len(population))
        self._cpu_samples.append(cpu_percent)
        
        # Set generation start time for next iteration
        self._generation_start_time = time.time()
        
        # Heartbeat
        self.metrics_collector.heartbeat()
        
        # Log progress every 10 generations or at the end
        if generation % 10 == 0 or is_initial:
            logger.info(f"Generation {generation}: best={best_fitness:.4f}, avg={avg_fitness:.4f}, diversity={diversity:.4f}",
                       correlation_id=self.metrics_collector.performance_metrics_correlation_id)
    
    async def _record_evolution_summary(self, best_genome: ConversationalGenome) -> None:
        """Record comprehensive evolution summary metrics."""
        
        # Overall success
        self.metrics_collector.record_success(True, "evolution_run")
        
        # Calculate final statistics
        total_generations = len(self._generation_times)
        total_time = sum(self._generation_times) if self._generation_times else 0
        avg_generation_time = total_time / total_generations if total_generations > 0 else 0
        
        # Performance summary metrics
        self.metrics_collector.record_custom_metric(
            name="total_generations",
            value=total_generations,
            unit="count"
        )
        
        self.metrics_collector.record_custom_metric(
            name="total_evolution_time",
            value=total_time,
            unit="seconds"
        )
        
        self.metrics_collector.record_custom_metric(
            name="average_generation_time",
            value=avg_generation_time,
            unit="seconds"
        )
        
        self.metrics_collector.record_custom_metric(
            name="peak_memory_usage",
            value=self._peak_memory,
            unit="MB"
        )
        
        self.metrics_collector.record_custom_metric(
            name="memory_increase",
            value=self._peak_memory - self._start_memory,
            unit="MB"
        )
        
        # Final fitness metrics
        if self._fitness_history:
            initial_fitness = self._fitness_history[0]
            final_fitness = self._fitness_history[-1]
            total_improvement = final_fitness - initial_fitness
            
            self.metrics_collector.record_custom_metric(
                name="total_fitness_improvement",
                value=total_improvement,
                unit="score"
            )
            
            self.metrics_collector.record_custom_metric(
                name="fitness_improvement_rate",
                value=total_improvement / total_generations if total_generations > 0 else 0,
                unit="score_per_generation"
            )
        
        # Diversity evolution
        if len(self._diversity_history) > 1:
            diversity_change = self._diversity_history[-1] - self._diversity_history[0]
            self.metrics_collector.record_custom_metric(
                name="diversity_change",
                value=diversity_change,
                unit="score"
            )
        
        # CPU usage summary
        if self._cpu_samples:
            avg_cpu = sum(self._cpu_samples) / len(self._cpu_samples)
            max_cpu = max(self._cpu_samples)
            
            self.metrics_collector.record_custom_metric(
                name="average_cpu_usage",
                value=avg_cpu,
                unit="percent"
            )
            
            self.metrics_collector.record_custom_metric(
                name="peak_cpu_usage",
                value=max_cpu,
                unit="percent"
            )
        
        logger.info(f"Evolution completed: {total_generations} generations, final fitness: {best_genome.fitness_score:.4f}",
                   correlation_id=self.metrics_collector.performance_metrics_correlation_id)
    
    async def _cleanup_evolution_metrics(self) -> None:
        """Clean up evolution metrics and perform final collection."""
        
        # Force garbage collection to measure memory cleanup
        gc.collect()
        
        # Record final memory state
        process = psutil.Process()
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        self.metrics_collector.record_custom_metric(
            name="final_memory_usage",
            value=final_memory,
            unit="MB"
        )
        
        # Update final component performance metrics
        if hasattr(self, 'metrics_collector') and self.metrics_collector.performance_metrics:
            perf = self.metrics_collector.performance_metrics
            perf.generation_number = len(self._generation_times)
            perf.fitness_improvement = (
                self._fitness_history[-1] - self._fitness_history[0] 
                if len(self._fitness_history) >= 2 else 0
            )
            perf.diversity_score = self._diversity_history[-1] if self._diversity_history else 0
        
        # Final heartbeat
        self.metrics_collector.heartbeat()


class ObservableKrakenLNN(KrakenLNN):
    """Kraken LNN with integrated observability and metrics collection."""
    
    def __init__(self, 
                 reservoir_size: int = 100,
                 connectivity: float = 0.1,
                 memory_buffer_size: int = 1000,
                 dynamics=None,
                 component_id: Optional[str] = None):
        """Initialize observable Kraken LNN.
        
        Args:
            reservoir_size: Size of the liquid reservoir
            connectivity: Connection density
            memory_buffer_size: Size of temporal memory buffer
            dynamics: Liquid dynamics configuration
            component_id: Component ID for observability (auto-generated if None)
        """
        super().__init__(reservoir_size, connectivity, memory_buffer_size, dynamics)
        
        # Generate component ID if not provided
        self.component_id = component_id or f"kraken_lnn_{uuid.uuid4().hex[:12]}"
        
        # Initialize observability
        self.observability_engine = get_observability_engine()
        self.metrics_collector = self.observability_engine.register_kraken_lnn(self.component_id)
        
        # Set correlation ID for tracing
        correlation_id = f"kraken_{uuid.uuid4().hex[:12]}"
        self.metrics_collector.set_correlation_id(correlation_id)
        
        # Kraken-specific metrics
        self._sequence_times: List[float] = []
        self._memory_utilization_history: List[float] = []
        
        logger.info(f"Observable Kraken LNN initialized: {self.component_id}",
                   reservoir_size=reservoir_size, connectivity=connectivity)
    
    async def process_sequence(self, 
                             input_sequence: List[float],
                             learning_enabled: bool = True,
                             memory_consolidation: bool = True) -> Dict[str, Any]:
        """Process sequence with comprehensive observability."""
        
        correlation_id = self.metrics_collector.performance_metrics_correlation_id
        start_time = time.time()
        
        try:
            logger.debug(f"Processing sequence: {len(input_sequence)} inputs",
                        correlation_id=correlation_id)
            
            # Record pre-processing metrics
            await self._record_pre_processing_metrics(input_sequence)
            
            # Process through parent method
            result = await super().process_sequence(input_sequence, learning_enabled, memory_consolidation)
            
            # Record processing metrics
            processing_time = time.time() - start_time
            await self._record_processing_metrics(processing_time, input_sequence, result)
            
            return result
            
        except Exception as e:
            # Record failure metrics
            processing_time = time.time() - start_time
            self.metrics_collector.record_latency(processing_time * 1000, "sequence_processing")
            self.metrics_collector.record_success(False, "sequence_processing")
            
            logger.error(f"Sequence processing failed: {e}",
                        correlation_id=correlation_id, exc_info=True)
            raise
    
    async def _record_pre_processing_metrics(self, input_sequence: List[float]) -> None:
        """Record metrics before sequence processing."""
        
        # Input sequence metrics
        self.metrics_collector.record_custom_metric(
            name="input_sequence_length",
            value=len(input_sequence),
            unit="count"
        )
        
        # Current memory utilization
        memory_utilization = len(self.temporal_memory.memories) / self.temporal_memory.buffer_size
        self._memory_utilization_history.append(memory_utilization)
        
        self.metrics_collector.record_custom_metric(
            name="memory_utilization",
            value=memory_utilization,
            unit="ratio"
        )
        
        # Reservoir state metrics
        reservoir_activity = float(self.liquid_reservoir.state.sum())
        self.metrics_collector.record_custom_metric(
            name="reservoir_activity",
            value=reservoir_activity,
            unit="sum"
        )
        
        # Record current resource usage
        process = psutil.Process()
        current_memory = process.memory_info().rss / (1024 * 1024)  # MB
        cpu_percent = process.cpu_percent()
        
        self.metrics_collector.record_resource_usage(
            memory_mb=current_memory,
            cpu_percent=cpu_percent
        )
        
        # Heartbeat
        self.metrics_collector.heartbeat()
    
    async def _record_processing_metrics(self, 
                                       processing_time: float,
                                       input_sequence: List[float],
                                       result: Dict[str, Any]) -> None:
        """Record metrics after sequence processing."""
        
        # Timing metrics
        self._sequence_times.append(processing_time)
        self.metrics_collector.record_latency(processing_time * 1000, "sequence_processing")
        
        # Success metrics
        self.metrics_collector.record_success(result.get("success", False), "sequence_processing")
        
        # Processing statistics from result
        if result.get("success"):
            # Output metrics
            liquid_outputs = result.get("liquid_outputs", [])
            self.metrics_collector.record_custom_metric(
                name="output_sequence_length",
                value=len(liquid_outputs),
                unit="count"
            )
            
            # Memory metrics
            memory_entries = result.get("memory_entries", 0)
            self.metrics_collector.record_custom_metric(
                name="memory_entries",
                value=memory_entries,
                unit="count"
            )
            
            # Dynamics metrics
            dynamics = result.get("dynamics", {})
            if dynamics:
                for param, value in dynamics.items():
                    self.metrics_collector.record_custom_metric(
                        name=f"dynamics_{param}",
                        value=float(value),
                        unit="value"
                    )
            
            # Learning metrics
            if hasattr(self.liquid_reservoir, 'adaptive_weights'):
                weight_stats = self._calculate_weight_statistics()
                for stat_name, stat_value in weight_stats.items():
                    self.metrics_collector.record_custom_metric(
                        name=f"weight_{stat_name}",
                        value=stat_value,
                        unit="value"
                    )
        
        # Resource metrics after processing
        process = psutil.Process()
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        final_cpu = process.cpu_percent()
        
        self.metrics_collector.record_resource_usage(
            memory_mb=final_memory,
            cpu_percent=final_cpu
        )
        
        # Performance metrics
        throughput = len(input_sequence) / processing_time if processing_time > 0 else 0
        self.metrics_collector.record_custom_metric(
            name="processing_throughput",
            value=throughput,
            unit="inputs_per_second"
        )
        
        # Update component performance metrics
        if self.metrics_collector.performance_metrics:
            perf = self.metrics_collector.performance_metrics
            perf.reservoir_utilization = memory_utilization if 'memory_utilization' in locals() else 0
            perf.learning_rate = self._estimate_learning_rate()
        
        logger.debug(f"Sequence processed: {processing_time:.3f}s, {len(input_sequence)} inputs",
                    correlation_id=self.metrics_collector.performance_metrics_correlation_id)
    
    def _calculate_weight_statistics(self) -> Dict[str, float]:
        """Calculate adaptive weight statistics."""
        if not hasattr(self.liquid_reservoir, 'adaptive_weights'):
            return {}
        
        weights = self.liquid_reservoir.adaptive_weights.weights
        if weights.size == 0:
            return {}
        
        return {
            "mean": float(weights.mean()),
            "std": float(weights.std()),
            "min": float(weights.min()),
            "max": float(weights.max()),
            "non_zero_count": int((weights != 0).sum()),
            "sparsity": float((weights == 0).sum() / weights.size)
        }
    
    def _estimate_learning_rate(self) -> float:
        """Estimate current learning rate from weight changes."""
        # Simple heuristic based on recent weight changes
        if len(self._sequence_times) < 2:
            return 0.0
        
        # Estimate learning rate from processing speed changes
        recent_times = self._sequence_times[-5:]  # Last 5 sequences
        if len(recent_times) >= 2:
            time_variance = sum((t - sum(recent_times)/len(recent_times))**2 for t in recent_times) / len(recent_times)
            # Higher variance suggests more learning activity
            return float(time_variance)
        
        return 0.0
    
    async def get_network_state(self) -> Dict[str, Any]:
        """Get network state with observability metrics."""
        
        # Get base state from parent
        state = await super().get_network_state()
        
        # Add observability metrics
        state["observability"] = {
            "component_id": self.component_id,
            "sequences_processed": len(self._sequence_times),
            "average_processing_time": sum(self._sequence_times) / len(self._sequence_times) if self._sequence_times else 0,
            "memory_utilization_history": self._memory_utilization_history[-10:],  # Last 10
            "current_correlation_id": self.metrics_collector.performance_metrics_correlation_id
        }
        
        return state


class ObservableNLPAgent(NLPAgent):
    """NLP Agent with integrated observability and metrics collection."""
    
    def __init__(self,
                 genome: ConversationalGenome,
                 config: AgentConfig,
                 component_id: Optional[str] = None):
        """Initialize observable NLP agent.
        
        Args:
            genome: Conversational genome defining agent personality traits
            config: Comprehensive agent configuration
            component_id: Component ID for observability (auto-generated if None)
        """
        super().__init__(genome, config)
        
        # Generate component ID if not provided
        self.component_id = component_id or f"nlp_agent_{uuid.uuid4().hex[:12]}"
        
        # Initialize observability
        self.observability_engine = get_observability_engine()
        self.metrics_collector = self.observability_engine.register_nlp_agent(self.component_id)
        
        # Set correlation ID for tracing
        correlation_id = f"agent_{uuid.uuid4().hex[:12]}"
        self.metrics_collector.set_correlation_id(correlation_id)
        
        # Agent-specific metrics
        self._conversation_times: List[float] = []
        self._response_lengths: List[int] = []
        self._token_usage_history: Dict[str, List[int]] = {}
        
        logger.info(f"Observable NLP agent initialized: {self.component_id}",
                   genome_id=genome.genome_id, llm_provider=config.llm_provider)
    
    async def chat(self,
                   message: str,
                   context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        """Enhanced chat with comprehensive observability."""
        
        correlation_id = self.metrics_collector.performance_metrics_correlation_id
        start_time = time.time()
        
        try:
            logger.debug(f"Processing chat message: {len(message)} characters",
                        correlation_id=correlation_id)
            
            # Record pre-chat metrics
            await self._record_pre_chat_metrics(message, context)
            
            # Process chat through parent method
            async for chunk in super().chat(message, context):
                yield chunk
            
            # Record post-chat metrics
            processing_time = time.time() - start_time
            await self._record_post_chat_metrics(processing_time, message, context)
            
        except Exception as e:
            # Record failure metrics
            processing_time = time.time() - start_time
            self.metrics_collector.record_latency(processing_time * 1000, "chat")
            self.metrics_collector.record_success(False, "chat")
            
            logger.error(f"Chat failed: {e}",
                        correlation_id=correlation_id, exc_info=True)
            raise
    
    async def _record_pre_chat_metrics(self, 
                                     message: str,
                                     context: Optional[Dict[str, Any]] = None) -> None:
        """Record metrics before chat processing."""
        
        # Message metrics
        self.metrics_collector.record_custom_metric(
            name="input_message_length",
            value=len(message),
            unit="characters"
        )
        
        self.metrics_collector.record_custom_metric(
            name="input_word_count",
            value=len(message.split()),
            unit="words"
        )
        
        # Conversation state metrics
        self.metrics_collector.record_custom_metric(
            name="conversation_length",
            value=len(self.conversation_buffer),
            unit="turns"
        )
        
        # Context metrics
        if context:
            self.metrics_collector.record_custom_metric(
                name="has_context",
                value=1,
                unit="boolean"
            )
        
        # Current resource usage
        process = psutil.Process()
        current_memory = process.memory_info().rss / (1024 * 1024)  # MB
        cpu_percent = process.cpu_percent()
        
        self.metrics_collector.record_resource_usage(
            memory_mb=current_memory,
            cpu_percent=cpu_percent
        )
        
        # Heartbeat
        self.metrics_collector.heartbeat()
    
    async def _record_post_chat_metrics(self,
                                      processing_time: float,
                                      message: str,
                                      context: Optional[Dict[str, Any]] = None) -> None:
        """Record metrics after chat processing."""
        
        # Timing metrics
        self._conversation_times.append(processing_time)
        self.metrics_collector.record_latency(processing_time * 1000, "chat")
        self.metrics_collector.record_success(True, "chat")
        
        # Response metrics (if available)
        if hasattr(self, 'conversation_buffer') and self.conversation_buffer:
            last_turn = self.conversation_buffer[-1]
            if last_turn.agent_response:
                response_length = len(last_turn.agent_response)
                self._response_lengths.append(response_length)
                
                self.metrics_collector.record_custom_metric(
                    name="response_length",
                    value=response_length,
                    unit="characters"
                )
                
                self.metrics_collector.record_custom_metric(
                    name="response_word_count",
                    value=len(last_turn.agent_response.split()),
                    unit="words"
                )
                
                # Response ratio
                input_length = len(message)
                ratio = response_length / input_length if input_length > 0 else 0
                self.metrics_collector.record_custom_metric(
                    name="response_ratio",
                    value=ratio,
                    unit="ratio"
                )
        
        # LLM metrics (if available from client)
        if self.llm_client and hasattr(self.llm_client, 'metrics'):
            llm_metrics = self.llm_client.metrics
            self.metrics_collector.record_custom_metric(
                name="llm_tokens_used",
                value=llm_metrics.total_tokens,
                unit="tokens"
            )
            
            self.metrics_collector.record_custom_metric(
                name="llm_cost",
                value=llm_metrics.total_cost,
                unit="USD"
            )
            
            # Track token usage history
            provider = self.config.llm_provider
            if provider not in self._token_usage_history:
                self._token_usage_history[provider] = []
            self._token_usage_history[provider].append(llm_metrics.total_tokens)
        
        # Performance metrics
        throughput = 1.0 / processing_time if processing_time > 0 else 0
        self.metrics_collector.record_custom_metric(
            name="chat_throughput",
            value=throughput,
            unit="chats_per_second"
        )
        
        # Update component performance metrics
        if self.metrics_collector.performance_metrics:
            perf = self.metrics_collector.performance_metrics
            
            # Update LLM-specific metrics
            if self.llm_client and hasattr(self.llm_client, 'metrics'):
                perf.token_usage = {
                    "total": self.llm_client.metrics.total_tokens,
                    "prompt": getattr(self.llm_client.metrics, 'prompt_tokens', 0),
                    "completion": getattr(self.llm_client.metrics, 'completion_tokens', 0)
                }
                perf.cost_usd = self.llm_client.metrics.total_cost
                perf.api_response_time_ms = processing_time * 1000
        
        logger.debug(f"Chat completed: {processing_time:.3f}s",
                    correlation_id=self.metrics_collector.performance_metrics_correlation_id)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive agent metrics including observability data."""
        
        # Get base metrics from parent
        base_metrics = await super().get_metrics()
        
        # Add observability metrics
        base_metrics["observability"] = {
            "component_id": self.component_id,
            "conversations_processed": len(self._conversation_times),
            "average_chat_time": sum(self._conversation_times) / len(self._conversation_times) if self._conversation_times else 0,
            "average_response_length": sum(self._response_lengths) / len(self._response_lengths) if self._response_lengths else 0,
            "token_usage_history": {
                provider: {
                    "total_tokens": sum(usage),
                    "average_tokens": sum(usage) / len(usage) if usage else 0,
                    "total_conversations": len(usage)
                }
                for provider, usage in self._token_usage_history.items()
            },
            "current_correlation_id": self.metrics_collector.performance_metrics_correlation_id
        }
        
        return base_metrics


# Utility functions for easy integration

def create_observable_evolution_engine(config: EvolutionConfig) -> ObservableEvolutionEngine:
    """Create an observable evolution engine.
    
    Args:
        config: Evolution configuration
        
    Returns:
        ObservableEvolutionEngine instance
    """
    return ObservableEvolutionEngine(config)


def create_observable_kraken_lnn(reservoir_size: int = 100,
                                connectivity: float = 0.1,
                                memory_buffer_size: int = 1000,
                                dynamics=None) -> ObservableKrakenLNN:
    """Create an observable Kraken LNN.
    
    Args:
        reservoir_size: Size of the liquid reservoir
        connectivity: Connection density
        memory_buffer_size: Size of temporal memory buffer
        dynamics: Liquid dynamics configuration
        
    Returns:
        ObservableKrakenLNN instance
    """
    return ObservableKrakenLNN(reservoir_size, connectivity, memory_buffer_size, dynamics)


def create_observable_agent(genome: ConversationalGenome,
                          config: AgentConfig) -> ObservableNLPAgent:
    """Create an observable NLP agent.
    
    Args:
        genome: Conversational genome
        config: Agent configuration
        
    Returns:
        ObservableNLPAgent instance
    """
    return ObservableNLPAgent(genome, config)


async def setup_observability_monitoring() -> None:
    """Set up comprehensive observability monitoring for all components."""
    
    engine = get_observability_engine()
    await engine.start()
    
    logger.info("Observability monitoring started for all components")


async def shutdown_observability_monitoring() -> None:
    """Shutdown observability monitoring."""
    
    engine = get_observability_engine()
    await engine.stop()
    
    logger.info("Observability monitoring stopped")
