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
# GNU Affero General Public License, either express or implied.
# See the GNU Affero General Public License for more details.
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

"""Configuration management for Allele matrix benchmarking system.

This module provides configuration classes and settings for the matrix
benchmarking system.

Author: Bravetto AI Systems
Version: 1.0.0
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .types import BenchmarkType, ComponentUnderTest, ParameterSet


@dataclass
class MatrixBenchmarkSettings:
    """Global matrix benchmarking settings."""

    # Matrix configuration
    population_sizes: List[int] = field(default_factory=lambda: [50, 100, 200, 500, 1000])
    mutation_rates: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.3])
    reservoir_sizes: List[int] = field(default_factory=lambda: [50, 100, 200, 500])
    temperatures: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.7, 1.0])

    # LLM providers and models
    llm_providers: List[str] = field(default_factory=lambda: ["openai", "ollama"])
    llm_models: List[str] = field(default_factory=lambda: ["gpt-4", "gpt-3.5-turbo"])

    # Test configuration
    runs_per_config: int = 3
    timeout_seconds: int = 300
    warmup_runs: int = 1
    parallel_workers: int = 1
    max_concurrent_tests: int = 4

    # Measurement settings
    measure_memory: bool = True
    measure_cpu: bool = True
    measure_gpu: bool = False
    measure_network: bool = False

    # Output settings
    save_results: bool = True
    results_path: str = "benchmark_results"
    generate_report: bool = True
    export_format: str = "json"  # json, csv, parquet

    # Performance thresholds
    performance_thresholds: Dict[str, float] = field(default_factory=dict)

    # Suite settings
    parallel_execution: bool = False
    stop_on_failure: bool = True
    aggregate_results: bool = True

    # Regression testing
    regression_testing: bool = True
    regression_baseline_path: str = "regression_baseline.json"
    regression_tolerance_percent: float = 10.0

    # MLflow integration
    mlflow_integration: bool = True
    mlflow_experiment_name: str = "allele_matrix_benchmarks"

    def get_default_evolution_config(self) -> Dict[str, Any]:
        """Get default evolution benchmark configuration."""
        return {
            "benchmark_type": BenchmarkType.EVOLUTION,
            "component": ComponentUnderTest.EVOLUTION_ENGINE,
            "parameter_sets": [
                ParameterSet(
                    name="population_size_variation",
                    parameters={
                        "population_size": 100,
                        "generations": 50,
                        "mutation_rate": 0.1,
                        "crossover_rate": 0.8
                    }
                ),
                ParameterSet(
                    name="mutation_rate_variation",
                    parameters={
                        "population_size": 100,
                        "generations": 50,
                        "mutation_rate": 0.2,
                        "crossover_rate": 0.8
                    }
                )
            ],
            "runs_per_config": self.runs_per_config,
            "timeout_seconds": self.timeout_seconds,
            "measure_memory": self.measure_memory,
            "measure_cpu": self.measure_cpu
        }

    def get_default_kraken_config(self) -> Dict[str, Any]:
        """Get default Kraken LNN benchmark configuration."""
        return {
            "benchmark_type": BenchmarkType.KRAKEN_PROCESSING,
            "component": ComponentUnderTest.KRAKEN_LNN,
            "parameter_sets": [
                ParameterSet(
                    name="reservoir_size_variation",
                    parameters={
                        "reservoir_size": 100,
                        "connectivity": 0.1,
                        "memory_buffer_size": 1000
                    }
                ),
                ParameterSet(
                    name="connectivity_variation",
                    parameters={
                        "reservoir_size": 100,
                        "connectivity": 0.2,
                        "memory_buffer_size": 1000
                    }
                )
            ],
            "runs_per_config": self.runs_per_config,
            "timeout_seconds": self.timeout_seconds,
            "measure_memory": self.measure_memory,
            "measure_cpu": self.measure_cpu
        }

    def get_default_agent_config(self) -> Dict[str, Any]:
        """Get default NLP agent benchmark configuration."""
        return {
            "benchmark_type": BenchmarkType.AGENT_CHAT,
            "component": ComponentUnderTest.NLP_AGENT,
            "parameter_sets": [
                ParameterSet(
                    name="temperature_variation",
                    parameters={
                        "temperature": 0.7,
                        "max_tokens": 2048,
                        "streaming": True
                    }
                ),
                ParameterSet(
                    name="streaming_variation",
                    parameters={
                        "temperature": 0.7,
                        "max_tokens": 2048,
                        "streaming": False
                    }
                )
            ],
            "runs_per_config": self.runs_per_config,
            "timeout_seconds": self.timeout_seconds,
            "measure_memory": self.measure_memory,
            "measure_cpu": self.measure_cpu
        }

    def create_matrix_combinations(self) -> List[Dict[str, Any]]:
        """Create matrix combinations from settings.
        
        Returns:
            List of parameter combinations for matrix testing
        """
        combinations = []

        # Evolution matrix combinations
        for pop_size in self.population_sizes:
            for mutation_rate in self.mutation_rates:
                combinations.append({
                    "component": ComponentUnderTest.EVOLUTION_ENGINE,
                    "benchmark_type": BenchmarkType.EVOLUTION,
                    "population_size": pop_size,
                    "generations": 50,
                    "mutation_rate": mutation_rate,
                    "crossover_rate": 0.8,
                    "test_category": "evolution_matrix"
                })

        # Kraken matrix combinations
        for reservoir_size in self.reservoir_sizes:
            for connectivity in [0.1, 0.2, 0.3]:
                combinations.append({
                    "component": ComponentUnderTest.KRAKEN_LNN,
                    "benchmark_type": BenchmarkType.KRAKEN_PROCESSING,
                    "reservoir_size": reservoir_size,
                    "connectivity": connectivity,
                    "memory_buffer_size": 1000,
                    "test_category": "kraken_matrix"
                })

        # Agent matrix combinations
        for temperature in self.temperatures:
            for provider in self.llm_providers:
                combinations.append({
                    "component": ComponentUnderTest.NLP_AGENT,
                    "benchmark_type": BenchmarkType.AGENT_CHAT,
                    "temperature": temperature,
                    "llm_provider": provider,
                    "max_tokens": 2048,
                    "streaming": True,
                    "test_category": "agent_matrix"
                })

        return combinations

    def get_regression_thresholds(self) -> Dict[str, float]:
        """Get default regression testing thresholds."""
        return {
            "evolution_latency_ms": 5000.0,
            "kraken_processing_ms": 1000.0,
            "agent_response_ms": 10000.0,
            "memory_usage_mb": 1024.0,
            "cpu_usage_percent": 80.0,
            "error_rate": 0.05
        }

    @classmethod
    def from_env(cls) -> "MatrixBenchmarkSettings":
        """Create settings from environment variables."""
        return cls(
            runs_per_config=int(os.getenv("ALLELE_BENCHMARK_RUNS", "3")),
            timeout_seconds=int(os.getenv("ALLELE_BENCHMARK_TIMEOUT", "300")),
            parallel_workers=int(os.getenv("ALLELE_BENCHMARK_WORKERS", "1")),
            max_concurrent_tests=int(os.getenv("ALLELE_BENCHMARK_MAX_CONCURRENT", "4")),
            measure_memory=os.getenv("ALLELE_BENCHMARK_MEASURE_MEMORY", "true").lower() == "true",
            measure_cpu=os.getenv("ALLELE_BENCHMARK_MEASURE_CPU", "true").lower() == "true",
            measure_gpu=os.getenv("ALLELE_BENCHMARK_MEASURE_GPU", "false").lower() == "true",
            save_results=os.getenv("ALLELE_BENCHMARK_SAVE_RESULTS", "true").lower() == "true",
            results_path=os.getenv("ALLELE_BENCHMARK_RESULTS_PATH", "benchmark_results"),
            generate_report=os.getenv("ALLELE_BENCHMARK_GENERATE_REPORT", "true").lower() == "true",
            export_format=os.getenv("ALLELE_BENCHMARK_EXPORT_FORMAT", "json"),
            parallel_execution=os.getenv("ALLELE_BENCHMARK_PARALLEL", "false").lower() == "true",
            regression_testing=os.getenv("ALLELE_BENCHMARK_REGRESSION", "true").lower() == "true",
            mlflow_integration=os.getenv("ALLELE_BENCHMARK_MLFLOW", "true").lower() == "true"
        )


# Global singleton instance
_matrix_benchmark_settings: Optional[MatrixBenchmarkSettings] = None


def get_matrix_benchmark_settings() -> MatrixBenchmarkSettings:
    """Get the global matrix benchmark settings instance.
    
    Returns:
        Global MatrixBenchmarkSettings instance
    """
    global _matrix_benchmark_settings
    if _matrix_benchmark_settings is None:
        _matrix_benchmark_settings = MatrixBenchmarkSettings.from_env()
    return _matrix_benchmark_settings


def set_matrix_benchmark_settings(settings: MatrixBenchmarkSettings) -> None:
    """Set the global matrix benchmark settings instance.
    
    Args:
        settings: MatrixBenchmarkSettings instance to use
    """
    global _matrix_benchmark_settings
    _matrix_benchmark_settings = settings
