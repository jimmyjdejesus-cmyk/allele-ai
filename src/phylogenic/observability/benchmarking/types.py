# Copyright (C) 2025 Phylogenic AI Labs & Jimmy De Jesus
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

"""Benchmarking types and data structures for Allele matrix benchmarking.

This module defines types and data structures specific to matrix benchmarking
operations and performance profiling.

Author: Bravetto AI Systems
Version: 1.0.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from ..types import BenchmarkResult

logger = logging.getLogger(__name__)


class BenchmarkType(str, Enum):
    """Types of benchmarks that can be run."""

    EVOLUTION = "evolution"
    KRAKEN_PROCESSING = "kraken_processing"
    AGENT_CHAT = "agent_chat"
    LLM_INTEGRATION = "llm_integration"
    GENOME_OPERATIONS = "genome_operations"
    SYSTEM_PERFORMANCE = "system_performance"
    MEMORY_SCALING = "memory_scaling"
    CONCURRENT_OPERATIONS = "concurrent_operations"


class ComponentUnderTest(str, Enum):
    """Components that can be benchmarked."""

    EVOLUTION_ENGINE = "evolution_engine"
    KRAKEN_LNN = "kraken_lnn"
    NLP_AGENT = "nlp_agent"
    LLM_CLIENT = "llm_client"
    GENOME = "genome"
    SYSTEM = "system"


@dataclass
class ParameterSet:
    """A set of parameters for a benchmark configuration."""

    name: str
    parameters: Dict[str, Any]
    description: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate parameter set."""
        if not self.name:
            raise ValueError("Parameter set name cannot be empty")
        if not self.parameters:
            raise ValueError("Parameter set must have parameters")


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    benchmark_id: str
    benchmark_type: BenchmarkType
    component: ComponentUnderTest
    parameter_sets: List[ParameterSet]

    # Execution settings
    runs_per_config: int = 3
    timeout_seconds: int = 300
    warmup_runs: int = 1
    parallel_execution: bool = False

    # Measurement settings
    measure_memory: bool = True
    measure_cpu: bool = True
    measure_gpu: bool = False
    measure_network: bool = False

    # Output settings
    save_raw_data: bool = True
    generate_plots: bool = True
    export_format: str = "json"  # json, csv, parquet

    # Environment info
    environment_info: Dict[str, Any] = field(default_factory=dict)

    def generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for matrix testing.

        Returns:
            List of parameter dictionaries for each test combination
        """
        combinations = []

        # Start with base combination
        base_combination = {}
        for param_set in self.parameter_sets:
            base_combination.update(param_set.parameters)

        # Generate combinations by varying one parameter at a time
        for param_set in self.parameter_sets:
            for param_name, param_value in param_set.parameters.items():
                # Create combination with this parameter varied
                combination = base_combination.copy()
                combination[param_name] = param_value
                combination["_benchmark_source"] = param_set.name
                combination["_benchmark_id"] = self.benchmark_id
                combinations.append(combination)

        return combinations

    def get_test_count(self) -> int:
        """Get total number of test configurations.

        Returns:
            Total number of test configurations
        """
        return len(self.parameter_sets)  # Each parameter set creates one test


@dataclass
class PerformanceProfile:
    """Detailed performance profile from benchmark execution."""

    benchmark_id: str
    test_name: str
    parameters: Dict[str, Any]

    # Execution timing
    execution_times: List[float]
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    median_time: float
    p95_time: float
    p99_time: float

    # Resource utilization
    memory_usage: List[float]
    cpu_usage: List[float]
    gpu_usage: Optional[List[float]] = None
    peak_memory_mb: float = 0.0
    average_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    average_cpu_percent: float = 0.0

    # Throughput metrics
    operations_per_second: float = 0.0
    throughput_mb_per_second: float = 0.0

    # Quality metrics
    success_rate: float = 0.0
    error_rate: float = 0.0
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0

    # System environment
    environment: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_seconds: float = 0.0

    def __post_init__(self) -> None:
        """Calculate derived metrics from raw data."""
        if self.execution_times:
            times = np.array(self.execution_times)
            self.mean_time = float(np.mean(times))
            self.std_time = float(np.std(times))
            self.min_time = float(np.min(times))
            self.max_time = float(np.max(times))
            self.median_time = float(np.median(times))
            self.p95_time = float(np.percentile(times, 95))
            self.p99_time = float(np.percentile(times, 99))

            # Calculate operations per second
            if self.mean_time > 0:
                self.operations_per_second = 1.0 / self.mean_time

        if self.memory_usage:
            memory_array = np.array(self.memory_usage)
            self.peak_memory_mb = float(np.max(memory_array))
            self.average_memory_mb = float(np.mean(memory_array))

        if self.cpu_usage:
            cpu_array = np.array(self.cpu_usage)
            self.peak_cpu_percent = float(np.max(cpu_array))
            self.average_cpu_percent = float(np.mean(cpu_array))

    def to_benchmark_result(self) -> BenchmarkResult:
        """Convert to standard BenchmarkResult format."""
        return BenchmarkResult(
            benchmark_id=self.benchmark_id,
            test_name=self.test_name,
            parameters=self.parameters,
            mean_execution_time=self.mean_time,
            std_execution_time=self.std_time,
            min_execution_time=self.min_time,
            max_execution_time=self.max_time,
            p50_execution_time=self.median_time,
            p95_execution_time=self.p95_time,
            p99_execution_time=self.p99_time,
            operations_per_second=self.operations_per_second,
            throughput_mb_per_second=self.throughput_mb_per_second,
            peak_memory_mb=self.peak_memory_mb,
            average_memory_mb=self.average_memory_mb,
            cpu_utilization_percent=self.peak_cpu_percent,
            total_runs=self.total_operations,
            successful_runs=self.successful_operations,
            failed_runs=self.failed_operations,
            error_rate=self.error_rate,
            environment=self.environment,
            timestamp=self.timestamp,
        )


@dataclass
class BenchmarkSuite:
    """Collection of related benchmarks."""

    suite_id: str
    name: str
    description: str
    benchmarks: List[BenchmarkConfig]

    # Suite-level settings
    parallel_execution: bool = False
    stop_on_failure: bool = True
    aggregate_results: bool = True

    # Suite metadata
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def get_total_test_count(self) -> int:
        """Get total number of tests across all benchmarks."""
        return sum(benchmark.get_test_count() for benchmark in self.benchmarks)

    def get_benchmark_by_type(
        self, benchmark_type: BenchmarkType
    ) -> Optional[BenchmarkConfig]:
        """Find a benchmark of specific type."""
        for benchmark in self.benchmarks:
            if benchmark.benchmark_type == benchmark_type:
                return benchmark
        return None


@dataclass
class BenchmarkComparison:
    """Comparison between two benchmark results."""

    comparison_id: str
    baseline_result: PerformanceProfile
    comparison_result: PerformanceProfile

    # Comparison metrics
    performance_ratio: float
    time_improvement: float  # negative means slower
    memory_improvement: float  # negative means more memory used
    throughput_improvement: float  # positive means better throughput

    # Statistical significance
    t_statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float = 0.05

    # Summary
    improvement_summary: str = ""
    recommendations: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Calculate comparison metrics."""
        # Performance ratio (baseline / comparison)
        if self.comparison_result.mean_time > 0:
            self.performance_ratio = (
                self.baseline_result.mean_time / self.comparison_result.mean_time
            )

        # Time improvement (negative means slower)
        self.time_improvement = (
            (self.baseline_result.mean_time - self.comparison_result.mean_time)
            / self.baseline_result.mean_time
            * 100
        )

        # Memory improvement (negative means more memory used)
        self.memory_improvement = (
            (
                self.baseline_result.peak_memory_mb
                - self.comparison_result.peak_memory_mb
            )
            / max(self.baseline_result.peak_memory_mb, 1)
            * 100
        )

        # Throughput improvement
        if self.baseline_result.operations_per_second > 0:
            self.throughput_improvement = (
                (
                    self.comparison_result.operations_per_second
                    - self.baseline_result.operations_per_second
                )
                / self.baseline_result.operations_per_second
                * 100
            )

        # Statistical test (simplified)
        try:
            from scipy import stats  # type: ignore[import-untyped]

            t_stat, p_val = stats.ttest_ind(
                self.baseline_result.execution_times,
                self.comparison_result.execution_times,
            )
            self.t_statistic = float(t_stat)
            self.p_value = float(p_val)
            self.is_significant = self.p_value < self.confidence_level
        except ImportError:
            # scipy not available, skip statistical test
            self.t_statistic = 0.0
            self.p_value = 1.0
            self.is_significant = False
        except (ValueError, TypeError, AttributeError) as e:
            # Invalid data for statistical test
            logger.warning(f"Statistical test failed due to data issues: {e}")
            self.t_statistic = 0.0
            self.p_value = 1.0
            self.is_significant = False

        # Generate summary and recommendations
        self._generate_summary()

    def _generate_summary(self) -> None:
        """Generate summary and recommendations."""
        summary_parts = []
        recommendations = []

        # Performance summary
        if self.time_improvement > 5:
            summary_parts.append(
                f"Performance improved by {self.time_improvement:.1f}%"
            )
        elif self.time_improvement < -5:
            summary_parts.append(
                f"Performance degraded by {abs(self.time_improvement):.1f}%"
            )
        else:
            summary_parts.append("Performance change is minimal")

        # Memory summary
        if self.memory_improvement > 5:
            summary_parts.append(
                f"Memory usage reduced by {self.memory_improvement:.1f}%"
            )
        elif self.memory_improvement < -5:
            summary_parts.append(
                f"Memory usage increased by {abs(self.memory_improvement):.1f}%"
            )

        # Statistical significance
        if self.is_significant:
            summary_parts.append("Change is statistically significant")
        else:
            summary_parts.append("Change is not statistically significant")

        self.improvement_summary = "; ".join(summary_parts)

        # Generate recommendations
        if self.time_improvement < -10:
            recommendations.append(
                "Consider optimizing performance - significant slowdown detected"
            )

        if self.memory_improvement < -20:
            recommendations.append(
                "Memory usage increased significantly - review memory management"
            )

        if not self.is_significant and abs(self.time_improvement) > 5:
            recommendations.append(

                    "Performance change detected but not statistically"
                    " significant - increase sample size"

            )

        self.recommendations = recommendations


@dataclass
class BenchmarkReport:
    """Comprehensive benchmark report."""

    report_id: str
    suite_name: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Results
    profiles: List[PerformanceProfile] = field(default_factory=list)
    comparisons: List[BenchmarkComparison] = field(default_factory=list)

    # Summary statistics
    total_tests_run: int = 0
    total_tests_passed: int = 0
    total_tests_failed: int = 0
    overall_success_rate: float = 0.0

    # Performance summaries
    best_performing_config: Optional[Dict[str, Any]] = None
    worst_performing_config: Optional[Dict[str, Any]] = None
    performance_trends: Dict[str, float] = field(default_factory=dict)

    # Report metadata
    environment: Dict[str, Any] = field(default_factory=dict)
    report_format: str = "comprehensive"

    def add_profile(self, profile: PerformanceProfile) -> None:
        """Add a performance profile to the report."""
        self.profiles.append(profile)
        self._update_summary_stats()

    def add_comparison(self, comparison: BenchmarkComparison) -> None:
        """Add a benchmark comparison to the report."""
        self.comparisons.append(comparison)

    def _update_summary_stats(self) -> None:
        """Update summary statistics."""
        if not self.profiles:
            return

        self.total_tests_run = len(self.profiles)
        self.total_tests_passed = sum(
            1 for p in self.profiles if p.success_rate >= 0.95
        )
        self.total_tests_failed = sum(1 for p in self.profiles if p.success_rate < 0.95)

        if self.total_tests_run > 0:
            self.overall_success_rate = self.total_tests_passed / self.total_tests_run

        # Find best and worst performing configs
        if self.profiles:
            fastest = min(self.profiles, key=lambda p: p.mean_time)
            slowest = max(self.profiles, key=lambda p: p.mean_time)

            self.best_performing_config = {
                "test_name": fastest.test_name,
                "parameters": fastest.parameters,
                "mean_time": fastest.mean_time,
                "throughput": fastest.operations_per_second,
            }

            self.worst_performing_config = {
                "test_name": slowest.test_name,
                "parameters": slowest.parameters,
                "mean_time": slowest.mean_time,
                "throughput": slowest.operations_per_second,
            }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of benchmark results."""
        return {
            "report_id": self.report_id,
            "suite_name": self.suite_name,
            "created_at": self.created_at.isoformat(),
            "total_tests": self.total_tests_run,
            "tests_passed": self.total_tests_passed,
            "tests_failed": self.total_tests_failed,
            "success_rate": self.overall_success_rate,
            "best_config": self.best_performing_config,
            "worst_config": self.worst_performing_config,
            "performance_trends": self.performance_trends,
        }


@dataclass
class ResourceMeasurement:
    """Resource usage measurement during benchmark execution."""

    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    gpu_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    disk_io_read: Optional[float] = None
    disk_io_write: Optional[float] = None
    network_bytes_sent: Optional[float] = None
    network_bytes_recv: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "memory_percent": self.memory_percent,
            "gpu_percent": self.gpu_percent,
            "gpu_memory_mb": self.gpu_memory_mb,
            "disk_io_read": self.disk_io_read,
            "disk_io_write": self.disk_io_write,
            "network_bytes_sent": self.network_bytes_sent,
            "network_bytes_recv": self.network_bytes_recv,
        }
