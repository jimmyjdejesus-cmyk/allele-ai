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

"""Core types and data structures for Allele observability system.

This module defines the foundational types, metrics, and data structures
used throughout the observability and monitoring system.

Author: Bravetto AI Systems
Version: 1.0.0
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union


class MetricType(str, Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    DISTRIBUTION = "distribution"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ComponentType(str, Enum):
    """Types of components that can be monitored."""
    EVOLUTION_ENGINE = "evolution_engine"
    KRAKEN_LNN = "kraken_lnn"
    NLP_AGENT = "nlp_agent"
    LLM_CLIENT = "llm_client"
    GENOME = "genome"
    SYSTEM = "system"


@dataclass
class MetricValue:
    """Represents a single metric value with metadata."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None
    component: Optional[ComponentType] = None
    correlation_id: Optional[str] = None

    def __post_init__(self):
        """Validate metric value."""
        if self.value is None:
            raise ValueError("Metric value cannot be None")
        if not isinstance(self.value, (int, float)):
            raise ValueError(f"Metric value must be numeric, got {type(self.value)}")


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a component."""
    component_type: ComponentType
    component_id: str

    # Timing metrics
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    average_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    # Throughput metrics
    throughput_per_second: float = 0.0
    tokens_per_second: float = 0.0
    requests_per_minute: float = 0.0

    # Memory and resource metrics
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0
    disk_usage_mb: float = 0.0

    # Error and quality metrics
    error_rate: float = 0.0
    success_rate: float = 0.0
    quality_score: float = 0.0

    # Evolution-specific metrics
    generation_number: int = 0
    fitness_improvement: float = 0.0
    diversity_score: float = 0.0
    convergence_rate: float = 0.0

    # LLM-specific metrics
    token_usage: Dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    api_response_time_ms: float = 0.0

    # Kraken LNN metrics
    reservoir_utilization: float = 0.0
    learning_rate: float = 0.0
    memory_consolidation_time_ms: float = 0.0

    # Timestamp and correlation
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = None

    def update_latency(self, latency_ms: float) -> None:
        """Update latency metrics with a new measurement."""
        self.total_operations += 1

        # Update latency statistics
        if self.min_latency_ms == 0.0 or latency_ms < self.min_latency_ms:
            self.min_latency_ms = latency_ms
        if latency_ms > self.max_latency_ms:
            self.max_latency_ms = latency_ms

        # Update average using simple moving average
        if self.total_operations == 1:
            self.average_latency_ms = latency_ms
        else:
            # Calculate running average: (sum + new_value) / count
            old_sum = self.average_latency_ms * (self.total_operations - 1)
            new_sum = old_sum + latency_ms
            self.average_latency_ms = new_sum / self.total_operations

        # Store for percentile calculation
        if not hasattr(self, '_latency_history'):
            self._latency_history = []
        self._latency_history.append(latency_ms)

        # Keep only recent history for memory efficiency
        if len(self._latency_history) > 1000:
            self._latency_history = self._latency_history[-1000:]

        self._update_percentiles()

    def _update_percentiles(self) -> None:
        """Update percentile metrics from latency history."""
        if hasattr(self, '_latency_history') and self._latency_history:
            latencies = sorted(self._latency_history)
            n = len(latencies)

            if n >= 2:
                self.p50_latency_ms = latencies[int(0.5 * n)]
                self.p95_latency_ms = latencies[int(0.95 * n)]
                self.p99_latency_ms = latencies[int(0.99 * n)]

    def update_success(self, success: bool) -> None:
        """Update success/failure metrics."""
        if success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1

        total = self.successful_operations + self.failed_operations
        if total > 0:
            self.success_rate = self.successful_operations / total
            self.error_rate = self.failed_operations / total

    def update_throughput(self, time_delta_seconds: float, operations: int = 1) -> None:
        """Update throughput metrics."""
        if time_delta_seconds > 0:
            self.throughput_per_second = operations / time_delta_seconds

    def update_resource_usage(self, **kwargs) -> None:
        """Update resource usage metrics."""
        for metric, value in kwargs.items():
            if hasattr(self, metric):
                setattr(self, metric, value)


@dataclass
class BenchmarkResult:
    """Results from a matrix benchmark run."""
    benchmark_id: str
    test_name: str
    parameters: Dict[str, Any]

    # Performance metrics
    mean_execution_time: float
    std_execution_time: float
    min_execution_time: float
    max_execution_time: float
    p50_execution_time: float
    p95_execution_time: float
    p99_execution_time: float

    # Throughput metrics
    operations_per_second: float
    peak_memory_mb: float
    average_memory_mb: float

    # Success metrics
    total_runs: int
    successful_runs: int
    failed_runs: int
    error_rate: float

    # Optional fields with defaults (must come last)
    throughput_mb_per_second: float = 0.0
    cpu_utilization_percent: float = 0.0
    gpu_utilization_percent: float = 0.0

    # Metadata
    environment: Dict[str, Any] = field(default_factory=dict)
    component_info: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_runs == 0:
            return 0.0
        return self.successful_runs / self.total_runs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'benchmark_id': self.benchmark_id,
            'test_name': self.test_name,
            'parameters': self.parameters,
            'mean_execution_time': self.mean_execution_time,
            'std_execution_time': self.std_execution_time,
            'min_execution_time': self.min_execution_time,
            'max_execution_time': self.max_execution_time,
            'p50_execution_time': self.p50_execution_time,
            'p95_execution_time': self.p95_execution_time,
            'p99_execution_time': self.p99_execution_time,
            'operations_per_second': self.operations_per_second,
            'throughput_mb_per_second': self.throughput_mb_per_second,
            'peak_memory_mb': self.peak_memory_mb,
            'average_memory_mb': self.average_memory_mb,
            'cpu_utilization_percent': self.cpu_utilization_percent,
            'gpu_utilization_percent': self.gpu_utilization_percent,
            'total_runs': self.total_runs,
            'successful_runs': self.successful_runs,
            'failed_runs': self.failed_runs,
            'error_rate': self.error_rate,
            'success_rate': self.success_rate,
            'environment': self.environment,
            'component_info': self.component_info,
            'timestamp': self.timestamp.isoformat(),
            'correlation_id': self.correlation_id
        }


@dataclass
class SystemMetrics:
    """System-level metrics across all components."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # System resource metrics
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    memory_available_mb: float = 0.0
    disk_usage_mb: float = 0.0
    disk_available_mb: float = 0.0
    network_bytes_per_second: float = 0.0

    # GPU metrics (if available)
    gpu_usage_percent: float = 0.0
    gpu_memory_usage_mb: float = 0.0
    gpu_temperature_c: float = 0.0

    # Process metrics
    active_processes: int = 0
    thread_count: int = 0
    file_descriptors: int = 0

    # Component summaries
    total_components: int = 0
    healthy_components: int = 0
    degraded_components: int = 0
    failed_components: int = 0

    # Alert metrics
    active_alerts: int = 0
    critical_alerts: int = 0
    error_alerts: int = 0

    def health_percentage(self) -> float:
        """Calculate overall system health percentage."""
        if self.total_components == 0:
            return 100.0
        return (self.healthy_components / self.total_components) * 100.0


@dataclass
class ComponentMetrics:
    """Metrics for a specific component instance."""
    component_type: ComponentType
    component_id: str
    instance_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Status
    is_healthy: bool = True
    is_running: bool = True
    last_heartbeat: Optional[datetime] = None

    # Performance data
    performance_metrics: Optional[PerformanceMetrics] = None

    # Configuration info
    config: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def update_heartbeat(self) -> None:
        """Update the last heartbeat timestamp."""
        self.last_heartbeat = datetime.now(timezone.utc)
        self.updated_at = self.last_heartbeat


@dataclass
class AlertRule:
    """Rule definition for generating alerts."""
    rule_id: str
    name: str
    description: str
    component_type: ComponentType
    metric_name: str
    threshold: float
    condition: Literal["gt", "gte", "lt", "lte", "eq", "ne"]
    severity: AlertSeverity
    enabled: bool = True
    cooldown_seconds: int = 300  # 5 minutes default cooldown
    notification_channels: List[str] = field(default_factory=list)

    # Evaluation context
    evaluation_window: int = 60  # seconds
    minimum_samples: int = 5

    def evaluate(self, value: float, sample_count: int) -> bool:
        """Evaluate if the rule should trigger an alert."""
        if not self.enabled or sample_count < self.minimum_samples:
            return False

        if self.condition == "gt":
            return value > self.threshold
        elif self.condition == "gte":
            return value >= self.threshold
        elif self.condition == "lt":
            return value < self.threshold
        elif self.condition == "lte":
            return value <= self.threshold
        elif self.condition == "eq":
            return abs(value - self.threshold) < 1e-10
        elif self.condition == "ne":
            return abs(value - self.threshold) >= 1e-10

        return False


@dataclass
class Alert:
    """Alert instance generated by monitoring system."""
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rule_id: Optional[str] = None
    name: str = ""
    description: str = ""
    severity: AlertSeverity = AlertSeverity.INFO

    # Alert context
    component_type: ComponentType = ComponentType.SYSTEM
    component_id: str = ""
    metric_name: str = ""
    current_value: float = 0.0
    threshold: float = 0.0

    # Status
    status: Literal["active", "acknowledged", "resolved"] = "active"
    triggered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None

    # Additional context
    context: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None


@dataclass
class MonitoringConfig:
    """Configuration for the monitoring system."""
    enabled: bool = True
    collection_interval_seconds: int = 10
    retention_hours: int = 168  # 1 week

    # Component monitoring
    monitor_evolution: bool = True
    monitor_kraken: bool = True
    monitor_agents: bool = True
    monitor_system: bool = True

    # Alerting configuration
    alerting_enabled: bool = True
    alert_rules: List[AlertRule] = field(default_factory=list)

    # Performance thresholds
    performance_thresholds: Dict[str, float] = field(default_factory=dict)

    # Data storage
    storage_backend: str = "memory"  # memory, file, database
    storage_path: Optional[str] = None
    compression_enabled: bool = False

    # Rate limiting
    max_metrics_per_second: int = 1000
    max_alerts_per_minute: int = 100

    # Dashboard settings
    dashboard_enabled: bool = True
    dashboard_port: int = 8080
    dashboard_host: str = "localhost"

    # MLflow integration
    mlflow_enabled: bool = True
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "allele_benchmarks"


@dataclass
class DashboardConfig:
    """Configuration for monitoring dashboard."""
    enabled: bool = True
    host: str = "localhost"
    port: int = 8080
    title: str = "Allele Monitoring Dashboard"

    # Refresh settings
    refresh_interval_seconds: int = 30
    auto_refresh: bool = True

    # Layout configuration
    layout: Dict[str, Any] = field(default_factory=dict)

    # Component visibility
    show_evolution_metrics: bool = True
    show_kraken_metrics: bool = True
    show_agent_metrics: bool = True
    show_system_metrics: bool = True

    # Alert settings
    show_active_alerts: bool = True
    alert_sound_enabled: bool = False

    # Export settings
    export_enabled: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["json", "csv"])


@dataclass
class MatrixBenchmarkConfig:
    """Configuration for matrix benchmarking."""
    enabled: bool = True

    # Test parameters to vary
    population_sizes: List[int] = field(default_factory=lambda: [50, 100, 200, 500, 1000])
    mutation_rates: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.3])
    reservoir_sizes: List[int] = field(default_factory=lambda: [50, 100, 200, 500])
    temperatures: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.7, 1.0])

    # LLM providers and models to test
    llm_providers: List[str] = field(default_factory=lambda: ["openai", "ollama"])
    llm_models: List[str] = field(default_factory=lambda: ["gpt-4", "gpt-3.5-turbo"])

    # Test configuration
    runs_per_config: int = 3
    timeout_seconds: int = 300

    # Parallel execution
    parallel_workers: int = 1
    max_concurrent_tests: int = 4

    # Output settings
    save_results: bool = True
    results_path: str = "benchmark_results"
    generate_report: bool = True

    # Performance settings
    warmup_runs: int = 1
    measure_memory: bool = True
    measure_cpu: bool = True
    measure_gpu: bool = False


@dataclass
class MLflowConfig:
    """Configuration for MLflow integration."""
    enabled: bool = True
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "allele_observability"

    # Model registry
    model_registry_enabled: bool = True
    model_registry_uri: str = "sqlite:///mlflow_registry.db"

    # Artifact storage
    artifact_location: str = "./mlflow_artifacts"
    artifact_store_type: str = "local"

    # Auto-logging settings
    auto_log_evolution: bool = True
    auto_log_kraken: bool = True
    auto_log_agents: bool = True

    # Run configuration
    run_tags: Dict[str, str] = field(default_factory=dict)
    experiment_tags: Dict[str, str] = field(default_factory=dict)

    # Performance settings
    batch_logging: bool = True
    batch_size: int = 100
    log_interval_seconds: int = 30
