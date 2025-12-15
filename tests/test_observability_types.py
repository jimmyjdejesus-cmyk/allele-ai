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

"""Tests for observability types and data structures.

This module provides comprehensive testing for all observability types,
ensuring proper functionality and validation.

Author: Bravetto AI Systems
Version: 1.0.0
"""

from datetime import datetime, timezone

import pytest

from allele.observability.types import (
    Alert,
    AlertRule,
    AlertSeverity,
    BenchmarkResult,
    ComponentMetrics,
    ComponentType,
    DashboardConfig,
    MatrixBenchmarkConfig,
    MetricType,
    MetricValue,
    MLflowConfig,
    MonitoringConfig,
    PerformanceMetrics,
    SystemMetrics,
)


class TestMetricTypes:
    """Test metric type enumerations."""

    def test_metric_type_values(self):
        """Test that metric types have correct string values."""
        assert MetricType.COUNTER == "counter"
        assert MetricType.GAUGE == "gauge"
        assert MetricType.HISTOGRAM == "histogram"
        assert MetricType.TIMER == "timer"
        assert MetricType.DISTRIBUTION == "distribution"

    def test_alert_severity_values(self):
        """Test that alert severities have correct string values."""
        assert AlertSeverity.INFO == "info"
        assert AlertSeverity.WARNING == "warning"
        assert AlertSeverity.ERROR == "error"
        assert AlertSeverity.CRITICAL == "critical"

    def test_component_type_values(self):
        """Test that component types have correct string values."""
        assert ComponentType.EVOLUTION_ENGINE == "evolution_engine"
        assert ComponentType.KRAKEN_LNN == "kraken_lnn"
        assert ComponentType.NLP_AGENT == "nlp_agent"
        assert ComponentType.LLM_CLIENT == "llm_client"
        assert ComponentType.GENOME == "genome"
        assert ComponentType.SYSTEM == "system"


class TestMetricValue:
    """Test MetricValue dataclass."""

    def test_metric_value_creation(self):
        """Test basic MetricValue creation."""
        metric = MetricValue(
            name="test_metric",
            value=42.0,
            metric_type=MetricType.GAUGE
        )

        assert metric.name == "test_metric"
        assert metric.value == 42.0
        assert metric.metric_type == MetricType.GAUGE
        assert isinstance(metric.timestamp, datetime)
        assert metric.tags == {}
        assert metric.unit is None
        assert metric.component is None
        assert metric.correlation_id is None

    def test_metric_value_with_all_fields(self):
        """Test MetricValue creation with all fields."""
        timestamp = datetime.now(timezone.utc)
        metric = MetricValue(
            name="test_metric",
            value=100,
            metric_type=MetricType.COUNTER,
            timestamp=timestamp,
            tags={"env": "test", "region": "us-east-1"},
            unit="requests",
            component=ComponentType.EVOLUTION_ENGINE,
            correlation_id="req-12345"
        )

        assert metric.name == "test_metric"
        assert metric.value == 100
        assert metric.metric_type == MetricType.COUNTER
        assert metric.timestamp == timestamp
        assert metric.tags == {"env": "test", "region": "us-east-1"}
        assert metric.unit == "requests"
        assert metric.component == ComponentType.EVOLUTION_ENGINE
        assert metric.correlation_id == "req-12345"

    def test_metric_value_validation(self):
        """Test MetricValue validation."""
        # Test None value
        with pytest.raises(ValueError, match="Metric value cannot be None"):
            MetricValue(
                name="test_metric",
                value=None,
                metric_type=MetricType.GAUGE
            )

        # Test non-numeric value
        with pytest.raises(ValueError, match="Metric value must be numeric"):
            MetricValue(
                name="test_metric",
                value="not a number",
                metric_type=MetricType.GAUGE
            )


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_performance_metrics_creation(self):
        """Test basic PerformanceMetrics creation."""
        metrics = PerformanceMetrics(
            component_type=ComponentType.EVOLUTION_ENGINE,
            component_id="test-engine-1"
        )

        assert metrics.component_type == ComponentType.EVOLUTION_ENGINE
        assert metrics.component_id == "test-engine-1"
        assert metrics.total_operations == 0
        assert metrics.successful_operations == 0
        assert metrics.failed_operations == 0
        assert metrics.average_latency_ms == 0.0
        assert isinstance(metrics.timestamp, datetime)

    def test_latency_updates(self):
        """Test latency measurement updates."""
        metrics = PerformanceMetrics(
            component_type=ComponentType.KRAKEN_LNN,
            component_id="test-kraken-1"
        )

        # Record first latency
        metrics.update_latency(100.0)
        assert metrics.total_operations == 1
        assert metrics.average_latency_ms == 100.0
        assert metrics.min_latency_ms == 100.0
        assert metrics.max_latency_ms == 100.0

        # Record second latency
        metrics.update_latency(200.0)
        assert metrics.total_operations == 2
        assert metrics.average_latency_ms == 150.0  # EMA with alpha=0.1
        assert metrics.min_latency_ms == 100.0
        assert metrics.max_latency_ms == 200.0

    def test_success_updates(self):
        """Test success/failure updates."""
        metrics = PerformanceMetrics(
            component_type=ComponentType.NLP_AGENT,
            component_id="test-agent-1"
        )

        # Record success
        metrics.update_success(True)
        assert metrics.successful_operations == 1
        assert metrics.failed_operations == 0
        assert metrics.success_rate == 1.0
        assert metrics.error_rate == 0.0

        # Record failure
        metrics.update_success(False)
        assert metrics.successful_operations == 1
        assert metrics.failed_operations == 1
        assert metrics.success_rate == 0.5
        assert metrics.error_rate == 0.5

    def test_resource_usage_updates(self):
        """Test resource usage updates."""
        metrics = PerformanceMetrics(
            component_type=ComponentType.SYSTEM,
            component_id="test-system-1"
        )

        # Update resource usage
        metrics.update_resource_usage(
            memory_usage_mb=1024.0,
            cpu_usage_percent=75.5,
            gpu_usage_percent=50.0
        )

        assert metrics.memory_usage_mb == 1024.0
        assert metrics.cpu_usage_percent == 75.5
        assert metrics.gpu_usage_percent == 50.0


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_benchmark_result_creation(self):
        """Test basic BenchmarkResult creation."""
        result = BenchmarkResult(
            benchmark_id="bench-123",
            test_name="evolution_performance",
            parameters={"population_size": 100, "generations": 50},
            mean_execution_time=2.5,
            std_execution_time=0.1,
            min_execution_time=2.3,
            max_execution_time=2.8,
            p50_execution_time=2.4,
            p95_execution_time=2.7,
            p99_execution_time=2.75,
            operations_per_second=40.0,
            peak_memory_mb=512.0,
            average_memory_mb=256.0,
            cpu_utilization_percent=65.0,
            total_runs=10,
            successful_runs=10,
            failed_runs=0,
            error_rate=0.0
        )

        assert result.benchmark_id == "bench-123"
        assert result.test_name == "evolution_performance"
        assert result.parameters["population_size"] == 100
        assert result.mean_execution_time == 2.5
        assert result.success_rate == 1.0  # 10/10 successful

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        # All successful
        result = BenchmarkResult(
            benchmark_id="bench-1",
            test_name="test",
            parameters={},
            mean_execution_time=1.0,
            std_execution_time=0.0,
            min_execution_time=1.0,
            max_execution_time=1.0,
            p50_execution_time=1.0,
            p95_execution_time=1.0,
            p99_execution_time=1.0,
            operations_per_second=1.0,
            peak_memory_mb=100.0,
            average_memory_mb=100.0,
            total_runs=10,
            successful_runs=8,
            failed_runs=2,
            error_rate=0.2
        )

        assert result.success_rate == 0.8  # 8/10 successful

        # No runs
        result_no_runs = BenchmarkResult(
            benchmark_id="bench-2",
            test_name="test",
            parameters={},
            mean_execution_time=1.0,
            std_execution_time=0.0,
            min_execution_time=1.0,
            max_execution_time=1.0,
            p50_execution_time=1.0,
            p95_execution_time=1.0,
            p99_execution_time=1.0,
            operations_per_second=1.0,
            peak_memory_mb=100.0,
            average_memory_mb=100.0,
            total_runs=0,
            successful_runs=0,
            failed_runs=0,
            error_rate=0.0
        )

        assert result_no_runs.success_rate == 0.0

    def test_to_dict_serialization(self):
        """Test dictionary serialization."""
        result = BenchmarkResult(
            benchmark_id="bench-123",
            test_name="test",
            parameters={"param1": "value1"},
            mean_execution_time=1.0,
            std_execution_time=0.1,
            min_execution_time=0.9,
            max_execution_time=1.1,
            p50_execution_time=1.0,
            p95_execution_time=1.05,
            p99_execution_time=1.08,
            operations_per_second=10.0,
            peak_memory_mb=100.0,
            average_memory_mb=95.0,
            total_runs=5,
            successful_runs=5,
            failed_runs=0,
            error_rate=0.0
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["benchmark_id"] == "bench-123"
        assert result_dict["test_name"] == "test"
        assert result_dict["parameters"]["param1"] == "value1"
        assert "timestamp" in result_dict
        assert isinstance(result_dict["timestamp"], str)  # ISO format string


class TestSystemMetrics:
    """Test SystemMetrics dataclass."""

    def test_system_metrics_creation(self):
        """Test basic SystemMetrics creation."""
        metrics = SystemMetrics()

        assert metrics.cpu_usage_percent == 0.0
        assert metrics.memory_usage_mb == 0.0
        assert metrics.total_components == 0
        assert metrics.healthy_components == 0
        assert metrics.active_alerts == 0
        assert metrics.health_percentage() == 100.0  # No components = 100% healthy

    def test_health_percentage_calculation(self):
        """Test health percentage calculation."""
        metrics = SystemMetrics()

        # No components
        assert metrics.health_percentage() == 100.0

        # Some components
        metrics.total_components = 10
        metrics.healthy_components = 8
        metrics.degraded_components = 1
        metrics.failed_components = 1

        assert metrics.health_percentage() == 80.0  # 8/10 = 80%

        # All failed
        metrics.healthy_components = 0
        metrics.failed_components = 10
        assert metrics.health_percentage() == 0.0


class TestComponentMetrics:
    """Test ComponentMetrics dataclass."""

    def test_component_metrics_creation(self):
        """Test basic ComponentMetrics creation."""
        metrics = ComponentMetrics(
            component_type=ComponentType.EVOLUTION_ENGINE,
            component_id="engine-1"
        )

        assert metrics.component_type == ComponentType.EVOLUTION_ENGINE
        assert metrics.component_id == "engine-1"
        assert metrics.is_healthy is True
        assert metrics.is_running is True
        assert metrics.performance_metrics is None
        assert metrics.config == {}
        assert metrics.tags == {}
        assert isinstance(metrics.created_at, datetime)
        assert isinstance(metrics.updated_at, datetime)

    def test_heartbeat_update(self):
        """Test heartbeat updates."""
        metrics = ComponentMetrics(
            component_type=ComponentType.KRAKEN_LNN,
            component_id="kraken-1"
        )

        original_updated = metrics.updated_at

        # Small delay to ensure timestamp changes
        import time
        time.sleep(0.001)

        metrics.update_heartbeat()

        assert metrics.last_heartbeat is not None
        assert metrics.updated_at > original_updated


class TestAlertRule:
    """Test AlertRule dataclass."""

    def test_alert_rule_creation(self):
        """Test basic AlertRule creation."""
        rule = AlertRule(
            rule_id="test-rule",
            name="Test Alert",
            description="This is a test alert rule",
            component_type=ComponentType.EVOLUTION_ENGINE,
            metric_name="average_latency_ms",
            threshold=5000.0,
            condition="gt",
            severity=AlertSeverity.WARNING
        )

        assert rule.rule_id == "test-rule"
        assert rule.name == "Test Alert"
        assert rule.description == "This is a test alert rule"
        assert rule.component_type == ComponentType.EVOLUTION_ENGINE
        assert rule.metric_name == "average_latency_ms"
        assert rule.threshold == 5000.0
        assert rule.condition == "gt"
        assert rule.severity == AlertSeverity.WARNING
        assert rule.enabled is True
        assert rule.cooldown_seconds == 300

    def test_alert_rule_evaluation(self):
        """Test alert rule evaluation."""
        rule = AlertRule(
            rule_id="test-rule",
            name="Test Alert",
            description="Test",
            component_type=ComponentType.SYSTEM,
            metric_name="cpu_usage_percent",
            threshold=80.0,
            condition="gt",
            severity=AlertSeverity.WARNING,
            minimum_samples=3
        )

        # Should not trigger with insufficient samples
        assert not rule.evaluate(90.0, 2)  # Only 2 samples

        # Should trigger with sufficient samples
        assert rule.evaluate(90.0, 5)  # 5 samples, value > threshold

        # Should not trigger when condition not met
        assert not rule.evaluate(70.0, 5)  # Value < threshold

        # Test different conditions
        rule_lt = AlertRule(
            rule_id="lt-rule",
            name="LT Test",
            description="Test",
            component_type=ComponentType.SYSTEM,
            metric_name="memory_usage_mb",
            threshold=1000.0,
            condition="lt",
            severity=AlertSeverity.WARNING
        )

        assert rule_lt.evaluate(500.0, 5)  # Value < threshold
        assert not rule_lt.evaluate(1500.0, 5)  # Value > threshold


class TestAlert:
    """Test Alert dataclass."""

    def test_alert_creation(self):
        """Test basic Alert creation."""
        alert = Alert(
            rule_id="test-rule",
            name="Test Alert",
            description="This is a test alert",
            severity=AlertSeverity.ERROR,
            component_type=ComponentType.EVOLUTION_ENGINE,
            component_id="engine-1",
            metric_name="average_latency_ms",
            current_value=6000.0,
            threshold=5000.0
        )

        assert alert.rule_id == "test-rule"
        assert alert.name == "Test Alert"
        assert alert.description == "This is a test alert"
        assert alert.severity == AlertSeverity.ERROR
        assert alert.component_type == ComponentType.EVOLUTION_ENGINE
        assert alert.component_id == "engine-1"
        assert alert.metric_name == "average_latency_ms"
        assert alert.current_value == 6000.0
        assert alert.threshold == 5000.0
        assert alert.status == "active"
        assert alert.triggered_at is not None
        assert alert.resolved_at is None
        assert alert.acknowledged_at is None


class TestConfigurationClasses:
    """Test configuration dataclasses."""

    def test_monitoring_config_creation(self):
        """Test MonitoringConfig creation."""
        config = MonitoringConfig()

        assert config.enabled is True
        assert config.collection_interval_seconds == 10
        assert config.retention_hours == 168
        assert config.monitor_evolution is True
        assert config.monitor_kraken is True
        assert config.monitor_agents is True
        assert config.monitor_system is True
        assert config.alerting_enabled is True
        assert len(config.alert_rules) == 0
        assert config.dashboard_enabled is True
        assert config.mlflow_enabled is True

    def test_dashboard_config_creation(self):
        """Test DashboardConfig creation."""
        config = DashboardConfig()

        assert config.enabled is True
        assert config.host == "localhost"
        assert config.port == 8080
        assert config.title == "Allele Monitoring Dashboard"
        assert config.refresh_interval_seconds == 30
        assert config.auto_refresh is True
        assert config.show_evolution_metrics is True
        assert config.show_kraken_metrics is True
        assert config.show_agent_metrics is True
        assert config.show_system_metrics is True
        assert config.show_active_alerts is True
        assert config.export_enabled is True
        assert config.export_formats == ["json", "csv"]

    def test_matrix_benchmark_config_creation(self):
        """Test MatrixBenchmarkConfig creation."""
        config = MatrixBenchmarkConfig()

        assert config.enabled is True
        assert config.population_sizes == [50, 100, 200, 500, 1000]
        assert config.mutation_rates == [0.05, 0.1, 0.2, 0.3]
        assert config.reservoir_sizes == [50, 100, 200, 500]
        assert config.temperatures == [0.1, 0.3, 0.7, 1.0]
        assert config.llm_providers == ["openai", "ollama"]
        assert config.llm_models == ["gpt-4", "gpt-3.5-turbo"]
        assert config.runs_per_config == 3
        assert config.timeout_seconds == 300
        assert config.parallel_workers == 1
        assert config.max_concurrent_tests == 4
        assert config.save_results is True
        assert config.results_path == "benchmark_results"
        assert config.generate_report is True
        assert config.warmup_runs == 1
        assert config.measure_memory is True
        assert config.measure_cpu is True
        assert config.measure_gpu is False

    def test_mlflow_config_creation(self):
        """Test MLflowConfig creation."""
        config = MLflowConfig()

        assert config.enabled is True
        assert config.tracking_uri == "sqlite:///mlflow.db"
        assert config.experiment_name == "allele_observability"
        assert config.model_registry_enabled is True
        assert config.model_registry_uri == "sqlite:///mlflow_registry.db"
        assert config.artifact_location == "./mlflow_artifacts"
        assert config.artifact_store_type == "local"
        assert config.auto_log_evolution is True
        assert config.auto_log_kraken is True
        assert config.auto_log_agents is True
        assert config.run_tags == {}
        assert config.experiment_tags == {}
        assert config.batch_logging is True
        assert config.batch_size == 100
        assert config.log_interval_seconds == 30


# Additional utility tests
class TestUtilityFunctions:
    """Test utility functions and edge cases."""

    def test_datetime_handling(self):
        """Test proper datetime handling in metrics."""
        before = datetime.now(timezone.utc)

        metric = MetricValue(
            name="test",
            value=42.0,
            metric_type=MetricType.GAUGE
        )

        after = datetime.now(timezone.utc)

        assert before <= metric.timestamp <= after

    def test_nested_dataclass_relationships(self):
        """Test relationships between nested dataclasses."""
        # Create a performance metrics instance
        perf_metrics = PerformanceMetrics(
            component_type=ComponentType.EVOLUTION_ENGINE,
            component_id="test-engine"
        )

        # Create component metrics with performance metrics
        component_metrics = ComponentMetrics(
            component_type=ComponentType.EVOLUTION_ENGINE,
            component_id="test-engine",
            performance_metrics=perf_metrics
        )

        assert component_metrics.performance_metrics is perf_metrics
        assert component_metrics.performance_metrics.component_type == ComponentType.EVOLUTION_ENGINE

    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        config = MonitoringConfig(
            enabled=True,
            collection_interval_seconds=5,
            retention_hours=24
        )

        assert config.enabled is True
        assert config.collection_interval_seconds == 5
        assert config.retention_hours == 24

        # Test default values are reasonable
        dashboard_config = DashboardConfig()
        assert 1 <= dashboard_config.port <= 65535  # Valid port range

        benchmark_config = MatrixBenchmarkConfig()
        assert benchmark_config.runs_per_config > 0
        assert benchmark_config.timeout_seconds > 0
        assert len(benchmark_config.population_sizes) > 0
