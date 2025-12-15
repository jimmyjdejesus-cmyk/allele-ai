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

"""Configuration management for Allele observability system.

This module provides configuration classes and settings management for the
observability and monitoring system.

Author: Bravetto AI Systems
Version: 1.0.0
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

from .types import (
    AlertRule,
    AlertSeverity,
    ComponentType,
    DashboardConfig,
    MatrixBenchmarkConfig,
    MLflowConfig,
    MonitoringConfig,
)


@dataclass
class ObservabilitySettings:
    """Complete observability system configuration."""

    # Core monitoring settings
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    matrix_benchmark: MatrixBenchmarkConfig = field(default_factory=MatrixBenchmarkConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)

    # Default alert rules
    default_alert_rules: List[AlertRule] = field(default_factory=list)

    def __post_init__(self):
        """Initialize default settings and validate configuration."""
        self._setup_default_alert_rules()
        self._validate_settings()

    def _setup_default_alert_rules(self) -> None:
        """Set up default alert rules for monitoring."""
        if not self.monitoring.alert_rules:
            self.monitoring.alert_rules = [
                # Evolution engine alerts
                AlertRule(
                    rule_id="evolution_latency_high",
                    name="High Evolution Latency",
                    description="Evolution operations taking too long",
                    component_type=ComponentType.EVOLUTION_ENGINE,
                    metric_name="average_latency_ms",
                    threshold=5000.0,
                    condition="gt",
                    severity=AlertSeverity.WARNING,
                    notification_channels=["log"],
                    cooldown_seconds=300
                ),
                AlertRule(
                    rule_id="evolution_error_rate_high",
                    name="High Evolution Error Rate",
                    description="Too many evolution operations failing",
                    component_type=ComponentType.EVOLUTION_ENGINE,
                    metric_name="error_rate",
                    threshold=0.1,
                    condition="gt",
                    severity=AlertSeverity.ERROR,
                    notification_channels=["log"],
                    cooldown_seconds=600
                ),

                # Kraken LNN alerts
                AlertRule(
                    rule_id="kraken_memory_high",
                    name="High Kraken Memory Usage",
                    description="Kraken LNN memory usage too high",
                    component_type=ComponentType.KRAKEN_LNN,
                    metric_name="memory_usage_mb",
                    threshold=1000.0,
                    condition="gt",
                    severity=AlertSeverity.WARNING,
                    notification_channels=["log"],
                    cooldown_seconds=300
                ),
                AlertRule(
                    rule_id="kraken_latency_high",
                    name="High Kraken Processing Latency",
                    description="Kraken sequence processing taking too long",
                    component_type=ComponentType.KRAKEN_LNN,
                    metric_name="average_latency_ms",
                    threshold=1000.0,
                    condition="gt",
                    severity=AlertSeverity.WARNING,
                    notification_channels=["log"],
                    cooldown_seconds=300
                ),

                # NLP Agent alerts
                AlertRule(
                    rule_id="agent_latency_high",
                    name="High Agent Response Latency",
                    description="Agent response times too high",
                    component_type=ComponentType.NLP_AGENT,
                    metric_name="average_latency_ms",
                    threshold=10000.0,
                    condition="gt",
                    severity=AlertSeverity.WARNING,
                    notification_channels=["log"],
                    cooldown_seconds=300
                ),
                AlertRule(
                    rule_id="agent_error_rate_high",
                    name="High Agent Error Rate",
                    description="Too many agent operations failing",
                    component_type=ComponentType.NLP_AGENT,
                    metric_name="error_rate",
                    threshold=0.05,
                    condition="gt",
                    severity=AlertSeverity.ERROR,
                    notification_channels=["log"],
                    cooldown_seconds=600
                ),

                # System alerts
                AlertRule(
                    rule_id="system_memory_high",
                    name="High System Memory Usage",
                    description="System memory usage too high",
                    component_type=ComponentType.SYSTEM,
                    metric_name="memory_usage_mb",
                    threshold=8000.0,
                    condition="gt",
                    severity=AlertSeverity.WARNING,
                    notification_channels=["log"],
                    cooldown_seconds=300
                ),
                AlertRule(
                    rule_id="system_cpu_high",
                    name="High System CPU Usage",
                    description="System CPU usage too high",
                    component_type=ComponentType.SYSTEM,
                    metric_name="cpu_usage_percent",
                    threshold=80.0,
                    condition="gt",
                    severity=AlertSeverity.WARNING,
                    notification_channels=["log"],
                    cooldown_seconds=300
                ),
                AlertRule(
                    rule_id="system_health_degraded",
                    name="System Health Degraded",
                    description="System health percentage below threshold",
                    component_type=ComponentType.SYSTEM,
                    metric_name="health_percentage",
                    threshold=70.0,
                    condition="lt",
                    severity=AlertSeverity.ERROR,
                    notification_channels=["log"],
                    cooldown_seconds=600
                )
            ]

    def _validate_settings(self) -> None:
        """Validate configuration settings."""
        # Validate dashboard settings
        if self.dashboard.port <= 0 or self.dashboard.port > 65535:
            raise ValueError(f"Invalid dashboard port: {self.dashboard.port}")

        # Validate matrix benchmark settings
        if self.matrix_benchmark.runs_per_config <= 0:
            raise ValueError("runs_per_config must be positive")

        if self.matrix_benchmark.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        # Validate MLflow settings
        if self.mlflow.enabled:
            if not self.mlflow.tracking_uri:
                raise ValueError("MLflow tracking URI is required when MLflow is enabled")

    @classmethod
    def from_env(cls) -> "ObservabilitySettings":
        """Create settings from environment variables."""
        return cls(
            monitoring=MonitoringConfig(
                enabled=os.getenv("ALLELE_MONITORING_ENABLED", "true").lower() == "true",
                collection_interval_seconds=int(os.getenv("ALLELE_COLLECTION_INTERVAL", "10")),
                retention_hours=int(os.getenv("ALLELE_RETENTION_HOURS", "168")),
                monitor_evolution=os.getenv("ALLELE_MONITOR_EVOLUTION", "true").lower() == "true",
                monitor_kraken=os.getenv("ALLELE_MONITOR_KRAKEN", "true").lower() == "true",
                monitor_agents=os.getenv("ALLELE_MONITOR_AGENTS", "true").lower() == "true",
                monitor_system=os.getenv("ALLELE_MONITOR_SYSTEM", "true").lower() == "true",
                alerting_enabled=os.getenv("ALLELE_ALERTING_ENABLED", "true").lower() == "true",
                dashboard_enabled=os.getenv("ALLELE_DASHBOARD_ENABLED", "true").lower() == "true",
                dashboard_port=int(os.getenv("ALLELE_DASHBOARD_PORT", "8080")),
                dashboard_host=os.getenv("ALLELE_DASHBOARD_HOST", "localhost"),
                mlflow_enabled=os.getenv("ALLELE_MLFLOW_ENABLED", "true").lower() == "true",
                mlflow_tracking_uri=os.getenv("ALLELE_MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
                mlflow_experiment_name=os.getenv("ALLELE_MLFLOW_EXPERIMENT_NAME", "allele_benchmarks")
            ),
            dashboard=DashboardConfig(
                enabled=os.getenv("ALLELE_DASHBOARD_ENABLED", "true").lower() == "true",
                host=os.getenv("ALLELE_DASHBOARD_HOST", "localhost"),
                port=int(os.getenv("ALLELE_DASHBOARD_PORT", "8080")),
                title=os.getenv("ALLELE_DASHBOARD_TITLE", "Allele Monitoring Dashboard"),
                refresh_interval_seconds=int(os.getenv("ALLELE_DASHBOARD_REFRESH", "30")),
                auto_refresh=os.getenv("ALLELE_DASHBOARD_AUTO_REFRESH", "true").lower() == "true",
                show_evolution_metrics=os.getenv("ALLELE_SHOW_EVOLUTION_METRICS", "true").lower() == "true",
                show_kraken_metrics=os.getenv("ALLELE_SHOW_KRAKEN_METRICS", "true").lower() == "true",
                show_agent_metrics=os.getenv("ALLELE_SHOW_AGENT_METRICS", "true").lower() == "true",
                show_system_metrics=os.getenv("ALLELE_SHOW_SYSTEM_METRICS", "true").lower() == "true",
                show_active_alerts=os.getenv("ALLELE_SHOW_ACTIVE_ALERTS", "true").lower() == "true",
                export_enabled=os.getenv("ALLELE_DASHBOARD_EXPORT_ENABLED", "true").lower() == "true"
            ),
            matrix_benchmark=MatrixBenchmarkConfig(
                enabled=os.getenv("ALLELE_MATRIX_BENCHMARK_ENABLED", "true").lower() == "true",
                runs_per_config=int(os.getenv("ALLELE_BENCHMARK_RUNS_PER_CONFIG", "3")),
                timeout_seconds=int(os.getenv("ALLELE_BENCHMARK_TIMEOUT", "300")),
                parallel_workers=int(os.getenv("ALLELE_BENCHMARK_PARALLEL_WORKERS", "1")),
                max_concurrent_tests=int(os.getenv("ALLELE_BENCHMARK_MAX_CONCURRENT", "4")),
                save_results=os.getenv("ALLELE_BENCHMARK_SAVE_RESULTS", "true").lower() == "true",
                results_path=os.getenv("ALLELE_BENCHMARK_RESULTS_PATH", "benchmark_results"),
                generate_report=os.getenv("ALLELE_BENCHMARK_GENERATE_REPORT", "true").lower() == "true",
                measure_memory=os.getenv("ALLELE_BENCHMARK_MEASURE_MEMORY", "true").lower() == "true",
                measure_cpu=os.getenv("ALLELE_BENCHMARK_MEASURE_CPU", "true").lower() == "true"
            ),
            mlflow=MLflowConfig(
                enabled=os.getenv("ALLELE_MLFLOW_ENABLED", "true").lower() == "true",
                tracking_uri=os.getenv("ALLELE_MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
                experiment_name=os.getenv("ALLELE_MLFLOW_EXPERIMENT_NAME", "allele_observability"),
                model_registry_enabled=os.getenv("ALLELE_MLFLOW_MODEL_REGISTRY", "true").lower() == "true",
                artifact_location=os.getenv("ALLELE_MLFLOW_ARTIFACT_LOCATION", "./mlflow_artifacts"),
                auto_log_evolution=os.getenv("ALlELE_MLFLOW_AUTO_LOG_EVOLUTION", "true").lower() == "true",
                auto_log_kraken=os.getenv("ALLELE_MLFLOW_AUTO_LOG_KRAKEN", "true").lower() == "true",
                auto_log_agents=os.getenv("ALLELE_MLFLOW_AUTO_LOG_AGENTS", "true").lower() == "true",
                batch_logging=os.getenv("ALLELE_MLFLOW_BATCH_LOGGING", "true").lower() == "true",
                batch_size=int(os.getenv("ALLELE_MLFLOW_BATCH_SIZE", "100")),
                log_interval_seconds=int(os.getenv("ALLELE_MLFLOW_LOG_INTERVAL", "30"))
            )
        )


# Global singleton instance
_observability_settings: Optional[ObservabilitySettings] = None


def get_observability_settings() -> ObservabilitySettings:
    """Get the global observability settings instance.
    
    Returns:
        Global ObservabilitySettings instance
    """
    global _observability_settings
    if _observability_settings is None:
        _observability_settings = ObservabilitySettings.from_env()
    return _observability_settings


def set_observability_settings(settings: ObservabilitySettings) -> None:
    """Set the global observability settings instance.
    
    Args:
        settings: ObservabilitySettings instance to use
    """
    global _observability_settings
    _observability_settings = settings
