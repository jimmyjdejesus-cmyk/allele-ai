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

"""Central observability engine for Allele monitoring system.

This module provides the main observability engine that coordinates all
monitoring, metrics collection, and alerting activities.

Author: Bravetto AI Systems
Version: 1.0.0
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .collector import ComponentMetricsCollector, MetricsCollector
from .config import get_observability_settings
from .types import Alert, ComponentType, SystemMetrics

logger = logging.getLogger(__name__)


@dataclass
class ObservabilityEngine:
    """Central observability engine for Allele.
    
    This engine coordinates all monitoring activities, metrics collection,
    alerting, and provides a unified interface for observability features.
    """

    def __init__(self, settings=None):
        """Initialize observability engine.
        
        Args:
            settings: ObservabilitySettings instance (uses global if None)
        """
        self.settings = settings or get_observability_settings()
        self.metrics_collector = MetricsCollector()
        self.component_collectors: Dict[str, ComponentMetricsCollector] = {}
        self.is_running = False
        self._background_tasks: List[asyncio.Task] = []
        self._lock = threading.RLock()

        # Initialize engine with settings
        self._setup_alert_rules()
        self._setup_component_monitoring()

        logger.info("Observability engine initialized")

    def _setup_alert_rules(self) -> None:
        """Set up alert rules from configuration."""
        for rule in self.settings.monitoring.alert_rules:
            self.metrics_collector.alert_rules[rule.rule_id] = rule

        logger.info(f"Loaded {len(self.settings.monitoring.alert_rules)} alert rules")

    def _setup_component_monitoring(self) -> None:
        """Set up monitoring for configured components."""
        # System monitoring is always enabled
        self.metrics_collector.component_metrics["system"] = self._create_system_collector()

        # Evolution engine monitoring
        if self.settings.monitoring.monitor_evolution:
            evolution_id = f"evolution_engine_{int(time.time())}"
            self.metrics_collector.component_metrics[evolution_id] = self._create_evolution_collector(evolution_id)

        # Kraken LNN monitoring
        if self.settings.monitoring.monitor_kraken:
            kraken_id = f"kraken_lnn_{int(time.time())}"
            self.metrics_collector.component_metrics[kraken_id] = self._create_kraken_collector(kraken_id)

        # Agent monitoring
        if self.settings.monitoring.monitor_agents:
            agent_id = f"agents_{int(time.time())}"
            self.metrics_collector.component_metrics[agent_id] = self._create_agent_collector(agent_id)

        logger.info("Component monitoring setup completed")

    def _create_system_collector(self) -> ComponentMetricsCollector:
        """Create system metrics collector."""
        collector = ComponentMetricsCollector(
            component_type=ComponentType.SYSTEM,
            component_id="system",
            metrics_collector=self.metrics_collector
        )
        return collector

    def _create_evolution_collector(self, component_id: str) -> ComponentMetricsCollector:
        """Create evolution engine metrics collector."""
        collector = ComponentMetricsCollector(
            component_type=ComponentType.EVOLUTION_ENGINE,
            component_id=component_id,
            metrics_collector=self.metrics_collector
        )

        # Add evolution-specific metrics tracking
        self.component_collectors[component_id] = collector
        return collector

    def _create_kraken_collector(self, component_id: str) -> ComponentMetricsCollector:
        """Create Kraken LNN metrics collector."""
        collector = ComponentMetricsCollector(
            component_type=ComponentType.KRAKEN_LNN,
            component_id=component_id,
            metrics_collector=self.metrics_collector
        )

        # Add Kraken-specific metrics tracking
        self.component_collectors[component_id] = collector
        return collector

    def _create_agent_collector(self, component_id: str) -> ComponentMetricsCollector:
        """Create NLP agent metrics collector."""
        collector = ComponentMetricsCollector(
            component_type=ComponentType.NLP_AGENT,
            component_id=component_id,
            metrics_collector=self.metrics_collector
        )

        # Add agent-specific metrics tracking
        self.component_collectors[component_id] = collector
        return collector

    async def start(self) -> None:
        """Start the observability engine."""
        if self.is_running:
            logger.warning("Observability engine is already running")
            return

        self.is_running = True

        # Start background monitoring tasks
        if self.settings.monitoring.enabled:
            self._background_tasks.extend([
                asyncio.create_task(self._monitor_system_resources()),
                asyncio.create_task(self._cleanup_old_metrics()),
                asyncio.create_task(self._generate_heartbeat_metrics())
            ])

        logger.info("Observability engine started")

    async def stop(self) -> None:
        """Stop the observability engine."""
        if not self.is_running:
            return

        self.is_running = False

        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        self._background_tasks.clear()

        logger.info("Observability engine stopped")

    async def _monitor_system_resources(self) -> None:
        """Background task to monitor system resources."""
        while self.is_running:
            try:
                # Collect system resource metrics
                system_collector = self.component_collectors.get("system")
                if system_collector:
                    await self._collect_system_metrics(system_collector)

                # Wait for next collection interval
                await asyncio.sleep(self.settings.monitoring.collection_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in system resource monitoring: {e}")
                await asyncio.sleep(5)  # Wait before retry

    async def _collect_system_metrics(self, collector: ComponentMetricsCollector) -> None:
        """Collect system resource metrics."""
        try:
            import psutil

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            collector.record_resource_usage(cpu_percent=cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)  # Convert to MB
            collector.record_resource_usage(memory_mb=memory_mb)

            # GPU usage (if available)
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    collector.record_resource_usage(gpu_percent=gpu.load * 100)
            except ImportError:
                # GPUtil not available, skip GPU metrics
                pass

            # Update component health
            collector.heartbeat()

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")

    async def _cleanup_old_metrics(self) -> None:
        """Background task to clean up old metrics."""
        while self.is_running:
            try:
                # Clean up metrics older than retention period
                cutoff_hours = self.settings.monitoring.retention_hours / 24  # Convert to days
                cleared_count = self.metrics_collector.clear_metrics(older_than_hours=int(cutoff_hours * 24))

                if cleared_count > 0:
                    logger.debug(f"Cleaned up {cleared_count} old metrics")

                # Run cleanup every hour
                await asyncio.sleep(3600)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics cleanup: {e}")
                await asyncio.sleep(300)  # Wait before retry

    async def _generate_heartbeat_metrics(self) -> None:
        """Background task to generate heartbeat metrics for all components."""
        while self.is_running:
            try:
                # Update heartbeats for all components
                for collector in self.component_collectors.values():
                    collector.heartbeat()

                # Wait for next heartbeat interval
                await asyncio.sleep(60)  # Heartbeat every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat generation: {e}")
                await asyncio.sleep(30)  # Wait before retry

    def get_component_collector(self, component_type: ComponentType, component_id: str) -> Optional[ComponentMetricsCollector]:
        """Get a component metrics collector.
        
        Args:
            component_type: Type of component
            component_id: Component instance ID
            
        Returns:
            Component metrics collector if found
        """
        # Look for existing collector
        collector_key = f"{component_type.value}_{component_id}"
        return self.component_collectors.get(collector_key)

    def register_evolution_engine(self, component_id: str) -> ComponentMetricsCollector:
        """Register an evolution engine for monitoring.
        
        Args:
            component_id: Evolution engine instance ID
            
        Returns:
            Component metrics collector
        """
        collector = self._create_evolution_collector(component_id)
        self.component_collectors[f"evolution_engine_{component_id}"] = collector
        return collector

    def register_kraken_lnn(self, component_id: str) -> ComponentMetricsCollector:
        """Register a Kraken LNN for monitoring.
        
        Args:
            component_id: Kraken LNN instance ID
            
        Returns:
            Component metrics collector
        """
        collector = self._create_kraken_collector(component_id)
        self.component_collectors[f"kraken_lnn_{component_id}"] = collector
        return collector

    def register_nlp_agent(self, component_id: str) -> ComponentMetricsCollector:
        """Register an NLP agent for monitoring.
        
        Args:
            component_id: NLP agent instance ID
            
        Returns:
            Component metrics collector
        """
        collector = self._create_agent_collector(component_id)
        self.component_collectors[f"nlp_agent_{component_id}"] = collector
        return collector

    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        return self.metrics_collector.get_system_metrics()

    def get_component_metrics(self, component_id: Optional[str] = None) -> Dict[str, ComponentMetrics]:
        """Get component metrics."""
        return self.metrics_collector.get_component_metrics(component_id)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        return self.metrics_collector.get_performance_summary()

    def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts."""
        return [alert for alert in self.metrics_collector.alerts if alert.status == "active"]

    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of recent alerts
        """
        return sorted(
            self.metrics_collector.alerts,
            key=lambda a: a.triggered_at,
            reverse=True
        )[:limit]

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert.
        
        Args:
            alert_id: Alert ID to acknowledge
            acknowledged_by: User who acknowledged the alert
            
        Returns:
            True if alert was acknowledged, False if not found
        """
        for alert in self.metrics_collector.alerts:
            if alert.alert_id == alert_id:
                alert.status = "acknowledged"
                alert.acknowledged_at = datetime.now(timezone.utc)
                alert.acknowledged_by = acknowledged_by
                logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                return True

        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert.
        
        Args:
            alert_id: Alert ID to resolve
            
        Returns:
            True if alert was resolved, False if not found
        """
        for alert in self.metrics_collector.alerts:
            if alert.alert_id == alert_id:
                alert.status = "resolved"
                alert.resolved_at = datetime.now(timezone.utc)
                logger.info(f"Alert {alert_id} resolved")
                return True

        return False

    def export_metrics(self, format: str = "json", since: Optional[datetime] = None) -> str:
        """Export metrics.
        
        Args:
            format: Export format ("json", "pickle")
            since: Only export metrics since this timestamp
            
        Returns:
            Exported metrics as string
        """
        return self.metrics_collector.export_metrics(format=format, since=since)

    def get_observability_status(self) -> Dict[str, Any]:
        """Get comprehensive observability system status."""
        system_metrics = self.get_system_metrics()
        performance_summary = self.get_performance_summary()
        active_alerts = self.get_active_alerts()

        return {
            "engine_running": self.is_running,
            "settings": {
                "monitoring_enabled": self.settings.monitoring.enabled,
                "alerting_enabled": self.settings.monitoring.alerting_enabled,
                "dashboard_enabled": self.settings.dashboard.enabled,
                "mlflow_enabled": self.settings.mlflow.enabled
            },
            "system_metrics": {
                "health_percentage": system_metrics.health_percentage(),
                "cpu_usage_percent": system_metrics.cpu_usage_percent,
                "memory_usage_mb": system_metrics.memory_usage_mb,
                "components_total": system_metrics.total_components,
                "components_healthy": system_metrics.healthy_components,
                "active_alerts": system_metrics.active_alerts
            },
            "performance": performance_summary,
            "active_alerts": len(active_alerts),
            "components_monitored": len(self.component_collectors),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }


# Global singleton instance
_observability_engine: Optional[ObservabilityEngine] = None


def get_observability_engine() -> ObservabilityEngine:
    """Get the global observability engine instance.
    
    Returns:
        Global ObservabilityEngine instance
    """
    global _observability_engine
    if _observability_engine is None:
        _observability_engine = ObservabilityEngine()
    return _observability_engine


def set_observability_engine(engine: ObservabilityEngine) -> None:
    """Set the global observability engine instance.
    
    Args:
        engine: ObservabilityEngine instance to use
    """
    global _observability_engine
    _observability_engine = engine
