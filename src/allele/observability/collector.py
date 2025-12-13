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

"""Metrics collection and aggregation for Allele observability system.

This module provides metrics collection, aggregation, and storage capabilities
for monitoring Allele components in real-time.

Author: Bravetto AI Systems
Version: 1.0.0
"""

from typing import Dict, List, Any, Optional, Set, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import asyncio
import threading
import time
import json
import pickle
from pathlib import Path
from collections import defaultdict, deque
import logging

from .types import (
    MetricType,
    MetricValue,
    PerformanceMetrics,
    SystemMetrics,
    ComponentMetrics,
    ComponentType,
    AlertRule,
    Alert,
    AlertSeverity
)
from .config import get_observability_settings

logger = logging.getLogger(__name__)


@dataclass
class MetricsBuffer:
    """In-memory buffer for metrics storage."""
    metrics: Dict[str, List[MetricValue]] = field(default_factory=lambda: defaultdict(list))
    max_size: int = 10000
    max_age_hours: int = 24
    
    def add_metric(self, metric: MetricValue) -> None:
        """Add a metric to the buffer."""
        metric_key = f"{metric.component.value if metric.component else 'unknown'}:{metric.name}"
        self.metrics[metric_key].append(metric)
        
        # Clean up old metrics
        self._cleanup_old_metrics()
        
        # Enforce max size
        if len(self.metrics[metric_key]) > self.max_size:
            self.metrics[metric_key] = self.metrics[metric_key][-self.max_size:]
    
    def _cleanup_old_metrics(self) -> None:
        """Remove metrics older than max_age_hours."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.max_age_hours)
        
        for metric_key in list(self.metrics.keys()):
            self.metrics[metric_key] = [
                m for m in self.metrics[metric_key] 
                if m.timestamp > cutoff_time
            ]
            
            # Remove empty lists
            if not self.metrics[metric_key]:
                del self.metrics[metric_key]
    
    def get_metrics(self, 
                   component: Optional[ComponentType] = None,
                   metric_name: Optional[str] = None,
                   since: Optional[datetime] = None) -> List[MetricValue]:
        """Retrieve metrics based on filters."""
        all_metrics = []
        
        for metric_key, metric_list in self.metrics.items():
            # Apply filters
            if component:
                # Check if any metric in the list has the matching component
                if not any(m.component == component for m in metric_list):
                    continue
                    
            if metric_name and not any(m.name == metric_name for m in metric_list):
                continue
                
            for metric in metric_list:
                if since and metric.timestamp <= since:
                    continue
                all_metrics.append(metric)
                
        return all_metrics
    
    def get_latest_metrics(self) -> Dict[str, MetricValue]:
        """Get the latest metric for each metric type."""
        latest = {}
        
        for metric_key, metric_list in self.metrics.items():
            if metric_list:
                # Get most recent metric
                latest_metric = max(metric_list, key=lambda m: m.timestamp)
                latest[metric_key] = latest_metric
                
        return latest


class MetricsCollector:
    """Central metrics collection and storage system."""
    
    def __init__(self, buffer_size: int = 10000, max_age_hours: int = 24):
        """Initialize metrics collector.
        
        Args:
            buffer_size: Maximum number of metrics to store per type
            max_age_hours: Maximum age of metrics to retain
        """
        self.buffer = MetricsBuffer(max_size=buffer_size, max_age_hours=max_age_hours)
        self.component_metrics: Dict[str, ComponentMetrics] = {}
        self.alerts: List[Alert] = []
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_cooldowns: Dict[str, datetime] = {}
        self._lock = threading.RLock()
        
        # Performance tracking
        self.stats = {
            "metrics_collected": 0,
            "alerts_generated": 0,
            "components_monitored": 0,
            "start_time": time.time()
        }
        
        logger.info("Metrics collector initialized")
    
    def register_component(self, component: ComponentMetrics) -> None:
        """Register a component for monitoring.
        
        Args:
            component: Component metrics instance
        """
        with self._lock:
            self.component_metrics[component.instance_id] = component
            self.stats["components_monitored"] = len(self.component_metrics)
            
        logger.info(f"Component registered for monitoring: {component.instance_id}")
    
    def record_metric(self, 
                     name: str,
                     value: Union[int, float],
                     metric_type: MetricType,
                     component: Optional[ComponentType] = None,
                     component_id: Optional[str] = None,
                     unit: Optional[str] = None,
                     tags: Optional[Dict[str, str]] = None,
                     correlation_id: Optional[str] = None) -> None:
        """Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            component: Component type
            component_id: Component instance ID
            unit: Unit of measurement
            tags: Additional tags
            correlation_id: Correlation ID for tracing
        """
        metric = MetricValue(
            name=name,
            value=value,
            metric_type=metric_type,
            component=component,
            unit=unit,
            tags=tags or {},
            correlation_id=correlation_id
        )
        
        with self._lock:
            self.buffer.add_metric(metric)
            self.stats["metrics_collected"] += 1
            
        # Update component performance metrics if available
        if component_id and component_id in self.component_metrics:
            self._update_component_metrics(component_id, metric)
        
        # Check alert rules
        self._check_alert_rules(metric)
    
    def _update_component_metrics(self, component_id: str, metric: MetricValue) -> None:
        """Update component performance metrics."""
        component = self.component_metrics.get(component_id)
        if not component or not component.performance_metrics:
            return
            
        perf = component.performance_metrics
        
        # Map metric names to performance attributes
        metric_mapping = {
            "average_latency_ms": lambda x: perf.update_latency(x),
            "total_operations": lambda x: setattr(perf, "total_operations", int(x)),
            "successful_operations": lambda x: setattr(perf, "successful_operations", int(x)),
            "failed_operations": lambda x: setattr(perf, "failed_operations", int(x)),
            "memory_usage_mb": lambda x: perf.update_resource_usage(memory_usage_mb=x),
            "cpu_usage_percent": lambda x: perf.update_resource_usage(cpu_usage_percent=x),
            "error_rate": lambda x: setattr(perf, "error_rate", x),
            "success_rate": lambda x: setattr(perf, "success_rate", x),
            "throughput_per_second": lambda x: setattr(perf, "throughput_per_second", x)
        }
        
        updater = metric_mapping.get(metric.name)
        if updater:
            updater(metric.value)
    
    def _check_alert_rules(self, metric: MetricValue) -> None:
        """Check if any alert rules should be triggered."""
        if not self.alert_rules:
            return
            
        # Get recent metrics for the same metric name
        recent_metrics = self.buffer.get_metrics(
            component=metric.component,
            metric_name=metric.name,
            since=datetime.now(timezone.utc) - timedelta(minutes=1)
        )
        
        if not recent_metrics:
            return
            
        # Get the latest value
        latest_value = metric.value
        
        for rule in self.alert_rules.values():
            if rule.component_type != metric.component:
                continue
                
            if rule.metric_name != metric.name:
                continue
                
            # Check cooldown period
            cooldown_key = f"{rule.rule_id}:{metric.component.value if metric.component else 'unknown'}"
            if cooldown_key in self.alert_cooldowns:
                last_alert = self.alert_cooldowns[cooldown_key]
                if datetime.now(timezone.utc) - last_alert < timedelta(seconds=rule.cooldown_seconds):
                    continue
            
            # Evaluate rule
            if rule.evaluate(latest_value, len(recent_metrics)):
                self._generate_alert(rule, metric, latest_value)
                self.alert_cooldowns[cooldown_key] = datetime.now(timezone.utc)
    
    def _generate_alert(self, rule: AlertRule, metric: MetricValue, current_value: float) -> None:
        """Generate an alert based on rule violation."""
        alert = Alert(
            rule_id=rule.rule_id,
            name=rule.name,
            description=rule.description,
            severity=rule.severity,
            component_type=metric.component or ComponentType.SYSTEM,
            component_id=metric.component.value if metric.component else "unknown",
            metric_name=metric.name,
            current_value=current_value,
            threshold=rule.threshold,
            context={
                "condition": rule.condition,
                "evaluation_window": rule.evaluation_window,
                "tags": metric.tags
            },
            correlation_id=metric.correlation_id
        )
        
        with self._lock:
            self.alerts.append(alert)
            self.stats["alerts_generated"] += 1
            
        # Log alert
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }.get(rule.severity, logging.INFO)
        
        logger.log(log_level, 
                  f"ALERT: {alert.name} - {alert.description} "
                  f"(value: {current_value:.2f}, threshold: {rule.threshold:.2f})")
    
    def get_component_metrics(self, component_id: Optional[str] = None) -> Dict[str, ComponentMetrics]:
        """Get metrics for components."""
        with self._lock:
            if component_id:
                return {component_id: self.component_metrics[component_id]}
            return self.component_metrics.copy()
    
    def get_system_metrics(self) -> SystemMetrics:
        """Calculate aggregated system metrics."""
        with self._lock:
            # Get latest metrics
            latest_metrics = self.buffer.get_latest_metrics()
            
            # Initialize system metrics
            sys_metrics = SystemMetrics()
            
            # Aggregate component health
            total_components = len(self.component_metrics)
            if total_components > 0:
                healthy = sum(1 for c in self.component_metrics.values() if c.is_healthy)
                degraded = sum(1 for c in self.component_metrics.values() 
                             if c.is_running and not c.is_healthy)
                failed = sum(1 for c in self.component_metrics.values() if not c.is_running)
                
                sys_metrics.total_components = total_components
                sys_metrics.healthy_components = healthy
                sys_metrics.degraded_components = degraded
                sys_metrics.failed_components = failed
            
            # Aggregate system resource metrics
            system_component_metrics = [
                m for m in self.component_metrics.values() 
                if m.component_type == ComponentType.SYSTEM
            ]
            
            if system_component_metrics:
                # Get latest system metrics
                for comp in system_component_metrics:
                    if comp.performance_metrics:
                        perf = comp.performance_metrics
                        # Average CPU and memory usage
                        sys_metrics.cpu_usage_percent += perf.cpu_usage_percent
                        sys_metrics.memory_usage_mb += perf.memory_usage_mb
                
                # Calculate averages
                count = len(system_component_metrics)
                sys_metrics.cpu_usage_percent /= count
                sys_metrics.memory_usage_mb /= count
            
            # Alert metrics
            active_alerts = [a for a in self.alerts if a.status == "active"]
            sys_metrics.active_alerts = len(active_alerts)
            sys_metrics.critical_alerts = len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL])
            sys_metrics.error_alerts = len([a for a in active_alerts if a.severity == AlertSeverity.ERROR])
            
            return sys_metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        with self._lock:
            uptime_seconds = time.time() - self.stats["start_time"]
            
            return {
                "uptime_seconds": uptime_seconds,
                "metrics_collected": self.stats["metrics_collected"],
                "alerts_generated": self.stats["alerts_generated"],
                "components_monitored": self.stats["components_monitored"],
                "metrics_per_second": self.stats["metrics_collected"] / max(uptime_seconds, 1),
                "buffer_size": sum(len(metrics) for metrics in self.buffer.metrics.values()),
                "active_alerts": len([a for a in self.alerts if a.status == "active"]),
                "total_alerts": len(self.alerts)
            }
    
    def export_metrics(self, 
                      format: str = "json",
                      since: Optional[datetime] = None,
                      component: Optional[ComponentType] = None) -> str:
        """Export metrics in specified format.
        
        Args:
            format: Export format ("json", "pickle")
            since: Only export metrics since this timestamp
            component: Only export metrics for this component type
            
        Returns:
            Exported metrics as string
        """
        metrics = self.buffer.get_metrics(since=since, component=component)
        
        if format.lower() == "json":
            data = [metric.__dict__ for metric in metrics]
            for item in data:
                # Convert datetime objects to ISO strings
                if "timestamp" in item and isinstance(item["timestamp"], datetime):
                    item["timestamp"] = item["timestamp"].isoformat()
            return json.dumps(data, indent=2, default=str)
        elif format.lower() == "pickle":
            return pickle.dumps(metrics).hex()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def clear_metrics(self, older_than_hours: Optional[int] = None) -> int:
        """Clear old metrics from buffer.
        
        Args:
            older_than_hours: Clear metrics older than this many hours (None = all)
            
        Returns:
            Number of metrics cleared
        """
        with self._lock:
            if older_than_hours is None:
                # Clear all metrics
                total_cleared = sum(len(metrics) for metrics in self.buffer.metrics.values())
                self.buffer.metrics.clear()
                return total_cleared
            else:
                # Clear metrics older than specified time
                cutoff = datetime.now(timezone.utc) - timedelta(hours=older_than_hours)
                total_cleared = 0
                
                for metric_key in list(self.buffer.metrics.keys()):
                    old_count = len(self.buffer.metrics[metric_key])
                    self.buffer.metrics[metric_key] = [
                        m for m in self.buffer.metrics[metric_key] 
                        if m.timestamp > cutoff
                    ]
                    total_cleared += old_count - len(self.buffer.metrics[metric_key])
                    
                return total_cleared


class ComponentMetricsCollector:
    """Base class for component-specific metrics collectors."""
    
    def __init__(self, 
                 component_type: ComponentType,
                 component_id: str,
                 metrics_collector: MetricsCollector):
        """Initialize component metrics collector.
        
        Args:
            component_type: Type of component
            component_id: Component instance ID
            metrics_collector: Central metrics collector
        """
        self.component_type = component_type
        self.component_id = component_id
        self.metrics_collector = metrics_collector
        
        # Register component
        component_metrics = ComponentMetrics(
            component_type=component_type,
            component_id=component_id
        )
        self.metrics_collector.register_component(component_metrics)
        
        # Initialize performance metrics
        self.performance_metrics = PerformanceMetrics(
            component_type=component_type,
            component_id=component_id
        )
        
        self.performance_metrics_correlation_id = None
    
    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for metrics tracing."""
        self.performance_metrics_correlation_id = correlation_id
        self.performance_metrics.correlation_id = correlation_id
    
    def record_latency(self, latency_ms: float, operation: str = "operation") -> None:
        """Record operation latency."""
        self.performance_metrics.update_latency(latency_ms)
        
        self.metrics_collector.record_metric(
            name="average_latency_ms",
            value=latency_ms,
            metric_type=MetricType.TIMER,
            component=self.component_type,
            component_id=self.component_id,
            unit="ms",
            tags={"operation": operation},
            correlation_id=self.performance_metrics_correlation_id
        )
    
    def record_success(self, success: bool, operation: str = "operation") -> None:
        """Record operation success/failure."""
        self.performance_metrics.update_success(success)
        
        metric_name = "successful_operations" if success else "failed_operations"
        
        self.metrics_collector.record_metric(
            name=metric_name,
            value=1,
            metric_type=MetricType.COUNTER,
            component=self.component_type,
            component_id=self.component_id,
            tags={"operation": operation, "success": str(success)},
            correlation_id=self.performance_metrics_correlation_id
        )
        
        # Update error rate
        error_rate = self.performance_metrics.error_rate
        self.metrics_collector.record_metric(
            name="error_rate",
            value=error_rate,
            metric_type=MetricType.GAUGE,
            component=self.component_type,
            component_id=self.component_id,
            tags={"operation": operation},
            correlation_id=self.performance_metrics_correlation_id
        )
    
    def record_resource_usage(self, 
                             memory_mb: Optional[float] = None,
                             cpu_percent: Optional[float] = None,
                             gpu_percent: Optional[float] = None) -> None:
        """Record resource usage metrics."""
        if memory_mb is not None:
            self.performance_metrics.update_resource_usage(memory_usage_mb=memory_mb)
            self.metrics_collector.record_metric(
                name="memory_usage_mb",
                value=memory_mb,
                metric_type=MetricType.GAUGE,
                component=self.component_type,
                component_id=self.component_id,
                unit="MB",
                correlation_id=self.performance_metrics_correlation_id
            )
        
        if cpu_percent is not None:
            self.performance_metrics.update_resource_usage(cpu_usage_percent=cpu_percent)
            self.metrics_collector.record_metric(
                name="cpu_usage_percent",
                value=cpu_percent,
                metric_type=MetricType.GAUGE,
                component=self.component_type,
                component_id=self.component_id,
                unit="percent",
                correlation_id=self.performance_metrics_correlation_id
            )
        
        if gpu_percent is not None:
            self.performance_metrics.update_resource_usage(gpu_usage_percent=gpu_percent)
            self.metrics_collector.record_metric(
                name="gpu_usage_percent",
                value=gpu_percent,
                metric_type=MetricType.GAUGE,
                component=self.component_type,
                component_id=self.component_id,
                unit="percent",
                correlation_id=self.performance_metrics_correlation_id
            )
    
    def record_custom_metric(self, 
                           name: str,
                           value: Union[int, float],
                           unit: Optional[str] = None,
                           tags: Optional[Dict[str, str]] = None) -> None:
        """Record a custom metric."""
        self.metrics_collector.record_metric(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            component=self.component_type,
            component_id=self.component_id,
            unit=unit,
            tags=tags,
            correlation_id=self.performance_metrics_correlation_id
        )
    
    def heartbeat(self) -> None:
        """Send a heartbeat to indicate component is alive."""
        component = self.metrics_collector.component_metrics.get(self.component_id)
        if component:
            component.update_heartbeat()
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self.performance_metrics
