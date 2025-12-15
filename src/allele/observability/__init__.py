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

"""Observability and Monitoring System for Allele.

This module provides comprehensive observability, benchmarking, and monitoring
capabilities for the Allele ecosystem including real-time metrics collection,
performance analytics, and MLflow integration.

Author: Bravetto AI Systems
Version: 1.0.0
"""

from .collector import ComponentMetricsCollector, MetricsCollector
from .config import ObservabilitySettings, get_observability_settings
from .engine import ObservabilityEngine
from .types import (
    AlertRule,
    BenchmarkResult,
    ComponentMetrics,
    DashboardConfig,
    MetricType,
    MetricValue,
    MonitoringConfig,
    PerformanceMetrics,
    SystemMetrics,
)

__all__ = [
    # Types
    'MetricType',
    'MetricValue',
    'PerformanceMetrics',
    'BenchmarkResult',
    'SystemMetrics',
    'ComponentMetrics',
    'MonitoringConfig',
    'AlertRule',
    'DashboardConfig',

    # Core Classes
    'MetricsCollector',
    'ComponentMetricsCollector',
    'ObservabilityEngine',

    # Configuration
    'ObservabilitySettings',
    'get_observability_settings',
]
