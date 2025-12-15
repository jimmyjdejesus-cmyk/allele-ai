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

"""Machine Learning Analytics for Allele Observability.

This module provides advanced ML-based analytics including anomaly detection,
predictive analytics, and intelligent optimization recommendations.

Author: Bravetto AI Systems
Version: 1.0.0
"""

from .alert_intelligence import AlertCorrelator, IntelligentAlertManager
from .anomaly_detection import (
    AnomalyDetector,
    IsolationForestDetector,
    OneClassSVMDetector,
)
from .ml_config import MLAnalyticsConfig
from .optimization_engine import (
    ConfigurationRecommender,
    OptimizationEngine,
    PerformanceOptimizer,
)
from .predictive_analytics import (
    PerformancePredictor,
    PredictiveAnalyzer,
    TimeSeriesForecaster,
)
from .types import (
    AlertCluster,
    AnomalyResult,
    MLMetric,
    OptimizationRecommendation,
    PredictionResult,
)

__all__ = [
    # Anomaly Detection
    'AnomalyDetector',
    'IsolationForestDetector',
    'OneClassSVMDetector',

    # Predictive Analytics
    'PredictiveAnalyzer',
    'TimeSeriesForecaster',
    'PerformancePredictor',

    # Alert Intelligence
    'AlertCorrelator',
    'IntelligentAlertManager',

    # Optimization
    'OptimizationEngine',
    'PerformanceOptimizer',
    'ConfigurationRecommender',

    # Configuration
    'MLAnalyticsConfig',

    # Types
    'AnomalyResult',
    'PredictionResult',
    'AlertCluster',
    'OptimizationRecommendation',
    'MLMetric'
]
