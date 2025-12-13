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

"""ML Analytics types and data structures for Allele.

This module defines types and data structures for machine learning analytics
including anomaly detection, predictive analytics, and optimization.

Author: Bravetto AI Systems
Version: 1.0.0
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import numpy as np

from ..types import ComponentType, AlertSeverity


class AnomalyType(str, Enum):
    """Types of anomalies that can be detected."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_SPIKE = "resource_spike"
    MEMORY_LEAK = "memory_leak"
    CPU_OVERLOAD = "cpu_overload"
    LATENCY_SPIKE = "latency_spike"
    ERROR_RATE_SPIKE = "error_rate_spike"
    THROUGHPUT_DROP = "throughput_drop"
    UNUSUAL_PATTERN = "unusual_pattern"


class PredictionType(str, Enum):
    """Types of predictions that can be made."""
    PERFORMANCE_FORECAST = "performance_forecast"
    RESOURCE_USAGE = "resource_usage"
    ERROR_PROBABILITY = "error_probability"
    CAPACITY_PLANNING = "capacity_planning"
    PERFORMANCE_TREND = "performance_trend"
    ANOMALY_RISK = "anomaly_risk"


class OptimizationCategory(str, Enum):
    """Categories of optimization recommendations."""
    CONFIGURATION_TUNING = "configuration_tuning"
    RESOURCE_ALLOCATION = "resource_allocation"
    PERFORMANCE_TUNING = "performance_tuning"
    CAPACITY_SCALING = "capacity_scaling"
    ALERT_THRESHOLDS = "alert_thresholds"
    SYSTEM_SETTINGS = "system_settings"


class ModelStatus(str, Enum):
    """Status of ML models."""
    TRAINING = "training"
    READY = "ready"
    DEGRADED = "degraded"
    ERROR = "error"
    RETRAINING = "retraining"


@dataclass
class MLMetric:
    """ML metric with timestamp and component information."""
    timestamp: datetime
    component_type: ComponentType
    component_id: str
    metric_name: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_vector(self) -> np.ndarray:
        """Convert to numerical vector for ML models."""
        # Create a feature vector from the metric
        features = [
            self.value,
            self.timestamp.timestamp(),
            hash(self.component_type.value) % 1000,
            hash(self.metric_name) % 1000
        ]
        return np.array(features, dtype=float)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "component_type": self.component_type.value,
            "component_id": self.component_id,
            "metric_name": self.metric_name,
            "value": self.value,
            "metadata": self.metadata
        }


@dataclass
class AnomalyResult:
    """Result from anomaly detection."""
    timestamp: datetime
    component_type: ComponentType
    component_id: str
    metric_name: str
    anomaly_type: AnomalyType
    anomaly_score: float
    confidence: float
    severity: AlertSeverity
    
    # Anomaly details
    actual_value: float
    expected_value: float
    deviation: float
    threshold: float
    
    # Context information
    context: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    # Model information
    model_name: str = ""
    model_version: str = ""
    
    def __post_init__(self):
        """Calculate derived fields."""
        self.deviation = self.actual_value - self.expected_value
        self.severity = self._calculate_severity()
    
    def _calculate_severity(self) -> AlertSeverity:
        """Calculate alert severity based on anomaly score and confidence."""
        score_confidence_product = self.anomaly_score * self.confidence
        
        if score_confidence_product > 0.8:
            return AlertSeverity.CRITICAL
        elif score_confidence_product > 0.6:
            return AlertSeverity.ERROR
        elif score_confidence_product > 0.4:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "component_type": self.component_type.value,
            "component_id": self.component_id,
            "metric_name": self.metric_name,
            "anomaly_type": self.anomaly_type.value,
            "anomaly_score": self.anomaly_score,
            "confidence": self.confidence,
            "severity": self.severity.value,
            "actual_value": self.actual_value,
            "expected_value": self.expected_value,
            "deviation": self.deviation,
            "threshold": self.threshold,
            "context": self.context,
            "recommendations": self.recommendations,
            "model_name": self.model_name,
            "model_version": self.model_version
        }


@dataclass
class PredictionResult:
    """Result from predictive analytics."""
    timestamp: datetime
    prediction_type: PredictionType
    component_type: ComponentType
    component_id: str
    metric_name: str
    
    # Prediction details
    predicted_value: float
    confidence_interval: Tuple[float, float]  # (lower, upper)
    prediction_horizon_minutes: int
    
    # Model information
    model_name: str = ""
    model_version: str = ""
    model_accuracy: float = 0.0
    
    # Additional metrics
    feature_importance: Dict[str, float] = field(default_factory=dict)
    prediction_explanation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "prediction_type": self.prediction_type.value,
            "component_type": self.component_type.value,
            "component_id": self.component_id,
            "metric_name": self.metric_name,
            "predicted_value": self.predicted_value,
            "confidence_interval": self.confidence_interval,
            "prediction_horizon_minutes": self.prediction_horizon_minutes,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_accuracy": self.model_accuracy,
            "feature_importance": self.feature_importance,
            "prediction_explanation": self.prediction_explanation
        }


@dataclass
class AlertCluster:
    """Cluster of related alerts."""
    cluster_id: str
    cluster_type: str  # "root_cause", "cascading", "independent"
    
    # Cluster information
    alerts: List[Dict[str, Any]]  # Alert data
    common_attributes: Dict[str, Any] = field(default_factory=dict)
    root_cause_candidates: List[str] = field(default_factory=list)
    
    # Analysis results
    confidence: float = 0.0
    priority_score: float = 0.0
    impact_assessment: str = ""
    
    # Timing information
    first_alert_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_alert_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_minutes: float = 0.0
    
    def update_cluster(self, alert_data: Dict[str, Any]) -> None:
        """Update cluster with new alert."""
        self.alerts.append(alert_data)
        self.last_alert_time = datetime.now(timezone.utc)
        self.duration_minutes = (
            self.last_alert_time - self.first_alert_time
        ).total_seconds() / 60.0
        
        # Recalculate priority score
        self.priority_score = self._calculate_priority_score()
    
    def _calculate_priority_score(self) -> float:
        """Calculate priority score for the cluster."""
        if not self.alerts:
            return 0.0
        
        # Base score from number of alerts
        alert_count_score = min(len(self.alerts) / 10.0, 1.0)
        
        # Severity score
        severity_scores = {
            AlertSeverity.CRITICAL: 1.0,
            AlertSeverity.ERROR: 0.8,
            AlertSeverity.WARNING: 0.5,
            AlertSeverity.INFO: 0.2
        }
        
        severity_score = 0.0
        for alert in self.alerts:
            severity = alert.get("severity", AlertSeverity.INFO)
            severity_score += severity_scores.get(severity, 0.0)
        severity_score /= len(self.alerts)
        
        # Duration score (longer duration = higher priority for resolution)
        duration_score = min(self.duration_minutes / 60.0, 1.0)  # Cap at 1 hour
        
        # Combined score
        self.priority_score = (
            alert_count_score * 0.3 +
            severity_score * 0.5 +
            duration_score * 0.2
        )
        
        return self.priority_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cluster_id": self.cluster_id,
            "cluster_type": self.cluster_type,
            "alerts": self.alerts,
            "common_attributes": self.common_attributes,
            "root_cause_candidates": self.root_cause_candidates,
            "confidence": self.confidence,
            "priority_score": self.priority_score,
            "impact_assessment": self.impact_assessment,
            "first_alert_time": self.first_alert_time.isoformat(),
            "last_alert_time": self.last_alert_time.isoformat(),
            "duration_minutes": self.duration_minutes
        }


@dataclass
class OptimizationRecommendation:
    """Recommendation from optimization engine."""
    recommendation_id: str
    category: OptimizationCategory
    title: str
    description: str
    
    # Recommendation details
    current_value: Any
    recommended_value: Any
    expected_improvement: float  # percentage improvement
    confidence: float
    
    # Implementation details
    implementation_steps: List[str] = field(default_factory=list)
    estimated_effort: str = "low"  # low, medium, high
    risk_level: str = "low"  # low, medium, high
    
    # Context
    component_type: ComponentType = ComponentType.SYSTEM
    component_id: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    priority: int = 1  # 1=highest, 5=lowest
    
    def is_expired(self) -> bool:
        """Check if recommendation has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "recommendation_id": self.recommendation_id,
            "category": self.category.value,
            "title": self.title,
            "description": self.description,
            "current_value": str(self.current_value),
            "recommended_value": str(self.recommended_value),
            "expected_improvement": self.expected_improvement,
            "confidence": self.confidence,
            "implementation_steps": self.implementation_steps,
            "estimated_effort": self.estimated_effort,
            "risk_level": self.risk_level,
            "component_type": self.component_type.value,
            "component_id": self.component_id,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "priority": self.priority,
            "is_expired": self.is_expired()
        }


@dataclass
class ModelMetrics:
    """Metrics for ML models."""
    model_name: str
    model_version: str
    status: ModelStatus
    
    # Performance metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_score: float = 0.0
    
    # Training metrics
    training_samples: int = 0
    validation_samples: int = 0
    last_training_time: Optional[datetime] = None
    training_duration_seconds: float = 0.0
    
    # Inference metrics
    total_predictions: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0
    average_inference_time_ms: float = 0.0
    
    # Drift metrics
    data_drift_score: float = 0.0
    concept_drift_score: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def calculate_success_rate(self) -> float:
        """Calculate prediction success rate."""
        if self.total_predictions == 0:
            return 0.0
        return self.successful_predictions / self.total_predictions
    
    def is_healthy(self) -> bool:
        """Check if model is healthy."""
        return (
            self.status == ModelStatus.READY and
            self.accuracy > 0.7 and
            self.calculate_success_rate() > 0.9 and
            self.data_drift_score < 0.5
        )


@dataclass
class TimeSeriesData:
    """Time series data for ML models."""
    timestamps: List[datetime]
    values: List[float]
    component_type: ComponentType
    component_id: str
    metric_name: str
    
    # Additional metadata
    frequency_minutes: int = 1
    missing_values: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate time series data."""
        if len(self.timestamps) != len(self.values):
            raise ValueError("Timestamps and values must have same length")
        
        if len(self.timestamps) < 10:
            raise ValueError("Time series must have at least 10 data points")
    
    def to_numpy_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert to numpy arrays for ML models."""
        timestamps_float = np.array([ts.timestamp() for ts in self.timestamps])
        values_array = np.array(self.values)
        return timestamps_float, values_array
    
    def get_statistics(self) -> Dict[str, float]:
        """Get basic statistics of the time series."""
        if not self.values:
            return {}
        
        values_array = np.array(self.values)
        return {
            "mean": float(np.mean(values_array)),
            "std": float(np.std(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "median": float(np.median(values_array)),
            "count": len(self.values),
            "missing_count": len(self.missing_values),
            "frequency_minutes": self.frequency_minutes
        }
