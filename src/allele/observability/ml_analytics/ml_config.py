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

"""Configuration management for ML Analytics in Allele.

This module provides configuration classes and settings for ML-based analytics
including anomaly detection, predictive analytics, and optimization.

Author: Bravetto AI Systems
Version: 1.0.0
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import os
from .types import AnomalyType, PredictionType, OptimizationCategory


@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection."""
    enabled: bool = True
    
    # Model settings
    isolation_forest_contamination: float = 0.1
    isolation_forest_estimators: int = 100
    one_class_svm_nu: float = 0.1
    one_class_svm_kernel: str = "rbf"
    
    # Detection thresholds
    anomaly_threshold: float = 0.7
    confidence_threshold: float = 0.6
    
    # Component-specific settings
    component_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "evolution_engine": 0.8,
        "kraken_lnn": 0.7,
        "nlp_agent": 0.6,
        "system": 0.75
    })
    
    # Training settings
    min_training_samples: int = 100
    retrain_interval_hours: int = 24
    retrain_on_anomaly: bool = True
    
    # Performance settings
    max_processing_delay_seconds: int = 30
    batch_size: int = 50
    
    # Model persistence
    model_persistence_path: str = "./ml_models/anomaly_detection"
    auto_save_models: bool = True


@dataclass
class PredictiveAnalyticsConfig:
    """Configuration for predictive analytics."""
    enabled: bool = True
    
    # Time series forecasting settings
    forecast_horizon_hours: int = 24
    training_window_days: int = 30
    
    # Model settings
    arima_order: tuple = (1, 1, 1)
    lstm_sequence_length: int = 24
    lstm_units: int = 50
    lstm_epochs: int = 100
    
    # Prediction thresholds
    confidence_threshold: float = 0.7
    prediction_accuracy_threshold: float = 0.8
    
    # Component-specific settings
    component_horizons: Dict[str, int] = field(default_factory=lambda: {
        "evolution_engine": 6,  # hours
        "kraken_lnn": 2,
        "nlp_agent": 1,
        "system": 12
    })
    
    # Training settings
    min_training_samples: int = 200
    retrain_interval_hours: int = 48
    
    # Performance settings
    max_prediction_time_seconds: int = 10
    batch_prediction_size: int = 20
    
    # Model persistence
    model_persistence_path: str = "./ml_models/predictive_analytics"
    auto_save_models: bool = True


@dataclass
class AlertIntelligenceConfig:
    """Configuration for intelligent alert management."""
    enabled: bool = True
    
    # Clustering settings
    clustering_algorithm: str = "dbscan"  # "kmeans", "dbscan", "hierarchical"
    clustering_eps: float = 0.5
    clustering_min_samples: int = 2
    
    # Correlation settings
    correlation_window_minutes: int = 15
    correlation_threshold: float = 0.7
    
    # Alert deduplication
    deduplication_window_minutes: int = 5
    similarity_threshold: float = 0.8
    
    # Priority scoring
    priority_factors: Dict[str, float] = field(default_factory=lambda: {
        "severity": 0.4,
        "duration": 0.3,
        "frequency": 0.2,
        "component_criticality": 0.1
    })
    
    # Root cause analysis
    enable_root_cause_analysis: bool = True
    max_root_cause_candidates: int = 5
    
    # Alert management
    auto_acknowledge_duplicates: bool = True
    escalation_time_minutes: int = 60
    
    # Performance settings
    max_processing_time_seconds: int = 5
    batch_processing_size: int = 100


@dataclass
class OptimizationEngineConfig:
    """Configuration for optimization engine."""
    enabled: bool = True
    
    # Optimization categories
    enabled_categories: List[str] = field(default_factory=lambda: [
        "configuration_tuning",
        "resource_allocation",
        "performance_tuning"
    ])
    
    # Recommendation settings
    min_confidence_threshold: float = 0.6
    min_expected_improvement: float = 0.05  # 5%
    
    # Risk assessment
    high_risk_threshold: float = 0.8
    medium_risk_threshold: float = 0.5
    
    # Performance settings
    max_optimization_time_seconds: int = 30
    batch_optimization_size: int = 10
    
    # Rule-based optimization
    enable_rule_based: bool = True
    rule_file_path: str = "./optimization_rules.json"
    
    # ML-based optimization
    enable_ml_based: bool = True
    ml_model_path: str = "./ml_models/optimization"
    
    # Recommendation persistence
    recommendation_retention_days: int = 30
    auto_cleanup_expired: bool = True


@dataclass
class MLAnalyticsConfig:
    """Complete ML Analytics configuration."""
    # Core settings
    enabled: bool = True
    debug_mode: bool = False
    
    # Dependency checking
    check_optional_dependencies: bool = True
    fallback_to_basic_analytics: bool = True
    
    # Component configurations
    anomaly_detection: AnomalyDetectionConfig = field(default_factory=AnomalyDetectionConfig)
    predictive_analytics: PredictiveAnalyticsConfig = field(default_factory=PredictiveAnalyticsConfig)
    alert_intelligence: AlertIntelligenceConfig = field(default_factory=AlertIntelligenceConfig)
    optimization_engine: OptimizationEngineConfig = field(default_factory=OptimizationEngineConfig)
    
    # Global settings
    processing_interval_seconds: int = 60
    max_concurrent_models: int = 5
    
    # Data settings
    metrics_retention_days: int = 90
    model_retention_days: int = 365
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    performance_logging: bool = True
    
    # Integration settings
    integration_with_observability: bool = True
    integration_with_benchmarking: bool = True
    
    # Model versioning
    enable_model_versioning: bool = True
    auto_increment_versions: bool = True
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a specific ML feature is enabled."""
        if not self.enabled:
            return False
        
        feature_configs = {
            "anomaly_detection": self.anomaly_detection,
            "predictive_analytics": self.predictive_analytics,
            "alert_intelligence": self.alert_intelligence,
            "optimization_engine": self.optimization_engine
        }
        
        config = feature_configs.get(feature_name)
        return config.enabled if config else False
    
    def get_component_config(self, component_type: str) -> Dict[str, Any]:
        """Get configuration for a specific component."""
        configs = {
            "anomaly_detection": {
                "threshold": self.anomaly_detection.component_thresholds.get(component_type, 0.7),
                "enabled": self.anomaly_detection.enabled
            },
            "predictive_analytics": {
                "horizon_hours": self.predictive_analytics.component_horizons.get(component_type, 6),
                "enabled": self.predictive_analytics.enabled
            }
        }
        
        return configs.get(component_type, {"enabled": False})
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Check if any features are enabled
        if not any([
            self.anomaly_detection.enabled,
            self.predictive_analytics.enabled,
            self.alert_intelligence.enabled,
            self.optimization_engine.enabled
        ]):
            issues.append("No ML analytics features are enabled")
        
        # Check thresholds
        if self.anomaly_detection.anomaly_threshold <= 0 or self.anomaly_detection.anomaly_threshold >= 1:
            issues.append("Anomaly threshold must be between 0 and 1")
        
        if self.predictive_analytics.confidence_threshold <= 0 or self.predictive_analytics.confidence_threshold >= 1:
            issues.append("Prediction confidence threshold must be between 0 and 1")
        
        # Check processing intervals
        if self.processing_interval_seconds < 1:
            issues.append("Processing interval must be at least 1 second")
        
        # Check retention periods
        if self.metrics_retention_days < 1:
            issues.append("Metrics retention must be at least 1 day")
        
        return issues
    
    @classmethod
    def from_env(cls) -> "MLAnalyticsConfig":
        """Create configuration from environment variables."""
        return cls(
            enabled=os.getenv("ALLELE_ML_ANALYTICS_ENABLED", "true").lower() == "true",
            debug_mode=os.getenv("ALLELE_ML_DEBUG", "false").lower() == "true",
            processing_interval_seconds=int(os.getenv("ALLELE_ML_PROCESSING_INTERVAL", "60")),
            metrics_retention_days=int(os.getenv("ALLELE_ML_METRICS_RETENTION", "90")),
            
            # Anomaly detection settings
            anomaly_detection=AnomalyDetectionConfig(
                enabled=os.getenv("ALLELE_ML_ANOMALY_DETECTION", "true").lower() == "true",
                isolation_forest_contamination=float(os.getenv("ALLELE_ML_IF_CONTAMINATION", "0.1")),
                anomaly_threshold=float(os.getenv("ALLELE_ML_ANOMALY_THRESHOLD", "0.7")),
                retrain_interval_hours=int(os.getenv("ALLELE_ML_RETRAIN_INTERVAL", "24"))
            ),
            
            # Predictive analytics settings
            predictive_analytics=PredictiveAnalyticsConfig(
                enabled=os.getenv("ALLELE_ML_PREDICTIVE", "true").lower() == "true",
                forecast_horizon_hours=int(os.getenv("ALLELE_ML_FORECAST_HORIZON", "24")),
                confidence_threshold=float(os.getenv("ALLELE_ML_PREDICTION_CONFIDENCE", "0.7")),
                retrain_interval_hours=int(os.getenv("ALLELE_ML_PREDICTIVE_RETRAIN", "48"))
            ),
            
            # Alert intelligence settings
            alert_intelligence=AlertIntelligenceConfig(
                enabled=os.getenv("ALLELE_ML_ALERT_INTELLIGENCE", "true").lower() == "true",
                clustering_algorithm=os.getenv("ALLELE_ML_CLUSTERING_ALGO", "dbscan"),
                correlation_threshold=float(os.getenv("ALLELE_ML_CORRELATION_THRESHOLD", "0.7")),
                deduplication_window_minutes=int(os.getenv("ALLELE_ML_DEDUPLICATION_WINDOW", "5"))
            ),
            
            # Optimization engine settings
            optimization_engine=OptimizationEngineConfig(
                enabled=os.getenv("ALLELE_ML_OPTIMIZATION", "true").lower() == "true",
                min_confidence_threshold=float(os.getenv("ALLELE_ML_OPTIMIZATION_CONFIDENCE", "0.6")),
                min_expected_improvement=float(os.getenv("ALLELE_ML_OPTIMIZATION_IMPROVEMENT", "0.05")),
                recommendation_retention_days=int(os.getenv("ALLELE_ML_RECOMMENDATION_RETENTION", "30"))
            )
        )


# Global singleton instance
_ml_analytics_config: Optional[MLAnalyticsConfig] = None


def get_ml_analytics_config() -> MLAnalyticsConfig:
    """Get the global ML analytics configuration instance.
    
    Returns:
        Global MLAnalyticsConfig instance
    """
    global _ml_analytics_config
    if _ml_analytics_config is None:
        _ml_analytics_config = MLAnalyticsConfig.from_env()
    return _ml_analytics_config


def set_ml_analytics_config(config: MLAnalyticsConfig) -> None:
    """Set the global ML analytics configuration instance.
    
    Args:
        config: MLAnalyticsConfig instance to use
    """
    global _ml_analytics_config
    _ml_analytics_config = config
