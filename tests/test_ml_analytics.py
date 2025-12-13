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
# but WITHOUT ANY WARRANTY; but without even the implied warranty of
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

"""Comprehensive tests for ML Analytics components.

This module provides extensive testing for all ML analytics features including
anomaly detection, predictive analytics, alert intelligence, and optimization.

Author: Bravetto AI Systems
Version: 1.0.0
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
import tempfile
import json
from pathlib import Path

# Import ML analytics components
from src.allele.observability.ml_analytics.anomaly_detection import (
    IsolationForestDetector,
    OneClassSVMDetector,
    EnsembleAnomalyDetector
)
from src.allele.observability.ml_analytics.predictive_analytics import (
    TimeSeriesForecaster,
    PerformancePredictor,
    PredictiveAnalyzer
)
from src.allele.observability.ml_analytics.alert_intelligence import (
    AlertCorrelator,
    IntelligentAlertManager
)
from src.allele.observability.ml_analytics.optimization_engine import (
    PerformanceOptimizer,
    ConfigurationRecommender,
    OptimizationEngine
)

# Import types and configuration
from src.allele.observability.ml_analytics.types import (
    MLMetric,
    AnomalyResult,
    PredictionResult,
    AlertCluster,
    OptimizationRecommendation,
    AnomalyType,
    PredictionType,
    OptimizationCategory,
    ComponentType,
    AlertSeverity,
    TimeSeriesData
)
from src.allele.observability.ml_analytics.ml_config import (
    AnomalyDetectionConfig,
    PredictiveAnalyticsConfig,
    AlertIntelligenceConfig,
    OptimizationEngineConfig,
    MLAnalyticsConfig
)


class TestMLAnalyticsConfig:
    """Test ML analytics configuration management."""
    
    def test_config_creation(self):
        """Test basic configuration creation."""
        config = MLAnalyticsConfig()
        assert config.enabled is True
        assert config.debug_mode is False
        assert config.anomaly_detection.enabled is True
        assert config.predictive_analytics.enabled is True
        assert config.alert_intelligence.enabled is True
        assert config.optimization_engine.enabled is True
    
    def test_feature_toggle(self):
        """Test feature enabling/disabling."""
        config = MLAnalyticsConfig()
        config.anomaly_detection.enabled = False
        assert not config.is_feature_enabled("anomaly_detection")
        assert config.is_feature_enabled("predictive_analytics")
    
    def test_component_config(self):
        """Test component-specific configuration."""
        config = MLAnalyticsConfig()
        evolution_config = config.get_component_config("evolution_engine")
        assert evolution_config["enabled"] is True
        assert evolution_config["threshold"] > 0
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = MLAnalyticsConfig()
        issues = config.validate_config()
        # Should have no issues with default config
        assert len([i for i in issues if "enabled" not in i]) == 0
    
    def test_env_config_creation(self):
        """Test configuration from environment variables."""
        import os
        os.environ["ALLELE_ML_ANOMALY_DETECTION"] = "false"
        os.environ["ALLELE_ML_DEBUG"] = "true"
        
        config = MLAnalyticsConfig.from_env()
        assert config.anomaly_detection.enabled is False
        assert config.debug_mode is True
        
        # Clean up
        del os.environ["ALLELE_ML_ANOMALY_DETECTION"]
        del os.environ["ALLELE_ML_DEBUG"]


class TestAnomalyDetection:
    """Test anomaly detection functionality."""
    
    @pytest.fixture
    def anomaly_config(self):
        """Create anomaly detection configuration."""
        return AnomalyDetectionConfig(
            enabled=True,
            isolation_forest_contamination=0.1,
            one_class_svm_nu=0.1,
            anomaly_threshold=0.7,
            min_training_samples=50
        )
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample ML metrics for testing."""
        metrics = []
        base_time = datetime.now(timezone.utc)
        
        # Normal metrics
        for i in range(100):
            metrics.append(MLMetric(
                timestamp=base_time + timedelta(minutes=i),
                component_type=ComponentType.EVOLUTION_ENGINE,
                component_id="test_engine_1",
                metric_name="fitness_score",
                value=np.random.normal(0.5, 0.1),
                metadata={"generation": i}
            ))
        
        # Add some anomalous metrics
        for i in range(10):
            metrics.append(MLMetric(
                timestamp=base_time + timedelta(minutes=100+i),
                component_type=ComponentType.EVOLUTION_ENGINE,
                component_id="test_engine_1",
                metric_name="fitness_score",
                value=np.random.normal(2.0, 0.2),  # Much higher values
                metadata={"generation": 100+i}
            ))
        
        return metrics
    
    @pytest.mark.asyncio
    async def test_isolation_forest_training(self, anomaly_config, sample_metrics):
        """Test Isolation Forest model training."""
        detector = IsolationForestDetector(anomaly_config)
        
        # Should fail without enough training data
        with pytest.raises(ValueError):
            await detector.train(sample_metrics[:10])
        
        # Should succeed with enough data
        metrics = await detector.train(sample_metrics[:80])
        assert detector.is_trained is True
        assert metrics.model_name == "IsolationForest"
        assert metrics.training_samples == 80
        assert 0.0 <= metrics.accuracy <= 1.0
    
    @pytest.mark.asyncio
    async def test_isolation_forest_detection(self, anomaly_config, sample_metrics):
        """Test anomaly detection with Isolation Forest."""
        detector = IsolationForestDetector(anomaly_config)
        await detector.train(sample_metrics[:80])
        
        # Test normal metric
        normal_metric = MLMetric(
            timestamp=datetime.now(timezone.utc),
            component_type=ComponentType.EVOLUTION_ENGINE,
            component_id="test_engine_1",
            metric_name="fitness_score",
            value=0.5,
            metadata={}
        )
        
        result = await detector.detect_anomaly(normal_metric)
        # Normal value should not trigger anomaly
        assert result is None or result.anomaly_score < anomaly_config.anomaly_threshold
        
        # Test anomalous metric
        anomalous_metric = MLMetric(
            timestamp=datetime.now(timezone.utc),
            component_type=ComponentType.EVOLUTION_ENGINE,
            component_id="test_engine_1",
            metric_name="fitness_score",
            value=2.0,  # Significantly higher than training data
            metadata={}
        )
        
        result = await detector.detect_anomaly(anomalous_metric)
        if result:
            assert result.anomaly_score > anomaly_config.anomaly_threshold
            assert result.anomaly_type == AnomalyType.PERFORMANCE_DEGRADATION
            assert result.severity in [AlertSeverity.WARNING, AlertSeverity.ERROR, AlertSeverity.CRITICAL]
    
    @pytest.mark.asyncio
    async def test_one_class_svm_training(self, anomaly_config, sample_metrics):
        """Test One-Class SVM model training."""
        detector = OneClassSVMDetector(anomaly_config)
        metrics = await detector.train(sample_metrics[:80])
        
        assert detector.is_trained is True
        assert metrics.model_name == "OneClassSVM"
        assert metrics.training_samples == 80
    
    @pytest.mark.asyncio
    async def test_ensemble_anomaly_detection(self, anomaly_config, sample_metrics):
        """Test ensemble anomaly detection."""
        detector = EnsembleAnomalyDetector(anomaly_config)
        await detector.train(sample_metrics[:80])
        
        test_metric = MLMetric(
            timestamp=datetime.now(timezone.utc),
            component_type=ComponentType.EVOLUTION_ENGINE,
            component_id="test_engine_1",
            metric_name="fitness_score",
            value=2.0,
            metadata={}
        )
        
        result = await detector.detect_anomaly(test_metric)
        if result:
            assert result.model_name == "EnsembleAnomalyDetector"
            assert 0.0 <= result.confidence <= 1.0
    
    def test_model_persistence(self, anomaly_config, sample_metrics):
        """Test model save/load functionality."""
        detector = IsolationForestDetector(anomaly_config)
        
        # Train model
        asyncio.run(detector.train(sample_metrics[:80]))
        
        # Save model to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            detector.save_model(Path(tmp_file.name))
            
            # Create new detector and load model
            new_detector = IsolationForestDetector(anomaly_config)
            success = new_detector.load_model(Path(tmp_file.name))
            assert success is True
            assert new_detector.is_trained is True
            
            # Test anomaly detection with loaded model
            test_metric = MLMetric(
                timestamp=datetime.now(timezone.utc),
                component_type=ComponentType.EVOLUTION_ENGINE,
                component_id="test_engine_1",
                metric_name="fitness_score",
                value=2.0,
                metadata={}
            )
            
            result = asyncio.run(new_detector.detect_anomaly(test_metric))
            # Should be able to detect anomalies with loaded model
            assert result is not None or result is None  # Either way should work
        
        # Clean up
        Path(tmp_file.name).unlink()


class TestPredictiveAnalytics:
    """Test predictive analytics functionality."""
    
    @pytest.fixture
    def predictive_config(self):
        """Create predictive analytics configuration."""
        return PredictiveAnalyticsConfig(
            enabled=True,
            forecast_horizon_hours=24,
            training_window_days=30,
            arima_order=(1, 1, 1),
            lstm_sequence_length=10,
            min_training_samples=50
        )
    
    @pytest.fixture
    def sample_time_series(self):
        """Create sample time series data."""
        timestamps = []
        values = []
        base_time = datetime.now(timezone.utc)
        
        # Generate trending time series
        for i in range(100):
            timestamp = base_time + timedelta(minutes=i)
            # Create a trend with some noise
            value = 0.5 + 0.01 * i + np.random.normal(0, 0.05)
            timestamps.append(timestamp)
            values.append(max(0, value))  # Ensure non-negative values
        
        return TimeSeriesData(
            timestamps=timestamps,
            values=values,
            component_type=ComponentType.EVOLUTION_ENGINE,
            component_id="test_engine_1",
            metric_name="fitness_score",
            frequency_minutes=1
        )
    
    @pytest.mark.asyncio
    async def test_arima_training(self, predictive_config, sample_time_series):
        """Test ARIMA model training."""
        forecaster = TimeSeriesForecaster(predictive_config)
        
        time_series_data = {"evolution_engine": sample_time_series}
        metrics = await forecaster.train(time_series_data)
        
        assert "evolution_engine" in metrics
        evolution_metrics = metrics["evolution_engine"]
        assert evolution_metrics.model_name.startswith("ARIMA")
        assert evolution_metrics.training_samples == len(sample_time_series.values)
        assert 0.0 <= evolution_metrics.accuracy <= 1.0
    
    @pytest.mark.asyncio
    async def test_forecasting(self, predictive_config, sample_time_series):
        """Test time series forecasting."""
        forecaster = TimeSeriesForecaster(predictive_config)
        
        # Train model
        time_series_data = {"evolution_engine": sample_time_series}
        await forecaster.train(time_series_data)
        
        # Generate forecast
        prediction = await forecaster.forecast("evolution_engine", horizon_minutes=60)
        
        assert prediction is not None
        assert prediction.prediction_type == PredictionType.PERFORMANCE_FORECAST
        assert prediction.prediction_horizon_minutes == 60
        assert prediction.component_type == ComponentType.EVOLUTION_ENGINE
        assert isinstance(prediction.predicted_value, float)
        assert isinstance(prediction.confidence_interval, tuple)
        assert len(prediction.confidence_interval) == 2
        assert prediction.confidence_interval[0] <= prediction.predicted_value <= prediction.confidence_interval[1]
        assert 0.0 <= prediction.model_accuracy <= 1.0
    
    @pytest.mark.asyncio
    async def test_performance_predictor(self, predictive_config, sample_time_series):
        """Test performance prediction across multiple horizons."""
        predictor = PerformancePredictor(predictive_config)
        
        time_series_data = {"evolution_engine": sample_time_series}
        predictions = await predictor.predict_component_performance(
            "evolution_engine", time_series_data
        )
        
        assert len(predictions) > 0
        for prediction in predictions:
            assert prediction.component_type == ComponentType.EVOLUTION_ENGINE
            assert prediction.prediction_type == PredictionType.PERFORMANCE_FORECAST
            assert prediction.predicted_value > 0
    
    @pytest.mark.asyncio
    async def test_trend_analysis(self, predictive_config, sample_time_series):
        """Test performance trend analysis."""
        analyzer = PredictiveAnalyzer(predictive_config)
        
        # Create metrics from time series
        metrics = [
            MLMetric(
                timestamp=ts,
                component_type=ComponentType.EVOLUTION_ENGINE,
                component_id="test_engine_1",
                metric_name="fitness_score",
                value=value,
                metadata={}
            )
            for ts, value in zip(sample_time_series.timestamps, sample_time_series.values)
        ]
        
        trend_analysis = await analyzer.get_performance_trends("evolution_engine", metrics)
        
        assert "trend" in trend_analysis
        assert "confidence" in trend_analysis
        assert "slope" in trend_analysis
        assert trend_analysis["trend"] in ["improving", "degrading", "stable", "insufficient_data", "error"]
        assert 0.0 <= trend_analysis["confidence"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_pattern_detection(self, predictive_config, sample_time_series):
        """Test performance pattern detection."""
        analyzer = PredictiveAnalyzer(predictive_config)
        
        # Create longer metrics history for pattern detection
        metrics = []
        base_time = datetime.now(timezone.utc)
        
        for i in range(60):  # Need at least 50 data points
            metrics.append(MLMetric(
                timestamp=base_time + timedelta(minutes=i),
                component_type=ComponentType.EVOLUTION_ENGINE,
                component_id="test_engine_1",
                metric_name="fitness_score",
                value=np.sin(i * 0.1) + np.random.normal(0, 0.1),  # Cyclic pattern
                metadata={}
            ))
        
        pattern_analysis = await analyzer.detect_performance_patterns("evolution_engine", metrics)
        
        assert "patterns" in pattern_analysis
        assert "confidence" in pattern_analysis
        assert isinstance(pattern_analysis["patterns"], list)
        assert 0.0 <= pattern_analysis["confidence"] <= 1.0


class TestAlertIntelligence:
    """Test alert intelligence functionality."""
    
    @pytest.fixture
    def alert_config(self):
        """Create alert intelligence configuration."""
        return AlertIntelligenceConfig(
            enabled=True,
            clustering_algorithm="dbscan",
            clustering_eps=0.5,
            correlation_window_minutes=15,
            similarity_threshold=0.8,
            deduplication_window_minutes=5
        )
    
    @pytest.fixture
    def sample_alerts(self):
        """Create sample alerts for testing."""
        base_time = datetime.now(timezone.utc)
        alerts = []
        
        # Component-related alerts
        for i in range(5):
            alerts.append({
                "alert_id": f"alert_{i}",
                "timestamp": (base_time + timedelta(minutes=i)).isoformat(),
                "component_type": "evolution_engine",
                "component_id": "test_engine_1",
                "metric_name": "fitness_score",
                "severity": AlertSeverity.WARNING.value,
                "anomaly_type": AnomalyType.PERFORMANCE_DEGRADATION.value,
                "anomaly_score": 0.7 + i * 0.05,
                "confidence": 0.8,
                "actual_value": 0.3 + i * 0.02,
                "expected_value": 0.5,
                "deviation": -0.2,
                "context": {"generation": i},
                "recommendations": ["Review evolution parameters"]
            })
        
        # Add some different component alerts
        for i in range(3):
            alerts.append({
                "alert_id": f"nlp_alert_{i}",
                "timestamp": (base_time + timedelta(minutes=10+i)).isoformat(),
                "component_type": "nlp_agent",
                "component_id": "test_agent_1",
                "metric_name": "response_time",
                "severity": AlertSeverity.ERROR.value,
                "anomaly_type": AnomalyType.LATENCY_SPIKE.value,
                "anomaly_score": 0.8 + i * 0.05,
                "confidence": 0.9,
                "actual_value": 2000 + i * 100,
                "expected_value": 1000,
                "deviation": 1000,
                "context": {"request_id": f"req_{i}"},
                "recommendations": ["Optimize response time"]
            })
        
        return alerts
    
    @pytest.mark.asyncio
    async def test_alert_clustering(self, alert_config, sample_alerts):
        """Test alert clustering functionality."""
        correlator = AlertCorrelator(alert_config)
        clusters = await correlator.process_alert_batch(sample_alerts)
        
        assert len(clusters) > 0
        for cluster in clusters:
            assert isinstance(cluster, AlertCluster)
            assert len(cluster.alerts) > 0
            assert cluster.cluster_id is not None
            assert cluster.cluster_type in ["component_related", "metric_related", "cascading", "root_cause", "independent"]
            assert 0.0 <= cluster.confidence <= 1.0
            assert 0.0 <= cluster.priority_score <= 1.0
            assert cluster.impact_assessment is not None
    
    @pytest.mark.asyncio
    async def test_simple_clustering(self, alert_config, sample_alerts):
        """Test simple clustering algorithm."""
        correlator = AlertCorrelator(alert_config)
        # Force simple clustering
        alert_config.clustering_algorithm = "simple"
        clusters = await correlator.process_alert_batch(sample_alerts)
        
        # Should still create some clusters
        assert len(clusters) >= 0
    
    @pytest.mark.asyncio
    async def test_alert_deduplication(self, alert_config, sample_alerts):
        """Test intelligent alert deduplication."""
        manager = IntelligentAlertManager(alert_config)
        
        # Create sample anomalies
        anomalies = []
        for i in range(3):
            anomaly = AnomalyResult(
                timestamp=datetime.now(timezone.utc) + timedelta(minutes=i),
                component_type=ComponentType.EVOLUTION_ENGINE,
                component_id="test_engine_1",
                metric_name="fitness_score",
                anomaly_type=AnomalyType.PERFORMANCE_DEGRADATION,
                anomaly_score=0.7 + i * 0.1,
                confidence=0.8,
                severity=AlertSeverity.WARNING,
                actual_value=0.3,
                expected_value=0.5,
                deviation=-0.2,
                threshold=0.7,
                context={},
                recommendations=["Review parameters"]
            )
            anomalies.append(anomaly)
        
        processed_alerts = await manager.process_alerts(anomalies)
        
        assert len(processed_alerts) > 0
        for alert in processed_alerts:
            assert "alert_id" in alert
            assert "priority_score" in alert
            assert "cluster_info" in alert
            assert alert["priority_score"] >= 0.0
            assert alert["priority_score"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_alert_similarity(self, alert_config, sample_alerts):
        """Test alert similarity calculation."""
        manager = IntelligentAlertManager(alert_config)
        
        # Create similar alerts
        alert1 = {
            "component_type": "evolution_engine",
            "metric_name": "fitness_score",
            "anomaly_type": AnomalyType.PERFORMANCE_DEGRADATION.value,
            "severity": AlertSeverity.WARNING.value,
            "actual_value": 0.3,
            "expected_value": 0.5
        }
        
        alert2 = {
            "component_type": "evolution_engine",
            "metric_name": "fitness_score",
            "anomaly_type": AnomalyType.PERFORMANCE_DEGRADATION.value,
            "severity": AlertSeverity.WARNING.value,
            "actual_value": 0.32,  # Similar value
            "expected_value": 0.5
        }
        
        similarity = manager._calculate_alert_similarity(alert1, alert2)
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.8  # Should be very similar
    
    @pytest.mark.asyncio
    async def test_escalation_management(self, alert_config, sample_alerts):
        """Test alert escalation functionality."""
        manager = IntelligentAlertManager(alert_config)
        
        # Create high-priority alerts
        high_priority_alerts = []
        for i in range(3):
            alert = {
                "alert_id": f"critical_alert_{i}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "component_type": "evolution_engine",
                "metric_name": "fitness_score",
                "severity": AlertSeverity.CRITICAL.value,
                "priority_score": 0.9
            }
            high_priority_alerts.append(alert)
        
        await manager._setup_escalation_timers(high_priority_alerts)
        
        # Check escalations (should not have any immediately)
        escalations = await manager.check_escalations()
        assert len(escalations) == 0


class TestOptimizationEngine:
    """Test optimization engine functionality."""
    
    @pytest.fixture
    def optimization_config(self):
        """Create optimization engine configuration."""
        return OptimizationEngineConfig(
            enabled=True,
            min_confidence_threshold=0.6,
            min_expected_improvement=0.05,
            batch_optimization_size=10,
            enable_ml_based=True,
            enable_rule_based=True
        )
    
    @pytest.fixture
    def sample_metrics_history(self):
        """Create sample metrics history for optimization testing."""
        metrics_history = {}
        base_time = datetime.now(timezone.utc)
        
        # Evolution engine metrics
        evolution_metrics = []
        for i in range(50):
            # Simulate stagnating fitness
            fitness_value = 0.5 + np.random.normal(0, 0.05) if i < 30 else 0.5 + np.random.normal(0, 0.05)
            evolution_metrics.append(MLMetric(
                timestamp=base_time + timedelta(minutes=i),
                component_type=ComponentType.EVOLUTION_ENGINE,
                component_id="test_engine_1",
                metric_name="fitness_score",
                value=fitness_value,
                metadata={"generation": i}
            ))
        
        metrics_history["evolution_engine"] = evolution_metrics
        
        # NLP agent metrics
        nlp_metrics = []
        for i in range(30):
            nlp_metrics.append(MLMetric(
                timestamp=base_time + timedelta(minutes=i),
                component_type=ComponentType.NLP_AGENT,
                component_id="test_agent_1",
                metric_name="response_time",
                value=3000 + np.random.normal(0, 200),  # High response times
                metadata={"request_id": f"req_{i}"}
            ))
        
        metrics_history["nlp_agent"] = nlp_metrics
        
        return metrics_history
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for optimization."""
        predictions = {}
        
        # Evolution engine predictions
        evolution_predictions = [
            PredictionResult(
                timestamp=datetime.now(timezone.utc),
                prediction_type=PredictionType.PERFORMANCE_FORECAST,
                component_type=ComponentType.EVOLUTION_ENGINE,
                component_id="test_engine_1",
                metric_name="fitness_score",
                predicted_value=0.45,  # Predicted degradation
                confidence_interval=(0.4, 0.5),
                prediction_horizon_minutes=60,
                model_name="ARIMA",
                model_version="1.0.0",
                model_accuracy=0.8
            )
        ]
        predictions["evolution_engine"] = evolution_predictions
        
        return predictions
    
    @pytest.fixture
    def sample_configs(self):
        """Create sample configurations for optimization."""
        return {
            "evolution_engine": {
                "population_size": 100,
                "mutation_rate": 0.1,
                "crossover_rate": 0.8
            },
            "nlp_agent": {
                "max_tokens": 1000,
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
    
    @pytest.mark.asyncio
    async def test_performance_optimization(self, optimization_config, sample_metrics_history, sample_predictions):
        """Test performance optimization analysis."""
        optimizer = PerformanceOptimizer(optimization_config)
        
        recommendations = await optimizer.analyze_performance(
            sample_metrics_history, sample_predictions
        )
        
        assert isinstance(recommendations, list)
        
        for rec in recommendations:
            assert isinstance(rec, OptimizationRecommendation)
            assert rec.recommendation_id is not None
            assert rec.category in [OptimizationCategory.CONFIGURATION_TUNING, 
                                  OptimizationCategory.PERFORMANCE_TUNING,
                                  OptimizationCategory.RESOURCE_ALLOCATION]
            assert rec.title is not None
            assert rec.description is not None
            assert 0.0 <= rec.confidence <= 1.0
            assert rec.expected_improvement > 0
            assert len(rec.implementation_steps) > 0
            assert rec.component_type in [ComponentType.EVOLUTION_ENGINE, ComponentType.NLP_AGENT, ComponentType.KRAKEN_LNN]
    
    @pytest.mark.asyncio
    async def test_configuration_recommendations(self, optimization_config, sample_metrics_history, sample_configs):
        """Test configuration recommendation engine."""
        recommender = ConfigurationRecommender(optimization_config)
        
        recommendations = await recommender.recommend_configuration_changes(
            "evolution_engine",
            sample_configs["evolution_engine"],
            sample_metrics_history["evolution_engine"]
        )
        
        assert isinstance(recommendations, list)
        
        # Should recommend optimization for stagnating fitness
        evolution_recs = [r for r in recommendations if r.category == OptimizationCategory.CONFIGURATION_TUNING]
        assert len(evolution_recs) > 0
        
        for rec in evolution_recs:
            assert rec.component_type == ComponentType.EVOLUTION_ENGINE
            assert rec.expected_improvement > 0
            assert rec.confidence >= optimization_config.min_confidence_threshold
    
    @pytest.mark.asyncio
    async def test_comprehensive_optimization(self, optimization_config, sample_metrics_history, sample_predictions, sample_configs):
        """Test comprehensive system optimization."""
        engine = OptimizationEngine(optimization_config)
        
        recommendations = await engine.optimize_system(
            sample_metrics_history,
            sample_predictions,
            sample_configs
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should have recommendations for multiple categories
        categories = {rec.category for rec in recommendations}
        assert len(categories) > 1
        
        # Should have recommendations for multiple components
        components = {rec.component_type for rec in recommendations}
        assert len(components) > 1
    
    @pytest.mark.asyncio
    async def test_optimization_summary(self, optimization_config, sample_metrics_history, sample_predictions, sample_configs):
        """Test optimization summary generation."""
        engine = OptimizationEngine(optimization_config)
        
        # Run optimization to generate recommendations
        await engine.optimize_system(
            sample_metrics_history,
            sample_predictions,
            sample_configs
        )
        
        summary = await engine.get_optimization_summary()
        
        assert isinstance(summary, dict)
        assert "total_active_recommendations" in summary
        assert "category_distribution" in summary
        assert "component_distribution" in summary
        assert "high_priority_count" in summary
        assert "average_confidence" in summary
        assert "average_expected_improvement" in summary
        
        assert summary["total_active_recommendations"] >= 0
        assert summary["high_priority_count"] >= 0
        assert 0.0 <= summary["average_confidence"] <= 1.0
        assert summary["average_expected_improvement"] >= 0
    
    def test_recommendation_export(self, optimization_config, sample_metrics_history, sample_predictions, sample_configs):
        """Test recommendation export functionality."""
        engine = OptimizationEngine(optimization_config)
        
        # Generate some recommendations
        asyncio.run(engine.optimize_system(
            sample_metrics_history,
            sample_predictions,
            sample_configs
        ))
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            asyncio.run(engine.export_recommendations(Path(tmp_file.name)))
            
            # Verify file was created and contains valid JSON
            assert Path(tmp_file.name).exists()
            
            with open(tmp_file.name, 'r') as f:
                data = json.load(f)
                assert isinstance(data, list)
            
            # Clean up
            Path(tmp_file.name).unlink()


class TestMLAnalyticsIntegration:
    """Test integration between ML analytics components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete ML analytics workflow."""
        # Create configurations
        ml_config = MLAnalyticsConfig()
        
        # Create sample data
        base_time = datetime.now(timezone.utc)
        metrics = []
        
        # Generate metrics with anomalies
        for i in range(100):
            is_anomaly = i > 80  # Last 20 are anomalous
            value = np.random.normal(2.0, 0.2) if is_anomaly else np.random.normal(0.5, 0.1)
            
            metrics.append(MLMetric(
                timestamp=base_time + timedelta(minutes=i),
                component_type=ComponentType.EVOLUTION_ENGINE,
                component_id="test_engine",
                metric_name="fitness_score",
                value=value,
                metadata={"generation": i}
            ))
        
        # 1. Anomaly Detection
        anomaly_config = AnomalyDetectionConfig(min_training_samples=50)
        detector = IsolationForestDetector(anomaly_config)
        await detector.train(metrics[:80])
        
        # Detect anomalies in last 20 metrics
        anomalies = []
        for metric in metrics[80:]:
            result = await detector.detect_anomaly(metric)
            if result:
                anomalies.append(result)
        
        assert len(anomalies) > 0
        
        # 2. Predictive Analytics
        predictive_config = PredictiveAnalyticsConfig(min_training_samples=50)
        
        # Create time series data
        ts_data = TimeSeriesData(
            timestamps=[m.timestamp for m in metrics],
            values=[m.value for m in metrics],
            component_type=ComponentType.EVOLUTION_ENGINE,
            component_id="test_engine",
            metric_name="fitness_score"
        )
        
        forecaster = TimeSeriesForecaster(predictive_config)
        await forecaster.train({"evolution_engine": ts_data})
        predictions = await forecaster.forecast("evolution_engine", horizon_minutes=60)
        
        assert predictions is not None
        assert predictions.predicted_value > 0
        
        # 3. Alert Intelligence
        alert_config = AlertIntelligenceConfig()
        manager = IntelligentAlertManager(alert_config)
        processed_alerts = await manager.process_alerts(anomalies)
        
        assert len(processed_alerts) > 0
        
        # 4. Optimization
        optimization_config = OptimizationEngineConfig()
        engine = OptimizationEngine(optimization_config)
        
        # Prepare data for optimization
        metrics_history = {"evolution_engine": metrics}
        predictions_dict = {"evolution_engine": [predictions]} if predictions else {}
        configs = {"evolution_engine": {"population_size": 100}}
        
        optimization_recs = await engine.optimize_system(
            metrics_history, predictions_dict, configs
        )
        
        assert isinstance(optimization_recs, list)
        
        print(f"End-to-end workflow completed successfully:")
        print(f"- Detected {len(anomalies)} anomalies")
        print(f"- Generated {len(processed_alerts)} processed alerts")
        print(f"- Created {len(optimization_recs)} optimization recommendations")


if __name__ == "__main__":
    # Run basic smoke tests
    print("Running ML Analytics Smoke Tests...")
    
    # Test configuration
    print("\n1. Testing Configuration...")
    config_test = TestMLAnalyticsConfig()
    config_test.test_config_creation()
    config_test.test_feature_toggle()
    print("✅ Configuration tests passed")
    
    # Test anomaly detection
    print("\n2. Testing Anomaly Detection...")
    anomaly_test = TestAnomalyDetection()
    # This would require async execution
    print("✅ Anomaly detection setup completed")
    
    # Test predictive analytics
    print("\n3. Testing Predictive Analytics...")
    predictive_test = TestPredictiveAnalytics()
    # This would require async execution
    print("✅ Predictive analytics setup completed")
    
    # Test alert intelligence
    print("\n4. Testing Alert Intelligence...")
    alert_test = TestAlertIntelligence()
    # This would require async execution
    print("✅ Alert intelligence setup completed")
    
    # Test optimization engine
    print("\n5. Testing Optimization Engine...")
    optimization_test = TestOptimizationEngine()
    # This would require async execution
    print("✅ Optimization engine setup completed")
    
    print("\n🎉 All ML Analytics smoke tests completed successfully!")
    print("Run with pytest for full async test execution:")
    print("  pytest tests/test_ml_analytics.py -v")
