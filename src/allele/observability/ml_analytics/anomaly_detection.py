# Copyright (C) 2025 Bravetto AI Systems & Jimmy De Jesus
#
# This file is part of Allele.
#
# Allele is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of License, or
# (at your option) any later version.
#
# Allele is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General License for more details.
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

"""Anomaly Detection Engine for Allele ML Analytics.

This module provides machine learning-based anomaly detection using
Isolation Forest and One-Class SVM algorithms.

Author: Bravetto AI Systems
Version: 1.0.0
"""

import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

from .ml_config import AnomalyDetectionConfig
from .types import AnomalyResult, AnomalyType, MLMetric, ModelMetrics, ModelStatus

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Base anomaly detection interface."""

    def __init__(self, config: AnomalyDetectionConfig):
        """Initialize anomaly detector.
        
        Args:
            config: Anomaly detection configuration
        """
        self.config = config
        self.is_trained = False
        self.model = None
        self.scaler = None
        self.training_data = []
        self.last_training_time = None

    async def train(self, training_data: List[MLMetric]) -> ModelMetrics:
        """Train the anomaly detection model.
        
        Args:
            training_data: List of ML metrics for training
            
        Returns:
            Model training metrics
        """
        raise NotImplementedError

    async def detect_anomaly(self, metric: MLMetric) -> Optional[AnomalyResult]:
        """Detect anomaly in a single metric.
        
        Args:
            metric: ML metric to analyze
            
        Returns:
            Anomaly result if anomaly detected, None otherwise
        """
        raise NotImplementedError

    async def detect_anomalies_batch(self, metrics: List[MLMetric]) -> List[AnomalyResult]:
        """Detect anomalies in a batch of metrics.
        
        Args:
            metrics: List of ML metrics to analyze
            
        Returns:
            List of anomaly results
        """
        results = []
        for metric in metrics:
            result = await self.detect_anomaly(metric)
            if result:
                results.append(result)
        return results

    def save_model(self, filepath: Path) -> None:
        """Save trained model to file.
        
        Args:
            filepath: Path to save model
        """
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "config": self.config,
            "is_trained": self.is_trained,
            "last_training_time": self.last_training_time,
            "training_data_count": len(self.training_data)
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Anomaly detection model saved to {filepath}")

    def load_model(self, filepath: Path) -> bool:
        """Load trained model from file.
        
        Args:
            filepath: Path to model file
            
        Returns:
            True if model loaded successfully
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.is_trained = model_data["is_trained"]
            self.last_training_time = model_data["last_training_time"]

            logger.info(f"Anomaly detection model loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model from {filepath}: {e}")
            return False

    def _prepare_features(self, metrics: List[MLMetric]) -> np.ndarray:
        """Prepare feature matrix from metrics.
        
        Args:
            metrics: List of ML metrics
            
        Returns:
            Feature matrix
        """
        features = []
        for metric in metrics:
            feature_vector = metric.to_vector()
            features.append(feature_vector)

        return np.array(features)

    def _scale_features(self, features: np.ndarray) -> np.ndarray:
        """Scale features using fitted scaler.
        
        Args:
            features: Feature matrix
            
        Returns:
            Scaled feature matrix
        """
        if self.scaler is None:
            # Create and fit scaler
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(features)
        else:
            return self.scaler.transform(features)

    def _calculate_anomaly_score(self, features: np.ndarray) -> np.ndarray:
        """Calculate anomaly scores using the model.
        
        Args:
            features: Feature matrix
            
        Returns:
            Anomaly scores
        """
        if hasattr(self.model, 'decision_function'):
            # For One-Class SVM
            scores = self.model.decision_function(features)
            # Convert to anomaly scores (higher = more anomalous)
            return -scores  # Negative scores become positive anomalies
        elif hasattr(self.model, 'score_samples'):
            # For Isolation Forest
            scores = self.model.score_samples(features)
            # Lower scores indicate anomalies
            return -scores
        else:
            raise ValueError("Model does not support anomaly scoring")

    def _determine_anomaly_type(self, metric_name: str, anomaly_score: float) -> AnomalyType:
        """Determine type of anomaly based on metric name and score.
        
        Args:
            metric_name: Name of the metric
            anomaly_score: Anomaly score
            
        Returns:
            Type of anomaly
        """
        metric_lower = metric_name.lower()

        if "latency" in metric_lower or "response_time" in metric_lower:
            return AnomalyType.LATENCY_SPIKE
        elif "cpu" in metric_lower or "processor" in metric_lower:
            return AnomalyType.CPU_OVERLOAD
        elif "memory" in metric_lower:
            return AnomalyType.MEMORY_LEAK
        elif "error" in metric_lower or "failure" in metric_lower:
            return AnomalyType.ERROR_RATE_SPIKE
        elif "throughput" in metric_lower or "ops" in metric_lower:
            return AnomalyType.THROUGHPUT_DROP
        elif "fitness" in metric_lower:
            return AnomalyType.PERFORMANCE_DEGRADATION
        else:
            return AnomalyType.UNUSUAL_PATTERN


class IsolationForestDetector(AnomalyDetector):
    """Anomaly detector using Isolation Forest algorithm."""

    def __init__(self, config: AnomalyDetectionConfig):
        """Initialize Isolation Forest detector.
        
        Args:
            config: Anomaly detection configuration
        """
        super().__init__(config)
        self.model_name = "IsolationForest"
        self.model_version = "1.0.0"

    async def train(self, training_data: List[MLMetric]) -> ModelMetrics:
        """Train Isolation Forest model.
        
        Args:
            training_data: List of ML metrics for training
            
        Returns:
            Model training metrics
        """
        try:
            # Check if scikit-learn is available
            from sklearn.ensemble import IsolationForest

            if len(training_data) < self.config.min_training_samples:
                raise ValueError(
                    f"Insufficient training data: {len(training_data)} < "
                    f"{self.config.min_training_samples}"
                )

            # Prepare features
            features = self._prepare_features(training_data)
            scaled_features = self._scale_features(features)

            # Train Isolation Forest
            self.model = IsolationForest(
                contamination=self.config.isolation_forest_contamination,
                n_estimators=self.config.isolation_forest_estimators,
                random_state=42,
                n_jobs=-1
            )

            self.model.fit(scaled_features)
            self.is_trained = True
            self.training_data = training_data
            self.last_training_time = datetime.now(timezone.utc)

            # Calculate training metrics
            anomaly_scores = self._calculate_anomaly_scores(scaled_features)
            anomaly_count = np.sum(anomaly_scores > 0)

            training_metrics = ModelMetrics(
                model_name=self.model_name,
                model_version=self.model_version,
                status=ModelStatus.READY,
                training_samples=len(training_data),
                validation_samples=0,
                last_training_time=self.last_training_time,
                accuracy=1.0 - anomaly_count / len(training_data),  # Accuracy as inverse of anomaly rate
                total_predictions=0,
                successful_predictions=0
            )

            logger.info(
                f"Isolation Forest model trained with {len(training_data)} samples, "
                f"detected {anomaly_count} anomalies"
            )

            return training_metrics

        except ImportError:
            raise ImportError(
                "scikit-learn is required for Isolation Forest anomaly detection. "
                "Install with: pip install scikit-learn"
            )
        except Exception as e:
            logger.error(f"Isolation Forest training failed: {e}")
            raise

    async def detect_anomaly(self, metric: MLMetric) -> Optional[AnomalyResult]:
        """Detect anomaly using Isolation Forest.
        
        Args:
            metric: ML metric to analyze
            
        Returns:
            Anomaly result if anomaly detected, None otherwise
        """
        if not self.is_trained:
            return None

        try:
            # Prepare features
            features = self._prepare_features([metric])
            scaled_features = self._scale_features(features)

            # Get anomaly score
            anomaly_score = self._calculate_anomaly_score(scaled_features)[0]

            # Check if anomaly
            component_threshold = self.config.component_thresholds.get(
                metric.component_type.value, self.config.anomaly_threshold
            )

            if anomaly_score > component_threshold:
                # Calculate expected value (use recent training data average)
                expected_value = self._calculate_expected_value(metric)
                confidence = min(anomaly_score / (component_threshold * 2), 1.0)

                anomaly_result = AnomalyResult(
                    timestamp=metric.timestamp,
                    component_type=metric.component_type,
                    component_id=metric.component_id,
                    metric_name=metric.metric_name,
                    anomaly_type=self._determine_anomaly_type(metric.metric_name, anomaly_score),
                    anomaly_score=anomaly_score,
                    confidence=confidence,
                    severity=None,  # Will be calculated in __post_init__
                    actual_value=metric.value,
                    expected_value=expected_value,
                    deviation=metric.value - expected_value,
                    threshold=component_threshold,
                    context=metric.metadata,
                    recommendations=self._generate_recommendations(metric, anomaly_score),
                    model_name=self.model_name,
                    model_version=self.model_version
                )

                return anomaly_result

            return None

        except Exception as e:
            logger.error(f"Anomaly detection failed for metric {metric.metric_name}: {e}")
            return None

    def _calculate_anomaly_scores(self, features: np.ndarray) -> np.ndarray:
        """Calculate anomaly scores for features.
        
        Args:
            features: Feature matrix
            
        Returns:
            Anomaly scores
        """
        return self._calculate_anomaly_score(features)

    def _calculate_expected_value(self, metric: MLMetric) -> float:
        """Calculate expected value for metric based on recent data.
        
        Args:
            metric: ML metric
            
        Returns:
            Expected value
        """
        # Use recent training data to calculate expected value
        if not self.training_data:
            return metric.value  # Fallback to actual value

        # Filter training data for same metric type
        similar_metrics = [
            m for m in self.training_data
            if m.component_type == metric.component_type and m.metric_name == metric.metric_name
        ]

        if similar_metrics:
            # Calculate mean of similar metrics
            values = [m.value for m in similar_metrics[-50:]]  # Use last 50 similar metrics
            return float(np.mean(values))
        else:
            # Use overall mean
            values = [m.value for m in self.training_data[-100:]]  # Use last 100 metrics
            return float(np.mean(values)) if values else metric.value

    def _generate_recommendations(self, metric: MLMetric, anomaly_score: float) -> List[str]:
        """Generate recommendations based on anomaly.
        
        Args:
            metric: Metric that triggered anomaly
            anomaly_score: Anomaly score
            
        Returns:
            List of recommendations
        """
        recommendations = []

        # Component-specific recommendations
        if metric.component_type.value == "evolution_engine":
            recommendations.append("Consider adjusting evolution parameters (population size, mutation rate)")
            recommendations.append("Review fitness function effectiveness")

        elif metric.component_type.value == "kraken_lnn":
            recommendations.append("Check reservoir size and connectivity settings")
            recommendations.append("Monitor memory buffer utilization")

        elif metric.component_type.value == "nlp_agent":
            recommendations.append("Review LLM configuration and API response times")
            recommendations.append("Check conversation context management")

        # Metric-specific recommendations
        metric_lower = metric.metric_name.lower()
        if "latency" in metric_lower:
            recommendations.append("Investigate performance bottlenecks")
            recommendations.append("Consider scaling resources")
        elif "memory" in metric_lower:
            recommendations.append("Review memory allocation and garbage collection")
            recommendations.append("Check for memory leaks")
        elif "cpu" in metric_lower:
            recommendations.append("Optimize CPU-intensive operations")
            recommendations.append("Consider load balancing")

        return recommendations[:3]  # Limit to 3 recommendations


class OneClassSVMDetector(AnomalyDetector):
    """Anomaly detector using One-Class SVM algorithm."""

    def __init__(self, config: AnomalyDetectionConfig):
        """Initialize One-Class SVM detector.
        
        Args:
            config: Anomaly detection configuration
        """
        super().__init__(config)
        self.model_name = "OneClassSVM"
        self.model_version = "1.0.0"

    async def train(self, training_data: List[MLMetric]) -> ModelMetrics:
        """Train One-Class SVM model.
        
        Args:
            training_data: List of ML metrics for training
            
        Returns:
            Model training metrics
        """
        try:
            # Check if scikit-learn is available
            from sklearn.svm import OneClassSVM

            if len(training_data) < self.config.min_training_samples:
                raise ValueError(
                    f"Insufficient training data: {len(training_data)} < "
                    f"{self.config.min_training_samples}"
                )

            # Prepare features
            features = self._prepare_features(training_data)
            scaled_features = self._scale_features(features)

            # Train One-Class SVM
            self.model = OneClassSVM(
                nu=self.config.one_class_svm_nu,
                kernel=self.config.one_class_svm_kernel,
                gamma='scale'
            )

            self.model.fit(scaled_features)
            self.is_trained = True
            self.training_data = training_data
            self.last_training_time = datetime.now(timezone.utc)

            # Calculate training metrics
            predictions = self.model.predict(scaled_features)
            anomaly_count = np.sum(predictions == -1)

            training_metrics = ModelMetrics(
                model_name=self.model_name,
                model_version=self.model_version,
                status=ModelStatus.READY,
                training_samples=len(training_data),
                validation_samples=0,
                last_training_time=self.last_training_time,
                accuracy=1.0 - anomaly_count / len(training_data),
                total_predictions=0,
                successful_predictions=0
            )

            logger.info(
                f"One-Class SVM model trained with {len(training_data)} samples, "
                f"detected {anomaly_count} anomalies"
            )

            return training_metrics

        except ImportError:
            raise ImportError(
                "scikit-learn is required for One-Class SVM anomaly detection. "
                "Install with: pip install scikit-learn"
            )
        except Exception as e:
            logger.error(f"One-Class SVM training failed: {e}")
            raise

    async def detect_anomaly(self, metric: MLMetric) -> Optional[AnomalyResult]:
        """Detect anomaly using One-Class SVM.
        
        Args:
            metric: ML metric to analyze
            
        Returns:
            Anomaly result if anomaly detected, None otherwise
        """
        if not self.is_trained:
            return None

        try:
            # Prepare features
            features = self._prepare_features([metric])
            scaled_features = self._scale_features(features)

            # Get prediction and score
            prediction = self.model.predict(scaled_features)[0]
            anomaly_score = self._calculate_anomaly_score(scaled_features)[0]

            # Check if anomaly (SVM returns -1 for anomalies)
            if prediction == -1:
                component_threshold = self.config.component_thresholds.get(
                    metric.component_type.value, self.config.anomaly_threshold
                )

                expected_value = self._calculate_expected_value(metric)
                confidence = min(abs(anomaly_score) / (component_threshold * 2), 1.0)

                anomaly_result = AnomalyResult(
                    timestamp=metric.timestamp,
                    component_type=metric.component_type,
                    component_id=metric.component_id,
                    metric_name=metric.metric_name,
                    anomaly_type=self._determine_anomaly_type(metric.metric_name, anomaly_score),
                    anomaly_score=abs(anomaly_score),
                    confidence=confidence,
                    severity=None,
                    actual_value=metric.value,
                    expected_value=expected_value,
                    deviation=metric.value - expected_value,
                    threshold=component_threshold,
                    context=metric.metadata,
                    recommendations=self._generate_recommendations(metric, anomaly_score),
                    model_name=self.model_name,
                    model_version=self.model_version
                )

                return anomaly_result

            return None

        except Exception as e:
            logger.error(f"One-Class SVM anomaly detection failed: {e}")
            return None

    def _calculate_anomaly_score(self, features: np.ndarray) -> np.ndarray:
        """Calculate anomaly scores for features.
        
        Args:
            features: Feature matrix
            
        Returns:
            Anomaly scores
        """
        return self._calculate_anomaly_score(features)

    def _calculate_expected_value(self, metric: MLMetric) -> float:
        """Calculate expected value for metric."""
        # Similar implementation to Isolation Forest
        if not self.training_data:
            return metric.value

        similar_metrics = [
            m for m in self.training_data
            if m.component_type == metric.component_type and m.metric_name == metric.metric_name
        ]

        if similar_metrics:
            values = [m.value for m in similar_metrics[-50:]]
            return float(np.mean(values))
        else:
            values = [m.value for m in self.training_data[-100:]]
            return float(np.mean(values)) if values else metric.value

    def _generate_recommendations(self, metric: MLMetric, anomaly_score: float) -> List[str]:
        """Generate recommendations based on anomaly."""
        recommendations = []

        # One-Class SVM specific recommendations
        recommendations.append("Review training data for potential outliers")
        recommendations.append("Consider adjusting SVM parameters (nu, kernel)")

        # Add metric-specific recommendations
        if "latency" in metric.metric_name.lower():
            recommendations.append("Investigate performance bottlenecks")
        elif "memory" in metric.metric_name.lower():
            recommendations.append("Review memory allocation patterns")

        return recommendations[:3]


class EnsembleAnomalyDetector(AnomalyDetector):
    """Ensemble anomaly detector combining multiple algorithms."""

    def __init__(self, config: AnomalyDetectionConfig):
        """Initialize ensemble anomaly detector.
        
        Args:
            config: Anomaly detection configuration
        """
        super().__init__(config)
        self.detectors = [
            IsolationForestDetector(config),
            OneClassSVMDetector(config)
        ]
        self.model_name = "EnsembleAnomalyDetector"
        self.model_version = "1.0.0"

    async def train(self, training_data: List[MLMetric]) -> ModelMetrics:
        """Train ensemble of anomaly detection models.
        
        Args:
            training_data: List of ML metrics for training
            
        Returns:
            Model training metrics
        """
        training_metrics = []

        # Train all detectors
        for detector in self.detectors:
            try:
                metrics = await detector.train(training_data)
                training_metrics.append(metrics)
            except Exception as e:
                logger.warning(f"Failed to train detector {detector.model_name}: {e}")

        # Return metrics from first successful detector
        if training_metrics:
            metrics = training_metrics[0].__dict__.copy()
            metrics["model_name"] = self.model_name
            metrics["model_version"] = self.model_version
            return ModelMetrics(**metrics)
        else:
            raise ValueError("No detectors could be trained successfully")

    async def detect_anomaly(self, metric: MLMetric) -> Optional[AnomalyResult]:
        """Detect anomaly using ensemble of detectors.
        
        Args:
            metric: ML metric to analyze
            
        Returns:
            Anomaly result if anomaly detected, None otherwise
        """
        anomaly_results = []
        anomaly_scores = []

        # Get predictions from all detectors
        for detector in self.detectors:
            try:
                result = await detector.detect_anomaly(metric)
                if result:
                    anomaly_results.append(result)
                    anomaly_scores.append(result.anomaly_score)
            except Exception as e:
                logger.warning(f"Detector {detector.model_name} failed: {e}")

        # If multiple detectors agree, it's more likely a real anomaly
        if len(anomaly_scores) >= 2:
            avg_score = np.mean(anomaly_scores)
            confidence = len(anomaly_scores) / len(self.detectors)

            # Use the result with highest confidence
            best_result = max(anomaly_results, key=lambda r: r.confidence)

            # Update with ensemble statistics
            best_result.anomaly_score = avg_score
            best_result.confidence = confidence
            best_result.model_name = self.model_name

            return best_result
        elif anomaly_results:
            return anomaly_results[0]
        else:
            return None
