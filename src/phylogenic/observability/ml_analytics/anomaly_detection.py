# Copyright (C) 2025 Phylogenic AI Labs & Jimmy De Jesus
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
from typing import Any, List, Optional, cast

import numpy as np
from numpy.typing import NDArray

from .ml_config import AnomalyDetectionConfig
from .types import AnomalyResult, AnomalyType, MLMetric, ModelMetrics, ModelStatus

logger = logging.getLogger(__name__)


# Fallback lightweight implementations when scikit-learn is not available.
# These are intentionally simple and only aim to provide predictable
# behavior for unit tests when scikit-learn is not installed in the
# execution environment.
class _SimpleStandardScaler:
    def __init__(self) -> None:
        self.mean_: Optional[NDArray[Any]] = None
        self.scale_: Optional[NDArray[Any]] = None

    def fit_transform(self, X: NDArray[Any]) -> NDArray[Any]:
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        # Avoid div by zero
        self.scale_[self.scale_ == 0] = 1.0
        return (X - self.mean_) / self.scale_

    def transform(self, X: NDArray[Any]) -> NDArray[Any]:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler not fitted")
        return (X - self.mean_) / self.scale_


class _SimpleIsolationForest:
    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self.contamination: float = contamination
        self.random_state: int = random_state
        self._trained: bool = False
        self._center: Optional[NDArray[Any]] = None

    def fit(self, X: NDArray[Any]) -> None:
        # Simple robust center: median
        self._center = np.median(X, axis=0)
        self._trained = True

    def score_samples(self, X: NDArray[Any]) -> NDArray[Any]:
        # Smaller scores for points far from center
        if not self._trained:
            raise ValueError("Model not trained")

        dists = np.linalg.norm(X - self._center, axis=1)

        # Use median absolute deviation (MAD) for robust scaling
        mad = float(np.median(np.abs(dists - np.median(dists))))
        if mad == 0:
            mad = float(np.mean(dists)) if float(np.mean(dists)) > 0 else 1.0

        normalized = dists / mad

        # Convert to score where higher values indicate more "normal" samples
        # (bounded in (0,1])
        scores = 1.0 / (1.0 + normalized)
        return scores


class _SimpleOneClassSVM:
    def __init__(self, nu: float = 0.1, kernel: str = "rbf", gamma: str = "scale") -> None:
        self.nu: float = nu
        self._trained: bool = False
        self._center: Optional[NDArray[Any]] = None

    def fit(self, X: NDArray[Any]) -> None:
        self._center = np.mean(X, axis=0)
        self._trained = True

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not self._trained:
            raise ValueError("Model not trained")
        dists = np.linalg.norm(X - self._center, axis=1)
        # Higher values indicate more inlier-ness; invert for anomaly scoring
        # Mypy can't reliably type numpy ops here; ignore no-any-return
        return cast(NDArray[Any], -(dists / (np.mean(dists) + np.std(dists) + 1e-6)))  # type: ignore[no-any-return]

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Predict returns 1 for inliers and -1 for outliers
        df = self.decision_function(X)
        # Use median as a threshold for a simple deterministic cutoff
        thresh = np.median(df)
        return cast(NDArray[Any], np.where(df >= thresh, 1, -1))  # type: ignore[no-any-return]


class AnomalyDetector:
    """Base anomaly detection interface."""

    def __init__(self, config: AnomalyDetectionConfig) -> None:
        """Initialize anomaly detector.

        Args:
            config: Anomaly detection configuration
        """
        self.config = config
        self.is_trained: bool = False
        self.model: Optional[Any] = None
        self.scaler: Optional[Any] = None
        self.training_data: List[MLMetric] = []
        self.last_training_time: Optional[datetime] = None
        # Reference timestamp used to convert absolute timestamps into
        # relative values during feature preparation. Set during training
        # to make timestamp features robust to large epoch magnitudes.
        self._timestamp_ref: Optional[float] = None

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

    async def detect_anomalies_batch(
        self, metrics: List[MLMetric]
    ) -> List[AnomalyResult]:
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
            "training_data_count": len(self.training_data),
        }

        with open(filepath, "wb") as f:
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
            with open(filepath, "rb") as f:
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

    def _prepare_features(self, metrics: List[MLMetric]) -> NDArray[Any]:
        """Prepare feature matrix from metrics.

        Args:
            metrics: List of ML metrics

        Returns:
            Feature matrix
        """
        features = []
        for metric in metrics:
            # Convert timestamp to relative seconds if a reference exists
            vec = metric.to_vector()
            if self._timestamp_ref is not None:
                # vec[1] is timestamp (seconds since epoch); convert to
                # relative minutes to reduce magnitude and improve scaling
                vec[1] = (metric.timestamp.timestamp() - self._timestamp_ref) / 60.0
            feature_vector = vec
            features.append(feature_vector)

        return np.array(features)

    def _scale_features(self, features: NDArray[Any]) -> NDArray[Any]:
        """Scale features using fitted scaler.

        Args:
            features: Feature matrix

        Returns:
            Scaled feature matrix
        """
        if self.scaler is None:
            # Create and fit scaler; fall back to a simple scaler if sklearn missing
            try:
                from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

                self.scaler = StandardScaler()
            except Exception:
                self.scaler = _SimpleStandardScaler()

            return self.scaler.fit_transform(features)
        else:
            return self.scaler.transform(features)

    def _calculate_anomaly_score(self, features: NDArray[Any]) -> NDArray[Any]:
        """Calculate anomaly scores using the model.

        Args:
            features: Feature matrix

        Returns:
            Anomaly scores
        """
        # If using the simple fallback IsolationForest implementation,
        # compute a robust normalized anomaly score where higher values
        # indicate more anomalous samples (bounded between 0 and 1).
        model = self.model
        if isinstance(model, _SimpleIsolationForest):
            # Prioritize the metric value (first feature) when computing
            # anomaly scores for the simple fallback to make detectors
            # sensitive to changes in metric value despite large
            # timestamp or hashed feature scales.
            try:
                if model._center is None:
                    raise AttributeError("Center not set")
                val_center = model._center[0]
                dists_val = np.abs(features[:, 0] - val_center)
                mad_val = float(np.median(np.abs(dists_val - np.median(dists_val))))
                if mad_val == 0:
                    mad_val = float(np.mean(dists_val)) if float(np.mean(dists_val)) > 0 else 1.0

                normalized = dists_val / mad_val
            except Exception:
                # Fallback to full-feature distance if something unexpected
                if model._center is None:
                    raise
                dists = np.linalg.norm(features - model._center, axis=1)
                mad = float(np.median(np.abs(dists - np.median(dists))))
                if mad == 0:
                    mad = float(np.mean(dists)) if float(np.mean(dists)) > 0 else 1.0
                normalized = dists / mad

            # Map to (0,1) with saturation: large normalized -> score ~1.0
            anomaly_scores = normalized / (1.0 + normalized)
            return cast(NDArray[Any], anomaly_scores)

        if model is None:
            raise ValueError("Model not set")

        if hasattr(model, "decision_function"):
            # For One-Class SVM, decision_function gives larger values for inliers
            scores = model.decision_function(features)
            raw = -scores  # Negative -> positive anomalies
        elif hasattr(model, "score_samples"):
            # For Isolation Forest (sklearn), lower score_samples indicate anomalies
            scores = model.score_samples(features)
            raw = -scores
        else:
            raise ValueError("Model does not support anomaly scoring")

        # Normalize anomaly scores to [0,1] relative to training distribution
        training_scores = getattr(self.model, "_training_anomaly_scores", None)
        try:
            if training_scores is not None and np.ptp(training_scores) > 0:
                min_ts = float(np.min(training_scores))
                max_ts = float(np.max(training_scores))
                normed = (raw - min_ts) / (max_ts - min_ts)
                return cast(NDArray[Any], normed)
        except Exception:
            pass

        # Fallback: return raw values
        return raw

    def _determine_anomaly_type(
        self, metric_name: str, anomaly_score: float
    ) -> AnomalyType:
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
            try:
                from sklearn.ensemble import IsolationForest  # type: ignore[import-untyped]
            except Exception:
                IsolationForest = _SimpleIsolationForest

            if len(training_data) < self.config.min_training_samples:
                raise ValueError(
                    f"Insufficient training data: {len(training_data)} < "
                    f"{self.config.min_training_samples}"
                )

            # Prepare features
            # Set timestamp reference to the earliest timestamp in training
            # so timestamp features are relative and do not dominate scaling.
            try:
                self._timestamp_ref = min(
                    [m.timestamp.timestamp() for m in training_data]
                )
            except Exception:
                self._timestamp_ref = None

            features = self._prepare_features(training_data)
            scaled_features = self._scale_features(features)

            # Train Isolation Forest
            self.model = IsolationForest(
                contamination=self.config.isolation_forest_contamination,
                n_estimators=self.config.isolation_forest_estimators,
                random_state=42,
                n_jobs=-1,
            )

            self.model.fit(scaled_features)
            self.is_trained = True
            self.training_data = training_data
            self.last_training_time = datetime.now(timezone.utc)

            # Calculate training metrics
            anomaly_scores = self._calculate_anomaly_scores(scaled_features)
            anomaly_count: int = int(np.sum(anomaly_scores > 0))

            # Determine dynamic detection threshold based on contamination
            try:
                quantile_thresh = float(
                    np.quantile(
                        anomaly_scores, 1.0 - self.config.isolation_forest_contamination
                    )
                )
            except Exception:
                quantile_thresh = self.config.anomaly_threshold

            # Use component-configured threshold as a safeguard but prefer
            # the data-driven quantile when it is more permissive. This
            # helps avoid overly strict thresholds in normal-production
            # settings while still guarding against noisy small-variance
            # training sets.
            component_key = (
                training_data[0].component_type.value if training_data else "unknown"
            )
            component_thr = self.config.component_thresholds.get(
                component_key, self.config.anomaly_threshold
            )
            detection_threshold = min(component_thr, quantile_thresh)

            # Store threshold on the model for use during detection
            try:
                self.model._detection_threshold = detection_threshold
                self.model._training_anomaly_scores = anomaly_scores
            except Exception:
                pass

            training_metrics = ModelMetrics(
                model_name=self.model_name,
                model_version=self.model_version,
                status=ModelStatus.READY,
                training_samples=len(training_data),
                validation_samples=0,
                last_training_time=self.last_training_time,
                accuracy=1.0
                - anomaly_count
                / len(training_data),  # Accuracy as inverse of anomaly rate
                total_predictions=0,
                successful_predictions=0,
            )

            logger.info(
                f"Isolation Forest model trained with {len(training_data)} samples, "
                f"detected {anomaly_count} anomalies"
            )

            return training_metrics

        except ImportError as e:
            # If import fails for reasons other than simple missing package, re-raise
            raise ImportError(
                "scikit-learn is required for Isolation Forest anomaly detection. "
                "Install with: pip install scikit-learn"
            ) from e
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
            anomaly_score = float(self._calculate_anomaly_score(scaled_features)[0])

            # Check if anomaly
            component_threshold = self.config.component_thresholds.get(
                metric.component_type.value, self.config.anomaly_threshold
            )

            # Prefer model-specific dynamic threshold if available (set during training)
            model_threshold = getattr(self.model, "_detection_threshold", None)
            threshold_to_use = (
                model_threshold if model_threshold is not None else component_threshold
            )

            if anomaly_score > threshold_to_use:
                # Calculate expected value (use recent training data average)
                expected_value = self._calculate_expected_value(metric)
                confidence = min(anomaly_score / (component_threshold * 2), 1.0)

                anomaly_result = AnomalyResult(
                    timestamp=metric.timestamp,
                    component_type=metric.component_type,
                    component_id=metric.component_id,
                    metric_name=metric.metric_name,
                    anomaly_type=self._determine_anomaly_type(
                        metric.metric_name, anomaly_score
                    ),
                    anomaly_score=anomaly_score,
                    confidence=confidence,
                    severity=None,  # Will be calculated in __post_init__
                    actual_value=metric.value,
                    expected_value=expected_value,
                    deviation=metric.value - expected_value,
                    threshold=component_threshold,
                    context=metric.metadata,
                    recommendations=self._generate_recommendations(
                        metric, anomaly_score
                    ),
                    model_name=self.model_name,
                    model_version=self.model_version,
                )

                return anomaly_result

            # As a fallback, also consider large deviations in the raw metric
            # value compared to recent historical values (z-score > 3).
            try:
                similar_values = [
                    m.value
                    for m in self.training_data
                    if m.component_type == metric.component_type
                    and m.metric_name == metric.metric_name
                ]
                if similar_values:
                    mean_val = float(np.mean(similar_values))
                    std_val = (
                        float(np.std(similar_values))
                        if float(np.std(similar_values)) > 0
                        else 1e-6
                    )
                    z_score = abs(metric.value - mean_val) / std_val
                    if z_score >= 3.0:
                        # Treat as anomaly even if model score was below threshold
                        expected_value = mean_val
                        confidence = min(1.0, z_score / 5.0)
                        anomaly_result = AnomalyResult(
                            timestamp=metric.timestamp,
                            component_type=metric.component_type,
                            component_id=metric.component_id,
                            metric_name=metric.metric_name,
                            anomaly_type=self._determine_anomaly_type(
                                metric.metric_name, anomaly_score
                            ),
                            anomaly_score=float(
                                max(anomaly_score, min(1.0, z_score / 5.0))
                            ),
                            confidence=confidence,
                            severity=None,
                            actual_value=metric.value,
                            expected_value=expected_value,
                            deviation=metric.value - expected_value,
                            threshold=component_threshold,
                            context=metric.metadata,
                            recommendations=self._generate_recommendations(
                                metric, anomaly_score
                            ),
                            model_name=self.model_name,
                            model_version=self.model_version,
                        )
                        return anomaly_result
            except Exception:
                # If anything goes wrong in fallback logic, do not raise; return None
                pass

            return None

        except Exception as e:
            logger.error(
                f"Anomaly detection failed for metric {metric.metric_name}: {e}"
            )
            return None

    def _calculate_anomaly_scores(self, features: np.ndarray) -> np.ndarray:
        """Calculate anomaly scores for features.

        Args:
            features: Feature matrix

        Returns:
            Anomaly scores
        """
        # Mypy may treat numpy operations as Any; ignore the specific no-any-return
        return cast(NDArray[Any], self._calculate_anomaly_score(features))  # type: ignore[no-any-return]

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
            m
            for m in self.training_data
            if m.component_type == metric.component_type
            and m.metric_name == metric.metric_name
        ]

        if similar_metrics:
            # Calculate mean of similar metrics
            values = [
                m.value for m in similar_metrics[-50:]
            ]  # Use last 50 similar metrics
            return float(np.mean(values))
        else:
            # Use overall mean
            values = [
                m.value for m in self.training_data[-100:]
            ]  # Use last 100 metrics
            return float(np.mean(values)) if values else metric.value

    def _generate_recommendations(
        self, metric: MLMetric, anomaly_score: float
    ) -> List[str]:
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
            recommendations.append(

                    "Consider adjusting evolution parameters (population size, "
                    "mutation rate)"

            )
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

    def __init__(self, config: AnomalyDetectionConfig) -> None:
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
            try:
                from sklearn.svm import OneClassSVM  # type: ignore[import-untyped]
            except Exception:
                OneClassSVM = _SimpleOneClassSVM

            if len(training_data) < self.config.min_training_samples:
                raise ValueError(
                    f"Insufficient training data: {len(training_data)} < "
                    f"{self.config.min_training_samples}"
                )

            # Prepare features
            try:
                self._timestamp_ref = min(
                    [m.timestamp.timestamp() for m in training_data]
                )
            except Exception:
                self._timestamp_ref = None

            features = self._prepare_features(training_data)
            scaled_features = self._scale_features(features)

            # Train One-Class SVM
            self.model = OneClassSVM(
                nu=self.config.one_class_svm_nu,
                kernel=self.config.one_class_svm_kernel,
                gamma="scale",
            )

            self.model.fit(scaled_features)
            self.is_trained = True
            self.training_data = training_data
            self.last_training_time = datetime.now(timezone.utc)

            # Calculate training metrics
            predictions = self.model.predict(scaled_features)
            anomaly_count: int = int(np.sum(predictions == -1))

            training_metrics = ModelMetrics(
                model_name=self.model_name,
                model_version=self.model_version,
                status=ModelStatus.READY,
                training_samples=len(training_data),
                validation_samples=0,
                last_training_time=self.last_training_time,
                accuracy=1.0 - anomaly_count / len(training_data),
                total_predictions=0,
                successful_predictions=0,
            )

            logger.info(
                f"One-Class SVM model trained with {len(training_data)} samples, "
                f"detected {anomaly_count} anomalies"
            )

            return training_metrics

        except ImportError as e:
            raise ImportError(
                "scikit-learn is required for One-Class SVM anomaly detection. "
                "Install with: pip install scikit-learn"
            ) from e
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
            model = self.model
            if model is None:
                return None
            prediction = model.predict(scaled_features)[0]
            anomaly_score = float(self._calculate_anomaly_score(scaled_features)[0])

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
                    anomaly_type=self._determine_anomaly_type(
                        metric.metric_name, anomaly_score
                    ),
                    anomaly_score=abs(anomaly_score),
                    confidence=confidence,
                    severity=None,
                    actual_value=metric.value,
                    expected_value=expected_value,
                    deviation=metric.value - expected_value,
                    threshold=component_threshold,
                    context=metric.metadata,
                    recommendations=self._generate_recommendations(
                        metric, anomaly_score
                    ),
                    model_name=self.model_name,
                    model_version=self.model_version,
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
            m
            for m in self.training_data
            if m.component_type == metric.component_type
            and m.metric_name == metric.metric_name
        ]

        if similar_metrics:
            values = [m.value for m in similar_metrics[-50:]]
            return float(np.mean(values))
        else:
            values = [m.value for m in self.training_data[-100:]]
            return float(np.mean(values)) if values else metric.value

    def _generate_recommendations(
        self, metric: MLMetric, anomaly_score: float
    ) -> List[str]:
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

    def __init__(self, config: AnomalyDetectionConfig) -> None:
        """Initialize ensemble anomaly detector.

        Args:
            config: Anomaly detection configuration
        """
        super().__init__(config)
        self.detectors = [IsolationForestDetector(config), OneClassSVMDetector(config)]
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
                logger.warning(
                    f"Failed to train detector {getattr(detector, 'model_name', '<unknown>')}: {e}"
                )

        # Return metrics from first successful detector
        if training_metrics:
            metrics_dict = training_metrics[0].__dict__.copy()
            metrics_dict["model_name"] = self.model_name
            metrics_dict["model_version"] = self.model_version
            return ModelMetrics(**metrics_dict)
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
                logger.warning(f"Detector {getattr(detector, 'model_name', '<unknown>')} failed: {e}")

        # If multiple detectors agree, it's more likely a real anomaly
        if len(anomaly_scores) >= 2:
            avg_score = float(np.mean(anomaly_scores))
            confidence = len(anomaly_scores) / len(self.detectors)

            # Use the result with highest confidence
            best_result = max(anomaly_results, key=lambda r: r.confidence)

            # Update with ensemble statistics
            best_result.anomaly_score = float(avg_score)
            best_result.confidence = confidence
            best_result.model_name = self.model_name

            return best_result
        elif anomaly_results:
            # Return a normalized ensemble-style result even if only one
            # detector flagged an anomaly so that callers can rely on
            # consistent metadata (model_name and confidence).
            result = anomaly_results[0]
            result.model_name = self.model_name
            result.confidence = len(anomaly_scores) / len(self.detectors)
            return result
        else:
            return None
