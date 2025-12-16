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

"""Intelligent Alert Management for Allele ML Analytics.

This module provides advanced alert clustering, correlation analysis,
and intelligent alert management using machine learning techniques.

Author: Bravetto AI Systems
Version: 1.0.0
"""

import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import numpy as np

from .ml_config import AlertIntelligenceConfig
from .types import AlertCluster, AlertSeverity, AnomalyResult, ComponentType

logger = logging.getLogger(__name__)


class AlertCorrelator:
    """Alert correlation and clustering engine."""

    def __init__(self, config: AlertIntelligenceConfig):
        """Initialize alert correlator.

        Args:
            config: Alert intelligence configuration
        """
        self.config = config
        self.alert_clusters = {}  # cluster_id -> AlertCluster
        self.alert_history = []  # Recent alerts for correlation
        self.correlation_cache = {}
        self.similarity_threshold = config.similarity_threshold

        # Stable component type encoding to ensure reproducible ML features
        self.component_type_encoding = {
            ComponentType.EVOLUTION_ENGINE.value: 10,
            ComponentType.KRAKEN_LNN.value: 20,
            ComponentType.NLP_AGENT.value: 30,
            ComponentType.LLM_CLIENT.value: 40,
            ComponentType.GENOME.value: 50,
            ComponentType.SYSTEM.value: 60,
            "unknown": 0  # Default for unrecognized types
        }

    async def process_alert_batch(self, alerts: List[Dict[str, Any]]) -> List[AlertCluster]:
        """Process a batch of alerts and create clusters.

        Args:
            alerts: List of alert dictionaries

        Returns:
            List of alert clusters
        """
        try:
            # Add alerts to history
            for alert in alerts:
                self.alert_history.append(alert)

            # Clean old alerts from history
            cutoff_time = datetime.now(timezone.utc) - timedelta(
                minutes=self.config.correlation_window_minutes * 2
            )
            self.alert_history = [
                alert for alert in self.alert_history
                if datetime.fromisoformat(alert.get('timestamp', datetime.now(timezone.utc).isoformat())) > cutoff_time
            ]

            # Perform clustering
            clusters = await self._cluster_alerts(alerts)

            # Update existing clusters
            await self._update_clusters(clusters)

            return list(self.alert_clusters.values())

        except Exception as e:
            logger.error(f"Alert batch processing failed: {e}")
            return []

    async def _cluster_alerts(self, alerts: List[Dict[str, Any]]) -> List[AlertCluster]:
        """Cluster alerts using the specified algorithm.

        Args:
            alerts: List of alert dictionaries

        Returns:
            List of alert clusters
        """
        if self.config.clustering_algorithm == "dbscan":
            return await self._dbscan_clustering(alerts)
        elif self.config.clustering_algorithm == "kmeans":
            return await self._kmeans_clustering(alerts)
        elif self.config.clustering_algorithm == "hierarchical":
            return await self._hierarchical_clustering(alerts)
        else:
            logger.warning(f"Unknown clustering algorithm: {self.config.clustering_algorithm}")
            return await self._simple_clustering(alerts)

    async def _dbscan_clustering(self, alerts: List[Dict[str, Any]]) -> List[AlertCluster]:
        """Perform DBSCAN clustering on alerts.

        Args:
            alerts: List of alert dictionaries

        Returns:
            List of alert clusters
        """
        try:
            from sklearn.cluster import DBSCAN

            if len(alerts) < 2:
                return []

            # Extract features from alerts
            features = self._extract_alert_features(alerts)

            # Perform DBSCAN clustering
            clustering = DBSCAN(
                eps=self.config.clustering_eps,
                min_samples=self.config.clustering_min_samples
            )
            cluster_labels = clustering.fit_predict(features)

            # Create clusters
            clusters = []
            for cluster_id in set(cluster_labels):
                if cluster_id == -1:  # Noise points
                    continue

                cluster_alerts = [
                    alerts[i] for i, label in enumerate(cluster_labels)
                    if label == cluster_id
                ]

                cluster = self._create_alert_cluster(cluster_alerts, f"dbscan_{cluster_id}")
                clusters.append(cluster)

            return clusters

        except ImportError:
            logger.warning("scikit-learn not available, falling back to simple clustering")
            return await self._simple_clustering(alerts)
        except Exception as e:
            logger.error(f"DBSCAN clustering failed: {e}")
            return await self._simple_clustering(alerts)

    async def _kmeans_clustering(self, alerts: List[Dict[str, Any]]) -> List[AlertCluster]:
        """Perform K-means clustering on alerts.

        Args:
            alerts: List of alert dictionaries

        Returns:
            List of alert clusters
        """
        try:
            from sklearn.cluster import KMeans

            if len(alerts) < 2:
                return []

            # Extract features from alerts
            features = self._extract_alert_features(alerts)

            # Determine number of clusters
            n_clusters = min(len(alerts) // 2, 5)  # Reasonable number of clusters

            # Perform K-means clustering
            clustering = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clustering.fit_predict(features)

            # Create clusters
            clusters = []
            for cluster_id in range(n_clusters):
                cluster_alerts = [
                    alerts[i] for i, label in enumerate(cluster_labels)
                    if label == cluster_id
                ]

                cluster = self._create_alert_cluster(cluster_alerts, f"kmeans_{cluster_id}")
                clusters.append(cluster)

            return clusters

        except ImportError:
            logger.warning("scikit-learn not available, falling back to simple clustering")
            return await self._simple_clustering(alerts)
        except Exception as e:
            logger.error(f"K-means clustering failed: {e}")
            return await self._simple_clustering(alerts)

    async def _hierarchical_clustering(self, alerts: List[Dict[str, Any]]) -> List[AlertCluster]:
        """Perform hierarchical clustering on alerts.

        Args:
            alerts: List of alert dictionaries

        Returns:
            List of alert clusters
        """
        try:
            from sklearn.cluster import AgglomerativeClustering

            if len(alerts) < 2:
                return []

            # Extract features from alerts
            features = self._extract_alert_features(alerts)

            # Determine number of clusters
            n_clusters = min(len(alerts) // 3, 4)

            # Perform hierarchical clustering
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
            cluster_labels = clustering.fit_predict(features)

            # Create clusters
            clusters = []
            for cluster_id in range(n_clusters):
                cluster_alerts = [
                    alerts[i] for i, label in enumerate(cluster_labels)
                    if label == cluster_id
                ]

                cluster = self._create_alert_cluster(cluster_alerts, f"hierarchical_{cluster_id}")
                clusters.append(cluster)

            return clusters

        except ImportError:
            logger.warning("scikit-learn not available, falling back to simple clustering")
            return await self._simple_clustering(alerts)
        except Exception as e:
            logger.error(f"Hierarchical clustering failed: {e}")
            return await self._simple_clustering(alerts)

    async def _simple_clustering(self, alerts: List[Dict[str, Any]]) -> List[AlertCluster]:
        """Simple clustering based on common attributes.

        Args:
            alerts: List of alert dictionaries

        Returns:
            List of alert clusters
        """
        if not alerts:
            return []

        # Group alerts by common attributes
        component_groups = defaultdict(list)
        severity_groups = defaultdict(list)

        for alert in alerts:
            component_type = alert.get('component_type', 'unknown')
            severity = alert.get('severity', AlertSeverity.INFO.value)

            component_groups[component_type].append(alert)
            severity_groups[severity].append(alert)

        clusters = []

        # Create clusters for component types with multiple alerts
        for component_type, component_alerts in component_groups.items():
            if len(component_alerts) >= 2:
                cluster = self._create_alert_cluster(
                    component_alerts,
                    f"component_{component_type}"
                )
                clusters.append(cluster)

        # Create clusters for severity groups with multiple alerts
        for severity, severity_alerts in severity_groups.items():
            if len(severity_alerts) >= 3:
                cluster = self._create_alert_cluster(
                    severity_alerts,
                    f"severity_{severity}"
                )
                clusters.append(cluster)

        return clusters

    def _extract_alert_features(self, alerts: List[Dict[str, Any]]) -> np.ndarray:
        """Extract numerical features from alerts for clustering.

        Args:
            alerts: List of alert dictionaries

        Returns:
            Feature matrix
        """
        features = []

        for alert in alerts:
            feature_vector = []

            # Severity (numerical)
            severity_map = {
                AlertSeverity.CRITICAL.value: 4,
                AlertSeverity.ERROR.value: 3,
                AlertSeverity.WARNING.value: 2,
                AlertSeverity.INFO.value: 1
            }
            feature_vector.append(float(severity_map.get(alert.get('severity', AlertSeverity.INFO.value), 1)))

            # Component type (stable encoding)
            component_type = alert.get('component_type', 'unknown')
            feature_vector.append(float(self.component_type_encoding.get(component_type, 0)))

            # Anomaly score (if available)
            anomaly_score = alert.get('anomaly_score', 0.0)
            feature_vector.append(float(anomaly_score))

            # Timestamp (hour of day)
            timestamp_str = alert.get('timestamp', datetime.now(timezone.utc).isoformat())
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                feature_vector.append(float(timestamp.hour))
            except Exception:
                feature_vector.append(12.0)  # Default to noon

            # Confidence (if available)
            confidence = alert.get('confidence', 0.5)
            feature_vector.append(float(confidence))

            features.append(feature_vector)

        return np.array(features, dtype=float)

    def _create_alert_cluster(self, alerts: List[Dict[str, Any]], cluster_id: str) -> AlertCluster:
        """Create an alert cluster from a list of alerts.

        Args:
            alerts: List of alert dictionaries
            cluster_id: Unique cluster identifier

        Returns:
            AlertCluster
        """
        # Determine cluster type
        cluster_type = self._determine_cluster_type(alerts)

        # Extract common attributes
        common_attributes = self._find_common_attributes(alerts)

        # Calculate priority score
        priority_score = self._calculate_cluster_priority(alerts)

        # Generate root cause candidates
        root_cause_candidates = self._generate_root_cause_candidates(alerts)

        # Determine first and last alert times
        timestamps = []
        for alert in alerts:
            try:
                ts = datetime.fromisoformat(alert.get('timestamp', datetime.now(timezone.utc).isoformat()).replace('Z', '+00:00'))
                timestamps.append(ts)
            except Exception:
                timestamps.append(datetime.now(timezone.utc))

        first_alert_time = min(timestamps) if timestamps else datetime.now(timezone.utc)
        last_alert_time = max(timestamps) if timestamps else datetime.now(timezone.utc)

        # Calculate confidence
        confidence = min(1.0, len(alerts) / 5.0)  # More alerts = higher confidence

        cluster = AlertCluster(
            cluster_id=cluster_id,
            cluster_type=cluster_type,
            alerts=alerts,
            common_attributes=common_attributes,
            root_cause_candidates=root_cause_candidates,
            confidence=confidence,
            priority_score=priority_score,
            impact_assessment=self._assess_impact(alerts),
            first_alert_time=first_alert_time,
            last_alert_time=last_alert_time,
            duration_minutes=(last_alert_time - first_alert_time).total_seconds() / 60.0
        )

        return cluster

    def _determine_cluster_type(self, alerts: List[Dict[str, Any]]) -> str:
        """Determine the type of alert cluster.

        Args:
            alerts: List of alerts

        Returns:
            Cluster type string
        """
        if len(alerts) <= 1:
            return "independent"

        # Check for same component
        components = {alert.get('component_type', 'unknown') for alert in alerts}
        if len(components) == 1:
            return "component_related"

        # Check for same metric
        metrics = {alert.get('metric_name', 'unknown') for alert in alerts}
        if len(metrics) == 1:
            return "metric_related"

        # Check for cascading alerts (time-based)
        timestamps = []
        for alert in alerts:
            try:
                ts = datetime.fromisoformat(alert.get('timestamp', datetime.now(timezone.utc).isoformat()).replace('Z', '+00:00'))
                timestamps.append(ts)
            except Exception:
                pass

        if len(timestamps) > 1:
            timestamps.sort()
            max_gap = max((timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1))
            if max_gap < 300:  # 5 minutes
                return "cascading"

        return "root_cause"

    def _find_common_attributes(self, alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find common attributes across alerts.

        Args:
            alerts: List of alerts

        Returns:
            Dictionary of common attributes
        """
        if not alerts:
            return {}

        common = {}
        all_keys = set()

        # Collect all keys
        for alert in alerts:
            all_keys.update(alert.keys())

        # Find common values
        for key in all_keys:
            values = [alert.get(key) for alert in alerts if key in alert]
            if len(set(values)) == 1:  # All same value
                common[key] = values[0]

        return common

    def _calculate_cluster_priority(self, alerts: List[Dict[str, Any]]) -> float:
        """Calculate priority score for alert cluster.

        Args:
            alerts: List of alerts

        Returns:
            Priority score between 0 and 1
        """
        if not alerts:
            return 0.0

        # Base score from number of alerts
        alert_count_score = min(len(alerts) / 10.0, 1.0)

        # Severity score
        severity_scores = {
            AlertSeverity.CRITICAL.value: 1.0,
            AlertSeverity.ERROR.value: 0.8,
            AlertSeverity.WARNING.value: 0.5,
            AlertSeverity.INFO.value: 0.2
        }

        severity_score = sum(
            severity_scores.get(alert.get('severity', AlertSeverity.INFO.value), 0.2)
            for alert in alerts
        ) / len(alerts)

        # Duration score
        timestamps = []
        for alert in alerts:
            try:
                ts = datetime.fromisoformat(alert.get('timestamp', datetime.now(timezone.utc).isoformat()).replace('Z', '+00:00'))
                timestamps.append(ts)
            except Exception:
                pass

        duration_score = 0.0
        if len(timestamps) > 1:
            timestamps.sort()
            duration_minutes = (max(timestamps) - min(timestamps)).total_seconds() / 60.0
            duration_score = min(duration_minutes / 60.0, 1.0)  # Cap at 1 hour

        # Combined score
        priority_score = (
            alert_count_score * self.config.priority_factors.get("frequency", 0.2) +
            severity_score * self.config.priority_factors.get("severity", 0.4) +
            duration_score * self.config.priority_factors.get("duration", 0.3) +
            0.1  # Component criticality factor
        )

        return min(priority_score, 1.0)

    def _generate_root_cause_candidates(self, alerts: List[Dict[str, Any]]) -> List[str]:
        """Generate potential root cause candidates.

        Args:
            alerts: List of alerts

        Returns:
            List of root cause candidates
        """
        candidates = []

        # Analyze alert patterns
        component_types = [alert.get('component_type', 'unknown') for alert in alerts]
        metric_names = [alert.get('metric_name', 'unknown') for alert in alerts]

        # Component-based candidates
        component_counter = Counter(component_types)
        if component_counter:
            most_common_component = component_counter.most_common(1)[0][0]
            candidates.append(f"Issue with {most_common_component} component")

        # Metric-based candidates
        metric_counter = Counter(metric_names)
        if metric_counter:
            most_common_metric = metric_counter.most_common(1)[0][0]
            candidates.append(f"Problem with {most_common_metric} metric")

        # Anomaly-based candidates
        anomaly_scores = [alert.get('anomaly_score', 0.0) for alert in alerts]
        if anomaly_scores:
            avg_anomaly_score = sum(anomaly_scores) / len(anomaly_scores)
            if avg_anomaly_score > 0.8:
                candidates.append("System-wide anomaly detected")
            elif avg_anomaly_score > 0.6:
                candidates.append("Significant performance degradation")

        # Time-based candidates
        timestamps = []
        for alert in alerts:
            try:
                ts = datetime.fromisoformat(
                    alert.get('timestamp', datetime.now(timezone.utc).isoformat()).replace('Z', '+00:00')
                )
                timestamps.append(ts)
            except Exception as e:
                # Skip invalid timestamps but keep a debug log for diagnostics
                logger.debug(f"Failed to parse alert timestamp: {e}", alert=alert)

        if len(timestamps) > 1:
            timestamps.sort()
            time_gaps = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
            avg_gap = sum(time_gaps) / len(time_gaps) if time_gaps else 0

            if avg_gap < 60:  # Less than 1 minute
                candidates.append("Rapid cascade of failures")
            elif avg_gap < 300:  # Less than 5 minutes
                candidates.append("Progressive system degradation")

        return candidates[:self.config.max_root_cause_candidates]

    def _assess_impact(self, alerts: List[Dict[str, Any]]) -> str:
        """Assess the impact of alert cluster.

        Args:
            alerts: List of alerts

        Returns:
            Impact assessment string
        """
        if not alerts:
            return "No impact"

        # Count by severity
        severity_counts = Counter(alert.get('severity', AlertSeverity.INFO.value) for alert in alerts)

        # Critical alerts
        if severity_counts.get(AlertSeverity.CRITICAL.value, 0) > 0:
            return "Critical - System functionality severely impacted"

        # Multiple error alerts
        if severity_counts.get(AlertSeverity.ERROR.value, 0) > 2:
            return "High - Multiple error conditions detected"

        # Multiple warning alerts
        if severity_counts.get(AlertSeverity.WARNING.value, 0) > 5:
            return "Medium - Multiple warning conditions detected"

        # Component-specific impact
        components = {alert.get('component_type', 'unknown') for alert in alerts}
        if len(components) == 1:
            component = list(components)[0]
            if component == "evolution_engine":
                return "High - Evolution process affected"
            elif component == "kraken_lnn":
                return "High - Neural network processing affected"
            elif component == "nlp_agent":
                return "Medium - Natural language processing affected"

        # Default assessment
        if len(alerts) > 5:
            return "Medium - Multiple related alerts detected"
        else:
            return "Low - Limited impact detected"

    async def _update_clusters(self, new_clusters: List[AlertCluster]) -> None:
        """Update existing clusters with new information.

        Args:
            new_clusters: List of new alert clusters
        """
        for cluster in new_clusters:
            # Check if cluster already exists
            existing_cluster = self.alert_clusters.get(cluster.cluster_id)

            if existing_cluster:
                # Update existing cluster
                for alert in cluster.alerts:
                    existing_cluster.update_cluster(alert)
            else:
                # Add new cluster
                self.alert_clusters[cluster.cluster_id] = cluster

        # Remove old clusters
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=2)
        expired_clusters = [
            cluster_id for cluster_id, cluster in self.alert_clusters.items()
            if cluster.last_alert_time < cutoff_time
        ]

        for cluster_id in expired_clusters:
            del self.alert_clusters[cluster_id]


class IntelligentAlertManager:
    """Intelligent alert management with deduplication and prioritization."""

    def __init__(self, config: AlertIntelligenceConfig):
        """Initialize intelligent alert manager.

        Args:
            config: Alert intelligence configuration
        """
        self.config = config
        self.correlator = AlertCorrelator(config)
        self.deduplication_cache = {}
        self.alert_priorities = {}
        self.escalation_timers = {}

    async def process_alerts(self, anomalies: List[AnomalyResult]) -> List[Dict[str, Any]]:
        """Process anomalies and create intelligent alerts.

        Args:
            anomalies: List of anomaly results

        Returns:
            List of processed alerts with intelligence metadata
        """
        try:
            # Convert anomalies to alert dictionaries
            alert_dicts = [self._anomaly_to_alert_dict(anomaly) for anomaly in anomalies]

            # Perform correlation and clustering
            clusters = await self.correlator.process_alert_batch(alert_dicts)

            # Deduplicate alerts
            deduplicated_alerts = await self._deduplicate_alerts(alert_dicts)

            # Prioritize alerts
            prioritized_alerts = await self._prioritize_alerts(deduplicated_alerts, clusters)

            # Set up escalation timers
            await self._setup_escalation_timers(prioritized_alerts)

            return prioritized_alerts

        except Exception as e:
            logger.error(f"Alert processing failed: {e}")
            return []

    def _anomaly_to_alert_dict(self, anomaly: AnomalyResult) -> Dict[str, Any]:
        """Convert anomaly result to alert dictionary.

        Args:
            anomaly: Anomaly result

        Returns:
            Alert dictionary
        """
        return {
            "alert_id": f"{anomaly.component_type}_{anomaly.metric_name}_{anomaly.timestamp.isoformat()}",
            "timestamp": anomaly.timestamp.isoformat(),
            "component_type": anomaly.component_type.value,
            "component_id": anomaly.component_id,
            "metric_name": anomaly.metric_name,
            "severity": anomaly.severity.value,
            "anomaly_type": anomaly.anomaly_type.value,
            "anomaly_score": anomaly.anomaly_score,
            "confidence": anomaly.confidence,
            "actual_value": anomaly.actual_value,
            "expected_value": anomaly.expected_value,
            "deviation": anomaly.deviation,
            "context": anomaly.context,
            "recommendations": anomaly.recommendations
        }

    async def _deduplicate_alerts(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate alerts based on similarity.

        Args:
            alerts: List of alert dictionaries

        Returns:
            List of deduplicated alerts
        """
        if self.config.auto_acknowledge_duplicates:
            return await self._smart_deduplication(alerts)
        else:
            return alerts

    async def _smart_deduplication(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform smart deduplication using similarity metrics.

        Args:
            alerts: List of alert dictionaries

        Returns:
            List of deduplicated alerts
        """
        deduplicated = []
        processed_signatures = set()

        for alert in alerts:
            # Create alert signature
            signature = self._create_alert_signature(alert)

            # Check if similar alert exists in recent history
            similar_alerts = await self._find_similar_alerts(alert)

            if not similar_alerts:
                # No similar alerts, add to deduplicated list
                deduplicated.append(alert)
                processed_signatures.add(signature)
            else:
                # Similar alert exists, potentially skip this one
                if self._should_suppress_alert(alert, similar_alerts[0]):
                    logger.info(f"Suppressed duplicate alert: {alert.get('alert_id')}")
                else:
                    deduplicated.append(alert)
                    processed_signatures.add(signature)

        return deduplicated

    def _create_alert_signature(self, alert: Dict[str, Any]) -> str:
        """Create a signature for alert deduplication.

        Args:
            alert: Alert dictionary

        Returns:
            Alert signature string
        """
        # Key attributes for signature
        key_attrs = [
            alert.get('component_type', ''),
            alert.get('metric_name', ''),
            alert.get('anomaly_type', ''),
            alert.get('severity', '')
        ]

        # Round values to reduce noise
        anomaly_score = round(alert.get('anomaly_score', 0.0), 1)
        confidence = round(alert.get('confidence', 0.0), 1)

        signature = f"{'_'.join(key_attrs)}_{anomaly_score}_{confidence}"
        return signature

    async def _find_similar_alerts(self, alert: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar alerts in recent history.

        Args:
            alert: Alert to compare

        Returns:
            List of similar alerts
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(
            minutes=self.config.deduplication_window_minutes
        )

        similar_alerts = []

        for cached_alert in self.deduplication_cache.values():
            try:
                alert_time = datetime.fromisoformat(cached_alert.get('timestamp', datetime.now(timezone.utc).isoformat()).replace('Z', '+00:00'))

                if alert_time > cutoff_time:
                    similarity = self._calculate_alert_similarity(alert, cached_alert)
                    if similarity >= self.config.similarity_threshold:
                        similar_alerts.append(cached_alert)
            except Exception:
                continue

        return similar_alerts

    def _calculate_alert_similarity(self, alert1: Dict[str, Any], alert2: Dict[str, Any]) -> float:
        """Calculate similarity between two alerts.

        Args:
            alert1: First alert
            alert2: Second alert

        Returns:
            Similarity score between 0 and 1
        """
        similarity_score = 0.0
        total_weight = 0.0

        # Component type similarity
        if alert1.get('component_type') == alert2.get('component_type'):
            similarity_score += 0.3
        total_weight += 0.3

        # Metric name similarity
        if alert1.get('metric_name') == alert2.get('metric_name'):
            similarity_score += 0.2
        total_weight += 0.2

        # Anomaly type similarity
        if alert1.get('anomaly_type') == alert2.get('anomaly_type'):
            similarity_score += 0.2
        total_weight += 0.2

        # Severity similarity
        if alert1.get('severity') == alert2.get('severity'):
            similarity_score += 0.1
        total_weight += 0.1

        # Value similarity (if available)
        val1 = alert1.get('actual_value', 0.0)
        val2 = alert2.get('actual_value', 0.0)
        if val1 > 0 and val2 > 0:
            relative_diff = abs(val1 - val2) / max(val1, val2)
            value_similarity = max(0.0, 1.0 - relative_diff)
            similarity_score += value_similarity * 0.2
            total_weight += 0.2

        return similarity_score / total_weight if total_weight > 0 else 0.0

    def _should_suppress_alert(self, current_alert: Dict[str, Any], similar_alert: Dict[str, Any]) -> bool:
        """Determine if current alert should be suppressed.

        Args:
            current_alert: Current alert
            similar_alert: Similar alert from history

        Returns:
            True if alert should be suppressed
        """
        # Check if similar alert is more recent and of equal or higher severity
        try:
            current_time = datetime.fromisoformat(current_alert.get('timestamp', datetime.now(timezone.utc).isoformat()).replace('Z', '+00:00'))
            similar_time = datetime.fromisoformat(similar_alert.get('timestamp', datetime.now(timezone.utc).isoformat()).replace('Z', '+00:00'))

            # If similar alert is more recent, suppress current
            if similar_time > current_time:
                return True

            # Check severity
            current_severity = current_alert.get('severity', AlertSeverity.INFO.value)
            similar_severity = similar_alert.get('severity', AlertSeverity.INFO.value)

            severity_order = {
                AlertSeverity.CRITICAL.value: 4,
                AlertSeverity.ERROR.value: 3,
                AlertSeverity.WARNING.value: 2,
                AlertSeverity.INFO.value: 1
            }

            if severity_order.get(similar_severity, 0) >= severity_order.get(current_severity, 0):
                return True

        except Exception:
            pass

        return False

    async def _prioritize_alerts(self, alerts: List[Dict[str, Any]],
                               clusters: List[AlertCluster]) -> List[Dict[str, Any]]:
        """Prioritize alerts based on various factors.

        Args:
            alerts: List of alerts
            clusters: Alert clusters

        Returns:
            List of prioritized alerts
        """
        # Create cluster priority lookup
        {cluster.cluster_id: cluster.priority_score for cluster in clusters}

        for alert in alerts:
            # Calculate base priority
            priority = self._calculate_alert_priority(alert)

            # Boost priority if part of a high-priority cluster
            for cluster in clusters:
                if any(a.get('alert_id') == alert.get('alert_id') for a in cluster.alerts):
                    priority *= (1.0 + cluster.priority_score)
                    break

            alert['priority_score'] = min(priority, 1.0)
            alert['cluster_info'] = self._get_cluster_info(alert, clusters)

        # Sort by priority
        alerts.sort(key=lambda x: x.get('priority_score', 0.0), reverse=True)

        return alerts

    def _calculate_alert_priority(self, alert: Dict[str, Any]) -> float:
        """Calculate priority score for an alert.

        Args:
            alert: Alert dictionary

        Returns:
            Priority score between 0 and 1
        """
        # Base priority from severity
        severity_scores = {
            AlertSeverity.CRITICAL.value: 1.0,
            AlertSeverity.ERROR.value: 0.8,
            AlertSeverity.WARNING.value: 0.5,
            AlertSeverity.INFO.value: 0.2
        }

        base_priority = severity_scores.get(alert.get('severity', AlertSeverity.INFO.value), 0.2)

        # Boost from anomaly score
        anomaly_score = alert.get('anomaly_score', 0.0)
        anomaly_boost = anomaly_score * 0.3

        # Boost from confidence
        confidence = alert.get('confidence', 0.0)
        confidence_boost = confidence * 0.2

        # Component criticality boost
        component_type = alert.get('component_type', '')
        component_boost = 0.0
        if component_type in ['evolution_engine', 'kraken_lnn']:
            component_boost = 0.2

        total_priority = base_priority + anomaly_boost + confidence_boost + component_boost

        return min(total_priority, 1.0)

    def _get_cluster_info(self, alert: Dict[str, Any], clusters: List[AlertCluster]) -> Dict[str, Any]:
        """Get cluster information for an alert.

        Args:
            alert: Alert dictionary
            clusters: List of alert clusters

        Returns:
            Cluster information dictionary
        """
        for cluster in clusters:
            if any(a.get('alert_id') == alert.get('alert_id') for a in cluster.alerts):
                return {
                    'cluster_id': cluster.cluster_id,
                    'cluster_type': cluster.cluster_type,
                    'priority_score': cluster.priority_score,
                    'confidence': cluster.confidence,
                    'impact_assessment': cluster.impact_assessment,
                    'root_cause_candidates': cluster.root_cause_candidates
                }

        return {}

    async def _setup_escalation_timers(self, prioritized_alerts: List[Dict[str, Any]]) -> None:
        """Set up escalation timers for high-priority alerts.

        Args:
            prioritized_alerts: List of prioritized alerts
        """
        current_time = datetime.now(timezone.utc)

        for alert in prioritized_alerts:
            if alert.get('priority_score', 0.0) > 0.7:  # High priority alerts
                escalation_time = current_time + timedelta(minutes=self.config.escalation_time_minutes)
                self.escalation_timers[alert.get('alert_id')] = escalation_time

    async def check_escalations(self) -> List[Dict[str, Any]]:
        """Check for alerts that need escalation.

        Returns:
            List of alerts to escalate
        """
        current_time = datetime.now(timezone.utc)
        escalations = []

        for alert_id, escalation_time in list(self.escalation_timers.items()):
            if current_time > escalation_time:
                escalations.append({
                    'alert_id': alert_id,
                    'escalation_time': escalation_time.isoformat(),
                    'reason': f"No acknowledgment within {self.config.escalation_time_minutes} minutes"
                })
                # Remove from timers
                del self.escalation_timers[alert_id]

        return escalations
