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

"""Optimization Engine for Allele ML Analytics.

This module provides automated performance optimization recommendations
using both rule-based and machine learning approaches.

Author: Bravetto AI Systems
Version: 1.0.0
"""

import json
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .ml_config import OptimizationEngineConfig
from .types import (
    ComponentType,
    MLMetric,
    OptimizationCategory,
    OptimizationRecommendation,
    PredictionResult,
)

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """ML-based performance optimization recommendations."""

    def __init__(self, config: OptimizationEngineConfig):
        """Initialize performance optimizer.

        Args:
            config: Optimization engine configuration
        """
        self.config = config
        self.optimization_rules = self._load_optimization_rules()
        self.performance_history = deque(maxlen=1000)
        self.optimization_models = {}
        self.recommendation_cache = {}

    def _load_optimization_rules(self) -> Dict[str, Any]:
        """Load optimization rules from file or create defaults.

        Returns:
            Optimization rules dictionary
        """
        try:
            rule_file = Path(self.config.rule_file_path)
            if rule_file.exists():
                with open(rule_file) as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load optimization rules: {e}")

        # Default optimization rules
        return self._create_default_rules()

    def _create_default_rules(self) -> Dict[str, Any]:
        """Create default optimization rules.

        Returns:
            Default optimization rules
        """
        return {
            "performance_thresholds": {
                "cpu_usage": {"warning": 70.0, "critical": 90.0},
                "memory_usage": {"warning": 75.0, "critical": 95.0},
                "latency_ms": {"warning": 100.0, "critical": 500.0},
                "error_rate": {"warning": 1.0, "critical": 5.0}
            },
            "component_specific_rules": {
                "evolution_engine": {
                    "population_size": {"min": 50, "max": 1000, "optimal": 200},
                    "mutation_rate": {"min": 0.01, "max": 0.5, "optimal": 0.1},
                    "crossover_rate": {"min": 0.5, "max": 1.0, "optimal": 0.8}
                },
                "kraken_lnn": {
                    "reservoir_size": {"min": 100, "max": 10000, "optimal": 1000},
                    "spectral_radius": {"min": 0.1, "max": 2.0, "optimal": 0.9},
                    "leaking_rate": {"min": 0.1, "max": 1.0, "optimal": 0.8}
                },
                "nlp_agent": {
                    "max_tokens": {"min": 100, "max": 4000, "optimal": 1000},
                    "temperature": {"min": 0.0, "max": 2.0, "optimal": 0.7},
                    "top_p": {"min": 0.1, "max": 1.0, "optimal": 0.9}
                }
            },
            "resource_allocation": {
                "cpu_intensive_tasks": ["evolution_engine", "kraken_lnn"],
                "memory_intensive_tasks": ["kraken_lnn"],
                "io_intensive_tasks": ["nlp_agent"]
            }
        }

    async def analyze_performance(self, metrics_history: Dict[str, List[MLMetric]],
                                predictions: Dict[str, List[PredictionResult]]) -> List[OptimizationRecommendation]:
        """Analyze performance and generate optimization recommendations.

        Args:
            metrics_history: Historical metrics by component
            predictions: Performance predictions by component

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        try:
            # Analyze each component
            for component_type, metrics in metrics_history.items():
                if not metrics:
                    continue

                # Rule-based analysis
                rule_recommendations = await self._rule_based_analysis(
                    component_type, metrics, predictions.get(component_type, [])
                )
                recommendations.extend(rule_recommendations)

                # ML-based analysis (if enabled)
                if self.config.enable_ml_based:
                    ml_recommendations = await self._ml_based_analysis(
                        component_type, metrics, predictions.get(component_type, [])
                    )
                    recommendations.extend(ml_recommendations)

            # Filter and rank recommendations
            filtered_recommendations = await self._filter_recommendations(recommendations)

            # Cache recommendations
            self.recommendation_cache = {
                rec.recommendation_id: rec for rec in filtered_recommendations
            }

            return filtered_recommendations

        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return []

    async def _rule_based_analysis(self, component_type: str,
                                 metrics: List[MLMetric],
                                 predictions: List[PredictionResult]) -> List[OptimizationRecommendation]:
        """Perform rule-based optimization analysis.

        Args:
            component_type: Type of component
            metrics: Historical metrics
            predictions: Performance predictions

        Returns:
            List of rule-based recommendations
        """
        recommendations = []

        try:
            # Analyze performance thresholds
            threshold_recommendations = await self._analyze_thresholds(component_type, metrics)
            recommendations.extend(threshold_recommendations)

            # Analyze component-specific parameters
            parameter_recommendations = await self._analyze_component_parameters(component_type, metrics)
            recommendations.extend(parameter_recommendations)

            # Analyze resource allocation
            resource_recommendations = await self._analyze_resource_allocation(component_type, metrics)
            recommendations.extend(resource_recommendations)

            # Analyze predictions for future optimization
            prediction_recommendations = await self._analyze_predictions(component_type, predictions)
            recommendations.extend(prediction_recommendations)

        except Exception as e:
            logger.error(f"Rule-based analysis failed for {component_type}: {e}")

        return recommendations

    async def _analyze_thresholds(self, component_type: str,
                                metrics: List[MLMetric]) -> List[OptimizationRecommendation]:
        """Analyze performance thresholds and generate recommendations.

        Args:
            component_type: Type of component
            metrics: Historical metrics

        Returns:
            List of threshold-based recommendations
        """
        recommendations = []

        # Group metrics by name
        metric_groups = defaultdict(list)
        for metric in metrics:
            metric_groups[metric.metric_name].append(metric)

        thresholds = self.optimization_rules.get("performance_thresholds", {})

        for metric_name, metric_list in metric_groups.items():
            if len(metric_list) < 10:
                continue

            values = [m.value for m in metric_list]
            avg_value = np.mean(values)
            max_value = np.max(values)
            np.std(values)

            # Check if metric exceeds thresholds
            threshold_key = metric_name.lower().replace(" ", "_")
            if threshold_key in thresholds:
                threshold = thresholds[threshold_key]

                # Critical threshold exceeded
                if max_value > threshold["critical"]:
                    recommendations.append(OptimizationRecommendation(
                        recommendation_id=f"{component_type}_{metric_name}_critical_threshold",
                        category=OptimizationCategory.PERFORMANCE_TUNING,
                        title=f"Critical Performance Threshold Exceeded: {metric_name}",
                        description=f"The {metric_name} has exceeded critical threshold ({threshold['critical']:.1f}). Current max: {max_value:.1f}",
                        current_value=max_value,
                        recommended_value=threshold["critical"] * 0.8,  # 20% below critical
                        expected_improvement=25.0,  # 25% improvement expected
                        confidence=0.9,
                        implementation_steps=[
                            f"Review {metric_name} configuration",
                            "Consider resource scaling",
                            "Implement performance monitoring"
                        ],
                        estimated_effort="high",
                        risk_level="medium",
                        component_type=ComponentType(component_type),
                        priority=1
                    ))

                # Warning threshold exceeded
                elif avg_value > threshold["warning"]:
                    recommendations.append(OptimizationRecommendation(
                        recommendation_id=f"{component_type}_{metric_name}_warning_threshold",
                        category=OptimizationCategory.PERFORMANCE_TUNING,
                        title=f"Performance Threshold Warning: {metric_name}",
                        description=f"The {metric_name} is approaching threshold. Current avg: {avg_value:.1f}, threshold: {threshold['warning']:.1f}",
                        current_value=avg_value,
                        recommended_value=threshold["warning"] * 0.7,  # 30% below warning
                        expected_improvement=15.0,  # 15% improvement expected
                        confidence=0.7,
                        implementation_steps=[
                            f"Monitor {metric_name} trends",
                            "Optimize related parameters"
                        ],
                        estimated_effort="medium",
                        risk_level="low",
                        component_type=ComponentType(component_type),
                        priority=2
                    ))

        return recommendations

    async def _analyze_component_parameters(self, component_type: str,
                                          metrics: List[MLMetric]) -> List[OptimizationRecommendation]:
        """Analyze component-specific parameters.

        Args:
            component_type: Type of component
            metrics: Historical metrics

        Returns:
            List of parameter optimization recommendations
        """
        recommendations = []

        component_rules = self.optimization_rules.get("component_specific_rules", {})
        if component_type not in component_rules:
            # Generate generic recommendations if no specific rules exist
            if metrics:
                avg_latency = sum(m.value for m in metrics if m.name == "latency") / len(metrics)
                if avg_latency > 100:  # Generic threshold
                    recommendations.append(OptimizationRecommendation(
                        recommendation_id=f"{component_type}_generic_latency",
                        category=OptimizationCategory.PERFORMANCE_TUNING,
                        title=f"High Latency Detected: {component_type}",
                        description=f"Average latency {avg_latency:.1f}ms exceeds generic threshold",
                        current_value=avg_latency,
                        recommended_value=100.0,
                        expected_improvement=20.0,
                        confidence=0.6,
                        implementation_steps=["Investigate performance bottlenecks"],
                        estimated_effort="medium",
                        risk_level="low",
                        component_type=ComponentType(component_type),
                        priority=2
                    ))
            return recommendations

        component_rules[component_type]

        # Analyze fitness/performance metrics for evolution
        if component_type == "evolution_engine":
            fitness_metrics = [m for m in metrics if "fitness" in m.metric_name.lower()]
            if fitness_metrics:
                recent_fitness = [m.value for m in fitness_metrics[-10:]]
                if len(recent_fitness) >= 5:
                    avg_fitness = np.mean(recent_fitness)

                    # Check if fitness is stagnating
                    if len(recent_fitness) >= 10:
                        fitness_trend = np.polyfit(range(len(recent_fitness)), recent_fitness, 1)[0]
                        if abs(fitness_trend) < 0.001:  # Near-zero trend
                            recommendations.append(OptimizationRecommendation(
                                recommendation_id=f"{component_type}_stagnating_fitness",
                                category=OptimizationCategory.CONFIGURATION_TUNING,
                                title="Evolution Fitness Stagnation Detected",
                                description=f"Evolution fitness has stagnated (trend: {fitness_trend:.6f}). Consider parameter adjustment.",
                                current_value=avg_fitness,
                                recommended_value=avg_fitness * 1.2,  # 20% improvement target
                                expected_improvement=20.0,
                                confidence=0.8,
                                implementation_steps=[
                                    "Increase mutation rate slightly",
                                    "Adjust selection pressure",
                                    "Review fitness function"
                                ],
                                estimated_effort="medium",
                                risk_level="medium",
                                component_type=ComponentType.EVOLUTION_ENGINE,
                                priority=1
                            ))

        # Analyze latency metrics for NLP agent
        elif component_type == "nlp_agent":
            latency_metrics = [m for m in metrics if "latency" in m.metric_name.lower() or "response_time" in m.metric_name.lower()]
            if latency_metrics:
                recent_latency = [m.value for m in latency_metrics[-10:]]
                if len(recent_latency) >= 5:
                    avg_latency = np.mean(recent_latency)

                    if avg_latency > 2000:  # 2 seconds
                        recommendations.append(OptimizationRecommendation(
                            recommendation_id=f"{component_type}_high_latency",
                            category=OptimizationCategory.PERFORMANCE_TUNING,
                            title="High Response Latency Detected",
                            description=f"Average response latency is {avg_latency:.0f}ms. Consider optimization.",
                            current_value=avg_latency,
                            recommended_value=1000.0,  # Target 1 second
                            expected_improvement=50.0,  # 50% improvement expected
                            confidence=0.8,
                            implementation_steps=[
                                "Reduce max_tokens parameter",
                                "Optimize prompt engineering",
                                "Consider model quantization"
                            ],
                            estimated_effort="medium",
                            risk_level="low",
                            component_type=ComponentType.NLP_AGENT,
                            priority=2
                        ))

        return recommendations

    async def _analyze_resource_allocation(self, component_type: str,
                                         metrics: List[MLMetric]) -> List[OptimizationRecommendation]:
        """Analyze resource allocation and generate recommendations.

        Args:
            component_type: Type of component
            metrics: Historical metrics

        Returns:
            List of resource allocation recommendations
        """
        recommendations = []

        # Group CPU and memory metrics
        cpu_metrics = [m for m in metrics if "cpu" in m.metric_name.lower()]
        memory_metrics = [m for m in metrics if "memory" in m.metric_name.lower()]

        if cpu_metrics:
            recent_cpu = [m.value for m in cpu_metrics[-10:]]
            avg_cpu = np.mean(recent_cpu)

            # High CPU usage
            if avg_cpu > 80.0:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"{component_type}_high_cpu_usage",
                    category=OptimizationCategory.RESOURCE_ALLOCATION,
                    title="High CPU Usage Detected",
                    description=f"Average CPU usage is {avg_cpu:.1f}%. Consider resource scaling.",
                    current_value=avg_cpu,
                    recommended_value=70.0,  # Target 70%
                    expected_improvement=20.0,
                    confidence=0.8,
                    implementation_steps=[
                        "Scale CPU resources",
                        "Optimize CPU-intensive operations",
                        "Review concurrent task limits"
                    ],
                    estimated_effort="high",
                    risk_level="medium",
                    component_type=ComponentType(component_type),
                    priority=1
                ))

        if memory_metrics:
            recent_memory = [m.value for m in memory_metrics[-10:]]
            avg_memory = np.mean(recent_memory)

            # High memory usage
            if avg_memory > 85.0:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"{component_type}_high_memory_usage",
                    category=OptimizationCategory.RESOURCE_ALLOCATION,
                    title="High Memory Usage Detected",
                    description=f"Average memory usage is {avg_memory:.1f}%. Consider memory optimization.",
                    current_value=avg_memory,
                    recommended_value=75.0,  # Target 75%
                    expected_improvement=15.0,
                    confidence=0.7,
                    implementation_steps=[
                        "Scale memory resources",
                        "Optimize memory-intensive operations",
                        "Review garbage collection settings"
                    ],
                    estimated_effort="high",
                    risk_level="medium",
                    component_type=ComponentType(component_type),
                    priority=1
                ))

        return recommendations

    async def _analyze_predictions(self, component_type: str,
                                 predictions: List[PredictionResult]) -> List[OptimizationRecommendation]:
        """Analyze performance predictions for optimization.

        Args:
            component_type: Type of component
            predictions: Performance predictions

        Returns:
            List of prediction-based recommendations
        """
        recommendations = []

        for prediction in predictions:
            try:
                # Check if predicted performance will degrade
                predicted_value = prediction.predicted_value

                # Simple heuristic: if predicted value is significantly worse
                # than current trend, suggest optimization
                if prediction.prediction_type.value == "performance_forecast":
                    if predicted_value > 1.5:  # Assuming normalized performance metric
                        recommendations.append(OptimizationRecommendation(
                            recommendation_id=f"{component_type}_predicted_degradation",
                            category=OptimizationCategory.PERFORMANCE_TUNING,
                            title="Predicted Performance Degradation",
                            description=f"Performance forecast shows degradation to {predicted_value:.2f} with confidence {prediction.model_accuracy:.2f}",
                            current_value=1.0,  # Current baseline
                            recommended_value=0.8,  # Target improvement
                            expected_improvement=20.0,
                            confidence=prediction.model_accuracy,
                            implementation_steps=[
                                "Proactive performance tuning",
                                "Increase monitoring frequency",
                                "Prepare scaling resources"
                            ],
                            estimated_effort="medium",
                            risk_level="low",
                            component_type=ComponentType(component_type),
                            priority=2,
                            expires_at=datetime.now(timezone.utc) + timedelta(hours=24)
                        ))
            except Exception as e:
                logger.warning(f"Prediction analysis failed: {e}")
                continue

        return recommendations

    async def _ml_based_analysis(self, component_type: str,
                                metrics: List[MLMetric],
                                predictions: List[PredictionResult]) -> List[OptimizationRecommendation]:
        """Perform ML-based optimization analysis.

        Args:
            component_type: Type of component
            metrics: Historical metrics
            predictions: Performance predictions

        Returns:
            List of ML-based recommendations
        """
        recommendations = []

        try:
            # Simple ML-based analysis using correlation and regression
            if len(metrics) >= 20:
                # Analyze metric correlations
                correlation_recommendations = await self._analyze_metric_correlations(component_type, metrics)
                recommendations.extend(correlation_recommendations)

                # Analyze performance patterns
                pattern_recommendations = await self._analyze_performance_patterns(component_type, metrics)
                recommendations.extend(pattern_recommendations)

        except Exception as e:
            logger.warning(f"ML-based analysis failed for {component_type}: {e}")

        return recommendations

    async def _analyze_metric_correlations(self, component_type: str,
                                         metrics: List[MLMetric]) -> List[OptimizationRecommendation]:
        """Analyze correlations between metrics for optimization insights.

        Args:
            component_type: Type of component
            metrics: Historical metrics

        Returns:
            List of correlation-based recommendations
        """
        recommendations = []

        # Group metrics by name
        metric_groups = defaultdict(list)
        for metric in metrics:
            metric_groups[metric.metric_name].append(metric)

        metric_names = list(metric_groups.keys())
        if len(metric_names) < 2:
            return recommendations

        # Calculate correlations between different metric types
        for i, name1 in enumerate(metric_names):
            for name2 in metric_names[i+1:]:
                values1 = [m.value for m in metric_groups[name1]]
                values2 = [m.value for m in metric_groups[name2]]

                if len(values1) >= 10 and len(values2) >= 10:
                    try:
                        correlation = np.corrcoef(values1, values2)[0, 1]

                        # Strong positive correlation (potential optimization target)
                        if correlation > 0.7:
                            recommendations.append(OptimizationRecommendation(
                                recommendation_id=f"{component_type}_correlation_{name1}_{name2}",
                                category=OptimizationCategory.PERFORMANCE_TUNING,
                                title=f"Strong Correlation Detected: {name1} ↔ {name2}",
                                description=f"Strong positive correlation ({correlation:.2f}) between {name1} and {name2}. Optimize the controlling metric.",
                                current_value=correlation,
                                recommended_value=0.5,  # Reduce correlation target
                                expected_improvement=15.0,
                                confidence=0.6,
                                implementation_steps=[
                                    f"Focus optimization efforts on {name1}",
                                    "Monitor both metrics closely",
                                    "Consider decoupling strategies"
                                ],
                                estimated_effort="medium",
                                risk_level="low",
                                component_type=ComponentType(component_type),
                                priority=3
                            ))
                    except KeyError as e:
                        logger.debug(f"Skipping unknown component value in optimization analysis: {e}")
                        continue
                    except AttributeError as e:
                        logger.debug(f"Component attribute error in optimization analysis: {e}")
                        continue

        return recommendations

    async def _analyze_performance_patterns(self, component_type: str,
                                          metrics: List[MLMetric]) -> List[OptimizationRecommendation]:
        """Analyze performance patterns for optimization opportunities.

        Args:
            component_type: Type of component
            metrics: Historical metrics

        Returns:
            List of pattern-based recommendations
        """
        recommendations = []

        # Analyze overall performance trend
        if len(metrics) >= 50:
            values = [m.value for m in metrics]
            timestamps = [m.timestamp.timestamp() for m in metrics]

            # Fit polynomial trend
            trend_coef = np.polyfit(timestamps, values, 2)

            # If trend is negative quadratic (performance degrading over time)
            if trend_coef[0] < -0.0001:  # Negative quadratic coefficient
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"{component_type}_degrading_trend",
                    category=OptimizationCategory.PERFORMANCE_TUNING,
                    title="Degrading Performance Trend Detected",
                    description=f"Performance shows accelerating degradation over time. Quadratic trend coefficient: {trend_coef[0]:.6f}",
                    current_value=values[-1],
                    recommended_value=np.mean(values[-10:]) * 1.1,  # Target 10% improvement
                    expected_improvement=25.0,
                    confidence=0.7,
                    implementation_steps=[
                        "Implement proactive maintenance",
                        "Review resource allocation",
                        "Consider architectural changes"
                    ],
                    estimated_effort="high",
                    risk_level="medium",
                    component_type=ComponentType(component_type),
                    priority=1
                ))

        return recommendations

    async def _filter_recommendations(self, recommendations: List[OptimizationRecommendation]) -> List[OptimizationRecommendation]:
        """Filter and rank optimization recommendations.

        Args:
            recommendations: Raw recommendations

        Returns:
            Filtered and ranked recommendations
        """
        # Remove expired recommendations
        datetime.now(timezone.utc)
        active_recommendations = [
            rec for rec in recommendations
            if not rec.is_expired()
        ]

        # Filter by confidence threshold
        min_confidence = self.config.min_confidence_threshold
        confident_recommendations = [
            rec for rec in active_recommendations
            if rec.confidence >= min_confidence
        ]

        # Filter by expected improvement threshold
        min_improvement = self.config.min_expected_improvement
        valuable_recommendations = [
            rec for rec in confident_recommendations
            if rec.expected_improvement >= min_improvement * 100  # Convert to percentage
        ]

        # Sort by priority, confidence, and expected improvement
        sorted_recommendations = sorted(
            valuable_recommendations,
            key=lambda x: (x.priority, -x.confidence, -x.expected_improvement)
        )

        # Limit to batch size
        return sorted_recommendations[:self.config.batch_optimization_size]


class ConfigurationRecommender:
    """Smart configuration tuning recommendations."""

    def __init__(self, config: OptimizationEngineConfig):
        """Initialize configuration recommender.

        Args:
            config: Optimization engine configuration
        """
        self.config = config
        self.config_history = deque(maxlen=100)
        self.performance_baselines = {}

    async def recommend_configuration_changes(self, component_type: str,
                                            current_config: Dict[str, Any],
                                            performance_metrics: List[MLMetric]) -> List[OptimizationRecommendation]:
        """Recommend configuration changes based on performance analysis.

        Args:
            component_type: Type of component
            current_config: Current configuration
            performance_metrics: Performance metrics

        Returns:
            List of configuration recommendations
        """
        recommendations = []

        try:
            # Analyze current performance
            baseline_performance = self._calculate_baseline_performance(component_type, performance_metrics)

            # Component-specific configuration analysis
            if component_type == "evolution_engine":
                recommendations.extend(await self._recommend_evolution_config(current_config, baseline_performance))
            elif component_type == "kraken_lnn":
                recommendations.extend(await self._recommend_kraken_config(current_config, baseline_performance))
            elif component_type == "nlp_agent":
                recommendations.extend(await self._recommend_nlp_config(current_config, baseline_performance))

        except Exception as e:
            logger.error(f"Configuration recommendation failed for {component_type}: {e}")

        return recommendations

    def _calculate_baseline_performance(self, component_type: str,
                                      metrics: List[MLMetric]) -> Dict[str, float]:
        """Calculate baseline performance metrics.

        Args:
            component_type: Type of component
            metrics: Performance metrics

        Returns:
            Baseline performance dictionary
        """
        baseline = {}

        # Group metrics by name
        metric_groups = defaultdict(list)
        for metric in metrics:
            metric_groups[metric.metric_name].append(metric.value)

        for metric_name, values in metric_groups.items():
            if len(values) >= 10:
                baseline[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "p50": np.percentile(values, 50),
                    "p95": np.percentile(values, 95),
                    "trend": np.polyfit(range(len(values)), values, 1)[0]
                }

        self.performance_baselines[component_type] = baseline
        return baseline

    async def _recommend_evolution_config(self, current_config: Dict[str, Any],
                                        baseline: Dict[str, float]) -> List[OptimizationRecommendation]:
        """Recommend evolution engine configuration changes.

        Args:
            current_config: Current configuration
            baseline: Baseline performance metrics

        Returns:
            List of configuration recommendations
        """
        recommendations = []

        # Analyze fitness metrics
        if "fitness" in baseline:
            fitness_baseline = baseline["fitness"]

            # Low average fitness
            if fitness_baseline["mean"] < 0.5:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id="evolution_config_low_fitness",
                    category=OptimizationCategory.CONFIGURATION_TUNING,
                    title="Low Average Fitness Detected",
                    description=f"Average fitness is {fitness_baseline['mean']:.3f}. Consider parameter optimization.",
                    current_value=current_config.get("population_size", 100),
                    recommended_value=200,  # Increase population for better exploration
                    expected_improvement=20.0,
                    confidence=0.7,
                    implementation_steps=[
                        "Increase population size to 200",
                        "Adjust mutation rate to 0.15",
                        "Review fitness function"
                    ],
                    estimated_effort="low",
                    risk_level="low",
                    priority=2
                ))

            # Stagnating fitness (low trend)
            if abs(fitness_baseline["trend"]) < 0.001:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id="evolution_config_stagnating",
                    category=OptimizationCategory.CONFIGURATION_TUNING,
                    title="Fitness Stagnation Detected",
                    description=f"Fitness trend is {fitness_baseline['trend']:.6f} (stagnating). Increase genetic diversity.",
                    current_value=current_config.get("mutation_rate", 0.1),
                    recommended_value=0.2,  # Increase mutation rate
                    expected_improvement=15.0,
                    confidence=0.8,
                    implementation_steps=[
                        "Increase mutation rate to 0.2",
                        "Increase crossover rate to 0.9",
                        "Consider immigration rate"
                    ],
                    estimated_effort="low",
                    risk_level="medium",
                    priority=1
                ))

        return recommendations

    async def _recommend_kraken_config(self, current_config: Dict[str, Any],
                                     baseline: Dict[str, float]) -> List[OptimizationRecommendation]:
        """Recommend Kraken LNN configuration changes.

        Args:
            current_config: Current configuration
            baseline: Baseline performance metrics

        Returns:
            List of configuration recommendations
        """
        recommendations = []

        # Analyze processing latency
        if "processing_latency" in baseline:
            latency_baseline = baseline["processing_latency"]

            # High latency
            if latency_baseline["mean"] > 1000:  # 1 second
                recommendations.append(OptimizationRecommendation(
                    recommendation_id="kraken_config_high_latency",
                    category=OptimizationCategory.PERFORMANCE_TUNING,
                    title="High Processing Latency",
                    description=f"Average processing latency is {latency_baseline['mean']:.0f}ms. Optimize reservoir parameters.",
                    current_value=current_config.get("reservoir_size", 1000),
                    recommended_value=800,  # Reduce reservoir size
                    expected_improvement=30.0,
                    confidence=0.8,
                    implementation_steps=[
                        "Reduce reservoir size to 800",
                        "Adjust spectral radius to 0.8",
                        "Optimize leaking rate"
                    ],
                    estimated_effort="medium",
                    risk_level="low",
                    priority=1
                ))

        # Analyze memory usage
        if "memory_usage" in baseline:
            memory_baseline = baseline["memory_usage"]

            # High memory usage
            if memory_baseline["mean"] > 80.0:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id="kraken_config_high_memory",
                    category=OptimizationCategory.RESOURCE_ALLOCATION,
                    title="High Memory Usage in Kraken LNN",
                    description=f"Average memory usage is {memory_baseline['mean']:.1f}%. Reduce reservoir complexity.",
                    current_value=current_config.get("reservoir_size", 1000),
                    recommended_value=600,  # Reduce reservoir size
                    expected_improvement=25.0,
                    confidence=0.7,
                    implementation_steps=[
                        "Reduce reservoir size to 600",
                        "Implement sparse connectivity",
                        "Consider hierarchical reservoirs"
                    ],
                    estimated_effort="high",
                    risk_level="medium",
                    priority=1
                ))

        return recommendations

    async def _recommend_nlp_config(self, current_config: Dict[str, Any],
                                  baseline: Dict[str, float]) -> List[OptimizationRecommendation]:
        """Recommend NLP agent configuration changes.

        Args:
            current_config: Current configuration
            baseline: Baseline performance metrics

        Returns:
            List of configuration recommendations
        """
        recommendations = []

        # Analyze response time
        if "response_time" in baseline:
            response_baseline = baseline["response_time"]

            # High response time
            if response_baseline["mean"] > 3000:  # 3 seconds
                recommendations.append(OptimizationRecommendation(
                    recommendation_id="nlp_config_high_response_time",
                    category=OptimizationCategory.PERFORMANCE_TUNING,
                    title="High Response Time",
                    description=f"Average response time is {response_baseline['mean']:.0f}ms. Optimize token and prompt settings.",
                    current_value=current_config.get("max_tokens", 1000),
                    recommended_value=800,  # Reduce max tokens
                    expected_improvement=35.0,
                    confidence=0.8,
                    implementation_steps=[
                        "Reduce max_tokens to 800",
                        "Optimize prompt engineering",
                        "Consider faster model variants"
                    ],
                    estimated_effort="low",
                    risk_level="low",
                    priority=2
                ))

        # Analyze token usage efficiency
        if "tokens_per_response" in baseline:
            token_baseline = baseline["tokens_per_response"]

            # Inefficient token usage
            if token_baseline["p95"] > current_config.get("max_tokens", 1000) * 0.9:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id="nlp_config_token_efficiency",
                    category=OptimizationCategory.CONFIGURATION_TUNING,
                    title="Inefficient Token Usage",
                    description=f"95th percentile token usage is {token_baseline['p95']:.0f}, near max limit.",
                    current_value=current_config.get("max_tokens", 1000),
                    recommended_value=int(token_baseline["p95"] * 1.2),  # 20% buffer
                    expected_improvement=10.0,
                    confidence=0.6,
                    implementation_steps=[
                        "Increase max_tokens to handle peak usage",
                        "Optimize prompt structure",
                        "Implement token budgeting"
                    ],
                    estimated_effort="low",
                    risk_level="low",
                    priority=3
                ))

        return recommendations


class OptimizationEngine:
    """Main optimization engine coordinator."""

    def __init__(self, config: OptimizationEngineConfig):
        """Initialize optimization engine.

        Args:
            config: Optimization engine configuration
        """
        self.config = config
        self.performance_optimizer = PerformanceOptimizer(config)
        self.configuration_recommender = ConfigurationRecommender(config)
        self.recommendation_history = deque(maxlen=1000)
        self.optimization_metrics = {}

    async def optimize_system(self, metrics_history: Dict[str, List[MLMetric]],
                            predictions: Dict[str, List[PredictionResult]],
                            current_configs: Dict[str, Dict[str, Any]]) -> List[OptimizationRecommendation]:
        """Perform comprehensive system optimization.

        Args:
            metrics_history: Historical metrics by component
            predictions: Performance predictions by component
            current_configs: Current configurations by component

        Returns:
            List of optimization recommendations
        """
        all_recommendations = []

        try:
            # Performance optimization analysis
            performance_recommendations = await self.performance_optimizer.analyze_performance(
                metrics_history, predictions
            )
            all_recommendations.extend(performance_recommendations)

            # Configuration optimization analysis
            for component_type, config in current_configs.items():
                if component_type in metrics_history:
                    config_recommendations = await self.configuration_recommender.recommend_configuration_changes(
                        component_type, config, metrics_history[component_type]
                    )
                    all_recommendations.extend(config_recommendations)

            # Store recommendations in history
            self.recommendation_history.extend(all_recommendations)

            # Clean up expired recommendations
            await self._cleanup_expired_recommendations()

            return all_recommendations

        except Exception as e:
            logger.error(f"System optimization failed: {e}")
            return []

    async def _cleanup_expired_recommendations(self) -> None:
        """Clean up expired recommendations."""
        datetime.now(timezone.utc)
        self.recommendation_history = deque(
            [rec for rec in self.recommendation_history if not rec.is_expired()],
            maxlen=1000
        )

    async def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary and statistics.

        Returns:
            Optimization summary dictionary
        """
        active_recommendations = [rec for rec in self.recommendation_history if not rec.is_expired()]

        # Group by category
        category_counts = defaultdict(int)
        for rec in active_recommendations:
            category_counts[rec.category.value] += 1

        # Group by component
        component_counts = defaultdict(int)
        for rec in active_recommendations:
            component_counts[rec.component_type.value] += 1

        # Calculate priority distribution
        priority_distribution = defaultdict(int)
        for rec in active_recommendations:
            priority_distribution[f"priority_{rec.priority}"] += 1

        return {
            "total_active_recommendations": len(active_recommendations),
            "category_distribution": dict(category_counts),
            "component_distribution": dict(component_counts),
            "priority_distribution": dict(priority_distribution),
            "high_priority_count": len([r for r in active_recommendations if r.priority <= 2]),
            "average_confidence": np.mean([r.confidence for r in active_recommendations]) if active_recommendations else 0.0,
            "average_expected_improvement": np.mean([r.expected_improvement for r in active_recommendations]) if active_recommendations else 0.0
        }

    async def export_recommendations(self, filepath: Path) -> None:
        """Export recommendations to file.

        Args:
            filepath: Path to export file
        """
        try:
            recommendations_data = [rec.to_dict() for rec in self.recommendation_history]

            with open(filepath, 'w') as f:
                json.dump(recommendations_data, f, indent=2, default=str)

            logger.info(f"Recommendations exported to {filepath}")

        except Exception as e:
            logger.error(f"Failed to export recommendations: {e}")
