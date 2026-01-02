# ML Analytics Documentation

Allele's Machine Learning Analytics system provides comprehensive monitoring, anomaly detection, predictive analytics, and intelligent optimization for AI-powered conversational agents.

## Overview

The ML Analytics system consists of four core components:

- **Anomaly Detection**: Real-time identification of unusual patterns in agent performance
- **Predictive Analytics**: Forecasting future performance trends and resource usage
- **Alert Intelligence**: Smart alert clustering, correlation, and escalation management
- **Optimization Engine**: Automated performance tuning and configuration recommendations

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  ML Analytics Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│  Data Collection → Preprocessing → Feature Engineering     │
│                           ↓                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Anomaly         │  │ Predictive      │  │ Alert       │ │
│  │ Detection       │  │ Analytics       │  │ Intelligence│ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
│                           ↓                                 │
│              ┌─────────────────────┐                         │
│              │ Optimization Engine │                         │
│              └─────────────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Anomaly Detection

Detects unusual patterns in agent behavior and system performance.

#### Supported Algorithms

- **Isolation Forest**: Effective for high-dimensional anomaly detection
- **One-Class SVM**: Robust outlier detection for normal behavior modeling
- **Ensemble Methods**: Combined approach for improved accuracy

#### Anomaly Types

- `performance_degradation`: Agent response quality decline
- `resource_spike`: Unusual CPU/memory/usage patterns
- `memory_leak`: Gradual memory consumption increase
- `cpu_overload`: Processing time spikes
- `latency_spike`: Response time anomalies
- `error_rate_spike`: Increased error frequency
- `throughput_drop`: Reduced processing capacity
- `unusual_pattern`: Generic behavioral anomalies

#### Example Usage

```python
from src.allele.observability.ml_analytics import AnomalyDetectionConfig, IsolationForestDetector

# Configure anomaly detection
config = AnomalyDetectionConfig(
    contamination=0.1,  # Expected fraction of anomalies
    min_training_samples=100
)

# Initialize detector
detector = IsolationForestDetector(config)

# Train on historical data
await detector.train(metrics_history)

# Detect anomalies in new data
anomaly_result = await detector.detect_anomaly(metric)

if anomaly_result:
    print(f"Anomaly detected: {anomaly_result.anomaly_type}")
    print(f"Score: {anomaly_result.anomaly_score:.3f}")
    print(f"Severity: {anomaly_result.severity}")
```

### 2. Predictive Analytics

Forecasts future performance trends and resource requirements.

#### Prediction Types

- `performance_forecast`: Agent quality and response time predictions
- `resource_usage`: CPU, memory, and storage usage forecasts
- `error_probability`: Likelihood of system failures
- `capacity_planning`: Resource scaling recommendations
- `performance_trend`: Long-term performance trajectory analysis
- `anomaly_risk`: Probability of future anomalies

#### Example Usage

```python
from src.allele.observability.ml_analytics import PredictiveAnalyticsConfig, ARIMAPredictor

# Configure predictor
config = PredictiveAnalyticsConfig(
    prediction_horizon_minutes=60,
    confidence_level=0.95
)

# Initialize predictor
predictor = ARIMAPredictor(config)

# Train on time series data
await predictor.train(time_series_data)

# Generate predictions
prediction = await predictor.predict(
    component_type=ComponentType.EVOLUTION_ENGINE,
    metric_name="fitness_score"
)

print(f"Predicted value: {prediction.predicted_value}")
print(f"Confidence interval: {prediction.confidence_interval}")
```

### 3. Alert Intelligence

Provides intelligent alert management with clustering and correlation.

#### Clustering Algorithms

- **DBSCAN**: Density-based clustering for anomaly grouping
- **K-Means**: Centroid-based clustering for similar alerts
- **Hierarchical**: Tree-based clustering for nested relationships
- **Simple Clustering**: Rule-based grouping for basic scenarios

#### Alert Features

- **Correlation**: Identifies related alerts across components
- **Deduplication**: Removes redundant alerts based on similarity
- **Escalation**: Automatic priority-based escalation
- **Root Cause Analysis**: Intelligent root cause candidate generation

#### Example Usage

```python
from src.allele.observability.ml_analytics import (
    AlertIntelligenceConfig, IntelligentAlertManager
)

# Configure alert intelligence
config = AlertIntelligenceConfig(
    clustering_algorithm="dbscan",
    similarity_threshold=0.8,
    deduplication_window_minutes=30
)

# Initialize alert manager
manager = IntelligentAlertManager(config)

# Process anomalies into intelligent alerts
alerts = await manager.process_alerts(anomalies)

# Cluster related alerts
clusters = await manager.correlator.process_alert_batch(alerts)

for cluster in clusters:
    print(f"Cluster: {cluster.cluster_id}")
    print(f"Type: {cluster.cluster_type}")
    print(f"Confidence: {cluster.confidence:.2f}")
    print(f"Priority: {cluster.priority_score:.2f}")
```

### 4. Optimization Engine

Automatically recommends configuration and performance optimizations.

#### Optimization Categories

- `configuration_tuning`: Parameter optimization recommendations
- `resource_allocation`: CPU/memory distribution suggestions
- `performance_tuning`: Speed and efficiency improvements
- `capacity_scaling`: Infrastructure scaling recommendations
- `alert_thresholds`: Monitoring threshold adjustments
- `system_settings`: Configuration parameter updates

#### Recommendation Features

- **Confidence Scoring**: Reliability assessment of recommendations
- **Impact Estimation**: Expected improvement predictions
- **Risk Assessment**: Implementation risk evaluation
- **Priority Ranking**: Importance-based recommendation ordering

#### Example Usage

```python
from src.allele.observability.ml_analytics import (
    OptimizationEngineConfig, OptimizationEngine
)

# Configure optimization engine
config = OptimizationEngineConfig(
    enable_ml_based=True,
    min_confidence_threshold=0.7,
    min_expected_improvement=10.0
)

# Initialize optimization engine
optimizer = OptimizationEngine(config)

# Perform system optimization
recommendations = await optimizer.optimize_system(
    metrics_history=metrics_history,
    predictions=predictions,
    current_configs=configs
)

for rec in recommendations:
    print(f"Recommendation: {rec.title}")
    print(f"Category: {rec.category}")
    print(f"Expected improvement: {rec.expected_improvement:.1f}%")
    print(f"Confidence: {rec.confidence:.2f}")
    print(f"Priority: {rec.priority}")
```

## Configuration

### AlertIntelligenceConfig

```python
@dataclass
class AlertIntelligenceConfig:
    # Clustering settings
    clustering_algorithm: str = "dbscan"  # dbscan, kmeans, hierarchical, simple
    clustering_eps: float = 0.5
    clustering_min_samples: int = 3
    similarity_threshold: float = 0.8
    
    # Correlation settings
    correlation_window_minutes: int = 60
    deduplication_window_minutes: int = 30
    auto_acknowledge_duplicates: bool = True
    
    # Alert management
    max_root_cause_candidates: int = 5
    escalation_time_minutes: int = 60
    priority_factors: Dict[str, float] = field(default_factory=dict)
```

### OptimizationEngineConfig

```python
@dataclass
class OptimizationEngineConfig:
    # ML-based optimization
    enable_ml_based: bool = True
    min_confidence_threshold: float = 0.7
    min_expected_improvement: float = 5.0  # percentage
    batch_optimization_size: int = 10
    
    # Rule-based optimization
    rule_file_path: Optional[Path] = None
    
    # Analysis settings
    analysis_window_hours: int = 24
    prediction_horizon_hours: int = 6
```

## Data Models

### MLMetric

Represents a single metric observation with temporal and component information.

```python
@dataclass
class MLMetric:
    timestamp: datetime
    component_type: ComponentType
    component_id: str
    metric_name: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### AnomalyResult

Contains detailed information about detected anomalies.

```python
@dataclass
class AnomalyResult:
    timestamp: datetime
    component_type: ComponentType
    component_id: str
    metric_name: str
    anomaly_type: AnomalyType
    anomaly_score: float
    confidence: float
    severity: AlertSeverity
    actual_value: float
    expected_value: float
    deviation: float
    threshold: float
    context: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
```

### AlertCluster

Groups related alerts for efficient analysis and response.

```python
@dataclass
class AlertCluster:
    cluster_id: str
    cluster_type: str
    alerts: List[Dict[str, Any]]
    common_attributes: Dict[str, Any] = field(default_factory=dict)
    root_cause_candidates: List[str] = field(default_factory=list)
    confidence: float = 0.0
    priority_score: float = 0.0
    impact_assessment: str = ""
    first_alert_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_alert_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_minutes: float = 0.0
```

### OptimizationRecommendation

Provides actionable optimization suggestions with implementation details.

```python
@dataclass
class OptimizationRecommendation:
    recommendation_id: str
    category: OptimizationCategory
    title: str
    description: str
    current_value: Any
    recommended_value: Any
    expected_improvement: float
    confidence: float
    implementation_steps: List[str] = field(default_factory=list)
    estimated_effort: str = "low"
    risk_level: str = "low"
    component_type: ComponentType = ComponentType.SYSTEM
    priority: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
```

## Integration Examples

### End-to-End Workflow

```python
import asyncio
from datetime import datetime, timedelta, timezone

async def ml_analytics_workflow():
    """Complete ML analytics workflow example."""
    
    # 1. Initialize components
    anomaly_config = AnomalyDetectionConfig(contamination=0.1)
    detector = IsolationForestDetector(anomaly_config)
    
    alert_config = AlertIntelligenceConfig(clustering_algorithm="dbscan")
    alert_manager = IntelligentAlertManager(alert_config)
    
    opt_config = OptimizationEngineConfig(enable_ml_based=True)
    optimizer = OptimizationEngine(opt_config)
    
    # 2. Collect metrics (simulated)
    base_time = datetime.now(timezone.utc)
    metrics = []
    
    for i in range(100):
        # Normal performance with occasional anomalies
        is_anomaly = i > 80 and i % 10 == 0
        value = 0.8 + (0.1 * np.random.normal()) if not is_anomaly else 2.0
        
        metric = MLMetric(
            timestamp=base_time + timedelta(minutes=i),
            component_type=ComponentType.EVOLUTION_ENGINE,
            component_id="test_engine",
            metric_name="fitness_score",
            value=value
        )
        metrics.append(metric)
    
    # 3. Train anomaly detector
    await detector.train(metrics[:80])
    
    # 4. Detect anomalies in recent data
    anomalies = []
    for metric in metrics[80:]:
        result = await detector.detect_anomaly(metric)
        if result:
            anomalies.append(result)
    
    print(f"Detected {len(anomalies)} anomalies")
    
    # 5. Process alerts
    alerts = await alert_manager.process_alerts(anomalies)
    print(f"Generated {len(alerts)} alerts")
    
    # 6. Generate optimization recommendations
    predictions = {}  # Would contain actual predictions
    configs = {"evolution_engine": {"population_size": 100}}
    
    recommendations = await optimizer.optimize_system(
        metrics_history={"evolution_engine": metrics},
        predictions=predictions,
        current_configs=configs
    )
    
    print(f"Generated {len(recommendations)} optimization recommendations")
    
    # 7. Display results
    for rec in recommendations[:3]:  # Top 3 recommendations
        print(f"\nRecommendation: {rec.title}")
        print(f"Description: {rec.description}")
        print(f"Expected improvement: {rec.expected_improvement:.1f}%")
        print(f"Confidence: {rec.confidence:.2f}")

# Run the workflow
if __name__ == "__main__":
    asyncio.run(ml_analytics_workflow())
```

### Monitoring Dashboard Integration

```python
class MLAnalyticsDashboard:
    """Integration example for monitoring dashboards."""
    
    def __init__(self):
        self.detector = IsolationForestDetector(AnomalyDetectionConfig())
        self.alert_manager = IntelligentAlertManager(AlertIntelligenceConfig())
        self.optimizer = OptimizationEngine(OptimizationEngineConfig())
    
    async def get_system_health(self):
        """Get comprehensive system health metrics."""
        
        # Collect current metrics
        current_metrics = await self._collect_current_metrics()
        
        # Detect anomalies
        anomalies = []
        for metric in current_metrics:
            result = await self.detector.detect_anomaly(metric)
            if result:
                anomalies.append(result)
        
        # Generate alerts
        alerts = await self.alert_manager.process_alerts(anomalies)
        
        # Get optimization recommendations
        recommendations = await self._get_optimization_recommendations()
        
        return {
            "health_score": self._calculate_health_score(anomalies),
            "active_anomalies": len(anomalies),
            "active_alerts": len(alerts),
            "pending_recommendations": len(recommendations),
            "top_recommendations": recommendations[:5]
        }
    
    async def _collect_current_metrics(self):
        """Collect current system metrics."""
        # Implementation would collect real metrics
        pass
    
    def _calculate_health_score(self, anomalies):
        """Calculate overall system health score."""
        if not anomalies:
            return 100.0
        
        # Deduct points based on anomaly severity and count
        penalty = sum(
            20 if a.severity == AlertSeverity.CRITICAL else
            15 if a.severity == AlertSeverity.ERROR else
            10 if a.severity == AlertSeverity.WARNING else 5
            for a in anomalies
        )
        
        return max(0.0, 100.0 - penalty)
```

## Best Practices

### Data Quality

1. **Ensure sufficient training data**: Minimum 100 samples for reliable anomaly detection
2. **Regular model retraining**: Update models weekly or when performance degrades
3. **Feature engineering**: Include relevant contextual features for better accuracy
4. **Data validation**: Validate input data ranges and formats

### Performance Optimization

1. **Batch processing**: Process multiple metrics together for efficiency
2. **Caching**: Cache model predictions and clustering results
3. **Sampling**: Use statistical sampling for large datasets
4. **Parallel processing**: Leverage async/await for concurrent operations

### Alert Management

1. **Threshold tuning**: Adjust similarity thresholds based on environment
2. **Escalation policies**: Define clear escalation timelines and responsibilities
3. **Noise reduction**: Use intelligent deduplication to reduce alert fatigue
4. **Context preservation**: Maintain relevant context for root cause analysis

### Monitoring and Maintenance

1. **Model drift detection**: Monitor for performance degradation over time
2. **A/B testing**: Test optimization recommendations before full deployment
3. **Feedback loops**: Incorporate user feedback to improve recommendations
4. **Regular audits**: Review and update configuration parameters periodically

## Troubleshooting

### Common Issues

**Anomaly Detection False Positives**
- Reduce `contamination` parameter in configuration
- Increase `min_training_samples` for better baseline
- Review feature engineering for noise reduction

**Clustering Algorithm Performance**
- DBSCAN: Tune `eps` parameter for density sensitivity
- K-Means: Specify appropriate `n_clusters` or use elbow method
- Consider switching to simpler algorithms for small datasets

**Optimization Recommendations Too Conservative**
- Lower `min_confidence_threshold` to accept riskier recommendations
- Adjust `min_expected_improvement` for more suggestions
- Enable ML-based optimization for data-driven recommendations

**High Memory Usage**
- Implement metric sampling for large time series
- Use sliding windows for historical data analysis
- Clear caches periodically in long-running processes

### Debugging

Enable debug logging to troubleshoot issues:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('allele.ml_analytics')

# Check model training status
logger.debug(f"Training samples: {len(training_data)}")
logger.debug(f"Model accuracy: {model_accuracy}")

# Monitor clustering results
logger.debug(f"Number of clusters: {len(clusters)}")
logger.debug(f"Cluster sizes: {[len(c.alerts) for c in clusters]}")
```

## API Reference

For complete API documentation and OpenAPI specifications, see:
- [ML Analytics Schemas](./api/schemas.yaml) - Data model definitions
- [API Reference](./api.md) - REST API documentation

## Performance Benchmarks

| Operation | Dataset Size | Typical Latency | Memory Usage |
|-----------|--------------|----------------|--------------|
| Anomaly Detection | 1K samples | ~50ms | ~10MB |
| Clustering | 500 alerts | ~200ms | ~5MB |
| Prediction | 100 time points | ~100ms | ~15MB |
| Optimization | 10 components | ~500ms | ~20MB |

*Performance may vary based on hardware configuration and data characteristics.*
