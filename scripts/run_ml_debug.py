import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np

from phylogenic.observability.ml_analytics.anomaly_detection import (
    IsolationForestDetector,
)
from phylogenic.observability.ml_analytics.ml_config import (
    AnomalyDetectionConfig,
    PredictiveAnalyticsConfig,
)
from phylogenic.observability.ml_analytics.predictive_analytics import (
    TimeSeriesForecaster,
)
from phylogenic.observability.ml_analytics.types import (
    ComponentType,
    MLMetric,
    TimeSeriesData,
)

# Create sample metrics
base_time = datetime.now(timezone.utc)
metrics = []
for i in range(100):
    is_anomaly = i > 80
    value = np.random.normal(2.0, 0.2) if is_anomaly else np.random.normal(0.5, 0.1)
    metrics.append(
        MLMetric(
            timestamp=base_time + timedelta(minutes=i),
            component_type=ComponentType.EVOLUTION_ENGINE,
            component_id="test_engine",
            metric_name="fitness_score",
            value=value,
            metadata={"generation": i},
        )
    )


async def run():
    anomaly_config = AnomalyDetectionConfig(min_training_samples=50)
    detector = IsolationForestDetector(anomaly_config)
    await detector.train(metrics[:80])

    normal_metric = MLMetric(
        timestamp=datetime.now(timezone.utc),
        component_type=ComponentType.EVOLUTION_ENGINE,
        component_id="test_engine",
        metric_name="fitness_score",
        value=0.5,
        metadata={},
    )
    result = await detector.detect_anomaly(normal_metric)
    print("Anomaly detection result for normal metric:", result)

    # Check anomalies in last 20 metrics
    anomalies = []
    for metric in metrics[80:]:
        r = await detector.detect_anomaly(metric)
        if r:
            anomalies.append(r)
    print("Anomalies detected in last 20:", len(anomalies))

    # Predictive
    predictive_config = PredictiveAnalyticsConfig(min_training_samples=50)
    ts_data = TimeSeriesData(
        timestamps=[m.timestamp for m in metrics],
        values=[m.value for m in metrics],
        component_type=ComponentType.EVOLUTION_ENGINE,
        component_id="test_engine",
        metric_name="fitness_score",
    )
    forecaster = TimeSeriesForecaster(predictive_config)
    await forecaster.train({"evolution_engine": ts_data})
    prediction = await forecaster.forecast("evolution_engine", horizon_minutes=60)
    print("Prediction:", prediction)


asyncio.run(run())
