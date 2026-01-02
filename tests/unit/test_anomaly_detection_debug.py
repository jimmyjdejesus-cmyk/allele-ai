from datetime import datetime, timedelta, timezone

import numpy as np

from phylogenic.observability.ml_analytics.anomaly_detection import (
    IsolationForestDetector,
)
from phylogenic.observability.ml_analytics.ml_config import AnomalyDetectionConfig
from phylogenic.observability.ml_analytics.types import ComponentType, MLMetric


def test_anomaly_detection_training_debug():
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

    cfg = AnomalyDetectionConfig(min_training_samples=50)
    detector = IsolationForestDetector(cfg)

    # Train on first 80
    import asyncio

    asyncio.run(detector.train(metrics[:80]))

    # Inspect model thresholds and training anomaly scores
    model = detector.model
    training_scores = getattr(model, "_training_anomaly_scores", None)
    thresh = getattr(model, "_detection_threshold", None)

    assert training_scores is not None, "Training scores should be stored on model"
    assert thresh is not None, "Detection threshold should be stored on model"

    # Compute anomaly scores for the last 20 metrics
    anomalies_scores = []
    for m in metrics[80:]:
        import asyncio

        result = asyncio.run(detector.detect_anomaly(m))
        if result:
            anomalies_scores.append(result.anomaly_score)

    # For debugging we expect at least one anomaly to be flagged
    min_s = training_scores.min()
    max_s = training_scores.max()
    mean_s = training_scores.mean()

    assert (
        len(anomalies_scores) > 0
    ), f"No anomalies detected; thresh={thresh}, stats=({min_s},{max_s},{mean_s})"
