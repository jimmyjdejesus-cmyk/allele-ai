import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np

from phylogenic.observability.ml_analytics.anomaly_detection import (
    IsolationForestDetector,
)
from phylogenic.observability.ml_analytics.ml_config import AnomalyDetectionConfig
from phylogenic.observability.ml_analytics.types import ComponentType, MLMetric

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

    model = detector.model
    print("model detection threshold:", getattr(model, "_detection_threshold", None))
    scores = getattr(model, "_training_anomaly_scores", None)
    if scores is not None:
        print(
            "training anomaly scores (min,max,median):",
            float(np.min(scores)),
            float(np.max(scores)),
            float(np.median(scores)),
        )
    # Inspect anomaly scores for last 20
    for m in metrics[80:90]:
        r = await detector.detect_anomaly(m)
        print("value", m.value, "->", (r.anomaly_score if r else None))


asyncio.run(run())
