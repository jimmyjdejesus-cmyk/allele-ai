import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure script is runnable from repository root by adding src/ to PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def main() -> None:
    import asyncio

    import numpy as np

    from phylogenic.observability.ml_analytics.ml_config import (
        PredictiveAnalyticsConfig,
    )
    from phylogenic.observability.ml_analytics.predictive_analytics import (
        TimeSeriesForecaster,
    )
    from phylogenic.observability.ml_analytics.types import (
        ComponentType,
        TimeSeriesData,
    )

    config = PredictiveAnalyticsConfig(min_training_samples=50)
    forecaster = TimeSeriesForecaster(config)
    base_time = datetime.now(timezone.utc)
    timestamps = [base_time + timedelta(minutes=i) for i in range(100)]
    values = [0.5 + 0.01 * i + float(np.random.normal(0, 0.05)) for i in range(100)]
    ts = TimeSeriesData(
        timestamps=timestamps,
        values=values,
        component_type=ComponentType.EVOLUTION_ENGINE,
        component_id="test_engine_1",
        metric_name="fitness_score",
        frequency_minutes=1,
    )

    metrics = asyncio.run(forecaster.train({"evolution_engine": ts}))
    print("metrics:", metrics)


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))
    main()
