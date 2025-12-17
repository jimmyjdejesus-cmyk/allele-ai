# Copyright (C) 2025 Phylogenic AI Labs & Jimmy De Jesus
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

"""Predictive Analytics Engine for Allele ML Analytics.

This module provides time series forecasting and performance prediction
using ARIMA, LSTM, and other advanced machine learning techniques.

Author: Bravetto AI Systems
Version: 1.0.0
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, cast

import numpy as np

from .ml_config import PredictiveAnalyticsConfig
from .types import (
    MLMetric,
    ModelMetrics,
    ModelStatus,
    PredictionResult,
    PredictionType,
    TimeSeriesData,
)

logger = logging.getLogger(__name__)


class TimeSeriesForecaster:
    """Time series forecasting using ARIMA and LSTM models."""

    def __init__(self, config: PredictiveAnalyticsConfig):
        """Initialize time series forecaster.

        Args:
            config: Predictive analytics configuration
        """
        self.config = config
        self.models: Dict[str, Any] = {}  # component_type -> model
        self.scalers: Dict[str, Any] = {}  # component_type -> scaler
        self.is_trained: Dict[str, bool] = {}
        self.last_training_time: Dict[str, datetime] = {}
        self.training_data: Dict[str, TimeSeriesData] = {}

    async def train(
        self, time_series_data: Dict[str, TimeSeriesData]
    ) -> Dict[str, ModelMetrics]:
        """Train forecasting models for different components.

        Args:
            time_series_data: Dictionary of component_type -> TimeSeriesData

        Returns:
            Dictionary of component_type -> ModelMetrics
        """
        training_metrics = {}

        for component_type, ts_data in time_series_data.items():
            try:
                if len(ts_data.values) < self.config.min_training_samples:
                    logger.warning(
                        f"Insufficient data for {component_type}: "
                        f"{len(ts_data.values)} < {self.config.min_training_samples}"
                    )
                    continue

                # Train ARIMA model; if ARIMA training fails, create a fallback
                try:
                    arima_metrics = await self._train_arima(component_type, ts_data)
                except Exception as e:
                    logger.warning(

                            f"ARIMA training failed for {component_type}: {e}. "
                            "Using fallback metrics."

                    )
                    # Populate a fallback model entry so forecasting can proceed
                    self.models[component_type] = {
                        "type": "arima",
                        "model": None,
                        "original_data": np.array(ts_data.values),
                        "is_stationary": False,
                    }
                    self.training_data[component_type] = ts_data
                    self.is_trained[component_type] = True
                    self.last_training_time[component_type] = datetime.now(timezone.utc)

                    arima_metrics = ModelMetrics(
                        model_name=f"ARIMA-{component_type}",
                        model_version="1.0.0",
                        status=ModelStatus.READY,
                        training_samples=len(ts_data.values),
                        last_training_time=self.last_training_time[component_type],
                        accuracy=0.5,
                        precision=0.0,
                        total_predictions=0,
                        successful_predictions=0,
                    )

                # Defensive: ensure arima_metrics is always a ModelMetrics instance
                if arima_metrics is None:
                    logger.warning(

                            f"ARIMA returned None for {component_type}; "
                            "using fallback metrics."

                    )
                    self.models[component_type] = {
                        "type": "arima",
                        "model": None,
                        "original_data": np.array(ts_data.values),
                        "is_stationary": False,
                    }
                    self.training_data[component_type] = ts_data
                    self.is_trained[component_type] = True
                    self.last_training_time[component_type] = datetime.now(timezone.utc)
                    arima_metrics = ModelMetrics(
                        model_name=f"ARIMA-{component_type}",
                        model_version="1.0.0",
                        status=ModelStatus.READY,
                        training_samples=len(ts_data.values),
                        last_training_time=self.last_training_time[component_type],
                        accuracy=0.5,
                        precision=0.0,
                        total_predictions=0,
                        successful_predictions=0,
                    )

                # Train LSTM model (if TensorFlow available)
                try:
                    lstm_metrics = await self._train_lstm(component_type, ts_data)
                    # Use the better performing model
                    if lstm_metrics.accuracy > arima_metrics.accuracy:
                        training_metrics[component_type] = lstm_metrics
                    else:
                        training_metrics[component_type] = arima_metrics
                except ImportError:
                    logger.warning("TensorFlow not available, using ARIMA only")
                    training_metrics[component_type] = arima_metrics

            except Exception as e:
                logger.error(f"Training failed for {component_type}: {e}")
                continue

        return training_metrics

    async def _train_arima(
        self, component_type: str, ts_data: TimeSeriesData
    ) -> ModelMetrics:
        """Train ARIMA model for time series forecasting.

        Args:
            component_type: Type of component
            ts_data: Time series data

        Returns:
            Model training metrics
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA  # type: ignore[import-untyped]
            from statsmodels.tsa.stattools import adfuller  # type: ignore[import-untyped]

            values = np.array(ts_data.values)

            # Check stationarity
            adf_test = adfuller(values)
            if adf_test[1] > 0.05:
                # Non-stationary, use differencing
                differenced_values = np.diff(values)
                p, d, q = self.config.arima_order
            else:
                # Stationary
                differenced_values = values
                p, d, q = (self.config.arima_order[0], 0, self.config.arima_order[2])

            # Fit ARIMA model
            model = ARIMA(differenced_values, order=(p, d, q))
            fitted_model = model.fit()

            # Store model
            self.models[component_type] = {
                "type": "arima",
                "model": fitted_model,
                "original_data": values,
                "is_stationary": adf_test[1] <= 0.05,
            }

            self.is_trained[component_type] = True
            self.last_training_time[component_type] = datetime.now(timezone.utc)
            self.training_data[component_type] = ts_data

            # Calculate training metrics (simplified)
            predictions = fitted_model.fittedvalues
            actual_values = differenced_values[len(predictions) :]

            # Initialize metrics with default values
            mae = 0.0
            mse = 0.0
            accuracy = 0.8

            if len(actual_values) > 0:
                mae = float(
                    np.mean(np.abs(actual_values - predictions[-len(actual_values) :]))
                )
                mse = float(
                    np.mean((actual_values - predictions[-len(actual_values) :]) ** 2)
                )

                # Safe division: check for zero mean absolute values
                # to avoid division by zero
                mean_abs_values = np.mean(np.abs(actual_values))
                if mean_abs_values > 0:
                    accuracy = float(1.0 / (1.0 + mae / mean_abs_values))
                else:
                    # Handle constant time series (all zeros) with fallback accuracy
                    accuracy = 0.5  # Conservative accuracy for constant data

            metrics = ModelMetrics(
                model_name=f"ARIMA-{component_type}",
                model_version="1.0.0",
                status=ModelStatus.READY,
                training_samples=len(values),
                last_training_time=self.last_training_time[component_type],
                accuracy=accuracy,
                precision=float(
                    max(0.0, min(1.0, 1.0 - mse / np.var(values)))
                    if np.var(values) > 0
                    else 0.0
                ),
                total_predictions=0,
                successful_predictions=0,
            )

            logger.info(
                f"ARIMA model trained for {component_type} with accuracy {accuracy:.3f}"
            )
            return metrics

        except ImportError:
            # Provide a lightweight fallback ARIMA-like model when
            # statsmodels is not available so tests and CI can run
            # in minimal environments. This simple trend model is not
            # intended for production use.
            logger.warning(
                "statsmodels not available, using simple linear-trend "
                "fallback for ARIMA"
            )

            values = np.array(ts_data.values)
            x = np.arange(len(values))
            A = np.vstack([x, np.ones(len(x))]).T
            slope, intercept = np.linalg.lstsq(A, values, rcond=None)[0]

            class _FallbackARIMA:
                def __init__(self, slope: float, intercept: float, values: np.ndarray, x: np.ndarray) -> None:
                    self.slope = slope
                    self.intercept = intercept
                    self.fittedvalues = values
                    self._x = x
                def forecast(self, steps: int = 1) -> Any:
                    last_x = int(self._x[-1])
                    return np.asarray(
                        [
                            self.intercept + self.slope * (last_x + i + 1)
                            for i in range(steps)
                        ],
                        dtype=float,
                    )
                def get_forecast(self, steps: int = 1) -> Any:
                    class _Res:
                        def __init__(self, arr: np.ndarray) -> None:
                            self._arr = arr

                        def conf_int(self) -> np.ndarray:
                            # Naive symmetric confidence interval (10% padding)
                            lower = self._arr * 0.9
                            upper = self._arr * 1.1
                            return np.asarray(np.vstack([lower, upper]).T, dtype=float)  # type: ignore[no-any-return]

                    return _Res(self.forecast(steps))

            fitted_model = _FallbackARIMA(slope, intercept, values, x)
            # Store fallback model and metadata similar to ARIMA branch
            self.models[component_type] = {
                "type": "arima_fallback",
                "model": fitted_model,
                "original_data": values,
                "is_stationary": False,
            }

            self.is_trained[component_type] = True
            self.last_training_time[component_type] = datetime.now(timezone.utc)
            self.training_data[component_type] = ts_data

            # Compute simple metrics for the fallback model
            preds = fitted_model.fittedvalues if hasattr(fitted_model, "fittedvalues") else fitted_model.forecast(1)
            mae = float(np.mean(np.abs(values[: len(preds)] - preds))) if len(preds) > 0 else 0.0
            mse = float(np.mean((values[: len(preds)] - preds) ** 2)) if len(preds) > 0 else 0.0
            accuracy = float(1.0 / (1.0 + mae / (np.mean(np.abs(values)) or 1.0)))

            metrics = ModelMetrics(
                model_name=f"ARIMA-Fallback-{component_type}",
                model_version="1.0.0",
                status=ModelStatus.READY,
                training_samples=len(values),
                last_training_time=self.last_training_time[component_type],
                accuracy=accuracy,
                precision=0.0,
                total_predictions=0,
                successful_predictions=0,
            )

            return metrics
        except Exception as e:
            logger.error(f"ARIMA training failed for {component_type}: {e}")
            raise

    async def _train_lstm(
        self, component_type: str, ts_data: TimeSeriesData
    ) -> ModelMetrics:
        """Train LSTM model for time series forecasting.

        Args:
            component_type: Type of component
            ts_data: Time series data

        Returns:
            Model training metrics
        """
        # Prefer using importlib to check availability of optional heavy deps
        import importlib.util

        try:
            if importlib.util.find_spec("tensorflow") is None:
                raise ImportError(

                        "TensorFlow is required for LSTM forecasting. "
                        "Install with: pip install tensorflow"

                )

            from sklearn.preprocessing import MinMaxScaler  # type: ignore[import-untyped]
            from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore[import-untyped]
            from tensorflow.keras.models import Sequential  # type: ignore[import-untyped]

            values = np.array(ts_data.values).reshape(-1, 1)

            # Scale data
            scaler = MinMaxScaler()
            scaled_values = scaler.fit_transform(values)
            self.scalers[component_type] = scaler

            # Prepare sequences
            sequence_length = self.config.lstm_sequence_length
            X, y = [], []
            for i in range(sequence_length, len(scaled_values)):
                X.append(scaled_values[i - sequence_length : i, 0])
                y.append(scaled_values[i, 0])

            if len(X) == 0:
                raise ValueError("Insufficient data for LSTM training")

            X, y = np.array(X), np.array(y)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            # Build LSTM model
            model = Sequential(
                [
                    LSTM(
                        self.config.lstm_units,
                        return_sequences=True,
                        input_shape=(sequence_length, 1),
                    ),
                    Dropout(0.2),
                    LSTM(self.config.lstm_units, return_sequences=False),
                    Dropout(0.2),
                    Dense(25),
                    Dense(1),
                ]
            )

            model.compile(optimizer="adam", loss="mse")

            # Train model
            history = model.fit(
                X,
                y,
                epochs=self.config.lstm_epochs,
                batch_size=32,
                validation_split=0.1,
                verbose=0,
            )

            # Store model
            self.models[component_type] = {
                "type": "lstm",
                "model": model,
                "sequence_length": sequence_length,
                "scaler": scaler,
            }

            self.is_trained[component_type] = True
            self.last_training_time[component_type] = datetime.now(timezone.utc)
            self.training_data[component_type] = ts_data

            # Calculate training metrics
            final_loss = history.history["loss"][-1]
            accuracy = max(0.0, min(1.0, 1.0 - final_loss))

            metrics = ModelMetrics(
                model_name=f"LSTM-{component_type}",
                model_version="1.0.0",
                status=ModelStatus.READY,
                training_samples=len(X),
                last_training_time=self.last_training_time[component_type],
                accuracy=accuracy,
                precision=accuracy,
                total_predictions=0,
                successful_predictions=0,
            )

            logger.info(
                f"LSTM model trained for {component_type} with accuracy {accuracy:.3f}"
            )
            return metrics

        except ImportError as e:
            raise ImportError(
                "TensorFlow is required for LSTM forecasting. "
                "Install with: pip install tensorflow"
            ) from e
        except Exception as e:
            logger.error(f"LSTM training failed for {component_type}: {e}")
            raise

    async def forecast(
        self, component_type: str, horizon_minutes: int
    ) -> Optional[PredictionResult]:
        """Generate forecast for a component.

        Args:
            component_type: Type of component
            horizon_minutes: Forecast horizon in minutes

        Returns:
            Prediction result
        """
        if not self.is_trained.get(component_type, False):
            logger.warning(f"No trained model for {component_type}")
            return None

        try:
            model_info = self.models[component_type]
            model = model_info["model"]
            ts_data = self.training_data[component_type]

            if model_info["type"] == "arima":
                return await self._arima_forecast(
                    component_type, model, ts_data, horizon_minutes
                )
            elif model_info["type"] == "lstm":
                return await self._lstm_forecast(
                    component_type, model, ts_data, horizon_minutes
                )
            else:
                raise ValueError(f"Unknown model type: {model_info['type']}")

        except Exception as e:
            logger.error(f"Forecasting failed for {component_type}: {e}")
            return None

    async def _arima_forecast(
        self,
        component_type: str,
        model: Any,
        ts_data: TimeSeriesData,
        horizon_minutes: int,
    ) -> PredictionResult:
        """Generate ARIMA forecast.

        Args:
            component_type: Type of component
            model: Fitted ARIMA model
            ts_data: Time series data
            horizon_minutes: Forecast horizon in minutes

        Returns:
            ARIMA prediction result
        """
        try:

            # Calculate number of steps to forecast
            steps = max(1, horizon_minutes // ts_data.frequency_minutes)

            # Generate forecast
            if model is None:
                # Simple fallback: use last value(s) as naive forecast
                last_value = float(ts_data.values[-1]) if ts_data.values else 0.0
                forecast_values = [last_value] * steps
                # Build a simple confidence interval using recent variance
                recent_std = (
                    float(np.std(ts_data.values[-min(len(ts_data.values), 10) :]))
                    if ts_data.values
                    else 0.0
                )
                lower_ci = forecast_values[-1] - 1.96 * recent_std
                upper_ci = forecast_values[-1] + 1.96 * recent_std
                predicted_value = float(forecast_values[-1])
            else:
                forecast = model.forecast(steps=steps)
                confidence_interval = model.get_forecast(steps=steps).conf_int()
                predicted_value = (
                    float(forecast.iloc[-1])
                    if hasattr(forecast, "iloc")
                    else float(forecast[-1])
                )
                lower_ci = (
                    float(confidence_interval.iloc[-1, 0])
                    if hasattr(confidence_interval, "iloc")
                    else float(confidence_interval[-1, 0])
                )
                upper_ci = (
                    float(confidence_interval.iloc[-1, 1])
                    if hasattr(confidence_interval, "iloc")
                    else float(confidence_interval[-1, 1])
                )

            # Get the most recent value for context
            ts_data.values[-1]

            # Calculate predicted value and confidence interval
            # Calculate confidence based on model accuracy and horizon

            # Calculate confidence based on model accuracy and horizon
            base_confidence = 0.8  # Base confidence
            horizon_penalty = min(
                0.3, horizon_minutes / (24 * 60)
            )  # Reduce confidence with longer horizon
            confidence = max(0.1, base_confidence - horizon_penalty)

            prediction_result = PredictionResult(
                timestamp=datetime.now(timezone.utc),
                prediction_type=PredictionType.PERFORMANCE_FORECAST,
                component_type=ts_data.component_type,
                component_id=ts_data.component_id,
                metric_name=ts_data.metric_name,
                predicted_value=predicted_value,
                confidence_interval=(lower_ci, upper_ci),
                prediction_horizon_minutes=horizon_minutes,
                model_name="ARIMA",
                model_version="1.0.0",
                model_accuracy=confidence,
                prediction_explanation=(
                    f"ARIMA forecast for next {horizon_minutes} minutes"
                ),
            )

            return prediction_result

        except Exception as e:
            logger.error(f"ARIMA forecast failed: {e}")
            raise

    async def _lstm_forecast(
        self,
        component_type: str,
        model: Any,
        ts_data: TimeSeriesData,
        horizon_minutes: int,
    ) -> PredictionResult:
        """Generate LSTM forecast.

        Args:
            component_type: Type of component
            model: Trained LSTM model
            ts_data: Time series data
            horizon_minutes: Forecast horizon in minutes

        Returns:
            LSTM prediction result
        """
        try:

            scaler = self.scalers[component_type]
            sequence_length = self.models[component_type]["sequence_length"]

            # Prepare last sequence
            recent_values = np.array(ts_data.values[-sequence_length:]).reshape(-1, 1)
            scaled_recent = scaler.transform(recent_values)

            # Generate forecast step by step
            forecast_steps = max(1, horizon_minutes // ts_data.frequency_minutes)
            forecast_scaled = []

            current_sequence = scaled_recent[-sequence_length:].reshape(
                1, sequence_length, 1
            )

            for _ in range(forecast_steps):
                # Predict next value
                next_pred = model.predict(current_sequence, verbose=0)[0, 0]
                forecast_scaled.append(next_pred)

                # Update sequence for next prediction
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = next_pred

            # Inverse transform
            forecast_values = scaler.inverse_transform(
                np.array(forecast_scaled).reshape(-1, 1)
            )

            # Get prediction and confidence interval
            predicted_value = float(forecast_values[-1])

            # Calculate confidence interval (simplified)
            recent_std = (
                np.std(ts_data.values[-50:])
                if len(ts_data.values) >= 50
                else np.std(ts_data.values)
            )
            lower_ci = float(predicted_value - 1.96 * recent_std)
            upper_ci = float(predicted_value + 1.96 * recent_std)

            # Calculate confidence
            base_confidence = 0.85  # LSTM typically has higher confidence
            horizon_penalty = min(0.2, horizon_minutes / (24 * 60))
            confidence = max(0.1, base_confidence - horizon_penalty)

            prediction_result = PredictionResult(
                timestamp=datetime.now(timezone.utc),
                prediction_type=PredictionType.PERFORMANCE_FORECAST,
                component_type=ts_data.component_type,
                component_id=ts_data.component_id,
                metric_name=ts_data.metric_name,
                predicted_value=predicted_value,
                confidence_interval=(lower_ci, upper_ci),
                prediction_horizon_minutes=horizon_minutes,
                model_name="LSTM",
                model_version="1.0.0",
                model_accuracy=confidence,
                prediction_explanation=(
                    f"LSTM forecast for next {horizon_minutes} minutes"
                ),
            )

            return prediction_result

        except Exception as e:
            logger.error(f"LSTM forecast failed: {e}")
            raise


class PerformancePredictor:
    """Performance prediction for system components."""

    def __init__(self, config: PredictiveAnalyticsConfig):
        """Initialize performance predictor.

        Args:
            config: Predictive analytics configuration
        """
        self.config = config
        self.forecaster = TimeSeriesForecaster(config)
        self.performance_profiles: Dict[str, Any] = {}

    async def predict_component_performance(
        self, component_type: str, time_series_data: Dict[str, TimeSeriesData]
    ) -> List[PredictionResult]:
        """Predict performance for a component across different horizons.

        Args:
            component_type: Type of component
            time_series_data: Time series data for the component

        Returns:
            List of prediction results for different horizons
        """
        try:
            # Train model if not already trained
            if not self.forecaster.is_trained.get(component_type, False):
                training_metrics = await self.forecaster.train(
                    {component_type: time_series_data[component_type]}
                )
                if not training_metrics:
                    logger.warning(f"No training metrics for {component_type}")
                    return []

            # Get forecast horizons based on component type
            horizons = self._get_forecast_horizons(component_type)

            predictions = []
            for horizon_minutes in horizons:
                prediction = await self.forecaster.forecast(
                    component_type, horizon_minutes
                )
                if prediction:
                    predictions.append(prediction)

            return predictions

        except Exception as e:
            logger.error(f"Performance prediction failed for {component_type}: {e}")
            return []

    def _get_forecast_horizons(self, component_type: str) -> List[int]:
        """Get forecast horizons for a component type.

        Args:
            component_type: Type of component

        Returns:
            List of forecast horizons in minutes
        """
        base_horizon = self.config.component_horizons.get(component_type, 6)

        # Return short, medium, and long-term forecasts
        return [
            base_horizon * 60,  # Short-term (hours)
            base_horizon * 60 * 3,  # Medium-term (3x)
            base_horizon * 60 * 6,  # Long-term (6x)
        ]

    async def predict_resource_usage(
        self, component_type: str, current_metrics: List[MLMetric]
    ) -> Optional[PredictionResult]:
        """Predict future resource usage.

        Args:
            component_type: Type of component
            current_metrics: Current metrics

        Returns:
            Resource usage prediction
        """
        # Convert current metrics to time series data
        ts_data = self._metrics_to_time_series(component_type, current_metrics)

        if not ts_data:
            return None

        # Predict CPU usage
        cpu_predictions = await self.predict_component_performance(
            component_type, {"cpu_usage": ts_data}
        )

        if cpu_predictions:
            return cpu_predictions[0]  # Return short-term prediction

        return None

    def _metrics_to_time_series(
        self, component_type: str, metrics: List[MLMetric]
    ) -> Optional[TimeSeriesData]:
        """Convert metrics to time series data.

        Args:
            component_type: Type of component
            metrics: List of metrics

        Returns:
            Time series data
        """
        if not metrics:
            return None

        # Group metrics by name
        metric_groups: Dict[str, List[MLMetric]] = {}
        for metric in metrics:
            if metric.metric_name not in metric_groups:
                metric_groups[metric.metric_name] = []
            metric_groups[metric.metric_name].append(metric)

        # Use the first metric group
        first_group = list(metric_groups.values())[0]

        # Sort by timestamp
        first_group.sort(key=lambda x: x.timestamp)

        timestamps = [m.timestamp for m in first_group]
        values = [m.value for m in first_group]

        if len(values) < 10:
            return None

        return TimeSeriesData(
            timestamps=timestamps,
            values=values,
            component_type=first_group[0].component_type,
            component_id=first_group[0].component_id,
            metric_name=first_group[0].metric_name,
            frequency_minutes=1,  # Assume 1-minute intervals
        )


class PredictiveAnalyzer:
    """Main predictive analytics coordinator."""

    def __init__(self, config: PredictiveAnalyticsConfig):
        """Initialize predictive analyzer.

        Args:
            config: Predictive analytics configuration
        """
        self.config = config
        self.forecaster = TimeSeriesForecaster(config)
        self.performance_predictor = PerformancePredictor(config)
        self.analysis_cache: Dict[str, Any] = {}

    async def analyze_and_predict(
        self, metrics_history: Dict[str, List[MLMetric]]
    ) -> Dict[str, List[PredictionResult]]:
        """Analyze metrics history and generate predictions.

        Args:
            metrics_history: Dictionary of component_type -> List[MLMetric]

        Returns:
            Dictionary of component_type -> List[PredictionResult]
        """
        all_predictions = {}

        for component_type, metrics in metrics_history.items():
            try:
                # Convert metrics to time series
                ts_data = self._prepare_time_series(component_type, metrics)
                if not ts_data:
                    continue

                # Generate predictions
                predictions = (
                    await self.performance_predictor.predict_component_performance(
                        component_type, {component_type: ts_data}
                    )
                )

                all_predictions[component_type] = predictions

            except Exception as e:
                logger.error(f"Analysis failed for {component_type}: {e}")
                continue

        return all_predictions

    def _prepare_time_series(
        self, component_type: str, metrics: List[MLMetric]
    ) -> Optional[TimeSeriesData]:
        """Prepare time series data from metrics.

        Args:
            component_type: Type of component
            metrics: List of metrics

        Returns:
            Time series data
        """
        if len(metrics) < self.config.min_training_samples:
            return None

        # Group by metric name and take the most recent one
            latest_metrics: Dict[str, MLMetric] = {}
        for metric in metrics:
            key = metric.metric_name
            if (
                key not in latest_metrics
                or metric.timestamp > latest_metrics[key].timestamp
            ):
                latest_metrics[key] = metric

        # Use the metric with most data points
        if not latest_metrics:
            return None

        # Sort metrics by timestamp
        sorted_metrics = sorted(metrics, key=lambda x: x.timestamp)

        timestamps = [m.timestamp for m in sorted_metrics]
        values = [m.value for m in sorted_metrics]

        return TimeSeriesData(
            timestamps=timestamps,
            values=values,
            component_type=sorted_metrics[0].component_type,
            component_id=sorted_metrics[0].component_id,
            metric_name=sorted_metrics[0].metric_name,
            frequency_minutes=1,  # Assume 1-minute intervals
        )

    async def get_performance_trends(
        self, component_type: str, metrics: List[MLMetric]
    ) -> Dict[str, Any]:
        """Analyze performance trends.

        Args:
            component_type: Type of component
            metrics: List of metrics

        Returns:
            Performance trend analysis
        """
        if len(metrics) < 20:
            return {"trend": "insufficient_data", "confidence": 0.0}

        try:
            # Calculate trend using linear regression
            values = np.array([m.value for m in metrics])
            timestamps = np.array([m.timestamp.timestamp() for m in metrics])

            # Normalize timestamps
            timestamps_norm = (timestamps - timestamps[0]) / (
                timestamps[-1] - timestamps[0]
            )

            # Fit linear trend
            trend_coef = np.polyfit(timestamps_norm, values, 1)[0]

            # Calculate trend strength
            trend_strength = (
                abs(trend_coef) / np.std(values) if np.std(values) > 0 else 0
            )

            # Determine trend direction
            if trend_coef > 0.01:
                trend_direction = "improving"
            elif trend_coef < -0.01:
                trend_direction = "degrading"
            else:
                trend_direction = "stable"

            # Calculate confidence
            confidence = min(1.0, trend_strength / 10.0)

            return {
                "trend": trend_direction,
                "confidence": confidence,
                "slope": float(trend_coef),
                "trend_strength": float(trend_strength),
                "analysis_period_hours": (timestamps[-1] - timestamps[0]) / 3600,
            }

        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return {"trend": "error", "confidence": 0.0}

    async def detect_performance_patterns(
        self, component_type: str, metrics: List[MLMetric]
    ) -> Dict[str, Any]:
        """Detect performance patterns and cycles.

        Args:
            component_type: Type of component
            metrics: List of metrics

        Returns:
            Pattern analysis results
        """
        if len(metrics) < 50:
            return {"patterns": [], "confidence": 0.0}

        try:
            values = np.array([m.value for m in metrics])

            # Simple periodicity detection using autocorrelation
            autocorr = np.correlate(values, values, mode="full")
            autocorr = autocorr[autocorr.size // 2 :]
            autocorr = autocorr / autocorr[0]

            # Find peaks in autocorrelation (indicating periodicity)
            peaks = []
            for i in range(1, len(autocorr) - 1):
                if (
                    autocorr[i] > autocorr[i - 1]
                    and autocorr[i] > autocorr[i + 1]
                    and autocorr[i] > 0.5
                ):
                    peaks.append((i, autocorr[i]))

            patterns = []
            for lag, strength in peaks[:3]:  # Top 3 patterns
                patterns.append(
                    {
                        "type": "periodic",
                        "period": lag,
                        "strength": float(strength),
                        "description": f"Cycle every {lag} data points",
                    }
                )

            confidence = (
                min(
                    1.0,
                    float(
                        sum(cast(float, p["strength"]) for p in patterns)
                        / len(patterns)
                    ),
                )
                if patterns
                else 0.0
            )

            return {
                "patterns": patterns,
                "confidence": confidence,
                "dominant_period": peaks[0][0] if peaks else None,
            }

        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            return {"patterns": [], "confidence": 0.0}
