#!/usr/bin/env python3
"""
Phase 5: Deployment & Operations - Intelligent Monitoring & Auto-Scaling
Implements AI-driven monitoring, predictive scaling, and performance optimization
for deployed LLM models with advanced analytics.
"""

import asyncio
import logging
import yaml
import json
import os
import sys
import time
import threading
import statistics
import pickle
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, asdict, field
from enum import Enum
import aiohttp
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
import prometheus_client as prom
from prometheus_client import CollectorRegistry, Gauge, Counter
import redis
import pymongo

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from scripts.utils.config_loader import load_config
from scripts.utils.logger import setup_logger

class ScalingDirection(Enum):
    """Auto-scaling directions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"

class PerformanceTier(Enum):
    """Performance classification tiers"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

class PredictionModel(Enum):
    """Types of prediction models"""
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    LSTM = "lstm"
    ENSEMBLE = "ensemble"

@dataclass
class PredictionResult:
    """Prediction result with confidence"""
    timestamp: datetime
    predicted_value: float
    confidence_interval: Tuple[float, float]
    model_used: PredictionModel
    accuracy_score: float
    features_used: List[str]

@dataclass
class ScalingDecision:
    """Auto-scaling decision"""
    timestamp: datetime
    direction: ScalingDirection
    current_replicas: int
    target_replicas: int
    reason: str
    confidence: float
    metrics_triggered: List[str]
    estimated_impact: Dict[str, float]

@dataclass
class PerformancePrediction:
    """Performance prediction"""
    timestamp: datetime
    metric_name: str
    current_value: float
    predicted_values: List[float]  # Next N time periods
    trend: str  # increasing, decreasing, stable
    anomaly_score: float
    recommendations: List[str]

@dataclass
class ResourceUtilization:
    """Resource utilization snapshot"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    disk_io: float
    network_io: float
    request_rate: float
    response_time: float
    queue_size: int
    error_rate: float

class IntelligentMonitoring:
    """AI-driven intelligent monitoring and auto-scaling system"""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.logger = setup_logger(
            "intelligent_monitoring",
            self.config.get("logging", {}).get("level", "INFO")
        )

        # Initialize components
        self.metrics_registry = prom.CollectorRegistry()

        # Storage and state
        self.historical_data: List[ResourceUtilization] = []
        self.predictions: Dict[str, List[PredictionResult]] = {}
        self.scaling_decisions: List[ScalingDecision] = []
        self.performance_predictions: Dict[str, PerformancePrediction] = {}

        # Machine learning models
        self.prediction_models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.is_trained = False

        # Performance thresholds and scaling policies
        self.performance_thresholds = self._load_performance_thresholds()
        self.scaling_policies = self._load_scaling_policies()

        # Monitoring state
        self.is_running = False
        self.prediction_thread = None
        self.scaling_thread = None
        self.optimization_thread = None

        # Setup metrics
        self._setup_intelligent_metrics()

        self.logger.info("IntelligentMonitoring initialized")

    def _setup_intelligent_metrics(self):
        """Setup intelligent monitoring metrics"""
        self.prediction_accuracy = prom.Gauge(
            'intelligent_monitoring_prediction_accuracy',
            'Prediction accuracy of ML models',
            ['model_type', 'metric'],
            registry=self.metrics_registry
        )

        self.scaling_decisions_total = prom.Counter(
            'intelligent_monitoring_scaling_decisions_total',
            'Total number of auto-scaling decisions',
            ['direction'],
            registry=self.metrics_registry
        )

        self.anomaly_count = prom.Counter(
            'intelligent_monitoring_anomalies_detected_total',
            'Total number of anomalies detected',
            ['severity'],
            registry=self.metrics_registry
        )

        self.optimization_savings = prom.Gauge(
            'intelligent_monitoring_optimization_savings',
            'Cost savings from optimizations',
            ['optimization_type'],
            registry=self.metrics_registry
        )

    def _load_performance_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load performance threshold configurations"""
        return {
            "response_time": {
                "excellent": 500,
                "good": 1000,
                "fair": 2000,
                "poor": 5000,
                "critical": 10000
            },
            "error_rate": {
                "excellent": 0.1,
                "good": 0.5,
                "fair": 1.0,
                "poor": 3.0,
                "critical": 5.0
            },
            "cpu_usage": {
                "excellent": 30,
                "good": 50,
                "fair": 70,
                "poor": 85,
                "critical": 95
            },
            "memory_usage": {
                "excellent": 40,
                "good": 60,
                "fair": 75,
                "poor": 85,
                "critical": 95
            },
            "throughput": {
                "excellent": 1000,
                "good": 500,
                "fair": 200,
                "poor": 50,
                "critical": 10
            }
        }

    def _load_scaling_policies(self) -> Dict[str, Any]:
        """Load auto-scaling policies"""
        return {
            "scale_up_triggers": {
                "cpu_usage": 80,
                "memory_usage": 85,
                "response_time_p95": 3000,
                "queue_size": 50,
                "error_rate": 2.0
            },
            "scale_down_triggers": {
                "cpu_usage": 30,
                "memory_usage": 40,
                "response_time_p95": 500,
                "queue_size": 5,
                "error_rate": 0.5
            },
            "scaling_limits": {
                "min_replicas": 1,
                "max_replicas": 20,
                "scale_up_cooldown": 300,  # 5 minutes
                "scale_down_cooldown": 600   # 10 minutes
            },
            "prediction_horizon": 15,  # minutes
            "confidence_threshold": 0.7
        }

    async def start_intelligent_monitoring(self):
        """Start the intelligent monitoring system"""
        if self.is_running:
            self.logger.warning("Intelligent monitoring is already running")
            return

        self.is_running = True

        # Start monitoring threads
        self.prediction_thread = threading.Thread(target=self._prediction_loop)
        self.prediction_thread.daemon = True
        self.prediction_thread.start()

        self.scaling_thread = threading.Thread(target=self._auto_scaling_loop)
        self.scaling_thread.daemon = True
        self.scaling_thread.start()

        self.optimization_thread = threading.Thread(target=self._optimization_loop)
        self.optimization_thread.daemon = True
        self.optimization_thread.start()

        self.logger.info("Intelligent monitoring system started")

    def stop_intelligent_monitoring(self):
        """Stop the intelligent monitoring system"""
        self.is_running = False
        self.logger.info("Intelligent monitoring system stopped")

    def _prediction_loop(self):
        """Main prediction loop"""
        while self.is_running:
            try:
                # Collect current metrics
                current_metrics = self._collect_current_metrics()

                # Store metrics
                self._store_metrics(current_metrics)

                # Update ML models
                if len(self.historical_data) >= 100:  # Minimum data for training
                    self._update_prediction_models()

                # Generate predictions
                self._generate_predictions()

                # Detect anomalies
                self._detect_anomalies()

                # Sleep for prediction interval
                interval = self.config.get("intelligent_monitoring", {}).get("prediction_interval", 60)
                time.sleep(interval)

            except Exception as e:
                self.logger.error(f"Error in prediction loop: {e}")
                time.sleep(10)

    def _auto_scaling_loop(self):
        """Main auto-scaling loop"""
        while self.is_running:
            try:
                # Evaluate scaling decisions
                scaling_decision = self._evaluate_scaling_decision()

                if scaling_decision and scaling_decision.direction != ScalingDirection.NO_ACTION:
                    # Execute scaling decision
                    self._execute_scaling_decision(scaling_decision)

                # Sleep for scaling evaluation interval
                interval = self.config.get("intelligent_monitoring", {}).get("scaling_interval", 120)
                time.sleep(interval)

            except Exception as e:
                self.logger.error(f"Error in auto-scaling loop: {e}")
                time.sleep(15)

    def _optimization_loop(self):
        """Main optimization loop"""
        while self.is_running:
            try:
                # Analyze performance patterns
                self._analyze_performance_patterns()

                # Identify optimization opportunities
                optimizations = self._identify_optimization_opportunities()

                # Apply optimizations if beneficial
                for optimization in optimizations:
                    if optimization["confidence"] > 0.8:
                        self._apply_optimization(optimization)

                # Sleep for optimization interval
                interval = self.config.get("intelligent_monitoring", {}).get("optimization_interval", 300)
                time.sleep(interval)

            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                time.sleep(30)

    def _collect_current_metrics(self) -> ResourceUtilization:
        """Collect current resource utilization metrics"""
        try:
            # This would typically collect from your monitoring system
            # For now, we'll simulate realistic metrics with some variation
            import random

            current_metrics = ResourceUtilization(
                timestamp=datetime.now(timezone.utc),
                cpu_usage=45 + random.uniform(-10, 20),
                memory_usage=60 + random.uniform(-15, 15),
                gpu_usage=70 + random.uniform(-20, 20),
                disk_io=50 + random.uniform(-20, 30),
                network_io=30 + random.uniform(-10, 20),
                request_rate=200 + random.uniform(-50, 100),
                response_time=800 + random.uniform(-200, 400),
                queue_size=10 + random.randint(0, 20),
                error_rate=0.5 + random.uniform(-0.3, 1.0)
            )

            # Ensure values are within reasonable bounds
            current_metrics.cpu_usage = max(0, min(100, current_metrics.cpu_usage))
            current_metrics.memory_usage = max(0, min(100, current_metrics.memory_usage))
            current_metrics.gpu_usage = max(0, min(100, current_metrics.gpu_usage))
            current_metrics.error_rate = max(0, current_metrics.error_rate)

            return current_metrics

        except Exception as e:
            self.logger.error(f"Error collecting current metrics: {e}")
            # Return default metrics
            return ResourceUtilization(
                timestamp=datetime.now(timezone.utc),
                cpu_usage=50, memory_usage=60, gpu_usage=70,
                disk_io=50, network_io=30, request_rate=200,
                response_time=800, queue_size=10, error_rate=0.5
            )

    def _store_metrics(self, metrics: ResourceUtilization):
        """Store metrics in historical data"""
        self.historical_data.append(metrics)

        # Keep only last 10000 data points
        if len(self.historical_data) > 10000:
            self.historical_data = self.historical_data[-5000:]

    def _update_prediction_models(self):
        """Update machine learning prediction models"""
        try:
            if len(self.historical_data) < 100:
                return

            # Prepare training data
            df = pd.DataFrame([asdict(m) for m in self.historical_data[-1000:]])

            # Feature engineering
            features = self._create_features(df)

            # Train models for different metrics
            target_metrics = ['cpu_usage', 'memory_usage', 'response_time', 'request_rate']

            for metric in target_metrics:
                if metric not in df.columns:
                    continue

                # Prepare training data
                X = features
                y = df[metric].values

                # Skip if insufficient data
                if len(X) < 50:
                    continue

                # Split data (use last 20% for validation)
                split_idx = int(len(X) * 0.8)
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]

                # Scale features
                if metric not in self.scalers:
                    self.scalers[metric] = StandardScaler()
                X_train_scaled = self.scalers[metric].fit_transform(X_train)
                X_val_scaled = self.scalers[metric].transform(X_val)

                # Train Random Forest model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train_scaled, y_train)

                # Validate model
                y_pred = model.predict(X_val_scaled)
                accuracy = 1 - (mean_absolute_error(y_val, y_pred) / np.mean(y_val))

                # Store model if good enough
                if accuracy > 0.7:
                    self.prediction_models[metric] = model
                    self.prediction_accuracy.labels(
                        model_type='random_forest',
                        metric=metric
                    ).set(accuracy)

                    self.logger.debug(f"Updated prediction model for {metric} with accuracy: {accuracy:.3f}")

        except Exception as e:
            self.logger.error(f"Error updating prediction models: {e}")

    def _create_features(self, df: pd.DataFrame) -> np.ndarray:
        """Create features for machine learning models"""
        features = []

        for i in range(len(df)):
            row = df.iloc[i]

            # Time-based features
            timestamp = pd.to_datetime(row['timestamp'])
            hour = timestamp.hour
            day_of_week = timestamp.dayofweek
            is_weekend = 1 if day_of_week >= 5 else 0

            # Lag features (previous values)
            if i > 0:
                prev_row = df.iloc[i-1]
                cpu_lag = prev_row['cpu_usage']
                memory_lag = prev_row['memory_usage']
                response_time_lag = prev_row['response_time']
                request_rate_lag = prev_row['request_rate']
            else:
                cpu_lag = memory_lag = response_time_lag = request_rate_lag = 0

            # Rolling statistics
            if i >= 5:
                window = df.iloc[max(0, i-5):i]
                cpu_ma = window['cpu_usage'].mean()
                memory_ma = window['memory_usage'].mean()
                response_time_ma = window['response_time'].mean()
            else:
                cpu_ma = memory_ma = response_time_ma = row['cpu_usage']

            # Combine features
            feature_vector = [
                hour, day_of_week, is_weekend,
                cpu_lag, memory_lag, response_time_lag, request_rate_lag,
                cpu_ma, memory_ma, response_time_ma,
                row['cpu_usage'], row['memory_usage'], row['request_rate'],
                row['response_time'], row['queue_size'], row['error_rate']
            ]

            features.append(feature_vector)

        return np.array(features)

    def _generate_predictions(self):
        """Generate predictions for future metrics"""
        try:
            if not self.historical_data or not self.prediction_models:
                return

            current_time = datetime.now(timezone.utc)
            prediction_horizon = self.scaling_policies["prediction_horizon"]

            for metric_name, model in self.prediction_models.items():
                # Create features for prediction
                current_features = self._create_prediction_features(metric_name)

                if current_features is None:
                    continue

                # Scale features
                if metric_name in self.scalers:
                    current_features_scaled = self.scalers[metric_name].transform([current_features])
                else:
                    current_features_scaled = [current_features]

                # Generate predictions for multiple time steps
                predictions = []
                for step in range(5):  # Predict next 5 time periods
                    pred_value = model.predict(current_features_scaled)[0]

                    # Create prediction result
                    confidence = max(0.5, min(0.95, model.score(current_features_scaled, [pred_value])))
                    confidence_interval = (
                        pred_value * (1 - confidence),
                        pred_value * (1 + confidence)
                    )

                    prediction = PredictionResult(
                        timestamp=current_time + timedelta(minutes=step * prediction_horizon),
                        predicted_value=pred_value,
                        confidence_interval=confidence_interval,
                        model_used=PredictionModel.RANDOM_FOREST,
                        accuracy_score=confidence,
                        features_used=[f"feature_{i}" for i in range(len(current_features))]
                    )

                    predictions.append(prediction)

                self.predictions[metric_name] = predictions

        except Exception as e:
            self.logger.error(f"Error generating predictions: {e}")

    def _create_prediction_features(self, metric_name: str) -> Optional[List[float]]:
        """Create features for prediction"""
        try:
            if not self.historical_data:
                return None

            current = self.historical_data[-1]
            timestamp = pd.to_datetime(current.timestamp)

            # Time features
            hour = timestamp.hour
            day_of_week = timestamp.dayofweek
            is_weekend = 1 if day_of_week >= 5 else 0

            # Recent metrics
            recent_metrics = [m for m in self.historical_data[-10:]]

            if len(recent_metrics) >= 2:
                prev = recent_metrics[-2]
                cpu_lag = prev.cpu_usage
                memory_lag = prev.memory_usage
                response_time_lag = prev.response_time
                request_rate_lag = prev.request_rate
            else:
                cpu_lag = memory_lag = response_time_lag = request_rate_lag = 0

            # Rolling averages
            if len(recent_metrics) >= 5:
                window = recent_metrics[-5:]
                cpu_ma = sum(m.cpu_usage for m in window) / len(window)
                memory_ma = sum(m.memory_usage for m in window) / len(window)
                response_time_ma = sum(m.response_time for m in window) / len(window)
            else:
                cpu_ma = current.cpu_usage
                memory_ma = current.memory_usage
                response_time_ma = current.response_time

            return [
                hour, day_of_week, is_weekend,
                cpu_lag, memory_lag, response_time_lag, request_rate_lag,
                cpu_ma, memory_ma, response_time_ma,
                current.cpu_usage, current.memory_usage, current.request_rate,
                current.response_time, current.queue_size, current.error_rate
            ]

        except Exception as e:
            self.logger.error(f"Error creating prediction features: {e}")
            return None

    def _detect_anomalies(self):
        """Detect anomalies in current metrics"""
        try:
            if len(self.historical_data) < 50:
                return

            # Prepare data for anomaly detection
            recent_data = self.historical_data[-50:]
            features = []

            for metrics in recent_data:
                feature_vector = [
                    metrics.cpu_usage,
                    metrics.memory_usage,
                    metrics.gpu_usage,
                    metrics.response_time,
                    metrics.request_rate,
                    metrics.error_rate
                ]
                features.append(feature_vector)

            features_array = np.array(features)

            # Train anomaly detector if not trained
            if not self.is_trained:
                self.anomaly_detector.fit(features_array)
                self.is_trained = True
                return

            # Detect anomalies
            anomaly_scores = self.anomaly_detector.decision_function(features_array)

            # Check if latest metrics are anomalous
            if anomaly_scores[-1] < 0:
                current_metrics = recent_data[-1]
                severity = "high" if anomaly_scores[-1] < -0.2 else "medium"

                self.anomaly_count.labels(severity=severity).inc()

                self.logger.warning(
                    f"Anomaly detected - Score: {anomaly_scores[-1]:.3f}, "
                    f"CPU: {current_metrics.cpu_usage:.1f}%, "
                    f"Memory: {current_metrics.memory_usage:.1f}%, "
                    f"Response Time: {current_metrics.response_time:.1f}ms"
                )

                # Create performance prediction with anomaly
                self._create_anomaly_prediction(current_metrics, anomaly_scores[-1])

        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")

    def _create_anomaly_prediction(self, metrics: ResourceUtilization, anomaly_score: float):
        """Create performance prediction for anomaly"""
        prediction = PerformancePrediction(
            timestamp=metrics.timestamp,
            metric_name="anomaly_detection",
            current_value=anomaly_score,
            predicted_values=[anomaly_score] * 5,  # Anomaly persists
            trend="anomalous",
            anomaly_score=abs(anomaly_score),
            recommendations=[
                "Investigate sudden metric changes",
                "Check for system load spikes",
                "Review recent deployments",
                "Monitor error rates closely"
            ]
        )

        self.performance_predictions["anomaly_detection"] = prediction

    def _evaluate_scaling_decision(self) -> Optional[ScalingDecision]:
        """Evaluate if auto-scaling is needed"""
        try:
            if not self.historical_data:
                return None

            current_metrics = self.historical_data[-1]
            current_replicas = self._get_current_replicas()

            # Check cooldown periods
            if not self._can_scale():
                return None

            # Check scale-up conditions
            scale_up_triggers = self.scaling_policies["scale_up_triggers"]
            scale_up_reasons = []

            if current_metrics.cpu_usage > scale_up_triggers["cpu_usage"]:
                scale_up_reasons.append(f"High CPU usage: {current_metrics.cpu_usage:.1f}%")

            if current_metrics.memory_usage > scale_up_triggers["memory_usage"]:
                scale_up_reasons.append(f"High memory usage: {current_metrics.memory_usage:.1f}%")

            if current_metrics.response_time > scale_up_triggers["response_time_p95"]:
                scale_up_reasons.append(f"High response time: {current_metrics.response_time:.1f}ms")

            if current_metrics.queue_size > scale_up_triggers["queue_size"]:
                scale_up_reasons.append(f"Large queue size: {current_metrics.queue_size}")

            if current_metrics.error_rate > scale_up_triggers["error_rate"]:
                scale_up_reasons.append(f"High error rate: {current_metrics.error_rate:.1f}%")

            # Check scale-down conditions
            scale_down_triggers = self.scaling_policies["scale_down_triggers"]
            scale_down_reasons = []

            if current_metrics.cpu_usage < scale_down_triggers["cpu_usage"]:
                scale_down_reasons.append(f"Low CPU usage: {current_metrics.cpu_usage:.1f}%")

            if current_metrics.memory_usage < scale_down_triggers["memory_usage"]:
                scale_down_reasons.append(f"Low memory usage: {current_metrics.memory_usage:.1f}%")

            if current_metrics.response_time < scale_down_triggers["response_time_p95"]:
                scale_down_reasons.append(f"Low response time: {current_metrics.response_time:.1f}ms")

            if current_metrics.queue_size < scale_down_triggers["queue_size"]:
                scale_down_reasons.append(f"Small queue size: {current_metrics.queue_size}")

            if current_metrics.error_rate < scale_down_triggers["error_rate"]:
                scale_down_reasons.append(f"Low error rate: {current_metrics.error_rate:.1f}%")

            # Consider predictions
            prediction_factors = self._evaluate_predictions_for_scaling()

            # Make decision
            if scale_up_reasons or prediction_factors["scale_up"]:
                target_replicas = min(
                    current_replicas + 1,
                    self.scaling_policies["scaling_limits"]["max_replicas"]
                )

                return ScalingDecision(
                    timestamp=datetime.now(timezone.utc),
                    direction=ScalingDirection.SCALE_UP,
                    current_replicas=current_replicas,
                    target_replicas=target_replicas,
                    reason="; ".join(scale_up_reasons),
                    confidence=0.8,
                    metrics_triggered=[reason.split(":")[0] for reason in scale_up_reasons],
                    estimated_impact=self._estimate_scaling_impact("up", current_replicas, target_replicas)
                )

            elif (scale_down_reasons and current_replicas > 1 and
                  len(scale_down_reasons) >= 3):  # Require multiple reasons for scale-down
                target_replicas = max(
                    current_replicas - 1,
                    self.scaling_policies["scaling_limits"]["min_replicas"]
                )

                return ScalingDecision(
                    timestamp=datetime.now(timezone.utc),
                    direction=ScalingDirection.SCALE_DOWN,
                    current_replicas=current_replicas,
                    target_replicas=target_replicas,
                    reason="; ".join(scale_down_reasons),
                    confidence=0.7,
                    metrics_triggered=[reason.split(":")[0] for reason in scale_down_reasons],
                    estimated_impact=self._estimate_scaling_impact("down", current_replicas, target_replicas)
                )

            return None

        except Exception as e:
            self.logger.error(f"Error evaluating scaling decision: {e}")
            return None

    def _can_scale(self) -> bool:
        """Check if scaling is allowed (cooldown periods)"""
        try:
            if not self.scaling_decisions:
                return True

            last_decision = self.scaling_decisions[-1]
            current_time = datetime.now(timezone.utc)
            time_since_last = (current_time - last_decision.timestamp).total_seconds()

            if last_decision.direction == ScalingDirection.SCALE_UP:
                cooldown = self.scaling_policies["scaling_limits"]["scale_up_cooldown"]
            else:
                cooldown = self.scaling_policies["scaling_limits"]["scale_down_cooldown"]

            return time_since_last >= cooldown

        except Exception as e:
            self.logger.error(f"Error checking scaling cooldown: {e}")
            return False

    def _get_current_replicas(self) -> int:
        """Get current number of replicas"""
        # This would typically query your orchestration system
        # For now, we'll return a simulated value
        return 3

    def _evaluate_predictions_for_scaling(self) -> Dict[str, bool]:
        """Evaluate predictions for scaling decisions"""
        result = {"scale_up": False, "scale_down": False}

        try:
            horizon = self.scaling_policies["prediction_horizon"]
            future_time = datetime.now(timezone.utc) + timedelta(minutes=horizon)
            confidence_threshold = self.scaling_policies["confidence_threshold"]

            for metric_name, predictions in self.predictions.items():
                if not predictions:
                    continue

                # Find prediction for target time
                target_prediction = None
                for pred in predictions:
                    if abs((pred.timestamp - future_time).total_seconds()) < 60:
                        target_prediction = pred
                        break

                if not target_prediction or target_prediction.accuracy_score < confidence_threshold:
                    continue

                # Check if prediction suggests scaling
                if metric_name == "cpu_usage" and target_prediction.predicted_value > 80:
                    result["scale_up"] = True
                elif metric_name == "memory_usage" and target_prediction.predicted_value > 85:
                    result["scale_up"] = True
                elif metric_name == "response_time" and target_prediction.predicted_value > 3000:
                    result["scale_up"] = True
                elif metric_name == "request_rate" and target_prediction.predicted_value > 500:
                    result["scale_up"] = True

        except Exception as e:
            self.logger.error(f"Error evaluating predictions for scaling: {e}")

        return result

    def _estimate_scaling_impact(self, direction: str, current: int, target: int) -> Dict[str, float]:
        """Estimate impact of scaling decision"""
        ratio = target / current

        if direction == "up":
            return {
                "expected_response_time_reduction": 0.3 * (ratio - 1),
                "expected_throughput_increase": 0.4 * (ratio - 1),
                "expected_cpu_increase": 0.1 * (ratio - 1),
                "expected_memory_increase": 0.15 * (ratio - 1),
                "cost_increase": 0.25 * (ratio - 1)
            }
        else:
            return {
                "expected_response_time_increase": 0.2 * (1 - ratio),
                "expected_throughput_decrease": 0.3 * (1 - ratio),
                "expected_cpu_decrease": 0.1 * (1 - ratio),
                "expected_memory_decrease": 0.12 * (1 - ratio),
                "cost_savings": 0.2 * (1 - ratio)
            }

    def _execute_scaling_decision(self, decision: ScalingDecision):
        """Execute auto-scaling decision"""
        try:
            self.logger.info(
                f"Executing scaling decision: {decision.direction.value} "
                f"from {decision.current_replicas} to {decision.target_replicas} replicas"
            )

            # This would typically call your orchestration API
            # For now, we'll simulate the scaling operation
            success = self._simulate_scaling(decision)

            if success:
                self.scaling_decisions.append(decision)
                self.scaling_decisions_total.labels(direction=decision.direction.value).inc()

                self.logger.info(f"Scaling decision executed successfully: {decision.direction.value}")
            else:
                self.logger.error(f"Scaling decision failed: {decision.direction.value}")

        except Exception as e:
            self.logger.error(f"Error executing scaling decision: {e}")

    def _simulate_scaling(self, decision: ScalingDecision) -> bool:
        """Simulate scaling operation"""
        # This would typically call Kubernetes API, cloud provider API, etc.
        import random
        return random.random() > 0.1  # 90% success rate

    def _analyze_performance_patterns(self):
        """Analyze performance patterns and trends"""
        try:
            if len(self.historical_data) < 100:
                return

            # Analyze different metrics
            metrics_to_analyze = ['cpu_usage', 'memory_usage', 'response_time', 'request_rate']

            for metric in metrics_to_analyze:
                values = [getattr(m, metric) for m in self.historical_data[-100:]]
                timestamps = [m.timestamp for m in self.historical_data[-100:]]

                # Calculate trend
                if len(values) >= 10:
                    recent_values = values[-10:]
                    older_values = values[-20:-10]

                    recent_avg = sum(recent_values) / len(recent_values)
                    older_avg = sum(older_values) / len(older_values)

                    trend = "increasing" if recent_avg > older_avg * 1.1 else "decreasing" if recent_avg < older_avg * 0.9 else "stable"

                    # Create performance prediction
                    prediction = PerformancePrediction(
                        timestamp=timestamps[-1],
                        metric_name=metric,
                        current_value=values[-1],
                        predicted_values=self._generate_simple_forecast(values, 5),
                        trend=trend,
                        anomaly_score=0.0,
                        recommendations=self._generate_metric_recommendations(metric, trend, recent_avg)
                    )

                    self.performance_predictions[metric] = prediction

        except Exception as e:
            self.logger.error(f"Error analyzing performance patterns: {e}")

    def _generate_simple_forecast(self, values: List[float], steps: int) -> List[float]:
        """Generate simple forecast using moving average"""
        if len(values) < 5:
            return [values[-1]] * steps

        # Use exponential moving average
        alpha = 0.3
        ema = values[-1]
        forecast = []

        for _ in range(steps):
            ema = alpha * values[-1] + (1 - alpha) * ema
            forecast.append(ema)

        return forecast

    def _generate_metric_recommendations(self, metric: str, trend: str, current_value: float) -> List[str]:
        """Generate recommendations for a metric"""
        recommendations = []

        if metric == "cpu_usage":
            if trend == "increasing" and current_value > 70:
                recommendations.extend([
                    "Consider scaling up to handle increased CPU load",
                    "Optimize application code for better CPU efficiency",
                    "Review and optimize background tasks"
                ])
            elif trend == "decreasing" and current_value < 30:
                recommendations.append("Consider scaling down to optimize costs")

        elif metric == "memory_usage":
            if trend == "increasing" and current_value > 80:
                recommendations.extend([
                    "Investigate potential memory leaks",
                    "Consider scaling up or optimizing memory usage",
                    "Review application memory management"
                ])

        elif metric == "response_time":
            if trend == "increasing" and current_value > 2000:
                recommendations.extend([
                    "Optimize application performance",
                    "Consider scaling up to reduce response times",
                    "Review database queries and external API calls"
                ])

        elif metric == "request_rate":
            if trend == "increasing" and current_value > 800:
                recommendations.extend([
                    "Prepare for increased load by scaling up",
                    "Implement caching to handle more requests",
                    "Review rate limiting policies"
                ])

        return recommendations

    def _identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify optimization opportunities"""
        opportunities = []

        try:
            if not self.historical_data:
                return opportunities

            current_metrics = self.historical_data[-1]

            # Check for over-provisioning
            if current_metrics.cpu_usage < 25 and current_metrics.memory_usage < 40:
                opportunities.append({
                    "type": "cost_optimization",
                    "description": "System appears over-provisioned",
                    "action": "scale_down",
                    "confidence": 0.8,
                    "estimated_savings": 0.3
                })

            # Check for performance issues
            if current_metrics.response_time > 3000:
                opportunities.append({
                    "type": "performance_optimization",
                    "description": "High response times detected",
                    "action": "optimize_performance",
                    "confidence": 0.9,
                    "estimated_improvement": 0.4
                })

            # Check for inefficient scaling patterns
            if len(self.scaling_decisions) >= 5:
                recent_decisions = self.scaling_decisions[-5:]
                scale_up_count = sum(1 for d in recent_decisions if d.direction == ScalingDirection.SCALE_UP)
                scale_down_count = sum(1 for d in recent_decisions if d.direction == ScalingDirection.SCALE_DOWN)

                if scale_up_count >= 3 and scale_down_count >= 2:
                    opportunities.append({
                        "type": "scaling_optimization",
                        "description": "Frequent scaling detected, consider different strategy",
                        "action": "adjust_scaling_policy",
                        "confidence": 0.7,
                        "estimated_improvement": 0.2
                    })

        except Exception as e:
            self.logger.error(f"Error identifying optimization opportunities: {e}")

        return opportunities

    def _apply_optimization(self, optimization: Dict[str, Any]):
        """Apply optimization"""
        try:
            self.logger.info(f"Applying optimization: {optimization['type']} - {optimization['description']}")

            # This would typically implement the actual optimization
            # For now, we'll just log the action

            if optimization["type"] == "cost_optimization":
                self.optimization_savings.labels(optimization_type="cost").set(optimization["estimated_savings"])
            elif optimization["type"] == "performance_optimization":
                self.optimization_savings.labels(optimization_type="performance").set(optimization["estimated_improvement"])

            self.logger.info(f"Optimization applied: {optimization['action']}")

        except Exception as e:
            self.logger.error(f"Error applying optimization: {e}")

    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data"""
        current_time = datetime.now(timezone.utc)

        # Current status
        current_metrics = self.historical_data[-1] if self.historical_data else None

        # Performance classification
        performance_status = self._classify_performance(current_metrics) if current_metrics else "unknown"

        # Recent predictions
        recent_predictions = {}
        for metric, predictions in self.predictions.items():
            if predictions:
                recent_predictions[metric] = {
                    "next_prediction": asdict(predictions[0]),
                    "trend": "increasing" if len(predictions) > 1 and predictions[0].predicted_value > predictions[1].predicted_value else "stable"
                }

        # Recent scaling decisions
        recent_scaling = [asdict(d) for d in self.scaling_decisions[-10:]]

        # Optimization opportunities
        optimization_opportunities = self._identify_optimization_opportunities()

        return {
            "timestamp": current_time.isoformat(),
            "current_metrics": asdict(current_metrics) if current_metrics else None,
            "performance_status": performance_status,
            "predictions": recent_predictions,
            "recent_scaling_decisions": recent_scaling,
            "optimization_opportunities": optimization_opportunities,
            "model_accuracy": {
                metric: model.score if hasattr(model, 'score') else 0.0
                for metric, model in self.prediction_models.items()
            },
            "anomaly_status": "anomaly_detected" if "anomaly_detection" in self.performance_predictions else "normal"
        }

    def _classify_performance(self, metrics: ResourceUtilization) -> str:
        """Classify current performance status"""
        scores = []

        # Check response time
        if metrics.response_time <= self.performance_thresholds["response_time"]["excellent"]:
            scores.append(4)
        elif metrics.response_time <= self.performance_thresholds["response_time"]["good"]:
            scores.append(3)
        elif metrics.response_time <= self.performance_thresholds["response_time"]["fair"]:
            scores.append(2)
        elif metrics.response_time <= self.performance_thresholds["response_time"]["poor"]:
            scores.append(1)
        else:
            scores.append(0)

        # Check error rate
        if metrics.error_rate <= self.performance_thresholds["error_rate"]["excellent"]:
            scores.append(4)
        elif metrics.error_rate <= self.performance_thresholds["error_rate"]["good"]:
            scores.append(3)
        elif metrics.error_rate <= self.performance_thresholds["error_rate"]["fair"]:
            scores.append(2)
        elif metrics.error_rate <= self.performance_thresholds["error_rate"]["poor"]:
            scores.append(1)
        else:
            scores.append(0)

        # Check CPU usage
        if metrics.cpu_usage <= self.performance_thresholds["cpu_usage"]["excellent"]:
            scores.append(4)
        elif metrics.cpu_usage <= self.performance_thresholds["cpu_usage"]["good"]:
            scores.append(3)
        elif metrics.cpu_usage <= self.performance_thresholds["cpu_usage"]["fair"]:
            scores.append(2)
        elif metrics.cpu_usage <= self.performance_thresholds["cpu_usage"]["poor"]:
            scores.append(1)
        else:
            scores.append(0)

        # Calculate average score
        avg_score = sum(scores) / len(scores) if scores else 0

        if avg_score >= 3.5:
            return "excellent"
        elif avg_score >= 2.5:
            return "good"
        elif avg_score >= 1.5:
            return "fair"
        elif avg_score >= 0.5:
            return "poor"
        else:
            return "critical"

async def main():
    """Main intelligent monitoring execution"""
    # Load configuration
    config_path = "configs/lifecycle/phase5_deployment_operations.yaml"

    # Initialize intelligent monitoring
    monitoring = IntelligentMonitoring(config_path)

    # Start intelligent monitoring
    await monitoring.start_intelligent_monitoring()

    try:
        # Keep system running
        while True:
            await asyncio.sleep(60)

            # Print dashboard summary
            dashboard = monitoring.get_monitoring_dashboard()
            print(f"Performance Status: {dashboard['performance_status']}")
            print(f"Active Predictions: {len(dashboard['predictions'])}")
            print(f"Recent Scaling Decisions: {len(dashboard['recent_scaling_decisions'])}")
            print(f"Optimization Opportunities: {len(dashboard['optimization_opportunities'])}")

    except KeyboardInterrupt:
        print("\nShutting down intelligent monitoring system...")
        monitoring.stop_intelligent_monitoring()

if __name__ == "__main__":
    asyncio.run(main())