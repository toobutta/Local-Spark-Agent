#!/usr/bin/env python3
"""
Phase 6: Maintenance & Evolution - Advanced Drift Detection System
Implements comprehensive data and concept drift detection with automated response
strategies for deployed LLM models.
"""

import asyncio
import logging
import yaml
import json
import os
import sys
import time
import threading
import pickle
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, NamedTuple
from pathlib import Path
from dataclasses import dataclass, asdict, field
from enum import Enum
import aiohttp
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
import prometheus_client as prom
from prometheus_client import CollectorRegistry, Gauge, Counter
import redis
import pymongo
from transformers import AutoTokenizer, AutoModel
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from scripts.utils.config_loader import load_config
from scripts.utils.logger import setup_logger
from scripts.utils.notification import NotificationManager

class DriftType(Enum):
    """Types of drift to detect"""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    LABEL_DRIFT = "label_drift"
    FEATURE_DRIFT = "feature_drift"
    PERFORMANCE_DRIFT = "performance_drift"
    COVARIATE_SHIFT = "covariate_shift"
    PRIOR_PROBABILITY_SHIFT = "prior_probability_shift"

class DriftSeverity(Enum):
    """Drift severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DetectionMethod(Enum):
    """Drift detection methods"""
    STATISTICAL_TEST = "statistical_test"
    MODEL_BASED = "model_based"
    DISTANCE_BASED = "distance_based"
    CLASSIFIER_BASED = "classifier_based"
    AUTOENCODER_BASED = "autoencoder_based"
    ENSEMBLE = "ensemble"

@dataclass
class DriftResult:
    """Drift detection result"""
    drift_type: DriftType
    detection_method: DetectionMethod
    test_statistic: float
    p_value: Optional[float]
    threshold: float
    drift_detected: bool
    severity: DriftSeverity
    confidence: float
    timestamp: datetime
    features_affected: List[str]
    magnitude: float
    description: str

@dataclass
class DriftAlert:
    """Drift alert information"""
    id: str
    drift_type: DriftType
    severity: DriftSeverity
    timestamp: datetime
    description: str
    affected_features: List[str]
    magnitude: float
    confidence: float
    recommended_actions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

class DriftDetector:
    """Advanced drift detection system for LLM models"""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.logger = setup_logger(
            "drift_detector",
            self.config.get("logging", {}).get("level", "INFO")
        )

        # Initialize components
        self.notification_manager = NotificationManager(self.config)

        # Storage and state
        self.reference_data: Optional[pd.DataFrame] = None
        self.current_data: List[Dict[str, Any]] = []
        self.drift_history: List[DriftResult] = []
        self.active_alerts: Dict[str, DriftAlert] = {}

        # Machine learning components
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.drift_models: Dict[str, Any] = {}
        self.is_trained = False

        # Detection thresholds
        self.detection_thresholds = self._load_detection_thresholds()

        # Monitoring metrics
        self.metrics_registry = prom.CollectorRegistry()
        self._setup_drift_metrics()

        # State
        self.is_running = False
        self.detection_thread = None

        self.logger.info("DriftDetector initialized")

    def _setup_drift_metrics(self):
        """Setup drift detection metrics"""
        self.drift_detections_total = prom.Counter(
            'drift_detections_total',
            'Total number of drift detections',
            ['drift_type', 'severity', 'method'],
            registry=self.metrics_registry
        )

        self.drift_magnitude = prom.Gauge(
            'drift_magnitude',
            'Magnitude of detected drift',
            ['drift_type', 'feature'],
            registry=self.metrics_registry
        )

        self.false_positive_rate = prom.Gauge(
            'drift_false_positive_rate',
            'False positive rate of drift detection',
            ['method'],
            registry=self.metrics_registry
        )

        self.detection_latency = prom.Histogram(
            'drift_detection_latency_seconds',
            'Time taken to detect drift',
            ['method'],
            registry=self.metrics_registry
        )

    def _load_detection_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load drift detection thresholds from configuration"""
        return {
            "statistical_tests": {
                "ks_test": 0.05,
                "chi_square_test": 0.05,
                "mann_whitney_test": 0.05,
                "t_test": 0.05
            },
            "distance_metrics": {
                "js_distance": 0.1,
                "kl_divergence": 0.2,
                "wasserstein_distance": 0.1,
                "hellinger_distance": 0.1
            },
            "model_based": {
                "classifier_accuracy_drop": 0.1,
                "autoencoder_reconstruction_error": 0.2,
                "isolation_forest_score": 0.1
            },
            "performance_drift": {
                "accuracy_drop": 0.05,
                "precision_drop": 0.05,
                "recall_drop": 0.05,
                "f1_drop": 0.05,
                "response_time_increase": 0.3
            }
        }

    async def start_drift_detection(self, reference_data_path: str):
        """Start the drift detection system"""
        if self.is_running:
            self.logger.warning("Drift detection is already running")
            return

        # Load reference data
        await self._load_reference_data(reference_data_path)

        self.is_running = True

        # Start detection thread
        self.detection_thread = threading.Thread(target=self._drift_detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()

        self.logger.info("Drift detection system started")

    def stop_drift_detection(self):
        """Stop the drift detection system"""
        self.is_running = False
        self.logger.info("Drift detection system stopped")

    async def _load_reference_data(self, reference_data_path: str):
        """Load reference data for drift detection"""
        try:
            if reference_data_path.endswith('.csv'):
                self.reference_data = pd.read_csv(reference_data_path)
            elif reference_data_path.endswith('.json'):
                with open(reference_data_path, 'r') as f:
                    data = json.load(f)
                self.reference_data = pd.DataFrame(data)
            else:
                raise ValueError(f"Unsupported file format: {reference_data_path}")

            # Preprocess reference data
            self._preprocess_reference_data()

            # Train drift detection models
            await self._train_drift_models()

            self.logger.info(f"Reference data loaded: {len(self.reference_data)} samples")

        except Exception as e:
            self.logger.error(f"Error loading reference data: {e}")
            raise

    def _preprocess_reference_data(self):
        """Preprocess reference data for drift detection"""
        try:
            # Identify numerical and categorical columns
            self.numerical_columns = self.reference_data.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_columns = self.reference_data.select_dtypes(include=['object']).columns.tolist()

            # Fit scalers for numerical columns
            for col in self.numerical_columns:
                if col in self.reference_data.columns:
                    self.scalers[col] = StandardScaler()
                    self.reference_data[col] = self.scalers[col].fit_transform(
                        self.reference_data[col].values.reshape(-1, 1)
                    ).flatten()

            # Fit encoders for categorical columns
            for col in self.categorical_columns:
                if col in self.reference_data.columns:
                    self.encoders[col] = LabelEncoder()
                    self.reference_data[col] = self.encoders[col].fit_transform(
                        self.reference_data[col].astype(str)
                    )

        except Exception as e:
            self.logger.error(f"Error preprocessing reference data: {e}")

    async def _train_drift_models(self):
        """Train drift detection models"""
        try:
            if self.reference_data is None:
                return

            # Train domain classifier for covariate shift detection
            await self._train_domain_classifier()

            # Train autoencoder for reconstruction-based drift detection
            await self._train_autoencoder()

            # Train isolation forest for anomaly-based drift detection
            await self._train_isolation_forest()

            self.is_trained = True
            self.logger.info("Drift detection models trained successfully")

        except Exception as e:
            self.logger.error(f"Error training drift models: {e}")

    async def _train_domain_classifier(self):
        """Train domain classifier for covariate shift detection"""
        try:
            if self.reference_data is None:
                return

            # Create synthetic domain labels
            reference_data_labeled = self.reference_data.copy()
            reference_data_labeled['domain'] = 0

            # Train classifier
            X = reference_data_labeled.drop(['domain'], axis=1, errors='ignore')
            y = reference_data_labeled['domain']

            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            classifier.fit(X, y)

            self.drift_models['domain_classifier'] = classifier

        except Exception as e:
            self.logger.error(f"Error training domain classifier: {e}")

    async def _train_autoencoder(self):
        """Train autoencoder for reconstruction-based drift detection"""
        try:
            if self.reference_data is None:
                return

            # Simple autoencoder using sklearn
            from sklearn.neural_network import MLPRegressor

            # Use only numerical columns for autoencoder
            numerical_data = self.reference_data[self.numerical_columns]

            autoencoder = MLPRegressor(
                hidden_layer_sizes=(len(self.numerical_columns) // 2, len(self.numerical_columns)),
                max_iter=1000,
                random_state=42
            )

            autoencoder.fit(numerical_data, numerical_data)
            self.drift_models['autoencoder'] = autoencoder

        except Exception as e:
            self.logger.error(f"Error training autoencoder: {e}")

    async def _train_isolation_forest(self):
        """Train isolation forest for anomaly detection"""
        try:
            if self.reference_data is None:
                return

            # Use only numerical columns
            numerical_data = self.reference_data[self.numerical_columns]

            isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            isolation_forest.fit(numerical_data)

            self.drift_models['isolation_forest'] = isolation_forest

        except Exception as e:
            self.logger.error(f"Error training isolation forest: {e}")

    def _drift_detection_loop(self):
        """Main drift detection loop"""
        while self.is_running:
            try:
                # Collect current data batch
                if len(self.current_data) >= 100:  # Minimum batch size
                    current_batch = self.current_data[-100:]
                    self.current_data = self.current_data[:-50]  # Keep some data

                    # Perform drift detection
                    drift_results = self._detect_drift_batch(current_batch)

                    # Process results
                    for result in drift_results:
                        self._process_drift_result(result)

                # Sleep for detection interval
                interval = self.config.get("drift_detection", {}).get("detection_interval", 300)
                time.sleep(interval)

            except Exception as e:
                self.logger.error(f"Error in drift detection loop: {e}")
                time.sleep(30)

    def add_data_sample(self, sample: Dict[str, Any]):
        """Add new data sample for drift detection"""
        try:
            # Preprocess sample
            processed_sample = self._preprocess_sample(sample)
            if processed_sample:
                self.current_data.append(processed_sample)

            # Keep buffer size manageable
            if len(self.current_data) > 1000:
                self.current_data = self.current_data[-500:]

        except Exception as e:
            self.logger.error(f"Error adding data sample: {e}")

    def _preprocess_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Preprocess a single data sample"""
        try:
            processed = sample.copy()

            # Apply scaling to numerical columns
            for col in self.numerical_columns:
                if col in processed and col in self.scalers:
                    processed[col] = self.scalers[col].transform([[processed[col]]])[0][0]

            # Apply encoding to categorical columns
            for col in self.categorical_columns:
                if col in processed and col in self.encoders:
                    try:
                        processed[col] = self.encoders[col].transform([str(processed[col])])[0]
                    except ValueError:
                        # Handle unseen categories
                        processed[col] = -1

            return processed

        except Exception as e:
            self.logger.error(f"Error preprocessing sample: {e}")
            return None

    def _detect_drift_batch(self, data_batch: List[Dict[str, Any]]) -> List[DriftResult]:
        """Detect drift in a batch of data"""
        results = []

        try:
            if not data_batch or self.reference_data is None:
                return results

            current_df = pd.DataFrame(data_batch)

            # Data drift detection
            data_drift_results = self._detect_data_drift(current_df)
            results.extend(data_drift_results)

            # Concept drift detection
            concept_drift_results = self._detect_concept_drift(current_df)
            results.extend(concept_drift_results)

            # Label drift detection
            label_drift_results = self._detect_label_drift(current_df)
            results.extend(label_drift_results)

            # Performance drift detection (if performance metrics available)
            performance_drift_results = self._detect_performance_drift(current_df)
            results.extend(performance_drift_results)

        except Exception as e:
            self.logger.error(f"Error detecting drift in batch: {e}")

        return results

    def _detect_data_drift(self, current_data: pd.DataFrame) -> List[DriftResult]:
        """Detect data drift using statistical tests"""
        results = []

        try:
            for column in self.numerical_columns:
                if column not in current_data.columns:
                    continue

                ref_values = self.reference_data[column].dropna()
                cur_values = current_data[column].dropna()

                if len(ref_values) < 30 or len(cur_values) < 30:
                    continue

                # Kolmogorov-Smirnov test
                ks_stat, ks_p_value = stats.ks_2samp(ref_values, cur_values)
                ks_threshold = self.detection_thresholds["statistical_tests"]["ks_test"]

                if ks_p_value < ks_threshold:
                    severity = self._calculate_drift_severity(ks_p_value, ks_threshold)
                    magnitude = abs(ks_stat)

                    result = DriftResult(
                        drift_type=DriftType.DATA_DRIFT,
                        detection_method=DetectionMethod.STATISTICAL_TEST,
                        test_statistic=ks_stat,
                        p_value=ks_p_value,
                        threshold=ks_threshold,
                        drift_detected=True,
                        severity=severity,
                        confidence=1 - ks_p_value,
                        timestamp=datetime.now(timezone.utc),
                        features_affected=[column],
                        magnitude=magnitude,
                        description=f"Kolmogorov-Smirnov test detected drift in feature {column}"
                    )
                    results.append(result)

                # Wasserstein distance
                w_distance = stats.wasserstein_distance(ref_values, cur_values)
                w_threshold = self.detection_thresholds["distance_metrics"]["wasserstein_distance"]

                if w_distance > w_threshold:
                    severity = self._calculate_drift_severity(w_distance, w_threshold, inverse=True)

                    result = DriftResult(
                        drift_type=DriftType.DATA_DRIFT,
                        detection_method=DetectionMethod.DISTANCE_BASED,
                        test_statistic=w_distance,
                        p_value=None,
                        threshold=w_threshold,
                        drift_detected=True,
                        severity=severity,
                        confidence=min(w_distance / w_threshold, 1.0),
                        timestamp=datetime.now(timezone.utc),
                        features_affected=[column],
                        magnitude=w_distance,
                        description=f"Wasserstein distance detected drift in feature {column}"
                    )
                    results.append(result)

        except Exception as e:
            self.logger.error(f"Error in data drift detection: {e}")

        return results

    def _detect_concept_drift(self, current_data: pd.DataFrame) -> List[DriftResult]:
        """Detect concept drift using model-based methods"""
        results = []

        try:
            # Domain classifier method
            if 'domain_classifier' in self.drift_models:
                classifier = self.drift_models['domain_classifier']

                # Prepare current data
                current_processed = current_data[self.numerical_columns].dropna()
                if len(current_processed) > 0:
                    # Predict domain labels
                    predictions = classifier.predict(current_processed)

                    # Calculate classifier accuracy
                    domain_accuracy = np.mean(predictions == 0)  # Should be reference domain

                    accuracy_threshold = self.detection_thresholds["model_based"]["classifier_accuracy_drop"]
                    accuracy_drop = 1 - domain_accuracy

                    if accuracy_drop > accuracy_threshold:
                        severity = self._calculate_drift_severity(accuracy_drop, accuracy_threshold, inverse=True)

                        result = DriftResult(
                            drift_type=DriftType.CONCEPT_DRIFT,
                            detection_method=DetectionMethod.CLASSIFIER_BASED,
                            test_statistic=accuracy_drop,
                            p_value=None,
                            threshold=accuracy_threshold,
                            drift_detected=True,
                            severity=severity,
                            confidence=accuracy_drop,
                            timestamp=datetime.now(timezone.utc),
                            features_affected=self.numerical_columns,
                            magnitude=accuracy_drop,
                            description="Domain classifier detected concept drift"
                        )
                        results.append(result)

            # Autoencoder method
            if 'autoencoder' in self.drift_models:
                autoencoder = self.drift_models['autoencoder']
                current_numerical = current_data[self.numerical_columns].dropna()

                if len(current_numerical) > 0:
                    # Reconstruct current data
                    reconstructed = autoencoder.predict(current_numerical)
                    reconstruction_error = np.mean(np.square(current_numerical - reconstructed), axis=1)

                    # Compare with reference reconstruction error
                    ref_numerical = self.reference_data[self.numerical_columns].dropna()
                    ref_reconstructed = autoencoder.predict(ref_numerical)
                    ref_reconstruction_error = np.mean(np.square(ref_numerical - ref_reconstructed), axis=1)

                    # Statistical test
                    error_stat, error_p_value = stats.ks_2samp(ref_reconstruction_error, reconstruction_error)
                    error_threshold = self.detection_thresholds["model_based"]["autoencoder_reconstruction_error"]

                    if error_p_value < error_threshold:
                        severity = self._calculate_drift_severity(error_p_value, error_threshold)

                        result = DriftResult(
                            drift_type=DriftType.CONCEPT_DRIFT,
                            detection_method=DetectionMethod.AUTOENCODER_BASED,
                            test_statistic=error_stat,
                            p_value=error_p_value,
                            threshold=error_threshold,
                            drift_detected=True,
                            severity=severity,
                            confidence=1 - error_p_value,
                            timestamp=datetime.now(timezone.utc),
                            features_affected=self.numerical_columns,
                            magnitude=error_stat,
                            description="Autoencoder detected concept drift"
                        )
                        results.append(result)

        except Exception as e:
            self.logger.error(f"Error in concept drift detection: {e}")

        return results

    def _detect_label_drift(self, current_data: pd.DataFrame) -> List[DriftResult]:
        """Detect label drift in target variable distribution"""
        results = []

        try:
            # Look for common label column names
            label_columns = ['target', 'label', 'y', 'output', 'prediction']
            label_column = None

            for col in label_columns:
                if col in current_data.columns and col in self.reference_data.columns:
                    label_column = col
                    break

            if label_column is None:
                return results

            ref_labels = self.reference_data[label_column].dropna()
            cur_labels = current_data[label_column].dropna()

            if len(ref_labels) < 30 or len(cur_labels) < 30:
                return results

            # Chi-square test for categorical labels
            if ref_labels.dtype == 'object' or len(ref_labels.unique()) < 10:
                # Create contingency table
                all_labels = sorted(set(ref_labels.unique()) | set(cur_labels.unique()))
                ref_counts = [ref_labels.value_counts().get(label, 0) for label in all_labels]
                cur_counts = [cur_labels.value_counts().get(label, 0) for label in all_labels]

                contingency_table = np.array([ref_counts, cur_counts])
                chi2_stat, chi2_p_value, _, _ = stats.chi2_contingency(contingency_table)

                chi2_threshold = self.detection_thresholds["statistical_tests"]["chi_square_test"]

                if chi2_p_value < chi2_threshold:
                    severity = self._calculate_drift_severity(chi2_p_value, chi2_threshold)

                    result = DriftResult(
                        drift_type=DriftType.LABEL_DRIFT,
                        detection_method=DetectionMethod.STATISTICAL_TEST,
                        test_statistic=chi2_stat,
                        p_value=chi2_p_value,
                        threshold=chi2_threshold,
                        drift_detected=True,
                        severity=severity,
                        confidence=1 - chi2_p_value,
                        timestamp=datetime.now(timezone.utc),
                        features_affected=[label_column],
                        magnitude=chi2_stat,
                        description=f"Chi-square test detected label drift in {label_column}"
                    )
                    results.append(result)

        except Exception as e:
            self.logger.error(f"Error in label drift detection: {e}")

        return results

    def _detect_performance_drift(self, current_data: pd.DataFrame) -> List[DriftResult]:
        """Detect performance drift using performance metrics"""
        results = []

        try:
            # Look for performance metric columns
            performance_metrics = ['accuracy', 'precision', 'recall', 'f1', 'response_time', 'error_rate']

            for metric in performance_metrics:
                if metric not in current_data.columns or metric not in self.reference_data.columns:
                    continue

                ref_values = self.reference_data[metric].dropna()
                cur_values = current_data[metric].dropna()

                if len(ref_values) < 10 or len(cur_values) < 10:
                    continue

                ref_mean = np.mean(ref_values)
                cur_mean = np.mean(cur_values)

                # Calculate relative change
                if ref_mean != 0:
                    relative_change = abs(cur_mean - ref_mean) / abs(ref_mean)
                else:
                    relative_change = abs(cur_mean - ref_mean)

                # Get threshold for this metric
                metric_threshold_key = f"{metric}_drop" if "accuracy" in metric or "precision" in metric or "recall" in metric or "f1" in metric else f"{metric}_increase"
                threshold = self.detection_thresholds["performance_drift"].get(metric_threshold_key, 0.1)

                if relative_change > threshold:
                    severity = self._calculate_drift_severity(relative_change, threshold, inverse=True)

                    result = DriftResult(
                        drift_type=DriftType.PERFORMANCE_DRIFT,
                        detection_method=DetectionMethod.STATISTICAL_TEST,
                        test_statistic=relative_change,
                        p_value=None,
                        threshold=threshold,
                        drift_detected=True,
                        severity=severity,
                        confidence=min(relative_change / threshold, 1.0),
                        timestamp=datetime.now(timezone.utc),
                        features_affected=[metric],
                        magnitude=relative_change,
                        description=f"Performance drift detected in {metric}: {ref_mean:.3f} -> {cur_mean:.3f}"
                    )
                    results.append(result)

        except Exception as e:
            self.logger.error(f"Error in performance drift detection: {e}")

        return results

    def _calculate_drift_severity(self, value: float, threshold: float, inverse: bool = False) -> DriftSeverity:
        """Calculate drift severity based on value and threshold"""
        try:
            if inverse:
                ratio = value / threshold
            else:
                ratio = threshold / value if value > 0 else float('inf')

            if ratio >= 3:
                return DriftSeverity.CRITICAL
            elif ratio >= 2:
                return DriftSeverity.HIGH
            elif ratio >= 1.5:
                return DriftSeverity.MEDIUM
            else:
                return DriftSeverity.LOW

        except Exception:
            return DriftSeverity.LOW

    def _process_drift_result(self, result: DriftResult):
        """Process drift detection result"""
        try:
            # Store result
            self.drift_history.append(result)

            # Update metrics
            self.drift_detections_total.labels(
                drift_type=result.drift_type.value,
                severity=result.severity.value,
                method=result.detection_method.value
            ).inc()

            for feature in result.features_affected:
                self.drift_magnitude.labels(
                    drift_type=result.drift_type.value,
                    feature=feature
                ).set(result.magnitude)

            # Create alert if severity is high or critical
            if result.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
                self._create_drift_alert(result)

            self.logger.warning(
                f"Drift detected: {result.drift_type.value} - {result.severity.value} - {result.description}"
            )

        except Exception as e:
            self.logger.error(f"Error processing drift result: {e}")

    def _create_drift_alert(self, result: DriftResult):
        """Create drift alert"""
        try:
            alert_id = f"drift_{result.drift_type.value}_{int(time.time())}"

            recommended_actions = self._generate_recommended_actions(result)

            alert = DriftAlert(
                id=alert_id,
                drift_type=result.drift_type,
                severity=result.severity,
                timestamp=result.timestamp,
                description=result.description,
                affected_features=result.features_affected,
                magnitude=result.magnitude,
                confidence=result.confidence,
                recommended_actions=recommended_actions,
                metadata={
                    "test_statistic": result.test_statistic,
                    "threshold": result.threshold,
                    "detection_method": result.detection_method.value
                }
            )

            self.active_alerts[alert_id] = alert

            # Send notification
            asyncio.create_task(self._send_drift_notification(alert))

        except Exception as e:
            self.logger.error(f"Error creating drift alert: {e}")

    def _generate_recommended_actions(self, result: DriftResult) -> List[str]:
        """Generate recommended actions for drift alert"""
        actions = []

        try:
            if result.drift_type == DriftType.DATA_DRIFT:
                actions.extend([
                    "Investigate data pipeline for changes",
                    "Validate data quality and preprocessing steps",
                    "Consider updating reference data distribution",
                    "Review data source configurations"
                ])

            elif result.drift_type == DriftType.CONCEPT_DRIFT:
                actions.extend([
                    "Schedule model retraining with recent data",
                    "Review model architecture for concept adaptation",
                    "Consider online learning approaches",
                    "Update feature engineering pipeline"
                ])

            elif result.drift_type == DriftType.LABEL_DRIFT:
                actions.extend([
                    "Validate labeling process and guidelines",
                    "Review labeler training and consistency",
                    "Consider active learning for label validation",
                    "Update labeling instructions if needed"
                ])

            elif result.drift_type == DriftType.PERFORMANCE_DRIFT:
                actions.extend([
                    "Analyze performance degradation root cause",
                    "Check system resources and infrastructure",
                    "Review model inference optimization",
                    "Consider model rollback or update"
                ])

            # Add severity-specific actions
            if result.severity == DriftSeverity.CRITICAL:
                actions.insert(0, "IMMEDIATE ACTION REQUIRED: Consider model rollback")
                actions.append("Escalate to model ops team immediately")

            elif result.severity == DriftSeverity.HIGH:
                actions.insert(0, "Schedule urgent model review")
                actions.append("Prepare emergency response plan")

        except Exception as e:
            self.logger.error(f"Error generating recommended actions: {e}")

        return actions

    async def _send_drift_notification(self, alert: DriftAlert):
        """Send drift notification"""
        try:
            await self.notification_manager.send_drift_alert(alert)
        except Exception as e:
            self.logger.error(f"Error sending drift notification: {e}")

    def get_drift_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get drift detection summary"""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            recent_drift = [d for d in self.drift_history if d.timestamp > cutoff_time]

            # Count by type and severity
            drift_by_type = {}
            drift_by_severity = {}

            for drift in recent_drift:
                drift_type = drift.drift_type.value
                drift_severity = drift.severity.value

                drift_by_type[drift_type] = drift_by_type.get(drift_type, 0) + 1
                drift_by_severity[drift_severity] = drift_by_severity.get(drift_severity, 0) + 1

            # Most affected features
            feature_counts = {}
            for drift in recent_drift:
                for feature in drift.features_affected:
                    feature_counts[feature] = feature_counts.get(feature, 0) + 1

            return {
                "total_drift_events": len(recent_drift),
                "drift_by_type": drift_by_type,
                "drift_by_severity": drift_by_severity,
                "most_affected_features": sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:5],
                "active_alerts": len(self.active_alerts),
                "last_detection": recent_drift[-1].timestamp.isoformat() if recent_drift else None
            }

        except Exception as e:
            self.logger.error(f"Error generating drift summary: {e}")
            return {}

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active drift alerts"""
        return [asdict(alert) for alert in self.active_alerts.values()]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge a drift alert"""
        if alert_id in self.active_alerts:
            # Here you would typically update alert status
            self.logger.info(f"Alert acknowledged: {alert_id}")
            return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve a drift alert"""
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]
            self.logger.info(f"Alert resolved: {alert_id}")
            return True
        return False

async def main():
    """Main drift detection execution"""
    # Load configuration
    config_path = "configs/lifecycle/phase6_maintenance_evolution.yaml"

    # Initialize drift detector
    drift_detector = DriftDetector(config_path)

    # Create some sample reference data
    reference_data_path = "reference_data.csv"
    if not os.path.exists(reference_data_path):
        # Generate sample data for demonstration
        np.random.seed(42)
        sample_data = {
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(5, 2, 1000),
            'feature3': np.random.exponential(1, 1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000),
            'target': np.random.choice([0, 1], 1000),
            'accuracy': np.random.normal(0.85, 0.05, 1000),
            'response_time': np.random.normal(1000, 200, 1000)
        }
        sample_df = pd.DataFrame(sample_data)
        sample_df.to_csv(reference_data_path, index=False)

    # Start drift detection
    await drift_detector.start_drift_detection(reference_data_path)

    try:
        # Simulate incoming data
        for i in range(100):
            # Simulate some data drift
            if i > 50:
                # Introduce drift in feature1
                feature1 = np.random.normal(0.5, 1, 1)[0]  # Shift mean
            else:
                feature1 = np.random.normal(0, 1, 1)[0]

            sample = {
                'feature1': feature1,
                'feature2': np.random.normal(5, 2, 1)[0],
                'feature3': np.random.exponential(1, 1)[0],
                'category': np.random.choice(['A', 'B', 'C'], 1)[0],
                'target': np.random.choice([0, 1], 1)[0],
                'accuracy': np.random.normal(0.85, 0.05, 1)[0],
                'response_time': np.random.normal(1000, 200, 1)[0]
            }

            drift_detector.add_data_sample(sample)

            await asyncio.sleep(1)

        # Print summary
        summary = drift_detector.get_drift_summary()
        print(f"Drift Summary: {json.dumps(summary, indent=2)}")

        active_alerts = drift_detector.get_active_alerts()
        print(f"Active Alerts: {len(active_alerts)}")

    except KeyboardInterrupt:
        print("\nShutting down drift detection system...")
        drift_detector.stop_drift_detection()

if __name__ == "__main__":
    asyncio.run(main())