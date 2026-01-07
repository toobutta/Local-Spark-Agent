#!/usr/bin/env python3
"""
Phase 5: Deployment & Operations - Comprehensive Monitoring System
Implements real-time monitoring, alerting, and performance optimization
for deployed LLM models.
"""

import asyncio
import logging
import yaml
import json
import os
import sys
import time
import statistics
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, asdict, field
from enum import Enum
import aiohttp
import psutil
import prometheus_client as prom
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, start_http_server
import redis
import pymongo
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from scripts.utils.config_loader import load_config
from scripts.utils.logger import setup_logger
from scripts.utils.notification import NotificationManager

class MetricType(Enum):
    """Types of metrics to collect"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class PerformanceTier(Enum):
    """Performance tiers for auto-scaling"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class MetricValue:
    """Single metric value with metadata"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE
    unit: str = ""

@dataclass
class AlertRule:
    """Alert rule definition"""
    name: str
    metric_name: str
    condition: str  # gt, lt, eq, ne
    threshold: float
    severity: AlertSeverity
    duration: int = 300  # seconds
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True

@dataclass
class Alert:
    """Alert instance"""
    id: str
    rule_name: str
    severity: AlertSeverity
    status: str  # firing, resolved
    started_at: datetime
    resolved_at: Optional[datetime] = None
    value: float = 0.0
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
    """Performance metrics aggregation"""
    timestamp: datetime
    response_time_p50: float
    response_time_p95: float
    response_time_p99: float
    throughput_rps: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float] = None
    disk_io: Optional[float] = None
    network_io: Optional[float] = None

class MonitoringSystem:
    """Comprehensive monitoring system for LLM deployments"""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.logger = setup_logger(
            "monitoring_system",
            self.config.get("logging", {}).get("level", "INFO")
        )

        # Initialize components
        self.metrics_registry = prom.CollectorRegistry()
        self.notification_manager = NotificationManager(self.config)

        # Storage
        self.metrics_buffer: List[MetricValue] = []
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.performance_history: List[PerformanceMetrics] = []

        # Anomaly detection
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

        # Monitoring state
        self.is_running = False
        self.monitoring_thread = None
        self.alerting_thread = None

        # Setup metrics and alert rules
        self._setup_prometheus_metrics()
        self._load_alert_rules()

        self.logger.info("MonitoringSystem initialized")

    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        # Application metrics
        self.request_count = prom.Counter(
            'llm_requests_total',
            'Total number of LLM requests',
            ['model', 'endpoint', 'status'],
            registry=self.metrics_registry
        )

        self.request_duration = prom.Histogram(
            'llm_request_duration_seconds',
            'LLM request duration in seconds',
            ['model', 'endpoint'],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0],
            registry=self.metrics_registry
        )

        self.active_connections = prom.Gauge(
            'llm_active_connections',
            'Number of active connections',
            ['model'],
            registry=self.metrics_registry
        )

        # System metrics
        self.cpu_usage = prom.Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.metrics_registry
        )

        self.memory_usage = prom.Gauge(
            'system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.metrics_registry
        )

        self.disk_usage = prom.Gauge(
            'system_disk_usage_percent',
            'System disk usage percentage',
            ['device'],
            registry=self.metrics_registry
        )

        # Model-specific metrics
        self.model_inference_time = prom.Histogram(
            'llm_inference_time_seconds',
            'Model inference time in seconds',
            ['model', 'batch_size'],
            registry=self.metrics_registry
        )

        self.model_queue_size = prom.Gauge(
            'llm_model_queue_size',
            'Number of requests in model queue',
            ['model'],
            registry=self.metrics_registry
        )

        self.gpu_usage = prom.Gauge(
            'system_gpu_usage_percent',
            'System GPU usage percentage',
            ['gpu_id'],
            registry=self.metrics_registry
        )

        # Custom metrics
        self.custom_metrics: Dict[str, Union[prom.Counter, prom.Gauge, prom.Histogram]] = {}

    def _load_alert_rules(self):
        """Load alert rules from configuration"""
        alert_rules_config = self.config.get("monitoring", {}).get("alert_rules", [])

        for rule_config in alert_rules_config:
            rule = AlertRule(
                name=rule_config["name"],
                metric_name=rule_config["metric_name"],
                condition=rule_config["condition"],
                threshold=rule_config["threshold"],
                severity=AlertSeverity(rule_config["severity"]),
                duration=rule_config.get("duration", 300),
                labels=rule_config.get("labels", {}),
                annotations=rule_config.get("annotations", {}),
                enabled=rule_config.get("enabled", True)
            )
            self.alert_rules[rule.name] = rule

        self.logger.info(f"Loaded {len(self.alert_rules)} alert rules")

    async def start_monitoring(self):
        """Start the monitoring system"""
        if self.is_running:
            self.logger.warning("Monitoring is already running")
            return

        self.is_running = True

        # Start Prometheus metrics server
        prom_port = self.config.get("monitoring", {}).get("prometheus_port", 8000)
        start_http_server(prom_port, registry=self.metrics_registry)
        self.logger.info(f"Prometheus metrics server started on port {prom_port}")

        # Start monitoring threads
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        self.alerting_thread = threading.Thread(target=self._alerting_loop)
        self.alerting_thread.daemon = True
        self.alerting_thread.start()

        self.logger.info("Monitoring system started")

    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.is_running = False
        self.logger.info("Monitoring system stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Collect system metrics
                self._collect_system_metrics()

                # Collect application metrics
                self._collect_application_metrics()

                # Store performance snapshot
                self._store_performance_snapshot()

                # Sleep for monitoring interval
                interval = self.config.get("monitoring", {}).get("collection_interval", 30)
                time.sleep(interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)

    def _alerting_loop(self):
        """Main alerting loop"""
        while self.is_running:
            try:
                # Evaluate alert rules
                self._evaluate_alert_rules()

                # Check for anomaly detection
                self._check_anomalies()

                # Sleep for alerting interval
                interval = self.config.get("monitoring", {}).get("alerting_interval", 60)
                time.sleep(interval)

            except Exception as e:
                self.logger.error(f"Error in alerting loop: {e}")
                time.sleep(10)

    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage.set(cpu_percent)
            self._add_metric("system_cpu_usage", cpu_percent, {"unit": "percent"})

            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.percent)
            self._add_metric("system_memory_usage", memory.percent, {"unit": "percent"})

            # Disk usage
            disk_partitions = psutil.disk_partitions()
            for partition in disk_partitions:
                try:
                    disk_usage = psutil.disk_usage(partition.mountpoint)
                    device = partition.device.replace(':', '').replace('\\', '_')
                    self.disk_usage.labels(device=device).set(disk_usage.percent)
                    self._add_metric("system_disk_usage", disk_usage.percent,
                                   {"device": device, "unit": "percent"})
                except PermissionError:
                    continue

            # Network I/O
            network_io = psutil.net_io_counters()
            network_bytes_sent = network_io.bytes_sent
            network_bytes_recv = network_io.bytes_recv
            self._add_metric("network_bytes_sent_total", network_bytes_sent, {"unit": "bytes"})
            self._add_metric("network_bytes_recv_total", network_bytes_recv, {"unit": "bytes"})

            # GPU usage (if available)
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_id = str(gpu.id)
                    self.gpu_usage.labels(gpu_id=gpu_id).set(gpu.load * 100)
                    self._add_metric("system_gpu_usage", gpu.load * 100,
                                   {"gpu_id": gpu_id, "unit": "percent"})
            except ImportError:
                pass

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")

    def _collect_application_metrics(self):
        """Collect application-specific metrics"""
        try:
            # This would typically collect metrics from your application
            # For now, we'll simulate some metrics

            # Simulate request metrics
            import random
            if random.random() > 0.7:  # Simulate occasional requests
                model_name = "example-model"
                endpoint = "/predict"
                status = "200" if random.random() > 0.1 else "500"

                self.request_count.labels(model=model_name, endpoint=endpoint, status=status).inc()

                # Simulate request duration
                duration = random.uniform(0.1, 2.0)
                self.request_duration.labels(model=model_name, endpoint=endpoint).observe(duration)

            # Simulate active connections
            active_conns = random.randint(1, 50)
            self.active_connections.labels(model="example-model").set(active_conns)

            # Simulate model inference metrics
            if random.random() > 0.8:  # Simulate occasional inference
                batch_size = random.choice([1, 4, 8, 16])
                inference_time = random.uniform(0.05, 0.5)
                self.model_inference_time.labels(model="example-model", batch_size=str(batch_size)).observe(inference_time)

            queue_size = random.randint(0, 20)
            self.model_queue_size.labels(model="example-model").set(queue_size)

        except Exception as e:
            self.logger.error(f"Error collecting application metrics: {e}")

    def _store_performance_snapshot(self):
        """Store performance metrics snapshot"""
        try:
            timestamp = datetime.now(timezone.utc)

            # Get recent metrics for aggregation
            recent_metrics = [m for m in self.metrics_buffer
                            if m.timestamp > timestamp - timedelta(minutes=5)]

            if not recent_metrics:
                return

            # Calculate performance metrics
            performance_metrics = PerformanceMetrics(
                timestamp=timestamp,
                response_time_p50=self._calculate_percentile(recent_metrics, "request_duration", 50),
                response_time_p95=self._calculate_percentile(recent_metrics, "request_duration", 95),
                response_time_p99=self._calculate_percentile(recent_metrics, "request_duration", 99),
                throughput_rps=self._calculate_throughput(recent_metrics),
                error_rate=self._calculate_error_rate(recent_metrics),
                cpu_usage=self._get_latest_metric_value(recent_metrics, "system_cpu_usage"),
                memory_usage=self._get_latest_metric_value(recent_metrics, "system_memory_usage"),
                gpu_usage=self._get_latest_metric_value(recent_metrics, "system_gpu_usage")
            )

            self.performance_history.append(performance_metrics)

            # Keep only last 1000 snapshots
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]

        except Exception as e:
            self.logger.error(f"Error storing performance snapshot: {e}")

    def _evaluate_alert_rules(self):
        """Evaluate all alert rules"""
        current_time = datetime.now(timezone.utc)

        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue

            try:
                # Get current metric value
                metric_value = self._get_current_metric_value(rule.metric_name)
                if metric_value is None:
                    continue

                # Evaluate condition
                is_firing = self._evaluate_condition(metric_value, rule.condition, rule.threshold)

                # Check if alert already exists
                existing_alert = self.alerts.get(rule_name)

                if is_firing:
                    if existing_alert is None:
                        # Create new alert
                        alert = Alert(
                            id=f"{rule_name}_{int(time.time())}",
                            rule_name=rule_name,
                            severity=rule.severity,
                            status="firing",
                            started_at=current_time,
                            value=metric_value,
                            labels=rule.labels,
                            annotations=rule.annotations
                        )
                        self.alerts[rule_name] = alert

                        # Send notification
                        asyncio.create_task(self._send_alert_notification(alert))

                        self.logger.warning(f"Alert fired: {rule_name} - {metric_value} {rule.condition} {rule.threshold}")

                    elif existing_alert.status == "firing":
                        # Check duration
                        duration = (current_time - existing_alert.started_at).total_seconds()
                        if duration >= rule.duration and existing_alert.id not in [a.id for a in self.alerts.values() if a.status == "firing" and a.started_at == existing_alert.started_at]:
                            # Re-send notification for persistent alert
                            asyncio.create_task(self._send_alert_notification(alert))

                else:
                    if existing_alert and existing_alert.status == "firing":
                        # Resolve alert
                        existing_alert.status = "resolved"
                        existing_alert.resolved_at = current_time

                        # Send resolution notification
                        asyncio.create_task(self._send_alert_resolution_notification(existing_alert))

                        self.logger.info(f"Alert resolved: {rule_name}")

            except Exception as e:
                self.logger.error(f"Error evaluating alert rule {rule_name}: {e}")

    def _check_anomalies(self):
        """Check for anomalies in performance metrics"""
        try:
            if len(self.performance_history) < 50:  # Need enough data
                return

            # Prepare data for anomaly detection
            recent_metrics = self.performance_history[-50:]
            features = []

            for metrics in recent_metrics:
                feature_vector = [
                    metrics.response_time_p95,
                    metrics.throughput_rps,
                    metrics.error_rate,
                    metrics.cpu_usage,
                    metrics.memory_usage
                ]
                if metrics.gpu_usage is not None:
                    feature_vector.append(metrics.gpu_usage)

                features.append(feature_vector)

            features_array = np.array(features)

            # Train if not trained
            if not self.is_trained:
                self.scaler.fit(features_array)
                scaled_features = self.scaler.transform(features_array)
                self.anomaly_detector.fit(scaled_features)
                self.is_trained = True
                return

            # Detect anomalies
            scaled_features = self.scaler.transform(features_array)
            anomaly_scores = self.anomaly_detector.decision_function(scaled_features)

            # Check if latest metrics are anomalous
            if anomaly_scores[-1] < 0:  # Anomaly detected
                latest_metrics = recent_metrics[-1]

                # Create anomaly alert
                alert = Alert(
                    id=f"anomaly_{int(time.time())}",
                    rule_name="anomaly_detection",
                    severity=AlertSeverity.WARNING,
                    status="firing",
                    started_at=datetime.now(timezone.utc),
                    value=float(anomaly_scores[-1]),
                    labels={"type": "anomaly"},
                    annotations={
                        "summary": "Anomalous performance detected",
                        "description": f"Anomaly score: {anomaly_scores[-1]:.3f}",
                        "metrics": json.dumps(asdict(latest_metrics), default=str)
                    }
                )

                self.alerts[f"anomaly_{int(time.time())}"] = alert
                asyncio.create_task(self._send_alert_notification(alert))

                self.logger.warning(f"Anomaly detected with score: {anomaly_scores[-1]:.3f}")

        except Exception as e:
            self.logger.error(f"Error checking anomalies: {e}")

    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        if condition == "gt":
            return value > threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "eq":
            return value == threshold
        elif condition == "ne":
            return value != threshold
        else:
            return False

    def _get_current_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value for a metric"""
        # This would typically query your metrics storage
        # For now, we'll simulate based on recent metrics
        recent_metrics = [m for m in self.metrics_buffer
                         if m.name == metric_name and
                         m.timestamp > datetime.now(timezone.utc) - timedelta(minutes=1)]

        if recent_metrics:
            return recent_metrics[-1].value
        return None

    def _add_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Add metric to buffer"""
        metric = MetricValue(
            name=name,
            value=value,
            timestamp=datetime.now(timezone.utc),
            labels=labels or {},
            metric_type=MetricType.GAUGE
        )
        self.metrics_buffer.append(metric)

        # Keep buffer size manageable
        if len(self.metrics_buffer) > 10000:
            self.metrics_buffer = self.metrics_buffer[-5000:]

    def _calculate_percentile(self, metrics: List[MetricValue], metric_name: str, percentile: float) -> float:
        """Calculate percentile for a metric"""
        values = [m.value for m in metrics if m.name == metric_name]
        if values:
            return np.percentile(values, percentile)
        return 0.0

    def _calculate_throughput(self, metrics: List[MetricValue]) -> float:
        """Calculate requests per second"""
        request_counts = [m for m in metrics if m.name == "llm_requests_total"]
        if len(request_counts) >= 2:
            time_diff = (request_counts[-1].timestamp - request_counts[0].timestamp).total_seconds()
            if time_diff > 0:
                count_diff = request_counts[-1].value - request_counts[0].value
                return count_diff / time_diff
        return 0.0

    def _calculate_error_rate(self, metrics: List[MetricValue]) -> float:
        """Calculate error rate"""
        total_requests = 0
        error_requests = 0

        for metric in metrics:
            if metric.name == "llm_requests_total":
                total_requests += metric.value
                if metric.labels.get("status") == "500":
                    error_requests += metric.value

        if total_requests > 0:
            return (error_requests / total_requests) * 100
        return 0.0

    def _get_latest_metric_value(self, metrics: List[MetricValue], metric_name: str) -> Optional[float]:
        """Get latest value for a metric"""
        for metric in reversed(metrics):
            if metric.name == metric_name:
                return metric.value
        return None

    async def _send_alert_notification(self, alert: Alert):
        """Send alert notification"""
        try:
            await self.notification_manager.send_alert(alert)
        except Exception as e:
            self.logger.error(f"Error sending alert notification: {e}")

    async def _send_alert_resolution_notification(self, alert: Alert):
        """Send alert resolution notification"""
        try:
            await self.notification_manager.send_alert_resolution(alert)
        except Exception as e:
            self.logger.error(f"Error sending alert resolution notification: {e}")

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_metrics": {
                "cpu_usage": self.cpu_usage._value.get(),
                "memory_usage": self.memory_usage._value.get(),
                "disk_usage": {label: gauge._value.get() for label, gauge in self.disk_usage._metrics.items()},
                "gpu_usage": {label: gauge._value.get() for label, gauge in self.gpu_usage._metrics.items()}
            },
            "application_metrics": {
                "total_requests": sum(counter._value.get() for counter in self.request_count._metrics.values()),
                "active_connections": sum(gauge._value.get() for gauge in self.active_connections._metrics.values()),
                "queue_sizes": {label: gauge._value.get() for label, gauge in self.model_queue_size._metrics.items()}
            },
            "active_alerts": len([a for a in self.alerts.values() if a.status == "firing"])
        }

    def get_performance_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get performance history"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_history = [m for m in self.performance_history if m.timestamp > cutoff_time]
        return [asdict(m) for m in recent_history]

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts"""
        return [asdict(alert) for alert in self.alerts.values() if alert.status == "firing"]

    def get_alert_rules(self) -> List[Dict[str, Any]]:
        """Get alert rules"""
        return [asdict(rule) for rule in self.alert_rules.values()]

    def create_custom_metric(self, name: str, metric_type: MetricType,
                           labels: List[str] = None, **kwargs) -> bool:
        """Create custom metric"""
        try:
            if metric_type == MetricType.COUNTER:
                metric = prom.Counter(name, kwargs.get("description", ""), labels or [],
                                    registry=self.metrics_registry)
            elif metric_type == MetricType.GAUGE:
                metric = prom.Gauge(name, kwargs.get("description", ""), labels or [],
                                  registry=self.metrics_registry)
            elif metric_type == MetricType.HISTOGRAM:
                metric = prom.Histogram(name, kwargs.get("description", ""), labels or [],
                                      buckets=kwargs.get("buckets", []),
                                      registry=self.metrics_registry)
            else:
                return False

            self.custom_metrics[name] = metric
            self.logger.info(f"Created custom metric: {name}")
            return True

        except Exception as e:
            self.logger.error(f"Error creating custom metric {name}: {e}")
            return False

    def update_custom_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Update custom metric value"""
        try:
            if name not in self.custom_metrics:
                return False

            metric = self.custom_metrics[name]

            if isinstance(metric, (prom.Counter, prom.Gauge)):
                if labels:
                    metric.labels(**labels).set(value)
                else:
                    metric.set(value)
            elif isinstance(metric, prom.Histogram):
                if labels:
                    metric.labels(**labels).observe(value)
                else:
                    metric.observe(value)

            return True

        except Exception as e:
            self.logger.error(f"Error updating custom metric {name}: {e}")
            return False

async def main():
    """Main monitoring system execution"""
    # Load configuration
    config_path = "configs/lifecycle/phase5_deployment_operations.yaml"

    # Initialize monitoring system
    monitoring = MonitoringSystem(config_path)

    # Start monitoring
    await monitoring.start_monitoring()

    try:
        # Keep monitoring running
        while True:
            await asyncio.sleep(60)

            # Print current status
            metrics = monitoring.get_current_metrics()
            print(f"Active alerts: {metrics['active_alerts']}")
            print(f"CPU usage: {metrics['system_metrics']['cpu_usage']:.1f}%")
            print(f"Memory usage: {metrics['system_metrics']['memory_usage']:.1f}%")

    except KeyboardInterrupt:
        print("\nShutting down monitoring system...")
        monitoring.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(main())