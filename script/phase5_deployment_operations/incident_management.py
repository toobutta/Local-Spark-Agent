#!/usr/bin/env python3
"""
Phase 5: Deployment & Operations - Incident Management System
Implements comprehensive incident detection, response, and resolution
for deployed LLM models with automated remediation.
"""

import asyncio
import logging
import yaml
import json
import os
import sys
import time
import threading
import smtplib
import subprocess
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, asdict, field
from enum import Enum
import aiohttp
import jinja2
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import redis
import pymongo
from sklearn.cluster import DBSCAN
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from scripts.utils.config_loader import load_config
from scripts.utils.logger import setup_logger
from scripts.utils.notification import NotificationManager

class IncidentSeverity(Enum):
    """Incident severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class IncidentStatus(Enum):
    """Incident status"""
    OPEN = "open"
    INVESTIGATING = "investigating"
    IDENTIFIED = "identified"
    MONITORING = "monitoring"
    RESOLVED = "resolved"
    CLOSED = "closed"

class IncidentType(Enum):
    """Types of incidents"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SERVICE_UNAVAILABLE = "service_unavailable"
    HIGH_ERROR_RATE = "high_error_rate"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SECURITY_INCIDENT = "security_incident"
    DATA_CORRUPTION = "data_corruption"
    MODEL_DRIFT = "model_drift"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"

class ActionStatus(Enum):
    """Remediation action status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Incident:
    """Incident definition"""
    id: str
    title: str
    description: str
    severity: IncidentSeverity
    type: IncidentType
    status: IncidentStatus
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    assigned_to: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    affected_services: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    root_cause: Optional[str] = None
    resolution: Optional[str] = None
    lessons_learned: Optional[str] = None
    custom_fields: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IncidentUpdate:
    """Incident update"""
    incident_id: str
    timestamp: datetime
    message: str
    author: str
    update_type: str  # status_change, note, action_taken
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RemediationAction:
    """Remediation action definition"""
    id: str
    incident_id: str
    name: str
    description: str
    action_type: str  # manual, automated, script
    status: ActionStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    script_path: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

@dataclass
class IncidentPattern:
    """Pattern for incident detection"""
    name: str
    description: str
    conditions: List[Dict[str, Any]]
    severity: IncidentSeverity
    incident_type: IncidentType
    auto_remediation: List[str] = field(default_factory=list)
    enabled: bool = True

class IncidentManager:
    """Comprehensive incident management system"""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.logger = setup_logger(
            "incident_manager",
            self.config.get("logging", {}).get("level", "INFO")
        )

        # Initialize components
        self.notification_manager = NotificationManager(self.config)

        # Storage
        self.incidents: Dict[str, Incident] = {}
        self.incident_updates: Dict[str, List[IncidentUpdate]] = {}
        self.remediation_actions: Dict[str, RemediationAction] = {}
        self.incident_patterns: Dict[str, IncidentPattern] = {}

        # State
        self.is_running = False
        self.detection_thread = None
        self.remediation_thread = None

        # Load incident patterns and setup
        self._load_incident_patterns()
        self._setup_escalation_policies()

        self.logger.info("IncidentManager initialized")

    def _load_incident_patterns(self):
        """Load incident detection patterns"""
        patterns_config = self.config.get("incident_management", {}).get("detection_patterns", [])

        # Default patterns if not configured
        if not patterns_config:
            patterns_config = [
                {
                    "name": "High Error Rate",
                    "description": "Detects high error rates in LLM requests",
                    "conditions": [
                        {"metric": "error_rate", "operator": "gt", "threshold": 5.0, "duration": 300},
                        {"metric": "request_count", "operator": "gt", "threshold": 10, "duration": 60}
                    ],
                    "severity": "high",
                    "incident_type": "high_error_rate",
                    "auto_remediation": ["restart_service", "scale_up"]
                },
                {
                    "name": "Service Unavailable",
                    "description": "Detects service unavailability",
                    "conditions": [
                        {"metric": "health_check", "operator": "eq", "threshold": 0, "duration": 60}
                    ],
                    "severity": "critical",
                    "incident_type": "service_unavailable",
                    "auto_remediation": ["restart_service", "check_dependencies"]
                },
                {
                    "name": "Performance Degradation",
                    "description": "Detects performance degradation",
                    "conditions": [
                        {"metric": "response_time_p95", "operator": "gt", "threshold": 5000, "duration": 300}
                    ],
                    "severity": "medium",
                    "incident_type": "performance_degradation",
                    "auto_remediation": ["scale_up", "clear_cache"]
                },
                {
                    "name": "Resource Exhaustion",
                    "description": "Detects resource exhaustion",
                    "conditions": [
                        {"metric": "cpu_usage", "operator": "gt", "threshold": 90, "duration": 300},
                        {"metric": "memory_usage", "operator": "gt", "threshold": 85, "duration": 300}
                    ],
                    "severity": "high",
                    "incident_type": "resource_exhaustion",
                    "auto_remediation": ["scale_up", "restart_service"]
                }
            ]

        for pattern_config in patterns_config:
            pattern = IncidentPattern(
                name=pattern_config["name"],
                description=pattern_config["description"],
                conditions=pattern_config["conditions"],
                severity=IncidentSeverity(pattern_config["severity"]),
                incident_type=IncidentType(pattern_config["incident_type"]),
                auto_remediation=pattern_config.get("auto_remediation", []),
                enabled=pattern_config.get("enabled", True)
            )
            self.incident_patterns[pattern.name] = pattern

        self.logger.info(f"Loaded {len(self.incident_patterns)} incident patterns")

    def _setup_escalation_policies(self):
        """Setup escalation policies"""
        self.escalation_policies = {
            IncidentSeverity.LOW: {
                "notification_delay": 300,  # 5 minutes
                "escalation_delay": 1800,   # 30 minutes
                "max_escalations": 2
            },
            IncidentSeverity.MEDIUM: {
                "notification_delay": 60,   # 1 minute
                "escalation_delay": 600,    # 10 minutes
                "max_escalations": 3
            },
            IncidentSeverity.HIGH: {
                "notification_delay": 30,   # 30 seconds
                "escalation_delay": 300,    # 5 minutes
                "max_escalations": 4
            },
            IncidentSeverity.CRITICAL: {
                "notification_delay": 0,    # immediate
                "escalation_delay": 120,    # 2 minutes
                "max_escalations": 5
            }
        }

    async def start_incident_management(self):
        """Start the incident management system"""
        if self.is_running:
            self.logger.warning("Incident management is already running")
            return

        self.is_running = True

        # Start detection and remediation threads
        self.detection_thread = threading.Thread(target=self._incident_detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()

        self.remediation_thread = threading.Thread(target=self._remediation_loop)
        self.remediation_thread.daemon = True
        self.remediation_thread.start()

        self.logger.info("Incident management system started")

    def stop_incident_management(self):
        """Stop the incident management system"""
        self.is_running = False
        self.logger.info("Incident management system stopped")

    def _incident_detection_loop(self):
        """Main incident detection loop"""
        while self.is_running:
            try:
                # Check all patterns for incident detection
                self._check_incident_patterns()

                # Check for incident escalations
                self._check_incident_escalations()

                # Sleep for detection interval
                interval = self.config.get("incident_management", {}).get("detection_interval", 60)
                time.sleep(interval)

            except Exception as e:
                self.logger.error(f"Error in incident detection loop: {e}")
                time.sleep(10)

    def _remediation_loop(self):
        """Main remediation loop"""
        while self.is_running:
            try:
                # Process pending remediation actions
                self._process_remediation_actions()

                # Sleep for remediation interval
                interval = self.config.get("incident_management", {}).get("remediation_interval", 30)
                time.sleep(interval)

            except Exception as e:
                self.logger.error(f"Error in remediation loop: {e}")
                time.sleep(5)

    def _check_incident_patterns(self):
        """Check all incident patterns for matches"""
        current_time = datetime.now(timezone.utc)

        for pattern_name, pattern in self.incident_patterns.items():
            if not pattern.enabled:
                continue

            try:
                # Check if pattern conditions are met
                if self._evaluate_pattern_conditions(pattern):
                    # Check if incident already exists for this pattern
                    existing_incident = self._find_existing_incident(pattern)

                    if existing_incident is None:
                        # Create new incident
                        incident_id = self._create_incident_from_pattern(pattern)
                        self.logger.warning(f"New incident detected: {incident_id} - {pattern.name}")

            except Exception as e:
                self.logger.error(f"Error checking incident pattern {pattern_name}: {e}")

    def _evaluate_pattern_conditions(self, pattern: IncidentPattern) -> bool:
        """Evaluate if pattern conditions are met"""
        try:
            for condition in pattern.conditions:
                metric_name = condition["metric"]
                operator = condition["operator"]
                threshold = condition["threshold"]
                duration = condition.get("duration", 0)

                # Get current metric value
                current_value = self._get_metric_value(metric_name)
                if current_value is None:
                    return False

                # Check historical values if duration is specified
                if duration > 0:
                    if not self._check_metric_duration(metric_name, operator, threshold, duration):
                        return False
                else:
                    # Check current value
                    if not self._evaluate_condition(current_value, operator, threshold):
                        return False

            return True

        except Exception as e:
            self.logger.error(f"Error evaluating pattern conditions: {e}")
            return False

    def _evaluate_condition(self, value: float, operator: str, threshold: float) -> bool:
        """Evaluate a single condition"""
        if operator == "gt":
            return value > threshold
        elif operator == "lt":
            return value < threshold
        elif operator == "eq":
            return value == threshold
        elif operator == "ne":
            return value != threshold
        elif operator == "gte":
            return value >= threshold
        elif operator == "lte":
            return value <= threshold
        else:
            return False

    def _check_metric_duration(self, metric_name: str, operator: str, threshold: float, duration: int) -> bool:
        """Check if metric has been meeting condition for specified duration"""
        try:
            # This would typically query your time-series database
            # For now, we'll simulate the check
            historical_values = self._get_metric_history(metric_name, duration)

            if len(historical_values) < duration // 60:  # Need at least some data points
                return False

            # Check if all values in the duration meet the condition
            for value in historical_values:
                if not self._evaluate_condition(value, operator, threshold):
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking metric duration: {e}")
            return False

    def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current metric value"""
        # This would typically query your metrics system
        # For now, we'll simulate some values
        simulated_metrics = {
            "error_rate": 1.5,
            "request_count": 100,
            "health_check": 1,
            "response_time_p95": 800,
            "cpu_usage": 65,
            "memory_usage": 70
        }
        return simulated_metrics.get(metric_name)

    def _get_metric_history(self, metric_name: str, duration: int) -> List[float]:
        """Get metric history for specified duration"""
        # This would typically query your time-series database
        # For now, we'll return simulated data
        import random
        base_value = self._get_metric_value(metric_name) or 0
        return [base_value + random.uniform(-10, 10) for _ in range(duration // 60)]

    def _find_existing_incident(self, pattern: IncidentPattern) -> Optional[Incident]:
        """Find existing incident for pattern"""
        for incident in self.incidents.values():
            if (incident.type == pattern.incident_type and
                incident.status not in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]):
                return incident
        return None

    def _create_incident_from_pattern(self, pattern: IncidentPattern) -> str:
        """Create incident from pattern"""
        incident_id = f"inc_{int(time.time())}"

        incident = Incident(
            id=incident_id,
            title=pattern.name,
            description=f"Automatically detected: {pattern.description}",
            severity=pattern.severity,
            type=pattern.incident_type,
            status=IncidentStatus.OPEN,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tags=["auto-detected"],
            affected_services=["llm-service"]
        )

        self.incidents[incident_id] = incident
        self.incident_updates[incident_id] = []

        # Add initial update
        self._add_incident_update(incident_id, "Incident created automatically", "system", "status_change")

        # Schedule auto-remediation
        if pattern.auto_remediation:
            self._schedule_auto_remediation(incident_id, pattern.auto_remediation)

        # Send notifications
        asyncio.create_task(self._send_incident_notification(incident))

        return incident_id

    def _schedule_auto_remediation(self, incident_id: str, actions: List[str]):
        """Schedule automatic remediation actions"""
        for action_name in actions:
            action_id = f"{incident_id}_{action_name}_{int(time.time())}"

            remediation_action = RemediationAction(
                id=action_id,
                incident_id=incident_id,
                name=action_name,
                description=f"Automatic remediation: {action_name}",
                action_type="automated",
                status=ActionStatus.PENDING,
                created_at=datetime.now(timezone.utc),
                parameters={}
            )

            self.remediation_actions[action_id] = remediation_action

    def _process_remediation_actions(self):
        """Process pending remediation actions"""
        current_time = datetime.now(timezone.utc)

        for action_id, action in self.remediation_actions.items():
            if action.status == ActionStatus.PENDING:
                # Check if it's time to execute
                delay = self.config.get("incident_management", {}).get("remediation_delay", 60)
                if (current_time - action.created_at).total_seconds() >= delay:
                    asyncio.create_task(self._execute_remediation_action(action))

            elif action.status == ActionStatus.IN_PROGRESS:
                # Check if action has timed out
                timeout = self.config.get("incident_management", {}).get("remediation_timeout", 300)
                if (action.started_at and
                    (current_time - action.started_at).total_seconds() > timeout):
                    action.status = ActionStatus.FAILED
                    action.error_message = "Remediation action timed out"
                    action.completed_at = current_time

    async def _execute_remediation_action(self, action: RemediationAction):
        """Execute a remediation action"""
        try:
            action.status = ActionStatus.IN_PROGRESS
            action.started_at = datetime.now(timezone.utc)

            self.logger.info(f"Executing remediation action: {action.name} for incident {action.incident_id}")

            # Execute based on action type
            if action.name == "restart_service":
                result = await self._restart_service(action)
            elif action.name == "scale_up":
                result = await self._scale_service(action, "up")
            elif action.name == "scale_down":
                result = await self._scale_service(action, "down")
            elif action.name == "clear_cache":
                result = await self._clear_cache(action)
            elif action.name == "check_dependencies":
                result = await self._check_dependencies(action)
            elif action.name == "rollback_deployment":
                result = await self._rollback_deployment(action)
            else:
                # Execute custom script
                result = await self._execute_custom_script(action)

            if result["success"]:
                action.status = ActionStatus.COMPLETED
                action.result = result
                self.logger.info(f"Remediation action completed successfully: {action.name}")

                # Add incident update
                self._add_incident_update(
                    action.incident_id,
                    f"Remediation action '{action.name}' completed successfully",
                    "system",
                    "action_taken",
                    {"action_id": action.id, "result": result}
                )
            else:
                action.status = ActionStatus.FAILED
                action.error_message = result.get("error", "Unknown error")
                action.result = result
                self.logger.error(f"Remediation action failed: {action.name} - {action.error_message}")

                # Add incident update
                self._add_incident_update(
                    action.incident_id,
                    f"Remediation action '{action.name}' failed: {action.error_message}",
                    "system",
                    "action_taken",
                    {"action_id": action.id, "error": action.error_message}
                )

            action.completed_at = datetime.now(timezone.utc)

        except Exception as e:
            action.status = ActionStatus.FAILED
            action.error_message = str(e)
            action.completed_at = datetime.now(timezone.utc)
            self.logger.error(f"Error executing remediation action {action.name}: {e}")

    async def _restart_service(self, action: RemediationAction) -> Dict[str, Any]:
        """Restart the service"""
        try:
            # This would typically use your service management system
            # For now, we'll simulate the restart
            result = subprocess.run(
                ["docker", "restart", "llm-service"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                return {"success": True, "output": result.stdout}
            else:
                return {"success": False, "error": result.stderr}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _scale_service(self, action: RemediationAction, direction: str) -> Dict[str, Any]:
        """Scale the service up or down"""
        try:
            # This would typically use your orchestration system
            # For now, we'll simulate the scaling
            current_replicas = 3
            new_replicas = current_replicas + 1 if direction == "up" else max(1, current_replicas - 1)

            result = subprocess.run(
                ["kubectl", "scale", "deployment", "llm-service", f"--replicas={new_replicas}"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                return {"success": True, "old_replicas": current_replicas, "new_replicas": new_replicas}
            else:
                return {"success": False, "error": result.stderr}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _clear_cache(self, action: RemediationAction) -> Dict[str, Any]:
        """Clear service cache"""
        try:
            # This would typically clear your application cache
            # For now, we'll simulate the operation
            return {"success": True, "cache_cleared": True}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _check_dependencies(self, action: RemediationAction) -> Dict[str, Any]:
        """Check service dependencies"""
        try:
            # This would typically check your service dependencies
            # For now, we'll simulate the check
            dependencies = ["database", "redis", "storage"]
            results = {dep: "healthy" for dep in dependencies}

            return {"success": True, "dependencies": results}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _rollback_deployment(self, action: RemediationAction) -> Dict[str, Any]:
        """Rollback to previous deployment"""
        try:
            # This would typically rollback your deployment
            # For now, we'll simulate the rollback
            result = subprocess.run(
                ["kubectl", "rollback", "deployment", "llm-service"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                return {"success": True, "output": result.stdout}
            else:
                return {"success": False, "error": result.stderr}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_custom_script(self, action: RemediationAction) -> Dict[str, Any]:
        """Execute custom remediation script"""
        try:
            if not action.script_path:
                return {"success": False, "error": "No script path specified"}

            # Execute the script
            result = subprocess.run(
                ["python", action.script_path] + [f"{k}={v}" for k, v in action.parameters.items()],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                return {"success": True, "output": result.stdout}
            else:
                return {"success": False, "error": result.stderr}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _check_incident_escalations(self):
        """Check for incident escalations"""
        current_time = datetime.now(timezone.utc)

        for incident in self.incidents.values():
            if incident.status in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]:
                continue

            policy = self.escalation_policies.get(incident.severity)
            if not policy:
                continue

            # Check if escalation is needed
            time_since_creation = (current_time - incident.created_at).total_seconds()

            if time_since_creation > policy["escalation_delay"]:
                # Escalate incident
                self._escalate_incident(incident)

    def _escalate_incident(self, incident: Incident):
        """Escalate an incident"""
        # This would typically implement your escalation logic
        # For now, we'll just add an update and send notification
        self._add_incident_update(
            incident.id,
            "Incident escalated due to delayed resolution",
            "system",
            "status_change"
        )

        asyncio.create_task(self._send_escalation_notification(incident))

    def _add_incident_update(self, incident_id: str, message: str, author: str,
                           update_type: str, metadata: Dict[str, Any] = None):
        """Add update to incident"""
        if incident_id not in self.incident_updates:
            self.incident_updates[incident_id] = []

        update = IncidentUpdate(
            incident_id=incident_id,
            timestamp=datetime.now(timezone.utc),
            message=message,
            author=author,
            update_type=update_type,
            metadata=metadata or {}
        )

        self.incident_updates[incident_id].append(update)

        # Update incident timestamp
        if incident_id in self.incidents:
            self.incidents[incident_id].updated_at = update.timestamp

    async def _send_incident_notification(self, incident: Incident):
        """Send incident notification"""
        try:
            await self.notification_manager.send_incident_alert(incident)
        except Exception as e:
            self.logger.error(f"Error sending incident notification: {e}")

    async def _send_escalation_notification(self, incident: Incident):
        """Send escalation notification"""
        try:
            await self.notification_manager.send_escalation_alert(incident)
        except Exception as e:
            self.logger.error(f"Error sending escalation notification: {e}")

    def create_manual_incident(self, title: str, description: str, severity: IncidentSeverity,
                             incident_type: IncidentType, affected_services: List[str] = None,
                             tags: List[str] = None) -> str:
        """Create a manual incident"""
        incident_id = f"inc_manual_{int(time.time())}"

        incident = Incident(
            id=incident_id,
            title=title,
            description=description,
            severity=severity,
            type=incident_type,
            status=IncidentStatus.OPEN,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tags=tags or [],
            affected_services=affected_services or []
        )

        self.incidents[incident_id] = incident
        self.incident_updates[incident_id] = []

        # Add initial update
        self._add_incident_update(incident_id, "Incident created manually", "user", "status_change")

        # Send notifications
        asyncio.create_task(self._send_incident_notification(incident))

        self.logger.info(f"Manual incident created: {incident_id}")
        return incident_id

    def update_incident_status(self, incident_id: str, new_status: IncidentStatus,
                             author: str, message: str = None) -> bool:
        """Update incident status"""
        if incident_id not in self.incidents:
            return False

        incident = self.incidents[incident_id]
        old_status = incident.status
        incident.status = new_status
        incident.updated_at = datetime.now(timezone.utc)

        if new_status == IncidentStatus.RESOLVED:
            incident.resolved_at = datetime.now(timezone.utc)

        # Add update
        update_message = message or f"Status changed from {old_status.value} to {new_status.value}"
        self._add_incident_update(incident_id, update_message, author, "status_change")

        self.logger.info(f"Incident {incident_id} status updated to {new_status.value}")
        return True

    def add_incident_note(self, incident_id: str, note: str, author: str) -> bool:
        """Add note to incident"""
        if incident_id not in self.incidents:
            return False

        self._add_incident_update(incident_id, note, author, "note")
        return True

    def assign_incident(self, incident_id: str, assignee: str, author: str) -> bool:
        """Assign incident to someone"""
        if incident_id not in self.incidents:
            return False

        incident = self.incidents[incident_id]
        incident.assigned_to = assignee
        incident.updated_at = datetime.now(timezone.utc)

        self._add_incident_update(
            incident_id,
            f"Incident assigned to {assignee}",
            author,
            "status_change"
        )

        return True

    def create_remediation_action(self, incident_id: str, name: str, description: str,
                                action_type: str, script_path: str = None,
                                parameters: Dict[str, Any] = None) -> str:
        """Create remediation action for incident"""
        action_id = f"{incident_id}_{name}_{int(time.time())}"

        remediation_action = RemediationAction(
            id=action_id,
            incident_id=incident_id,
            name=name,
            description=description,
            action_type=action_type,
            status=ActionStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            script_path=script_path,
            parameters=parameters or {}
        )

        self.remediation_actions[action_id] = remediation_action

        self.logger.info(f"Remediation action created: {action_id}")
        return action_id

    def get_incidents(self, status: IncidentStatus = None, severity: IncidentSeverity = None) -> List[Dict[str, Any]]:
        """Get incidents with optional filters"""
        incidents = list(self.incidents.values())

        if status:
            incidents = [i for i in incidents if i.status == status]

        if severity:
            incidents = [i for i in incidents if i.severity == severity]

        return [asdict(incident) for incident in incidents]

    def get_incident_details(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed incident information"""
        if incident_id not in self.incidents:
            return None

        incident = self.incidents[incident_id]
        updates = self.incident_updates.get(incident_id, [])
        actions = [a for a in self.remediation_actions.values() if a.incident_id == incident_id]

        return {
            "incident": asdict(incident),
            "updates": [asdict(update) for update in updates],
            "remediation_actions": [asdict(action) for action in actions]
        }

    def get_active_incidents_summary(self) -> Dict[str, Any]:
        """Get summary of active incidents"""
        active_incidents = [i for i in self.incidents.values()
                          if i.status not in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]]

        severity_counts = {}
        type_counts = {}

        for incident in active_incidents:
            severity_counts[incident.severity.value] = severity_counts.get(incident.severity.value, 0) + 1
            type_counts[incident.type.value] = type_counts.get(incident.type.value, 0) + 1

        return {
            "total_active": len(active_incidents),
            "by_severity": severity_counts,
            "by_type": type_counts,
            "critical_incidents": len([i for i in active_incidents if i.severity == IncidentSeverity.CRITICAL])
        }

async def main():
    """Main incident management execution"""
    # Load configuration
    config_path = "configs/lifecycle/phase5_deployment_operations.yaml"

    # Initialize incident manager
    incident_manager = IncidentManager(config_path)

    # Start incident management
    await incident_manager.start_incident_management()

    try:
        # Keep system running
        while True:
            await asyncio.sleep(60)

            # Print status
            summary = incident_manager.get_active_incidents_summary()
            print(f"Active incidents: {summary['total_active']}")
            print(f"Critical incidents: {summary['critical_incidents']}")

    except KeyboardInterrupt:
        print("\nShutting down incident management system...")
        incident_manager.stop_incident_management()

if __name__ == "__main__":
    asyncio.run(main())