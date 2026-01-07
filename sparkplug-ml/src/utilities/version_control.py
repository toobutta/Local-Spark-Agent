"""
Version Control Manager for SparkPlug ML Pipeline
Handles model versioning, experiment tracking, and artifact management
"""

import os
import hashlib
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class VersionControlManager:
    """Manages versioning and tracking of ML models and experiments"""

    def __init__(self, base_path: str = "data/versioning", versioning_backend: str = 'file'):
        self.base_path = Path(base_path)
        self.versioning_backend = versioning_backend

        # Create necessary directories
        self.models_dir = self.base_path / 'models'
        self.experiments_dir = self.base_path / 'experiments'
        self.artifacts_dir = self.base_path / 'artifacts'

        for dir_path in [self.models_dir, self.experiments_dir, self.artifacts_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def create_model_version(self, model_name: str, model_data: Any,
                           metadata: Dict[str, Any]) -> str:
        """Create a new version of a model"""
        version_id = self._generate_version_id()
        version_path = self.models_dir / model_name / version_id

        version_path.mkdir(parents=True, exist_ok=True)

        # Save model data
        model_file = version_path / 'model.pkl'
        # Note: Actual model saving would depend on the ML framework
        # For now, we'll just save metadata
        with open(model_file, 'w') as f:
            json.dump({'placeholder': 'model_data'}, f)

        # Save metadata
        metadata_file = version_path / 'metadata.json'
        metadata['version_id'] = version_id
        metadata['created_at'] = datetime.now().isoformat()
        metadata['model_name'] = model_name

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Created model version: {model_name}/{version_id}")
        return version_id

    def get_model_version(self, model_name: str, version_id: str) -> Dict[str, Any]:
        """Retrieve a specific model version"""
        version_path = self.models_dir / model_name / version_id
        metadata_file = version_path / 'metadata.json'

        if not metadata_file.exists():
            raise FileNotFoundError(f"Model version not found: {model_name}/{version_id}")

        with open(metadata_file, 'r') as f:
            return json.load(f)

    def list_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """List all versions of a model"""
        model_path = self.models_dir / model_name

        if not model_path.exists():
            return []

        versions = []
        for version_dir in model_path.iterdir():
            if version_dir.is_dir():
                metadata_file = version_dir / 'metadata.json'
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        versions.append(json.load(f))

        return sorted(versions, key=lambda x: x['created_at'], reverse=True)

    def create_experiment(self, experiment_name: str, config: Dict[str, Any]) -> str:
        """Create a new experiment"""
        experiment_id = self._generate_experiment_id()
        experiment_path = self.experiments_dir / experiment_id

        experiment_path.mkdir(parents=True, exist_ok=True)

        # Save experiment configuration
        config_file = experiment_path / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        # Create experiment metadata
        metadata = {
            'experiment_id': experiment_id,
            'name': experiment_name,
            'status': 'running',
            'created_at': datetime.now().isoformat(),
            'config': config
        }

        metadata_file = experiment_path / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Created experiment: {experiment_name} ({experiment_id})")
        return experiment_id

    def update_experiment(self, experiment_id: str, updates: Dict[str, Any]) -> None:
        """Update experiment metadata"""
        experiment_path = self.experiments_dir / experiment_id
        metadata_file = experiment_path / 'metadata.json'

        if not metadata_file.exists():
            raise FileNotFoundError(f"Experiment not found: {experiment_id}")

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        metadata.update(updates)
        metadata['updated_at'] = datetime.now().isoformat()

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def log_experiment_metric(self, experiment_id: str, metric_name: str,
                            metric_value: Any, step: Optional[int] = None) -> None:
        """Log a metric for an experiment"""
        experiment_path = self.experiments_dir / experiment_id
        metrics_file = experiment_path / 'metrics.json'

        # Load existing metrics
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = {}

        # Add new metric
        if metric_name not in metrics:
            metrics[metric_name] = []

        metric_entry = {
            'value': metric_value,
            'timestamp': datetime.now().isoformat()
        }

        if step is not None:
            metric_entry['step'] = step

        metrics[metric_name].append(metric_entry)

        # Save metrics
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

    def get_experiment_metrics(self, experiment_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get all metrics for an experiment"""
        experiment_path = self.experiments_dir / experiment_id
        metrics_file = experiment_path / 'metrics.json'

        if not metrics_file.exists():
            return {}

        with open(metrics_file, 'r') as f:
            return json.load(f)

    def save_artifact(self, experiment_id: str, artifact_name: str,
                     artifact_data: Any, artifact_type: str = 'file') -> str:
        """Save an artifact for an experiment"""
        artifact_id = self._generate_artifact_id()
        artifact_path = self.artifacts_dir / experiment_id / artifact_id

        artifact_path.mkdir(parents=True, exist_ok=True)

        # Save artifact metadata
        metadata = {
            'artifact_id': artifact_id,
            'name': artifact_name,
            'type': artifact_type,
            'experiment_id': experiment_id,
            'created_at': datetime.now().isoformat()
        }

        metadata_file = artifact_path / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save artifact data (simplified - in practice would handle different data types)
        if isinstance(artifact_data, dict):
            data_file = artifact_path / 'data.json'
            with open(data_file, 'w') as f:
                json.dump(artifact_data, f, indent=2)
        elif isinstance(artifact_data, str):
            data_file = artifact_path / 'data.txt'
            with open(data_file, 'w') as f:
                f.write(artifact_data)
        else:
            # For other types, just save as JSON representation
            data_file = artifact_path / 'data.json'
            with open(data_file, 'w') as f:
                json.dump({'data': str(artifact_data)}, f, indent=2)

        logger.info(f"Saved artifact: {artifact_name} ({artifact_id})")
        return artifact_id

    def compare_experiments(self, experiment_ids: List[str],
                          metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare multiple experiments"""
        comparison = {}

        for exp_id in experiment_ids:
            try:
                # Load experiment metadata
                exp_path = self.experiments_dir / exp_id
                metadata_file = exp_path / 'metadata.json'

                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                comparison[exp_id] = {
                    'metadata': metadata,
                    'metrics': {}
                }

                # Load metrics if requested
                if metrics:
                    exp_metrics = self.get_experiment_metrics(exp_id)
                    for metric in metrics:
                        if metric in exp_metrics:
                            comparison[exp_id]['metrics'][metric] = exp_metrics[metric]

            except FileNotFoundError:
                logger.warning(f"Experiment not found: {exp_id}")
                continue

        return comparison

    def _generate_version_id(self) -> str:
        """Generate a unique version ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_suffix = hashlib.md5(os.urandom(8)).hexdigest()[:6]
        return f"v_{timestamp}_{random_suffix}"

    def _generate_experiment_id(self) -> str:
        """Generate a unique experiment ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_suffix = hashlib.md5(os.urandom(8)).hexdigest()[:6]
        return f"exp_{timestamp}_{random_suffix}"

    def _generate_artifact_id(self) -> str:
        """Generate a unique artifact ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_suffix = hashlib.md5(os.urandom(8)).hexdigest()[:6]
        return f"art_{timestamp}_{random_suffix}"
