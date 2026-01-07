"""
Configuration Manager for SparkPlug ML Pipeline
Handles loading, validation, and management of configuration files
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for ML pipeline execution"""
    name: str
    version: str
    description: str

    # Data configuration
    data_source: str
    data_format: str
    preprocessing_steps: list

    # Model configuration
    model_type: str
    model_config: Dict[str, Any]

    # Training configuration
    training_config: Dict[str, Any]

    # Deployment configuration
    deployment_target: str
    deployment_config: Dict[str, Any]

    # Monitoring configuration
    monitoring_config: Dict[str, Any]

class ConfigurationManager:
    """Manages configuration loading and validation"""

    def __init__(self, config_path: Union[str, Path]):
        path = Path(config_path)
        self._single_config_file: Optional[Path] = path if path.is_file() else None
        self.config_path = path if path.is_dir() else path.parent
        self._config_cache: Dict[str, Any] = {}
        self._config_mtime: Dict[str, float] = {}
        self._single_file_cache: Optional[Dict[str, Any]] = None
        self._single_file_mtime: Optional[float] = None

    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load configuration from file with caching"""
        config_file = self.config_path / f"{config_name}.yaml"

        if not config_file.exists():
            config_file = self.config_path / f"{config_name}.yml"
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file {config_name}.yaml not found")

        # Check if file has been modified
        current_mtime = config_file.stat().st_mtime
        if config_name in self._config_mtime and self._config_mtime[config_name] == current_mtime:
            return self._config_cache[config_name]

        # Load and cache configuration
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        self._config_cache[config_name] = config
        self._config_mtime[config_name] = current_mtime

        logger.info(f"Loaded configuration: {config_name}")
        return config

    def _load_single_file(self) -> Dict[str, Any]:
        """Load the single configuration file when a direct file path is provided."""
        if not self._single_config_file:
            raise FileNotFoundError("No configuration file specified for direct loading")

        current_mtime = self._single_config_file.stat().st_mtime
        if self._single_file_cache is not None and self._single_file_mtime == current_mtime:
            return self._single_file_cache

        with open(self._single_config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}

        self._single_file_cache = config
        self._single_file_mtime = current_mtime
        return config

    def get_section(self, section_name: str, default: Optional[Any] = None) -> Any:
        """Retrieve a specific section from either a single config file or a named config."""
        fallback = {} if default is None else default

        if self._single_config_file:
            config = self._load_single_file()
            return config.get(section_name, fallback)

        try:
            return self.load_config(section_name)
        except FileNotFoundError:
            return fallback

    def save_config(self, config_name: str, config: Dict[str, Any]) -> None:
        """Save configuration to file"""
        config_file = self.config_path / f"{config_name}.yaml"

        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

        # Update cache
        self._config_cache[config_name] = config
        self._config_mtime[config_name] = config_file.stat().st_mtime

        logger.info(f"Saved configuration: {config_name}")

    def get_pipeline_config(self, pipeline_name: str) -> PipelineConfig:
        """Load and validate pipeline configuration"""
        config = self.load_config(f"pipelines/{pipeline_name}")

        # Validate required fields
        required_fields = [
            'name', 'version', 'data_source', 'model_type',
            'training_config', 'deployment_config'
        ]

        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")

        return PipelineConfig(**config)

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Load model-specific configuration"""
        return self.load_config(f"models/{model_name}")

    def get_environment_config(self, environment: str = 'development') -> Dict[str, Any]:
        """Load environment-specific configuration"""
        return self.load_config(f"environments/{environment}")

    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configurations with later configs taking precedence"""
        result = {}

        def deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    target[key] = deep_merge(target[key], value)
                else:
                    target[key] = value
            return target

        for config in configs:
            result = deep_merge(result, config)

        return result

    def validate_config(self, config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Validate configuration against schema"""
        # Simple validation - could be enhanced with JSON schema validation
        def validate_section(config_section: Dict[str, Any], schema_section: Dict[str, Any]) -> bool:
            for key, value_spec in schema_section.items():
                if key not in config_section:
                    if isinstance(value_spec, dict) and value_spec.get('required', False):
                        return False
                    continue

                config_value = config_section[key]

                # Type checking
                expected_type = value_spec.get('type')
                if expected_type and not isinstance(config_value, eval(expected_type)):
                    return False

                # Range checking for numbers
                if expected_type == 'int' or expected_type == 'float':
                    min_val = value_spec.get('min')
                    max_val = value_spec.get('max')
                    if min_val is not None and config_value < min_val:
                        return False
                    if max_val is not None and config_value > max_val:
                        return False

                # Recursive validation for nested objects
                if isinstance(value_spec, dict) and 'properties' in value_spec:
                    if not validate_section(config_value, value_spec['properties']):
                        return False

            return True

        return validate_section(config, schema)

    def create_default_config(self, config_type: str) -> Dict[str, Any]:
        """Create default configuration for given type"""
        defaults = {
            'pipeline': {
                'name': 'default_pipeline',
                'version': '1.0.0',
                'description': 'Default ML pipeline configuration',
                'data_source': 'local',
                'data_format': 'csv',
                'preprocessing_steps': ['normalize', 'encode'],
                'model_type': 'linear_regression',
                'model_config': {},
                'training_config': {
                    'epochs': 100,
                    'batch_size': 32,
                    'learning_rate': 0.001
                },
                'deployment_config': {
                    'target': 'local',
                    'format': 'pickle'
                },
                'monitoring_config': {
                    'enabled': True,
                    'metrics': ['accuracy', 'loss']
                }
            },
            'model': {
                'type': 'neural_network',
                'layers': [64, 32, 1],
                'activation': 'relu',
                'optimizer': 'adam',
                'loss': 'mse'
            },
            'environment': {
                'database_url': 'sqlite:///ml_pipeline.db',
                'cache_dir': './cache',
                'output_dir': './outputs',
                'log_level': 'INFO'
            }
        }

        return defaults.get(config_type, {})
