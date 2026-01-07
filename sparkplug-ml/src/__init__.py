"""
SparkPlug ML Pipeline Package
"""

__version__ = "0.1.0"
__author__ = "SparkPlug Team"

from .utilities.configuration_manager import ConfigurationManager
from .utilities.version_control import VersionControlManager
from .utilities.security_utils import SecurityManager, encrypt_sensitive_data, decrypt_sensitive_data

__all__ = [
    "ConfigurationManager",
    "VersionControlManager",
    "SecurityManager",
    "encrypt_sensitive_data",
    "decrypt_sensitive_data",
]
