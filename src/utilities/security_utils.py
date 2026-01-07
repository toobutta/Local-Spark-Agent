"""
Security Utilities for SparkPlug ML Pipeline
Handles encryption, secure storage, and access control for sensitive data
"""

import os
import base64
import hashlib
import secrets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from typing import Dict, Any, Optional, Union, List
import logging
import json

logger = logging.getLogger(__name__)

class SecurityManager:
    """Manages encryption and security operations"""

    def __init__(self, key_file: Optional[str] = None):
        self.key_file = key_file or os.environ.get('SPARKPLUG_KEY_FILE', '.sparkplug_key')
        self._fernet = None
        self._load_or_create_key()

    def _load_or_create_key(self) -> None:
        """Load existing key or create a new one"""
        if os.path.exists(self.key_file):
            with open(self.key_file, 'rb') as f:
                key = f.read()
        else:
            # Generate a new key
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            # Set restrictive permissions
            os.chmod(self.key_file, 0o600)
            logger.info(f"Created new encryption key: {self.key_file}")

        self._fernet = Fernet(key)

    def encrypt_sensitive_data(self, data: Union[str, Dict[str, Any], List[Any]]) -> str:
        """Encrypt sensitive data"""
        if isinstance(data, (dict, list)):
            data = json.dumps(data)

        if isinstance(data, str):
            data = data.encode('utf-8')

        encrypted = self._fernet.encrypt(data)
        return base64.b64encode(encrypted).decode('utf-8')

    def decrypt_sensitive_data(self, encrypted_data: str) -> Union[str, Dict[str, Any], List[Any]]:
        """Decrypt sensitive data"""
        try:
            encrypted = base64.b64decode(encrypted_data)
            decrypted = self._fernet.decrypt(encrypted)
            decrypted_str = decrypted.decode('utf-8')

            # Try to parse as JSON
            try:
                return json.loads(decrypted_str)
            except json.JSONDecodeError:
                return decrypted_str

        except Exception as e:
            logger.error(f"Failed to decrypt data: {e}")
            raise ValueError("Invalid encrypted data or key")

    def hash_data(self, data: Union[str, bytes], salt: Optional[bytes] = None) -> str:
        """Create a secure hash of data"""
        if isinstance(data, str):
            data = data.encode('utf-8')

        if salt is None:
            salt = secrets.token_bytes(16)

        # Use PBKDF2 for secure hashing
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )

        hash_bytes = kdf.derive(data)
        return base64.b64encode(salt + hash_bytes).decode('utf-8')

    def verify_hash(self, data: Union[str, bytes], hashed_data: str) -> bool:
        """Verify data against a hash"""
        try:
            decoded = base64.b64decode(hashed_data)
            salt = decoded[:16]
            expected_hash = decoded[16:]

            if isinstance(data, str):
                data = data.encode('utf-8')

            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )

            computed_hash = kdf.derive(data)
            return secrets.compare_digest(computed_hash, expected_hash)

        except Exception:
            return False

    def generate_secure_token(self, length: int = 32) -> str:
        """Generate a secure random token"""
        return secrets.token_urlsafe(length)

    def sanitize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Remove or encrypt sensitive information from configuration"""
        sensitive_keys = [
            'password', 'secret', 'key', 'token', 'api_key',
            'database_url', 'connection_string', 'private_key'
        ]

        sanitized = {}

        for key, value in config.items():
            key_lower = key.lower()

            # Check if this key contains sensitive information
            is_sensitive = any(sensitive_key in key_lower for sensitive_key in sensitive_keys)

            if is_sensitive and isinstance(value, str):
                # Encrypt sensitive string values
                sanitized[key] = self.encrypt_sensitive_data(value)
                sanitized[f"{key}_encrypted"] = True
            elif isinstance(value, dict):
                # Recursively sanitize nested dictionaries
                sanitized[key] = self.sanitize_config(value)
            else:
                sanitized[key] = value

        return sanitized

    def restore_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt encrypted values in configuration"""
        restored = {}

        for key, value in config.items():
            if key.endswith('_encrypted') and value is True:
                # This key was encrypted, decrypt its counterpart
                original_key = key[:-10]  # Remove '_encrypted' suffix
                if original_key in config:
                    try:
                        restored[original_key] = self.decrypt_sensitive_data(config[original_key])
                    except Exception:
                        logger.warning(f"Failed to decrypt {original_key}")
                        restored[original_key] = "[ENCRYPTED]"
            elif isinstance(value, dict):
                # Recursively restore nested dictionaries
                restored[key] = self.restore_config(value)
            elif not key.endswith('_encrypted'):
                # Regular key that wasn't encrypted
                restored[key] = value

        return restored

class AccessControl:
    """Manages access control and permissions"""

    def __init__(self, security_manager: SecurityManager):
        self.security = security_manager
        self.permissions_file = '.sparkplug_permissions'

    def grant_permission(self, user: str, resource: str, action: str) -> None:
        """Grant a permission to a user"""
        permissions = self._load_permissions()

        if user not in permissions:
            permissions[user] = {}

        if resource not in permissions[user]:
            permissions[user][resource] = []

        if action not in permissions[user][resource]:
            permissions[user][resource].append(action)

        self._save_permissions(permissions)
        logger.info(f"Granted permission: {user} can {action} on {resource}")

    def revoke_permission(self, user: str, resource: str, action: str) -> None:
        """Revoke a permission from a user"""
        permissions = self._load_permissions()

        if user in permissions and resource in permissions[user]:
            if action in permissions[user][resource]:
                permissions[user][resource].remove(action)

                # Clean up empty entries
                if not permissions[user][resource]:
                    del permissions[user][resource]
                if not permissions[user]:
                    del permissions[user]

        self._save_permissions(permissions)
        logger.info(f"Revoked permission: {user} can no longer {action} on {resource}")

    def check_permission(self, user: str, resource: str, action: str) -> bool:
        """Check if a user has a specific permission"""
        permissions = self._load_permissions()

        return (
            user in permissions and
            resource in permissions[user] and
            action in permissions[user][resource]
        )

    def get_user_permissions(self, user: str) -> Dict[str, list]:
        """Get all permissions for a user"""
        permissions = self._load_permissions()
        return permissions.get(user, {})

    def _load_permissions(self) -> Dict[str, Dict[str, list]]:
        """Load permissions from encrypted file"""
        if not os.path.exists(self.permissions_file):
            return {}

        try:
            with open(self.permissions_file, 'r') as f:
                encrypted_data = f.read().strip()

            return self.security.decrypt_sensitive_data(encrypted_data)
        except Exception:
            logger.warning("Failed to load permissions file, starting fresh")
            return {}

    def _save_permissions(self, permissions: Dict[str, Dict[str, list]]) -> None:
        """Save permissions to encrypted file"""
        encrypted_data = self.security.encrypt_sensitive_data(permissions)

        with open(self.permissions_file, 'w') as f:
            f.write(encrypted_data)

        # Set restrictive permissions
        os.chmod(self.permissions_file, 0o600)

def encrypt_sensitive_data(data: Any) -> str:
    """Convenience function for encrypting sensitive data"""
    security_manager = SecurityManager()
    return security_manager.encrypt_sensitive_data(data)

def decrypt_sensitive_data(encrypted_data: str) -> Any:
    """Convenience function for decrypting sensitive data"""
    security_manager = SecurityManager()
    return security_manager.decrypt_sensitive_data(encrypted_data)
