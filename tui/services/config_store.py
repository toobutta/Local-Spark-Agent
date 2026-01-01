"""
Configuration persistence layer with YAML storage and encrypted credentials.

Provides secure storage for API keys, workspace paths, and application settings.
"""

import os
import base64
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Generic
from dataclasses import dataclass, field, asdict
from datetime import datetime
import yaml
import json

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False


T = TypeVar('T')


@dataclass
class APICredentials:
    """Encrypted API credentials container."""
    anthropic_key: Optional[str] = None
    openai_key: Optional[str] = None
    gemini_key: Optional[str] = None
    ollama_host: str = "http://localhost:11434"
    custom_keys: Dict[str, str] = field(default_factory=dict)


@dataclass
class WorkspaceConfig:
    """Configuration for a workspace."""
    id: str
    name: str
    path: str
    active_agents: list = field(default_factory=list)
    mcp_servers: list = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AppConfig:
    """Main application configuration."""
    # UI Settings
    theme: str = "dark"
    font_size: int = 12
    show_sidebar: bool = True
    show_metrics: bool = True
    
    # Active workspace
    active_workspace_id: Optional[str] = None
    recent_workspaces: list = field(default_factory=list)
    
    # DGX/GPU Settings
    dgx_simulation_mode: bool = True
    dgx_refresh_interval: float = 1.0
    
    # Agent Settings
    auto_deploy_agents: bool = True
    verbose_logging: bool = False
    
    # Ollama Settings
    use_local_inference: bool = False
    default_model: str = "llama2"
    
    # Keyboard shortcuts (customizable)
    shortcuts: Dict[str, str] = field(default_factory=lambda: {
        "switch_workspace": "ctrl+w",
        "new_workspace": "ctrl+t",
        "plugin_manager": "ctrl+shift+p",
        "toggle_metrics": "ctrl+m",
        "tab_1": "ctrl+1",
        "tab_2": "ctrl+2",
        "tab_3": "ctrl+3",
        "tab_4": "ctrl+4",
        "tab_5": "ctrl+5",
    })


class ConfigStore:
    """
    Persistent configuration store with YAML backend and encrypted credentials.
    
    Usage:
        config = ConfigStore()
        config.set("theme", "dark")
        theme = config.get("theme", default="light")
        
        # Encrypted credentials
        config.set_credential("anthropic_key", "sk-ant-...")
        key = config.get_credential("anthropic_key")
    """
    
    CONFIG_DIR = Path.home() / ".sparkplug"
    CONFIG_FILE = CONFIG_DIR / "config.yaml"
    CREDENTIALS_FILE = CONFIG_DIR / ".credentials"
    WORKSPACES_FILE = CONFIG_DIR / "workspaces.yaml"
    
    _instance: Optional['ConfigStore'] = None
    
    def __new__(cls):
        """Singleton pattern for global config access."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._config: AppConfig = AppConfig()
        self._credentials: APICredentials = APICredentials()
        self._workspaces: Dict[str, WorkspaceConfig] = {}
        self._encryption_key: Optional[bytes] = None
        self._dirty = False
        self._debounce_timer = None
        
        # Ensure config directory exists
        try:
            self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create config directory {self.CONFIG_DIR}: {e}")
            # Continue with in-memory config only
        
        # Initialize encryption
        self._init_encryption()
        
        # Load existing configuration
        self._load_config()
        self._load_credentials()
        self._load_workspaces()
    
    def _init_encryption(self):
        """Initialize encryption key for credentials."""
        if not HAS_CRYPTOGRAPHY:
            return
            
        key_file = self.CONFIG_DIR / ".key"
        
        if key_file.exists():
            # Load existing key
            self._encryption_key = key_file.read_bytes()
        else:
            # Generate new key based on machine-specific data
            salt = os.urandom(16)
            machine_id = self._get_machine_id()
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=480000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(machine_id.encode()))
            
            # Store salt and key
            key_file.write_bytes(salt + key)
            self._encryption_key = key
    
    def _get_machine_id(self) -> str:
        """Get a machine-specific identifier for key derivation."""
        # Combine username and hostname for a semi-unique ID
        import socket
        return f"{os.getlogin()}@{socket.gethostname()}"
    
    def _encrypt(self, data: str) -> str:
        """Encrypt sensitive data."""
        if not HAS_CRYPTOGRAPHY or not self._encryption_key:
            # Fallback to base64 encoding (not secure, but functional)
            return base64.b64encode(data.encode()).decode()
        
        f = Fernet(self._encryption_key)
        return f.encrypt(data.encode()).decode()
    
    def _decrypt(self, data: str) -> str:
        """Decrypt sensitive data."""
        if not HAS_CRYPTOGRAPHY or not self._encryption_key:
            # Fallback to base64 decoding
            try:
                return base64.b64decode(data.encode()).decode()
            except Exception:
                return data
        
        try:
            f = Fernet(self._encryption_key)
            return f.decrypt(data.encode()).decode()
        except Exception:
            return ""
    
    def _load_config(self):
        """Load configuration from YAML file."""
        if self.CONFIG_FILE.exists():
            try:
                with open(self.CONFIG_FILE, 'r') as f:
                    data = yaml.safe_load(f) or {}
                
                # Update config with loaded values
                for key, value in data.items():
                    if hasattr(self._config, key):
                        setattr(self._config, key, value)
            except Exception as e:
                print(f"Warning: Failed to load config: {e}")
    
    def _load_credentials(self):
        """Load encrypted credentials."""
        if self.CREDENTIALS_FILE.exists():
            try:
                with open(self.CREDENTIALS_FILE, 'r') as f:
                    encrypted_data = yaml.safe_load(f) or {}
                
                # Decrypt each credential
                for key, value in encrypted_data.items():
                    if hasattr(self._credentials, key):
                        if isinstance(value, dict):
                            # Handle nested dicts like custom_keys
                            decrypted = {k: self._decrypt(v) for k, v in value.items()}
                            setattr(self._credentials, key, decrypted)
                        elif value:
                            setattr(self._credentials, key, self._decrypt(value))
            except Exception as e:
                print(f"Warning: Failed to load credentials: {e}")
    
    def _load_workspaces(self):
        """Load workspace configurations."""
        if self.WORKSPACES_FILE.exists():
            try:
                with open(self.WORKSPACES_FILE, 'r') as f:
                    data = yaml.safe_load(f) or {}
                
                for ws_id, ws_data in data.items():
                    self._workspaces[ws_id] = WorkspaceConfig(**ws_data)
            except Exception as e:
                print(f"Warning: Failed to load workspaces: {e}")
    
    def _save_config(self):
        """Save configuration to YAML file."""
        try:
            with open(self.CONFIG_FILE, 'w') as f:
                yaml.dump(asdict(self._config), f, default_flow_style=False)
        except Exception as e:
            print(f"Warning: Failed to save config: {e}")
    
    def _save_credentials(self):
        """Save encrypted credentials."""
        try:
            encrypted_data = {}
            creds_dict = asdict(self._credentials)
            
            for key, value in creds_dict.items():
                if isinstance(value, dict):
                    encrypted_data[key] = {k: self._encrypt(v) if v else "" for k, v in value.items()}
                elif value:
                    encrypted_data[key] = self._encrypt(value)
                else:
                    encrypted_data[key] = ""
            
            with open(self.CREDENTIALS_FILE, 'w') as f:
                yaml.dump(encrypted_data, f, default_flow_style=False)
            
            # Set restrictive permissions on credentials file
            os.chmod(self.CREDENTIALS_FILE, 0o600)
        except Exception as e:
            print(f"Warning: Failed to save credentials: {e}")
    
    def _save_workspaces(self):
        """Save workspace configurations."""
        try:
            data = {ws_id: asdict(ws) for ws_id, ws in self._workspaces.items()}
            with open(self.WORKSPACES_FILE, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        except Exception as e:
            print(f"Warning: Failed to save workspaces: {e}")
    
    def save(self):
        """Save all configuration immediately."""
        self._save_config()
        self._save_credentials()
        self._save_workspaces()
        self._dirty = False
    
    def mark_dirty(self):
        """Mark configuration as needing save (for debounced auto-save)."""
        self._dirty = True
    
    # ==================== General Config API ====================
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return getattr(self._config, key, default)
    
    def set(self, key: str, value: Any):
        """Set a configuration value."""
        if hasattr(self._config, key):
            setattr(self._config, key, value)
            self.mark_dirty()
    
    @property
    def config(self) -> AppConfig:
        """Get the full configuration object."""
        return self._config
    
    # ==================== Credentials API ====================
    
    def get_credential(self, key: str) -> Optional[str]:
        """Get a decrypted credential."""
        return getattr(self._credentials, key, None)
    
    def set_credential(self, key: str, value: str):
        """Set an encrypted credential."""
        if hasattr(self._credentials, key):
            setattr(self._credentials, key, value)
            self._save_credentials()
    
    def get_custom_credential(self, key: str) -> Optional[str]:
        """Get a custom credential by key."""
        return self._credentials.custom_keys.get(key)
    
    def set_custom_credential(self, key: str, value: str):
        """Set a custom credential."""
        self._credentials.custom_keys[key] = value
        self._save_credentials()
    
    # ==================== Workspace API ====================
    
    def get_workspace(self, workspace_id: str) -> Optional[WorkspaceConfig]:
        """Get a workspace by ID."""
        return self._workspaces.get(workspace_id)
    
    def get_active_workspace(self) -> Optional[WorkspaceConfig]:
        """Get the currently active workspace."""
        if self._config.active_workspace_id:
            return self._workspaces.get(self._config.active_workspace_id)
        return None
    
    def list_workspaces(self) -> list:
        """List all workspaces."""
        return list(self._workspaces.values())
    
    def create_workspace(self, name: str, path: str) -> WorkspaceConfig:
        """Create a new workspace."""
        import uuid
        ws_id = str(uuid.uuid4())[:8]
        
        workspace = WorkspaceConfig(
            id=ws_id,
            name=name,
            path=path,
        )
        
        self._workspaces[ws_id] = workspace
        self._save_workspaces()
        
        return workspace
    
    def delete_workspace(self, workspace_id: str) -> bool:
        """Delete a workspace."""
        if workspace_id in self._workspaces:
            del self._workspaces[workspace_id]
            
            # Clear active workspace if deleted
            if self._config.active_workspace_id == workspace_id:
                self._config.active_workspace_id = None
            
            # Remove from recent
            if workspace_id in self._config.recent_workspaces:
                self._config.recent_workspaces.remove(workspace_id)
            
            self._save_workspaces()
            self._save_config()
            return True
        return False
    
    def set_active_workspace(self, workspace_id: str) -> bool:
        """Set the active workspace."""
        if workspace_id in self._workspaces:
            self._config.active_workspace_id = workspace_id
            
            # Update last accessed
            self._workspaces[workspace_id].last_accessed = datetime.now().isoformat()
            
            # Update recent workspaces
            if workspace_id in self._config.recent_workspaces:
                self._config.recent_workspaces.remove(workspace_id)
            self._config.recent_workspaces.insert(0, workspace_id)
            self._config.recent_workspaces = self._config.recent_workspaces[:10]  # Keep last 10
            
            self._save_config()
            self._save_workspaces()
            return True
        return False
    
    def update_workspace(self, workspace_id: str, **kwargs) -> bool:
        """Update workspace properties."""
        if workspace_id in self._workspaces:
            ws = self._workspaces[workspace_id]
            for key, value in kwargs.items():
                if hasattr(ws, key):
                    setattr(ws, key, value)
            self._save_workspaces()
            return True
        return False


# Global singleton accessor
def get_config_store() -> ConfigStore:
    """Get the global ConfigStore instance."""
    return ConfigStore()

