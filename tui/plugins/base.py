"""
Plugin Base Class and Infrastructure.

Provides the foundation for SparkPlug TUI plugins.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)


class PluginState(Enum):
    """Plugin lifecycle states."""
    UNLOADED = auto()
    LOADING = auto()
    LOADED = auto()
    ACTIVE = auto()
    ERROR = auto()
    DISABLED = auto()


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str = ""
    author: str = "Unknown"
    homepage: str = ""
    license: str = "MIT"
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)  # Other plugin names
    python_dependencies: List[str] = field(default_factory=list)  # pip packages
    
    # Compatibility
    min_sparkplug_version: str = "1.0.0"
    max_sparkplug_version: Optional[str] = None
    
    # Categories/tags
    tags: List[str] = field(default_factory=list)


@dataclass
class Command:
    """
    A command that can be executed from the command palette.
    
    Attributes:
        name: Command name (e.g., "factory run")
        description: Human-readable description
        handler: Async function to execute
        shortcut: Optional keyboard shortcut
        category: Command category for grouping
    """
    name: str
    description: str
    handler: Callable[..., Any]
    shortcut: Optional[str] = None
    category: str = "General"
    args: List[str] = field(default_factory=list)  # Expected argument names
    
    def __post_init__(self):
        # Ensure handler is callable
        if not callable(self.handler):
            raise ValueError(f"Command handler must be callable: {self.name}")


@dataclass
class Tool:
    """
    A tool that can be used by agents.
    
    Attributes:
        name: Tool name (e.g., "web_search")
        description: Human-readable description
        handler: Function to execute the tool
        schema: JSON schema for tool parameters
    """
    name: str
    description: str
    handler: Callable[..., Any]
    schema: Dict[str, Any] = field(default_factory=dict)
    category: str = "General"
    requires_confirmation: bool = False
    
    def __post_init__(self):
        if not callable(self.handler):
            raise ValueError(f"Tool handler must be callable: {self.name}")


@dataclass
class Widget:
    """
    A UI widget contributed by a plugin.
    
    Attributes:
        name: Widget identifier
        widget_class: The Textual widget class
        location: Where to place the widget (e.g., "sidebar", "footer", "tab")
        priority: Order priority (higher = earlier)
    """
    name: str
    widget_class: Type
    location: str = "sidebar"
    priority: int = 0
    config: Dict[str, Any] = field(default_factory=dict)


class SparkPlugPlugin(ABC):
    """
    Base class for SparkPlug TUI plugins.
    
    Plugins can:
    - Register commands for the command palette
    - Register tools for agents to use
    - Register UI widgets
    - Subscribe to events
    - Access services (config, workspace, etc.)
    
    Example:
        class MyPlugin(SparkPlugPlugin):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="My Plugin",
                    version="1.0.0",
                    description="Does something cool"
                )
            
            def register_commands(self) -> List[Command]:
                return [
                    Command(
                        name="my command",
                        description="Does something",
                        handler=self.my_command_handler
                    )
                ]
            
            async def my_command_handler(self, **kwargs):
                # Command implementation
                pass
    """
    
    def __init__(self):
        self._state = PluginState.UNLOADED
        self._app = None
        self._config = None
        self._event_bus = None
        self._logger = logging.getLogger(f"plugin.{self.__class__.__name__}")
    
    # ==================== Required Properties ====================
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass
    
    # ==================== Lifecycle Methods ====================
    
    async def on_load(self):
        """
        Called when the plugin is loaded.
        
        Override to perform initialization that doesn't require the app.
        """
        pass
    
    async def on_activate(self, app):
        """
        Called when the plugin is activated.
        
        Override to perform initialization that requires the app.
        
        Args:
            app: The SparkPlug TUI application instance
        """
        self._app = app
    
    async def on_deactivate(self):
        """
        Called when the plugin is deactivated.
        
        Override to perform cleanup.
        """
        pass
    
    async def on_unload(self):
        """
        Called when the plugin is unloaded.
        
        Override to perform final cleanup.
        """
        pass
    
    # ==================== Registration Methods ====================
    
    def register_commands(self) -> List[Command]:
        """
        Register commands for the command palette.
        
        Override to provide commands.
        
        Returns:
            List of Command objects
        """
        return []
    
    def register_tools(self) -> List[Tool]:
        """
        Register tools for agents.
        
        Override to provide tools.
        
        Returns:
            List of Tool objects
        """
        return []
    
    def register_widgets(self) -> List[Widget]:
        """
        Register UI widgets.
        
        Override to provide widgets.
        
        Returns:
            List of Widget objects
        """
        return []
    
    # ==================== Event Handling ====================
    
    def get_event_subscriptions(self) -> Dict[str, Callable]:
        """
        Get event subscriptions.
        
        Override to subscribe to events.
        
        Returns:
            Dict mapping event types to handlers
        """
        return {}
    
    # ==================== Utility Methods ====================
    
    @property
    def state(self) -> PluginState:
        """Get current plugin state."""
        return self._state
    
    @property
    def app(self):
        """Get the application instance."""
        return self._app
    
    @property
    def logger(self) -> logging.Logger:
        """Get the plugin logger."""
        return self._logger
    
    def notify(self, message: str, title: str = "", severity: str = "info"):
        """Show a notification in the app."""
        if self._app:
            self._app.notify(message, title=title or self.metadata.name, severity=severity)
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a plugin configuration value."""
        if self._config:
            plugin_config = self._config.get(f"plugin.{self.metadata.name}", {})
            return plugin_config.get(key, default)
        return default
    
    def set_config(self, key: str, value: Any):
        """Set a plugin configuration value."""
        if self._config:
            plugin_key = f"plugin.{self.metadata.name}"
            plugin_config = self._config.get(plugin_key, {})
            plugin_config[key] = value
            self._config.set(plugin_key, plugin_config)


class PluginError(Exception):
    """Base exception for plugin errors."""
    pass


class PluginLoadError(PluginError):
    """Error loading a plugin."""
    pass


class PluginDependencyError(PluginError):
    """Error resolving plugin dependencies."""
    pass


class PluginCompatibilityError(PluginError):
    """Plugin is not compatible with current SparkPlug version."""
    pass


