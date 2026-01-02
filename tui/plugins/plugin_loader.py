"""
Plugin Loader Service.

Handles plugin discovery, loading, and lifecycle management.
"""

import asyncio
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
import logging

from .base import (
    SparkPlugPlugin,
    PluginMetadata,
    PluginState,
    PluginError,
    PluginLoadError,
    PluginDependencyError,
    Command,
    Tool,
    Widget,
)

logger = logging.getLogger(__name__)


class PluginLoader:
    """
    Manages plugin discovery, loading, and lifecycle.
    
    Features:
    - Scan plugin directories for plugins
    - Load/unload plugins dynamically
    - Hot-reload support for development
    - Dependency resolution between plugins
    
    Usage:
        loader = PluginLoader()
        
        # Discover and load all plugins
        await loader.discover_plugins()
        await loader.load_all()
        
        # Get loaded plugins
        plugins = loader.get_plugins()
        
        # Get all registered commands
        commands = loader.get_all_commands()
    """
    
    _instance: Optional['PluginLoader'] = None

    # Default plugin directories
    PLUGIN_DIRS = [
        Path.home() / ".sparkplug" / "plugins",
        Path(__file__).parent / "builtin",
    ]

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self._initialized = True
        self._plugins: Dict[str, SparkPlugPlugin] = {}
        self._plugin_classes: Dict[str, Type[SparkPlugPlugin]] = {}
        self._load_order: List[str] = []
        self._app = None
        self._config = None
        self._event_bus = None

        # Ensure plugin directories exist
        for plugin_dir in self.PLUGIN_DIRS:
            try:
                plugin_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not create plugin directory {plugin_dir}: {e}")
    
    
    def set_app(self, app):
        """Set the application instance for plugins."""
        self._app = app
    
    def set_config(self, config):
        """Set the config store for plugins."""
        self._config = config
    
    def set_event_bus(self, event_bus):
        """Set the event bus for plugins."""
        self._event_bus = event_bus
    
    # ==================== Plugin Discovery ====================
    
    async def discover_plugins(self) -> List[str]:
        """
        Discover all available plugins.
        
        Returns:
            List of discovered plugin names
        """
        discovered = []
        
        for plugin_dir in self.PLUGIN_DIRS:
            if not plugin_dir.exists():
                continue
            
            # Look for Python files and packages
            for item in plugin_dir.iterdir():
                try:
                    if item.is_file() and item.suffix == ".py" and not item.name.startswith("_"):
                        # Single file plugin
                        plugin_name = item.stem
                        plugin_class = self._load_plugin_from_file(item)
                        if plugin_class:
                            self._plugin_classes[plugin_name] = plugin_class
                            discovered.append(plugin_name)
                            logger.info(f"Discovered plugin: {plugin_name}")
                    
                    elif item.is_dir() and (item / "__init__.py").exists():
                        # Package plugin
                        plugin_name = item.name
                        plugin_class = self._load_plugin_from_package(item)
                        if plugin_class:
                            self._plugin_classes[plugin_name] = plugin_class
                            discovered.append(plugin_name)
                            logger.info(f"Discovered plugin package: {plugin_name}")
                
                except Exception as e:
                    logger.error(f"Error discovering plugin {item}: {e}")
        
        return discovered
    
    def _load_plugin_from_file(self, file_path: Path) -> Optional[Type[SparkPlugPlugin]]:
        """Load a plugin class from a Python file."""
        try:
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            if not spec or not spec.loader:
                return None
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[file_path.stem] = module
            spec.loader.exec_module(module)
            
            # Find the plugin class
            for name in dir(module):
                obj = getattr(module, name)
                if (isinstance(obj, type) and 
                    issubclass(obj, SparkPlugPlugin) and 
                    obj is not SparkPlugPlugin):
                    return obj
            
            return None
        
        except Exception as e:
            logger.error(f"Failed to load plugin from {file_path}: {e}")
            return None
    
    def _load_plugin_from_package(self, package_path: Path) -> Optional[Type[SparkPlugPlugin]]:
        """Load a plugin class from a package directory."""
        try:
            # Add package path to sys.path temporarily
            parent_path = str(package_path.parent)
            if parent_path not in sys.path:
                sys.path.insert(0, parent_path)
            
            module = importlib.import_module(package_path.name)
            
            # Find the plugin class
            for name in dir(module):
                obj = getattr(module, name)
                if (isinstance(obj, type) and 
                    issubclass(obj, SparkPlugPlugin) and 
                    obj is not SparkPlugPlugin):
                    return obj
            
            return None
        
        except Exception as e:
            logger.error(f"Failed to load plugin from {package_path}: {e}")
            return None
    
    # ==================== Plugin Loading ====================
    
    async def load_plugin(self, plugin_name: str) -> bool:
        """
        Load a specific plugin.
        
        Args:
            plugin_name: The plugin name to load
            
        Returns:
            True if loaded successfully
        """
        if plugin_name in self._plugins:
            logger.warning(f"Plugin already loaded: {plugin_name}")
            return True
        
        if plugin_name not in self._plugin_classes:
            logger.error(f"Plugin not found: {plugin_name}")
            return False
        
        try:
            plugin_class = self._plugin_classes[plugin_name]
            plugin = plugin_class()
            plugin._state = PluginState.LOADING
            
            # Check dependencies
            if not await self._resolve_dependencies(plugin):
                plugin._state = PluginState.ERROR
                return False
            
            # Set services
            plugin._config = self._config
            plugin._event_bus = self._event_bus
            
            # Call on_load
            await plugin.on_load()
            plugin._state = PluginState.LOADED
            
            # Store plugin
            self._plugins[plugin_name] = plugin
            self._load_order.append(plugin_name)
            
            # Subscribe to events
            if self._event_bus:
                subscriptions = plugin.get_event_subscriptions()
                for event_type, handler in subscriptions.items():
                    self._event_bus.subscribe(event_type, handler)
            
            logger.info(f"Loaded plugin: {plugin.metadata.name} v{plugin.metadata.version}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}")
            return False
    
    async def _resolve_dependencies(self, plugin: SparkPlugPlugin) -> bool:
        """Resolve plugin dependencies."""
        metadata = plugin.metadata
        
        for dep_name in metadata.dependencies:
            if dep_name not in self._plugins:
                # Try to load dependency
                if dep_name in self._plugin_classes:
                    if not await self.load_plugin(dep_name):
                        logger.error(f"Failed to load dependency {dep_name} for {metadata.name}")
                        return False
                else:
                    logger.error(f"Missing dependency {dep_name} for {metadata.name}")
                    return False
        
        return True
    
    async def load_all(self) -> int:
        """
        Load all discovered plugins.
        
        Returns:
            Number of plugins loaded successfully
        """
        loaded = 0
        
        for plugin_name in self._plugin_classes:
            if plugin_name not in self._plugins:
                if await self.load_plugin(plugin_name):
                    loaded += 1
        
        return loaded
    
    async def activate_plugin(self, plugin_name: str) -> bool:
        """
        Activate a loaded plugin.
        
        Args:
            plugin_name: The plugin to activate
            
        Returns:
            True if activated successfully
        """
        if plugin_name not in self._plugins:
            return False
        
        plugin = self._plugins[plugin_name]
        
        if plugin.state == PluginState.ACTIVE:
            return True
        
        try:
            await plugin.on_activate(self._app)
            plugin._state = PluginState.ACTIVE
            logger.info(f"Activated plugin: {plugin.metadata.name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to activate plugin {plugin_name}: {e}")
            plugin._state = PluginState.ERROR
            return False
    
    async def activate_all(self) -> int:
        """
        Activate all loaded plugins.
        
        Returns:
            Number of plugins activated
        """
        activated = 0
        
        for plugin_name in self._load_order:
            if await self.activate_plugin(plugin_name):
                activated += 1
        
        return activated
    
    # ==================== Plugin Unloading ====================
    
    async def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin.
        
        Args:
            plugin_name: The plugin to unload
            
        Returns:
            True if unloaded successfully
        """
        if plugin_name not in self._plugins:
            return False
        
        plugin = self._plugins[plugin_name]
        
        try:
            # Deactivate first if active
            if plugin.state == PluginState.ACTIVE:
                await plugin.on_deactivate()
            
            # Unload
            await plugin.on_unload()
            plugin._state = PluginState.UNLOADED
            
            # Remove from tracking
            del self._plugins[plugin_name]
            if plugin_name in self._load_order:
                self._load_order.remove(plugin_name)
            
            logger.info(f"Unloaded plugin: {plugin.metadata.name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to unload plugin {plugin_name}: {e}")
            return False
    
    async def unload_all(self):
        """Unload all plugins in reverse order."""
        for plugin_name in reversed(self._load_order.copy()):
            await self.unload_plugin(plugin_name)
    
    # ==================== Hot Reload ====================
    
    async def reload_plugin(self, plugin_name: str) -> bool:
        """
        Hot-reload a plugin.
        
        Args:
            plugin_name: The plugin to reload
            
        Returns:
            True if reloaded successfully
        """
        # Unload existing
        if plugin_name in self._plugins:
            await self.unload_plugin(plugin_name)
        
        # Re-discover (in case file changed)
        if plugin_name in self._plugin_classes:
            del self._plugin_classes[plugin_name]
        
        # Find and reload the module
        for plugin_dir in self.PLUGIN_DIRS:
            file_path = plugin_dir / f"{plugin_name}.py"
            if file_path.exists():
                # Remove from sys.modules to force reload
                if plugin_name in sys.modules:
                    del sys.modules[plugin_name]
                
                plugin_class = self._load_plugin_from_file(file_path)
                if plugin_class:
                    self._plugin_classes[plugin_name] = plugin_class
                    break
            
            package_path = plugin_dir / plugin_name
            if package_path.is_dir():
                if plugin_name in sys.modules:
                    del sys.modules[plugin_name]
                
                plugin_class = self._load_plugin_from_package(package_path)
                if plugin_class:
                    self._plugin_classes[plugin_name] = plugin_class
                    break
        
        # Load and activate
        if await self.load_plugin(plugin_name):
            return await self.activate_plugin(plugin_name)
        
        return False
    
    # ==================== Plugin Queries ====================
    
    def get_plugin(self, plugin_name: str) -> Optional[SparkPlugPlugin]:
        """Get a loaded plugin by name."""
        return self._plugins.get(plugin_name)
    
    def get_plugins(self) -> List[SparkPlugPlugin]:
        """Get all loaded plugins."""
        return list(self._plugins.values())
    
    def get_active_plugins(self) -> List[SparkPlugPlugin]:
        """Get all active plugins."""
        return [p for p in self._plugins.values() if p.state == PluginState.ACTIVE]
    
    def is_loaded(self, plugin_name: str) -> bool:
        """Check if a plugin is loaded."""
        return plugin_name in self._plugins
    
    def is_active(self, plugin_name: str) -> bool:
        """Check if a plugin is active."""
        plugin = self._plugins.get(plugin_name)
        return plugin is not None and plugin.state == PluginState.ACTIVE
    
    # ==================== Aggregate Registrations ====================
    
    def get_all_commands(self) -> List[Command]:
        """Get all commands from all active plugins."""
        commands = []
        for plugin in self.get_active_plugins():
            commands.extend(plugin.register_commands())
        return commands
    
    def get_all_tools(self) -> List[Tool]:
        """Get all tools from all active plugins."""
        tools = []
        for plugin in self.get_active_plugins():
            tools.extend(plugin.register_tools())
        return tools
    
    def get_all_widgets(self) -> List[Widget]:
        """Get all widgets from all active plugins."""
        widgets = []
        for plugin in self.get_active_plugins():
            widgets.extend(plugin.register_widgets())
        return widgets
    
    def get_commands_by_category(self) -> Dict[str, List[Command]]:
        """Get commands grouped by category."""
        commands = self.get_all_commands()
        by_category: Dict[str, List[Command]] = {}
        
        for cmd in commands:
            if cmd.category not in by_category:
                by_category[cmd.category] = []
            by_category[cmd.category].append(cmd)
        
        return by_category


# Global singleton accessor
def get_plugin_loader() -> PluginLoader:
    """Get the global PluginLoader instance."""
    return PluginLoader()

