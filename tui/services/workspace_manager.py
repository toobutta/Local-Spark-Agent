"""
Workspace Manager Service.

Provides multi-workspace support with create/switch/delete operations.
"""

import asyncio
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import uuid
import logging

from .config_store import ConfigStore, WorkspaceConfig, get_config_store
from .event_bus import EventBus, EventType, Event, get_event_bus

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceState:
    """Runtime state for a workspace."""
    workspace: WorkspaceConfig
    active_agents: List[str] = field(default_factory=list)
    mcp_connections: Dict[str, Any] = field(default_factory=dict)
    open_files: List[str] = field(default_factory=list)
    is_loaded: bool = False


class WorkspaceManager:
    """
    Manages multiple workspaces with their configurations and states.
    
    Features:
    - Create/delete/switch workspaces
    - Workspace-specific agent configurations
    - Workspace-specific MCP server connections
    - Recent workspaces tracking
    - Quick workspace search
    
    Usage:
        manager = WorkspaceManager()
        
        # Create workspace
        ws = await manager.create_workspace("My Project", "/path/to/project")
        
        # Switch workspace
        await manager.switch_workspace(ws.id)
        
        # Get active workspace
        active = manager.get_active_workspace()
    """
    
    _instance: Optional['WorkspaceManager'] = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._config_store = get_config_store()
        self._event_bus = get_event_bus()
        self._workspace_states: Dict[str, WorkspaceState] = {}
        self._active_workspace_id: Optional[str] = None
        self._switch_callbacks: List[Callable[[WorkspaceConfig], None]] = []
        
        # Load workspaces from config
        self._load_workspaces()
    
    def _load_workspaces(self):
        """Load workspaces from config store."""
        workspaces = self._config_store.list_workspaces()
        
        for ws in workspaces:
            self._workspace_states[ws.id] = WorkspaceState(workspace=ws)
        
        # Set active workspace from config
        active = self._config_store.get_active_workspace()
        if active:
            self._active_workspace_id = active.id
    
    # ==================== Workspace CRUD ====================
    
    async def create_workspace(
        self,
        name: str,
        path: str,
        *,
        template_id: Optional[str] = None,
        set_active: bool = True
    ) -> WorkspaceConfig:
        """
        Create a new workspace.
        
        Args:
            name: Workspace display name
            path: Path to the workspace directory
            template_id: Optional template to apply
            set_active: Whether to switch to the new workspace
            
        Returns:
            The created WorkspaceConfig
        """
        # Validate path
        workspace_path = Path(path)
        if not workspace_path.exists():
            workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Create workspace in config store
        workspace = self._config_store.create_workspace(name, str(workspace_path))
        
        # Create runtime state
        self._workspace_states[workspace.id] = WorkspaceState(workspace=workspace)
        
        # Emit event
        await self._event_bus.emit_async(
            EventType.WORKSPACE_CREATED,
            data={"workspace_id": workspace.id, "name": name, "path": path},
            source="workspace_manager"
        )
        
        # Set as active if requested
        if set_active:
            await self.switch_workspace(workspace.id)
        
        logger.info(f"Created workspace: {name} at {path}")
        return workspace
    
    async def delete_workspace(self, workspace_id: str) -> bool:
        """
        Delete a workspace.
        
        Args:
            workspace_id: The workspace ID to delete
            
        Returns:
            True if deleted successfully
        """
        if workspace_id not in self._workspace_states:
            return False
        
        # Get workspace info before deletion
        ws = self._workspace_states[workspace_id].workspace
        
        # If this is the active workspace, switch to another
        if self._active_workspace_id == workspace_id:
            other_workspaces = [w for w in self._workspace_states.keys() if w != workspace_id]
            if other_workspaces:
                await self.switch_workspace(other_workspaces[0])
            else:
                self._active_workspace_id = None
        
        # Remove from states
        del self._workspace_states[workspace_id]
        
        # Remove from config store
        self._config_store.delete_workspace(workspace_id)
        
        # Emit event
        await self._event_bus.emit_async(
            EventType.WORKSPACE_DELETED,
            data={"workspace_id": workspace_id, "name": ws.name},
            source="workspace_manager"
        )
        
        logger.info(f"Deleted workspace: {ws.name}")
        return True
    
    async def switch_workspace(self, workspace_id: str) -> bool:
        """
        Switch to a different workspace.
        
        Args:
            workspace_id: The workspace ID to switch to
            
        Returns:
            True if switched successfully
        """
        if workspace_id not in self._workspace_states:
            logger.warning(f"Workspace not found: {workspace_id}")
            return False
        
        old_workspace_id = self._active_workspace_id
        self._active_workspace_id = workspace_id
        
        # Update config store
        self._config_store.set_active_workspace(workspace_id)
        
        # Mark as loaded
        self._workspace_states[workspace_id].is_loaded = True
        
        # Get workspace config
        workspace = self._workspace_states[workspace_id].workspace
        
        # Call switch callbacks
        for callback in self._switch_callbacks:
            try:
                callback(workspace)
            except Exception as e:
                logger.error(f"Error in workspace switch callback: {e}")
        
        # Emit event
        await self._event_bus.emit_async(
            EventType.WORKSPACE_SWITCHED,
            data={
                "workspace_id": workspace_id,
                "name": workspace.name,
                "path": workspace.path,
                "previous_workspace_id": old_workspace_id
            },
            source="workspace_manager"
        )
        
        logger.info(f"Switched to workspace: {workspace.name}")
        return True
    
    async def update_workspace(
        self,
        workspace_id: str,
        *,
        name: Optional[str] = None,
        path: Optional[str] = None,
        active_agents: Optional[List[str]] = None,
        mcp_servers: Optional[List[str]] = None
    ) -> bool:
        """
        Update workspace properties.
        
        Args:
            workspace_id: The workspace ID to update
            name: New name (optional)
            path: New path (optional)
            active_agents: New active agents list (optional)
            mcp_servers: New MCP servers list (optional)
            
        Returns:
            True if updated successfully
        """
        if workspace_id not in self._workspace_states:
            return False
        
        updates = {}
        if name is not None:
            updates['name'] = name
        if path is not None:
            updates['path'] = path
        if active_agents is not None:
            updates['active_agents'] = active_agents
        if mcp_servers is not None:
            updates['mcp_servers'] = mcp_servers
        
        if updates:
            self._config_store.update_workspace(workspace_id, **updates)
            
            # Update local state
            ws = self._workspace_states[workspace_id].workspace
            for key, value in updates.items():
                setattr(ws, key, value)
            
            # Emit event
            await self._event_bus.emit_async(
                EventType.WORKSPACE_UPDATED,
                data={"workspace_id": workspace_id, "updates": updates},
                source="workspace_manager"
            )
        
        return True
    
    # ==================== Workspace Queries ====================
    
    def get_workspace(self, workspace_id: str) -> Optional[WorkspaceConfig]:
        """Get a workspace by ID."""
        if workspace_id in self._workspace_states:
            return self._workspace_states[workspace_id].workspace
        return None
    
    def get_active_workspace(self) -> Optional[WorkspaceConfig]:
        """Get the currently active workspace."""
        if self._active_workspace_id:
            return self.get_workspace(self._active_workspace_id)
        return None
    
    def get_active_workspace_id(self) -> Optional[str]:
        """Get the active workspace ID."""
        return self._active_workspace_id
    
    def get_workspace_state(self, workspace_id: str) -> Optional[WorkspaceState]:
        """Get the runtime state for a workspace."""
        return self._workspace_states.get(workspace_id)
    
    def list_workspaces(self) -> List[WorkspaceConfig]:
        """List all workspaces."""
        return [state.workspace for state in self._workspace_states.values()]
    
    def get_recent_workspaces(self, limit: int = 10) -> List[WorkspaceConfig]:
        """Get recently accessed workspaces."""
        recent_ids = self._config_store.config.recent_workspaces[:limit]
        return [
            self._workspace_states[ws_id].workspace
            for ws_id in recent_ids
            if ws_id in self._workspace_states
        ]
    
    def search_workspaces(self, query: str) -> List[WorkspaceConfig]:
        """
        Search workspaces by name or path.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching workspaces
        """
        query = query.lower()
        results = []
        
        for state in self._workspace_states.values():
            ws = state.workspace
            if query in ws.name.lower() or query in ws.path.lower():
                results.append(ws)
        
        return results
    
    # ==================== Workspace State Management ====================
    
    def add_active_agent(self, workspace_id: str, agent_id: str):
        """Add an active agent to a workspace."""
        if workspace_id in self._workspace_states:
            state = self._workspace_states[workspace_id]
            if agent_id not in state.active_agents:
                state.active_agents.append(agent_id)
    
    def remove_active_agent(self, workspace_id: str, agent_id: str):
        """Remove an active agent from a workspace."""
        if workspace_id in self._workspace_states:
            state = self._workspace_states[workspace_id]
            if agent_id in state.active_agents:
                state.active_agents.remove(agent_id)
    
    def get_active_agents(self, workspace_id: Optional[str] = None) -> List[str]:
        """Get active agents for a workspace (or active workspace if not specified)."""
        ws_id = workspace_id or self._active_workspace_id
        if ws_id and ws_id in self._workspace_states:
            return self._workspace_states[ws_id].active_agents.copy()
        return []
    
    # ==================== Callbacks ====================
    
    def on_workspace_switch(self, callback: Callable[[WorkspaceConfig], None]):
        """Register a callback for workspace switches."""
        self._switch_callbacks.append(callback)
    
    def remove_switch_callback(self, callback: Callable[[WorkspaceConfig], None]):
        """Remove a workspace switch callback."""
        if callback in self._switch_callbacks:
            self._switch_callbacks.remove(callback)


# Global singleton accessor
def get_workspace_manager() -> WorkspaceManager:
    """Get the global WorkspaceManager instance."""
    return WorkspaceManager()

