"""
Header Widget with workspace selector and quick actions.
"""

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static, Select, Label, Button, Input
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual import events
from typing import Optional, List, Tuple
import asyncio

# Import services (with fallback for standalone testing)
try:
    from ..services.workspace_manager import WorkspaceManager, get_workspace_manager
    from ..services.config_store import WorkspaceConfig
    from ..services.event_bus import EventBus, EventType, get_event_bus
    HAS_SERVICES = True
except ImportError:
    HAS_SERVICES = False
    WorkspaceManager = None
    get_workspace_manager = None
    WorkspaceConfig = None
    EventBus = None
    EventType = None
    get_event_bus = None


class WorkspaceChanged(Message):
    """Message emitted when workspace changes."""
    def __init__(self, workspace_id: str, workspace_name: str):
        self.workspace_id = workspace_id
        self.workspace_name = workspace_name
        super().__init__()


class HeaderWidget(Widget):
    """
    Enhanced header widget with workspace selector and quick actions.
    
    Features:
    - Workspace dropdown with recent workspaces
    - New workspace button
    - Import workspace button
    - Active agent count display
    - Quick workspace search (Ctrl+W)
    """
    
    DEFAULT_CSS = """
    HeaderWidget {
        height: 3;
        layout: horizontal;
        align-vertical: middle;
        padding: 0 1;
    }
    
    HeaderWidget #workspace-section {
        width: auto;
        layout: horizontal;
        align-vertical: middle;
    }
    
    HeaderWidget #workspace-select {
        width: 30;
        margin-right: 1;
    }
    
    HeaderWidget .header-btn {
        min-width: 3;
        margin: 0 1;
    }
    
    HeaderWidget #agent-count {
        color: #00f5d4;
        margin-left: 2;
    }
    
    HeaderWidget #workspace-search {
        display: none;
        width: 40;
    }
    
    HeaderWidget #workspace-search.visible {
        display: block;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._workspace_manager = None
        self._event_bus = None
        self._search_visible = False
        
        if HAS_SERVICES and get_workspace_manager is not None:
            self._workspace_manager = get_workspace_manager()
        if HAS_SERVICES and get_event_bus is not None:
            self._event_bus = get_event_bus()
    
    def compose(self) -> ComposeResult:
        yield Label("SPARKPLUG ADMIN", id="header-title")
        
        with Horizontal(id="workspace-section"):
            # Workspace selector
            yield Select(
                options=self._get_workspace_options(),
                value=self._get_active_workspace_id(),
                allow_blank=False,
                id="workspace-select"
            )
            
            # Quick action buttons
            yield Button("+", variant="success", id="new-workspace-btn", classes="header-btn")
            yield Button("⬇", variant="primary", id="import-workspace-btn", classes="header-btn")
            
            # Agent count indicator
            yield Label(self._get_agent_count_label(), id="agent-count")
        
        # Hidden search input (shown with Ctrl+W)
        yield Input(
            placeholder="Search workspaces... (Esc to close)",
            id="workspace-search"
        )
        
        yield Label("ADMIN MODE", id="admin-badge")
    
    def _get_workspace_options(self) -> List[Tuple[str, str]]:
        """Get workspace options for the select widget."""
        if not self._workspace_manager:
            # Fallback options for testing
            return [
                ("main", "Main Cluster (Default)"),
                ("research", "Research Node Alpha"),
                ("web", "Web Services Delta"),
            ]
        
        workspaces = self._workspace_manager.list_workspaces()
        if not workspaces:
            return [("default", "Default Workspace")]
        
        return [(ws.id, f"{ws.name}") for ws in workspaces]
    
    def _get_active_workspace_id(self) -> str:
        """Get the active workspace ID."""
        if not self._workspace_manager:
            return "main"
        
        active = self._workspace_manager.get_active_workspace()
        if active:
            return active.id
        
        # Return first workspace if no active
        workspaces = self._workspace_manager.list_workspaces()
        if workspaces:
            return workspaces[0].id
        
        return "default"
    
    def _get_agent_count_label(self) -> str:
        """Get the agent count label."""
        if not self._workspace_manager:
            return "⚡ 0 agents"
        
        agents = self._workspace_manager.get_active_agents()
        count = len(agents)
        return f"⚡ {count} agent{'s' if count != 1 else ''}"
    
    def on_mount(self):
        """Set up event subscriptions when mounted."""
        if self._event_bus and HAS_SERVICES and EventType is not None:
            # Subscribe to workspace events
            self._event_bus.subscribe(EventType.WORKSPACE_CREATED, self._on_workspace_event)
            self._event_bus.subscribe(EventType.WORKSPACE_DELETED, self._on_workspace_event)
            self._event_bus.subscribe(EventType.WORKSPACE_SWITCHED, self._on_workspace_event)
            
            # Subscribe to agent events
            self._event_bus.subscribe(EventType.AGENT_DEPLOYED, self._on_agent_event)
            self._event_bus.subscribe(EventType.AGENT_STOPPED, self._on_agent_event)
        
        # Set up periodic refresh for agent count
        self.set_interval(5.0, self._refresh_agent_count)
    
    def _on_workspace_event(self, event):
        """Handle workspace events."""
        self._refresh_workspace_selector()
    
    def _on_agent_event(self, event):
        """Handle agent events."""
        self._refresh_agent_count()
    
    def _refresh_workspace_selector(self):
        """Refresh the workspace selector options."""
        try:
            select = self.query_one("#workspace-select", Select)
            select.set_options(self._get_workspace_options())
            
            # Update selected value
            active_id = self._get_active_workspace_id()
            if active_id:
                select.value = active_id
        except Exception:
            pass
    
    def _refresh_agent_count(self):
        """Refresh the agent count display."""
        try:
            label = self.query_one("#agent-count", Label)
            label.update(self._get_agent_count_label())
        except Exception:
            pass
    
    async def on_select_changed(self, event: Select.Changed) -> None:
        """Handle workspace selection change."""
        if event.select.id == "workspace-select" and event.value:
            workspace_id = str(event.value)
            
            if self._workspace_manager:
                await self._workspace_manager.switch_workspace(workspace_id)
                
                # Get workspace name for message
                ws = self._workspace_manager.get_workspace(workspace_id)
                if ws:
                    self.post_message(WorkspaceChanged(workspace_id, ws.name))
                    self.app.notify(f"Switched to workspace: {ws.name}", title="Workspace")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "new-workspace-btn":
            await self._show_new_workspace_dialog()
        elif button_id == "import-workspace-btn":
            await self._show_import_workspace_dialog()
    
    async def _show_new_workspace_dialog(self):
        """Show dialog to create new workspace."""
        # For now, create a workspace with default name
        # In a full implementation, this would open a modal dialog
        if self._workspace_manager:
            import os
            default_path = os.path.expanduser("~/sparkplug-workspace")
            ws = await self._workspace_manager.create_workspace(
                f"Workspace {len(self._workspace_manager.list_workspaces()) + 1}",
                default_path
            )
            self.app.notify(f"Created workspace: {ws.name}", title="Workspace", severity="information")
            self._refresh_workspace_selector()
        else:
            self.app.notify("Create new workspace - feature coming soon!", title="Workspace")
    
    async def _show_import_workspace_dialog(self):
        """Show dialog to import existing workspace."""
        # Placeholder for import functionality
        self.app.notify("Import workspace - feature coming soon!", title="Workspace")
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle search input changes."""
        if event.input.id == "workspace-search":
            self._filter_workspaces(event.value)
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle search input submission."""
        if event.input.id == "workspace-search":
            # Select first matching workspace
            if self._workspace_manager:
                results = self._workspace_manager.search_workspaces(event.value)
                if results:
                    asyncio.create_task(self._workspace_manager.switch_workspace(results[0].id))
                    self._hide_search()
    
    def _filter_workspaces(self, query: str):
        """Filter workspace options based on search query."""
        if not self._workspace_manager or not query:
            self._refresh_workspace_selector()
            return
        
        results = self._workspace_manager.search_workspaces(query)
        try:
            select = self.query_one("#workspace-select", Select)
            if results:
                select.set_options([(ws.id, ws.name) for ws in results])
            else:
                select.set_options([("none", "No matching workspaces")])
        except Exception:
            pass
    
    def show_search(self):
        """Show the workspace search input."""
        try:
            search_input = self.query_one("#workspace-search", Input)
            search_input.add_class("visible")
            search_input.focus()
            self._search_visible = True
        except Exception:
            pass
    
    def _hide_search(self):
        """Hide the workspace search input."""
        try:
            search_input = self.query_one("#workspace-search", Input)
            search_input.remove_class("visible")
            search_input.value = ""
            self._search_visible = False
            self._refresh_workspace_selector()
        except Exception:
            pass
    
    def on_key(self, event: events.Key) -> None:
        """Handle key events."""
        if event.key == "escape" and self._search_visible:
            self._hide_search()
            event.prevent_default()
