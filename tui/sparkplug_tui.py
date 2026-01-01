"""
SparkPlug TUI - Advanced Agent Platform Interface.

Enhanced with keyboard navigation, tab switching, and integrated services.
"""

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Static, Input, TabbedContent
from textual.binding import Binding
from textual import events
import asyncio

from components.header import HeaderWidget
from components.sidebar import Sidebar
from components.systems import SystemsContent
from components.footer import FooterWidget
from components.command_palette import CommandPalette

# Import services with fallback
try:
    from services.config_store import get_config_store
    from services.event_bus import get_event_bus, EventType
    from services.workspace_manager import get_workspace_manager
    from plugins.plugin_loader import get_plugin_loader
    HAS_SERVICES = True
except ImportError:
    HAS_SERVICES = False


class SparkPlugTUI(App):
    """SparkPlug Advanced Agent Platform TUI."""
    
    CSS_PATH = "styles.tcss"
    
    BINDINGS = [
        # Tab switching - Direct number keys
        Binding("ctrl+1", "switch_tab_num(1)", "Tab 1", show=False),
        Binding("ctrl+2", "switch_tab_num(2)", "Tab 2", show=False),
        Binding("ctrl+3", "switch_tab_num(3)", "Tab 3", show=False),
        Binding("ctrl+4", "switch_tab_num(4)", "Tab 4", show=False),
        Binding("ctrl+5", "switch_tab_num(5)", "Tab 5", show=False),
        
        # Function key tab switching
        Binding("f1", "switch_tab('infra-tab')", "Infrastructure", show=True),
        Binding("f2", "switch_tab('memory-tab')", "Memory", show=True),
        Binding("f3", "switch_tab('orchestration-tab')", "Orchestration", show=True),
        Binding("f4", "switch_tab('communication-tab')", "Communication", show=True),
        Binding("f5", "switch_tab('code-tab')", "Code", show=True),

        # Workspace shortcuts
        Binding("ctrl+w", "show_workspace_search", "Switch Workspace", show=True),
        Binding("ctrl+t", "new_workspace_from_template", "New from Template", show=False),
        Binding("ctrl+shift+p", "show_plugin_manager", "Plugin Manager", show=False),

        # Advanced navigation
        Binding("ctrl+p", "show_command_palette", "Command Palette", show=True),
        Binding("ctrl+b", "toggle_sidebar", "Toggle Sidebar", show=True),
        Binding("ctrl+l", "clear_command_input", "Clear Command", show=False),
        Binding("escape", "clear_focus", "Clear Focus", show=False),

        # Metrics and monitoring
        Binding("ctrl+m", "toggle_metrics", "Toggle Metrics", show=False),
        Binding("ctrl+r", "refresh_metrics", "Refresh Metrics", show=True),
        
        # Configuration
        Binding("ctrl+s", "save_configuration", "Save Config", show=True),
        Binding("ctrl+h", "show_help", "Help", show=True),

        # Code assistant shortcuts
        Binding("ctrl+o", "open_file", "Open File", show=False),
        Binding("ctrl+shift+s", "save_file", "Save File", show=False),
        Binding("ctrl+g", "git_status", "Git Status", show=False),
        
        # Navigation
        Binding("ctrl+tab", "next_tab", "Next Tab", show=False),
        Binding("ctrl+shift+tab", "prev_tab", "Previous Tab", show=False),

        # Quit
        Binding("q", "quit", "Quit", show=True),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]
    
    def __init__(self):
        super().__init__()
        self._config_store = None
        self._event_bus = None
        self._workspace_manager = None
        self._plugin_loader = None
        self._metrics_visible = True
        self._current_tab_index = 0
        self._tab_ids = ['infra-tab', 'memory-tab', 'orchestration-tab', 'communication-tab', 'code-tab']
        
        # Initialize services
        if HAS_SERVICES:
            self._config_store = get_config_store()
            self._event_bus = get_event_bus()
            self._workspace_manager = get_workspace_manager()
            self._plugin_loader = get_plugin_loader()
    
    def compose(self) -> ComposeResult:
        yield Container(
            HeaderWidget(),
            Sidebar(),
            SystemsContent(),
            CommandPalette(id="command-palette"),
            FooterWidget(),
            id="app-grid"
        )

    async def on_mount(self):
        """Initialize services and plugins when app is mounted."""
        if HAS_SERVICES:
            # Set up plugin loader
            if self._plugin_loader:
                self._plugin_loader.set_app(self)
                self._plugin_loader.set_config(self._config_store)
                self._plugin_loader.set_event_bus(self._event_bus)
                
                # Discover and load plugins
                await self._plugin_loader.discover_plugins()
                await self._plugin_loader.load_all()
                await self._plugin_loader.activate_all()
            
            # Load saved configuration
            if self._config_store:
                self._metrics_visible = self._config_store.get("show_metrics", True)
            
            # Emit app started event
            if self._event_bus:
                await self._event_bus.emit_async(
                    EventType.CONFIG_LOADED,
                    data={"app": "sparkplug"},
                    source="app"
                )

    # ==================== Tab Navigation ====================
    
    def action_switch_tab(self, tab_id: str) -> None:
        """Switch to specified tab by ID."""
        try:
            tabbed_content = self.query_one(TabbedContent)
            tabbed_content.active = tab_id
            
            # Update current tab index
            if tab_id in self._tab_ids:
                self._current_tab_index = self._tab_ids.index(tab_id)
            
            # Get tab name for notification
            tab_names = {
                'infra-tab': 'Infrastructure',
                'memory-tab': 'Agent Memory',
                'orchestration-tab': 'Orchestration',
                'communication-tab': 'Communication',
                'code-tab': 'Code Assistant',
            }
            tab_name = tab_names.get(tab_id, tab_id)
            self.notify(f"Switched to {tab_name}", title="Navigation")
            
            # Emit event
            if self._event_bus:
                self._event_bus.emit(
                    EventType.TAB_SWITCHED,
                    data={"tab_id": tab_id, "tab_name": tab_name},
                    source="app"
                )
        except Exception as e:
            self.notify(f"Tab switch error: {e}", severity="error")
    
    def action_switch_tab_num(self, num: int) -> None:
        """Switch to tab by number (1-5)."""
        if 1 <= num <= len(self._tab_ids):
            self.action_switch_tab(self._tab_ids[num - 1])
    
    def action_next_tab(self) -> None:
        """Switch to next tab."""
        self._current_tab_index = (self._current_tab_index + 1) % len(self._tab_ids)
        self.action_switch_tab(self._tab_ids[self._current_tab_index])
    
    def action_prev_tab(self) -> None:
        """Switch to previous tab."""
        self._current_tab_index = (self._current_tab_index - 1) % len(self._tab_ids)
        self.action_switch_tab(self._tab_ids[self._current_tab_index])

    # ==================== Workspace Actions ====================
    
    def action_show_workspace_search(self) -> None:
        """Show workspace search/switch interface."""
        try:
            header = self.query_one(HeaderWidget)
            header.show_search()
            self.notify("Type to search workspaces...", title="Workspace")
        except Exception:
            self.notify("Workspace search - Ctrl+W", title="Workspace")
    
    async def action_new_workspace_from_template(self) -> None:
        """Create new workspace from template."""
        if self._workspace_manager:
            # For now, just create a default workspace
            # In full implementation, this would show a template picker
            import os
            default_path = os.path.expanduser("~/sparkplug-workspace")
            ws = await self._workspace_manager.create_workspace(
                f"New Workspace",
                default_path
            )
            self.notify(f"Created workspace: {ws.name}", title="Workspace", severity="success")
        else:
            self.notify("New workspace from template - Ctrl+T", title="Workspace")
    
    def action_show_plugin_manager(self) -> None:
        """Show plugin manager."""
        if self._plugin_loader:
            plugins = self._plugin_loader.get_plugins()
            plugin_list = "\n".join([f"• {p.metadata.name} v{p.metadata.version}" for p in plugins])
            self.notify(f"Loaded Plugins:\n{plugin_list or 'No plugins loaded'}", title="Plugin Manager")
        else:
            self.notify("Plugin manager - Ctrl+Shift+P", title="Plugins")

    # ==================== Command Palette ====================

    def action_show_command_palette(self) -> None:
        """Focus on the command palette input."""
        try:
            command_input = self.query_one("#command-input", Input)
            command_input.focus()
            self.notify("Type to search commands...", title="Command Palette")
        except Exception:
            pass

    # ==================== Sidebar ====================

    def action_toggle_sidebar(self) -> None:
        """Toggle sidebar visibility."""
        try:
            sidebar = self.query_one("Sidebar")
            if sidebar.styles.display == "none":
                sidebar.styles.display = "block"
                self.notify("Sidebar shown", title="Navigation")
            else:
                sidebar.styles.display = "none"
                self.notify("Sidebar hidden", title="Navigation")
            
            # Emit event
            if self._event_bus:
                self._event_bus.emit(
                    EventType.SIDEBAR_TOGGLED,
                    data={"visible": sidebar.styles.display != "none"},
                    source="app"
                )
        except Exception:
            pass

    # ==================== Metrics ====================

    def action_toggle_metrics(self) -> None:
        """Toggle metrics panel visibility."""
        self._metrics_visible = not self._metrics_visible
        
        # Save preference
        if self._config_store:
            self._config_store.set("show_metrics", self._metrics_visible)
        
        status = "shown" if self._metrics_visible else "hidden"
        self.notify(f"Metrics panel {status}", title="Metrics")

    def action_refresh_metrics(self) -> None:
        """Refresh all system metrics."""
        try:
            # Trigger DGX panel refresh
            from components.systems import DGXConfigPanel
            dgx_panel = self.query_one(DGXConfigPanel)
            if dgx_panel:
                dgx_panel.update_metrics()
            
            self.notify("Metrics refreshed", title="System", severity="success")
            
            # Emit event
            if self._event_bus:
                self._event_bus.emit(
                    EventType.METRICS_UPDATED,
                    data={},
                    source="app"
                )
        except Exception as e:
            self.notify(f"Refresh error: {e}", severity="error")

    # ==================== Configuration ====================

    def action_save_configuration(self) -> None:
        """Save current configuration."""
        if self._config_store:
            self._config_store.save()
            self.notify("Configuration saved", title="Configuration", severity="success")
            
            # Emit event
            if self._event_bus:
                self._event_bus.emit(
                    EventType.CONFIG_SAVED,
                    data={},
                    source="app"
                )
        else:
            self.notify("Configuration saved", title="Configuration", severity="success")

    # ==================== Help ====================

    def action_show_help(self) -> None:
        """Show help information."""
        help_text = """
SPARKPLUG ADVANCED AGENT PLATFORM

Tab Navigation:
• Ctrl+1-5: Switch to tab 1-5
• F1-F5: Switch tabs
• Ctrl+Tab/Shift+Tab: Next/Previous tab

Workspace:
• Ctrl+W: Search/switch workspace
• Ctrl+T: New from template
• Ctrl+Shift+P: Plugin manager

Navigation:
• Ctrl+P: Command palette
• Ctrl+B: Toggle sidebar
• Ctrl+M: Toggle metrics
• Escape: Clear focus

Actions:
• Ctrl+R: Refresh metrics
• Ctrl+S: Save configuration
• Ctrl+H: Show help
• Q/Ctrl+C: Quit

Code Assistant:
• Ctrl+O: Open file
• Ctrl+Shift+S: Save file
• Ctrl+G: Git status
        """
        self.notify(help_text.strip(), title="Help", severity="information")

    # ==================== Utility Actions ====================

    def action_clear_command_input(self) -> None:
        """Clear the command input."""
        try:
            command_input = self.query_one("#command-input", Input)
            command_input.value = ""
            command_input.focus()
        except Exception:
            pass

    def action_clear_focus(self) -> None:
        """Clear focus from current element."""
        self.screen.focused = None

    def action_open_file(self) -> None:
        """Open file in code assistant."""
        self.action_switch_tab('code-tab')
        self.notify("Use file browser to open files", title="File")

    def action_save_file(self) -> None:
        """Save current file in code assistant."""
        self.action_switch_tab('code-tab')
        self.notify("Use Save button to save files", title="File")

    def action_git_status(self) -> None:
        """Show git status."""
        self.action_switch_tab('code-tab')
        self.notify("Use Git Status button in Code tab", title="Git")


if __name__ == "__main__":
    app = SparkPlugTUI()
    app.run()
