from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static, Button, Input, Label, ListView, ListItem
from textual.containers import Vertical, Horizontal
import asyncio
from typing import Dict, List, Optional

class MemoryManager(Widget):
    """Agent memory management interface using Mem0"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.memory_client = None
        self.current_agent = "default"
        self.memories: List[Dict] = []

    def on_mount(self):
        """Initialize memory management"""
        self.initialize_memory_client()

    def initialize_memory_client(self):
        """Initialize Mem0 client"""
        try:
            from mem0 import Memory
            self.memory_client = Memory()
            self.notify("Memory system initialized", title="Memory", severity="information")
        except ImportError:
            self.notify("Mem0 not installed. Run: pip install mem0ai", title="Memory", severity="warning")
        except Exception as e:
            self.notify(f"Memory initialization failed: {e}", title="Memory", severity="error")

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("AGENT MEMORY MANAGEMENT", classes="section-header")

            # Agent selection
            with Horizontal():
                yield Label("Agent:")
                yield Input(
                    value=self.current_agent,
                    placeholder="agent-name",
                    id="agent-selector"
                )
                yield Button("Switch", variant="primary", id="switch-agent-btn")

            # Memory operations
            with Horizontal():
                yield Button("View Memory", variant="default", id="view-memory-btn")
                yield Button("Add Memory", variant="default", id="add-memory-btn")
                yield Button("Search Memory", variant="default", id="search-memory-btn")
                yield Button("Clear Memory", variant="warning", id="clear-memory-btn")

            # Memory input (for adding)
            with Vertical(id="memory-input-section"):
                yield Input(placeholder="Enter memory content...", id="memory-input")
                yield Button("Save Memory", variant="success", id="save-memory-btn")

            # Memory display
            with Vertical(id="memory-display"):
                yield Label("MEMORY ENTRIES", classes="section-header")
                yield ListView(id="memory-list")

            # Memory stats
            with Horizontal(id="memory-stats"):
                yield Static("Total Memories: 0", id="memory-count")
                yield Static("Last Updated: Never", id="memory-last-update")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        button_id = event.button.id

        if button_id == "switch-agent-btn":
            await self.switch_agent()
        elif button_id == "view-memory-btn":
            await self.view_memory()
        elif button_id == "add-memory-btn":
            self.toggle_memory_input(True)
        elif button_id == "save-memory-btn":
            await self.save_memory()
        elif button_id == "search-memory-btn":
            await self.search_memory()
        elif button_id == "clear-memory-btn":
            await self.clear_memory()

    async def switch_agent(self):
        """Switch to a different agent"""
        agent_input = self.query_one("#agent-selector", Input)
        new_agent = agent_input.value.strip()

        if new_agent:
            self.current_agent = new_agent
            self.notify(f"Switched to agent: {new_agent}", title="Memory", severity="information")
            await self.view_memory()
        else:
            self.notify("Please enter an agent name", title="Memory", severity="warning")

    async def view_memory(self):
        """View memories for current agent"""
        if not self.memory_client:
            self.notify("Memory client not initialized", title="Memory", severity="error")
            return

        try:
            # Get memories for current agent
            memories = self.memory_client.search(f"agent:{self.current_agent}", user_id=self.current_agent)
            self.memories = memories if isinstance(memories, list) else []

            # Update display
            await self.update_memory_display()

            self.notify(f"Loaded {len(self.memories)} memories", title="Memory", severity="information")

        except Exception as e:
            self.notify(f"Failed to load memories: {e}", title="Memory", severity="error")

    async def save_memory(self):
        """Save a new memory"""
        if not self.memory_client:
            self.notify("Memory client not initialized", title="Memory", severity="error")
            return

        memory_input = self.query_one("#memory-input", Input)
        content = memory_input.value.strip()

        if not content:
            self.notify("Please enter memory content", title="Memory", severity="warning")
            return

        try:
            # Add memory
            self.memory_client.add(content, user_id=self.current_agent)

            # Clear input and hide input section
            memory_input.value = ""
            self.toggle_memory_input(False)

            # Refresh memory display
            await self.view_memory()

            self.notify("Memory saved successfully", title="Memory", severity="information")

        except Exception as e:
            self.notify(f"Failed to save memory: {e}", title="Memory", severity="error")

    async def search_memory(self):
        """Search memories"""
        if not self.memory_client:
            self.notify("Memory client not initialized", title="Memory", severity="error")
            return

        # For now, just refresh the current view
        # In the future, add search input
        await self.view_memory()
        self.notify("Memory search completed", title="Memory", severity="information")

    async def clear_memory(self):
        """Clear all memories for current agent"""
        if not self.memory_client:
            self.notify("Memory client not initialized", title="Memory", severity="error")
            return

        # Note: Mem0 doesn't have a direct clear method
        # This would need to be implemented via their API
        self.notify("Memory clearing not yet implemented", title="Memory", severity="warning")

    def toggle_memory_input(self, show: bool):
        """Toggle memory input section visibility"""
        input_section = self.query_one("#memory-input-section")
        if show:
            input_section.styles.display = "block"
        else:
            input_section.styles.display = "none"

    async def update_memory_display(self):
        """Update the memory list display"""
        memory_list = self.query_one("#memory-list", ListView)
        memory_list.clear()

        for memory in self.memories[:10]:  # Show last 10 memories
            content = memory.get('content', str(memory))[:50] + "..."
            memory_list.append(ListItem(Label(content)))

        # Update stats
        count_label = self.query_one("#memory-count", Static)
        count_label.update(f"Total Memories: {len(self.memories)}")

        # Update last updated time
        import time
        update_label = self.query_one("#memory-last-update", Static)
        update_label.update(f"Last Updated: {time.strftime('%H:%M:%S')}")
