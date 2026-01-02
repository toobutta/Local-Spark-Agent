from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Input, ListView, ListItem, Label, Static
from textual.containers import Vertical, Horizontal
from textual import events
import asyncio

class CommandPalette(Widget):
    """Interactive command palette for agent operations"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.commands = [
            "deploy agent --name=<name> --gpu=<count>",
            "check status agent --name=<name>",
            "stop agent --name=<name>",
            "logs agent --name=<name>",
            "memory agent --name=<name>",
            "configure tools --add=<tool>",
            "backup configuration",
            "export metrics",
            "clear memory",
            "help"
        ]
        self.filtered_commands = self.commands.copy()
        self.selected_index = 0

    def compose(self) -> ComposeResult:
        with Vertical():
            # Command input
            yield Input(
                placeholder="Type a command...",
                id="command-input"
            )

            # Command suggestions
            with Vertical(id="suggestions-container"):
                yield ListView(id="command-suggestions")

    def on_mount(self):
        """Initialize the command palette"""
        self.update_suggestions("")

    def update_suggestions(self, query: str):
        """Update the command suggestions based on query"""
        if not query:
            self.filtered_commands = self.commands.copy()
        else:
            self.filtered_commands = [
                cmd for cmd in self.commands
                if query.lower() in cmd.lower()
            ]

        # Update the list view
        list_view = self.query_one("#command-suggestions", ListView)
        list_view.clear()

        for cmd in self.filtered_commands[:5]:  # Show top 5 matches
            list_view.append(ListItem(Label(cmd)))

    async def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes to filter commands"""
        if event.input.id == "command-input":
            self.update_suggestions(event.value)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle command execution"""
        if event.input.id == "command-input":
            command = event.value.strip()
            if command:
                await self.execute_command(command)
                # Clear the input
                event.input.value = ""

    async def execute_command(self, command: str):
        """Execute the given command"""
        # For now, just show a notification
        # In the future, this will integrate with agent frameworks
        self.app.notify(f"Executing: {command}", title="Command", severity="information")

        # Simulate command execution
        if command.startswith("deploy agent"):
            # Simulate agent deployment
            await asyncio.sleep(0.5)
            self.app.notify("Agent deployed successfully!", title="Success", severity="success")
        elif command.startswith("check status"):
            # Simulate status check
            await asyncio.sleep(0.2)
            self.app.notify("Agent status: RUNNING", title="Status", severity="info")
        elif command == "help":
            self.show_help()
        else:
            self.app.notify(f"Unknown command: {command}", title="Error", severity="error")

    def show_help(self):
        """Show available commands"""
        help_text = """
Available Commands:
• deploy agent --name=<name> --gpu=<count>  - Deploy a new agent
• check status agent --name=<name>          - Check agent status
• stop agent --name=<name>                  - Stop an agent
• logs agent --name=<name>                  - View agent logs
• memory agent --name=<name>                - Check agent memory usage
• configure tools --add=<tool>              - Add a tool to agent
• backup configuration                       - Backup current config
• export metrics                             - Export system metrics
• clear memory                               - Clear agent memory
• help                                       - Show this help
        """
        self.app.notify(help_text.strip(), title="Command Help", severity="info")
