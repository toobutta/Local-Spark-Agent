from textual.app import ComposeResult
from textual.containers import Horizontal, Container
from textual.widgets import Input, Static, Button, Widget
from textual.message import Message
from textual import events
import asyncio


class CommandBar(Container):
    """Functional command bar that connects to SparkPlug backend"""

    class CommandSubmitted(Message):
        """Message sent when a command is submitted"""
        def __init__(self, command: str):
            self.command = command
            super().__init__()

    class CommandResult(Message):
        """Message sent when a command result is received"""
        def __init__(self, result: dict):
            self.result = result
            super().__init__()

    def __init__(self, api_client=None, **kwargs):
        super().__init__(**kwargs)
        self.api_client = api_client
        self.placeholder = "Enter command..."
        self._suggestions = [
            "help", "status", "build", "deploy agent", "connect", "research",
            "agents", "mcp", "marketplace", "plugins", "browser", "ollama",
            "settings", "claude", "clear"
        ]
        self.input_widget = None

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Static("❯", id="prompt")
            yield Input(
                placeholder=self.placeholder,
                id="command-input"
            )
            yield Button("Execute", id="execute-btn", variant="primary")

    async def on_mount(self) -> None:
        """Set up event handlers when component mounts"""
        self.input_widget = self.query_one("#command-input", Input)
        self.input_widget.focus()

        # Set up WebSocket message handlers if API client is available
        if self.api_client:
            self.api_client.on_message("command_executed", self._handle_command_result)
            self.api_client.on_message("command_error", self._handle_command_error)
            self.api_client.on_message("agent_created", self._handle_agent_created)
            self.api_client.on_message("agent_thought", self._handle_agent_thought)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle execute button press"""
        if event.button.id == "execute-btn":
            await self._execute_command()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle command input submission"""
        await self._execute_command()

    async def on_key(self, event: events.Key) -> None:
        """Handle keyboard shortcuts"""
        if event.key == "ctrl+c":
            # Clear command
            if not self.input_widget:
                self.input_widget = self.query_one("#command-input", Input)
            self.input_widget.value = ""
            if self.app:
                self.app.notify("Command cleared", severity="information")

    async def _execute_command(self) -> None:
        """Execute the current command"""
        if not self.input_widget:
            self.input_widget = self.query_one("#command-input", Input)
        
        command = self.input_widget.value.strip()

        if not command:
            return

        # Post the command submitted message
        self.post_message(self.CommandSubmitted(command))

        # Clear the input
        self.input_widget.value = ""

        # Execute via API if available
        if self.api_client:
            try:
                result = await self.api_client.execute_command(command)
                self.post_message(self.CommandResult(result))
            except Exception as e:
                error_result = {
                    "success": False,
                    "output": "",
                    "error": f"Connection error: {e}"
                }
                self.post_message(self.CommandResult(error_result))
        else:
            # Fallback for when no API client is available
            self.post_message(self.CommandResult({
                "success": False,
                "output": "",
                "error": "No API connection available"
            }))

    async def _handle_command_result(self, data: dict) -> None:
        """Handle command execution results from WebSocket"""
        self.post_message(self.CommandResult(data))

    async def _handle_command_error(self, data: dict) -> None:
        """Handle command errors from WebSocket"""
        self.post_message(self.CommandResult({
            "success": False,
            "output": "",
            "error": data.get("error", "Unknown command error")
        }))

    async def _handle_agent_created(self, data: dict) -> None:
        """Handle agent creation notifications"""
        if self.app:
            self.app.notify(f"Agent {data.get('name', 'Unknown')} deployed", severity="success")

    async def _handle_agent_thought(self, data: dict) -> None:
        """Handle agent thought notifications"""
        agent = data.get('agent', 'Unknown')
        thought = data.get('thought', '')
        if self.app:
            self.app.notify(f"[{agent}] {thought}", severity="information")

    def get_suggestions(self, prefix: str) -> list[str]:
        """Get command suggestions based on prefix"""
        if not prefix:
            return self._suggestions[:5]  # Show first 5 suggestions

        return [cmd for cmd in self._suggestions if cmd.startswith(prefix.lower())]
