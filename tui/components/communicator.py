from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static, Button, Input, Label, ListView, ListItem, TextArea
from textual.containers import Vertical, Horizontal
import asyncio
import threading
import time
from typing import List, Dict, Optional
import json

class AgentCommunicator(Widget):
    """Agent-to-Agent communication interface using Redis"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.redis_client = None
        self.pubsub = None
        self.current_agent = "tui-controller"
        self.messages: List[Dict] = []
        self.is_listening = False
        self.listen_thread: Optional[threading.Thread] = None

    def on_mount(self):
        """Initialize Redis communication"""
        self.initialize_redis()

    def on_unmount(self):
        """Clean up resources"""
        self.stop_listening()

    def initialize_redis(self):
        """Initialize Redis client and pubsub"""
        try:
            import redis
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.pubsub = self.redis_client.pubsub()
            self.notify("Redis communication initialized", title="Communication", severity="success")
        except ImportError:
            self.notify("Redis not installed. Run: pip install redis", title="Communication", severity="warning")
        except Exception as e:
            self.notify(f"Redis connection failed: {e}", title="Communication", severity="error")

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("AGENT COMMUNICATION", classes="section-header")

            # Connection status
            with Horizontal():
                yield Static("Status: ", id="connection-status")
                yield Button("Connect", variant="primary", id="connect-btn")
                yield Button("Disconnect", variant="warning", id="disconnect-btn")

            # Agent identification
            with Horizontal():
                yield Label("Agent ID:")
                yield Input(value=self.current_agent, id="agent-id-input")
                yield Button("Update", variant="outline", id="update-agent-btn")

            # Message composition
            with Vertical(id="message-composer"):
                yield Label("SEND MESSAGE", classes="section-header")
                with Horizontal():
                    yield Label("To:")
                    yield Input(placeholder="target-agent", id="target-agent-input")
                yield TextArea(placeholder="Enter message...", id="message-input")
                yield Button("Send Message", variant="success", id="send-message-btn")

            # Message history
            with Vertical(id="message-history"):
                yield Label("MESSAGE HISTORY", classes="section-header")
                yield ListView(id="message-list")

            # Quick commands
            with Horizontal(id="quick-commands"):
                yield Button("Ping All", variant="outline", id="ping-all-btn")
                yield Button("Status Request", variant="outline", id="status-request-btn")
                yield Button("Clear History", variant="warning", id="clear-history-btn")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        button_id = event.button.id

        if button_id == "connect-btn":
            await self.connect_communication()
        elif button_id == "disconnect-btn":
            self.disconnect_communication()
        elif button_id == "update-agent-btn":
            self.update_agent_id()
        elif button_id == "send-message-btn":
            await self.send_message()
        elif button_id == "ping-all-btn":
            await self.ping_all_agents()
        elif button_id == "status-request-btn":
            await self.request_status()
        elif button_id == "clear-history-btn":
            self.clear_message_history()

    async def connect_communication(self):
        """Connect to Redis and start listening"""
        if not self.redis_client:
            self.initialize_redis()
            if not self.redis_client:
                return

        try:
            # Test connection
            self.redis_client.ping()

            # Start listening for messages
            self.start_listening()

            # Update status
            status_display = self.query_one("#connection-status", Static)
            status_display.update("Status: Connected ✅")

            self.notify("Connected to agent communication network", title="Communication", severity="success")

        except Exception as e:
            status_display = self.query_one("#connection-status", Static)
            status_display.update("Status: Connection Failed ❌")
            self.notify(f"Connection failed: {e}", title="Communication", severity="error")

    def disconnect_communication(self):
        """Disconnect from communication network"""
        self.stop_listening()

        status_display = self.query_one("#connection-status", Static)
        status_display.update("Status: Disconnected")

        self.notify("Disconnected from agent communication", title="Communication", severity="info")

    def update_agent_id(self):
        """Update the current agent ID"""
        agent_input = self.query_one("#agent-id-input", Input)
        new_id = agent_input.value.strip()

        if new_id:
            old_id = self.current_agent
            self.current_agent = new_id
            self.notify(f"Agent ID changed: {old_id} → {new_id}", title="Communication", severity="info")

            # Restart listening with new ID if connected
            if self.is_listening:
                self.stop_listening()
                self.start_listening()
        else:
            self.notify("Please enter a valid agent ID", title="Communication", severity="warning")

    async def send_message(self):
        """Send a message to another agent"""
        if not self.redis_client or not self.is_listening:
            self.notify("Not connected to communication network", title="Communication", severity="warning")
            return

        target_input = self.query_one("#target-agent-input", Input)
        message_input = self.query_one("#message-input", TextArea)

        target = target_input.value.strip()
        message = message_input.value.strip()

        if not target or not message:
            self.notify("Please specify target agent and message", title="Communication", severity="warning")
            return

        try:
            # Create message payload
            message_payload = {
                "from": self.current_agent,
                "to": target,
                "message": message,
                "timestamp": time.time(),
                "type": "direct_message"
            }

            # Publish to target agent's channel
            channel = f"agent:{target}"
            self.redis_client.publish(channel, json.dumps(message_payload))

            # Also publish to general channel
            self.redis_client.publish("agent:broadcast", json.dumps(message_payload))

            # Add to local message history
            self.add_message_to_history(message_payload, sent=True)

            # Clear inputs
            message_input.value = ""

            self.notify(f"Message sent to {target}", title="Communication", severity="success")

        except Exception as e:
            self.notify(f"Failed to send message: {e}", title="Communication", severity="error")

    async def ping_all_agents(self):
        """Send ping to all agents"""
        if not self.redis_client or not self.is_listening:
            self.notify("Not connected to communication network", title="Communication", severity="warning")
            return

        try:
            ping_message = {
                "from": self.current_agent,
                "to": "all",
                "message": "ping",
                "timestamp": time.time(),
                "type": "ping"
            }

            self.redis_client.publish("agent:broadcast", json.dumps(ping_message))
            self.add_message_to_history(ping_message, sent=True)

            self.notify("Ping sent to all agents", title="Communication", severity="info")

        except Exception as e:
            self.notify(f"Failed to send ping: {e}", title="Communication", severity="error")

    async def request_status(self):
        """Request status from all agents"""
        if not self.redis_client or not self.is_listening:
            self.notify("Not connected to communication network", title="Communication", severity="warning")
            return

        try:
            status_message = {
                "from": self.current_agent,
                "to": "all",
                "message": "status_request",
                "timestamp": time.time(),
                "type": "status_request"
            }

            self.redis_client.publish("agent:broadcast", json.dumps(status_message))
            self.add_message_to_history(status_message, sent=True)

            self.notify("Status request sent to all agents", title="Communication", severity="info")

        except Exception as e:
            self.notify(f"Failed to request status: {e}", title="Communication", severity="error")

    def start_listening(self):
        """Start listening for messages in a separate thread"""
        if self.is_listening:
            return

        self.is_listening = True

        # Subscribe to agent's personal channel and broadcast channel
        self.pubsub.subscribe(f"agent:{self.current_agent}")
        self.pubsub.subscribe("agent:broadcast")

        # Start listening thread
        self.listen_thread = threading.Thread(target=self.listen_for_messages, daemon=True)
        self.listen_thread.start()

    def stop_listening(self):
        """Stop listening for messages"""
        if not self.is_listening:
            return

        self.is_listening = False

        if self.pubsub:
            self.pubsub.unsubscribe()

        if self.listen_thread and self.listen_thread.is_alive():
            self.listen_thread.join(timeout=1.0)

    def listen_for_messages(self):
        """Listen for incoming messages (runs in separate thread)"""
        try:
            for message in self.pubsub.listen():
                if not self.is_listening:
                    break

                if message['type'] == 'message':
                    try:
                        # Parse message
                        data = json.loads(message['data'])
                        self.add_message_to_history(data, sent=False)
                    except json.JSONDecodeError:
                        # Handle non-JSON messages
                        raw_message = {
                            "from": "unknown",
                            "to": self.current_agent,
                            "message": message['data'],
                            "timestamp": time.time(),
                            "type": "raw"
                        }
                        self.add_message_to_history(raw_message, sent=False)

        except Exception as e:
            # Schedule UI notification on main thread to avoid race conditions
            asyncio.run_coroutine_threadsafe(
                self._notify_error(f"Message listening error: {e}"),
                self.app.loop
            )

    def add_message_to_history(self, message: Dict, sent: bool = False):
        """Add message to history and update display"""
        # Add to internal history
        self.messages.append({"data": message, "sent": sent})

        # Keep only last 50 messages
        if len(self.messages) > 50:
            self.messages = self.messages[-50:]

        # Update display (schedule to main thread)
        asyncio.run_coroutine_threadsafe(self.update_message_display(), self.app.loop)

    async def update_message_display(self):
        """Update the message list display"""
        message_list = self.query_one("#message-list", ListView)
        message_list.clear()

        for msg_data in self.messages[-20:]:  # Show last 20 messages
            message = msg_data["data"]
            sent = msg_data["sent"]

            # Format message display
            direction = "→" if sent else "←"
            from_agent = message.get("from", "unknown")
            to_agent = message.get("to", "unknown")
            msg_content = message.get("message", "")

            # Truncate long messages
            if len(msg_content) > 50:
                msg_content = msg_content[:47] + "..."

            # Color code based on direction
            if sent:
                display_text = f"[bold cyan]{direction} {to_agent}:[/bold cyan] {msg_content}"
            else:
                display_text = f"[bold green]{direction} {from_agent}:[/bold green] {msg_content}"

            message_list.append(ListItem(Label(display_text)))

    async def _notify_error(self, message: str):
        """Notify error from main thread"""
        self.app.notify(message, title="Communication", severity="error")

    def clear_message_history(self):
        """Clear message history"""
        self.messages.clear()
        message_list = self.query_one("#message-list", ListView)
        message_list.clear()
        self.notify("Message history cleared", title="Communication", severity="info")
