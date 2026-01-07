import httpx
import websockets
import asyncio
import json
from typing import Dict, List, Any, Optional, Callable


class SparkPlugAPI:
    """API client for connecting TUI to SparkPlug backend"""

    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=30.0
        )
        self.websocket = None
        self.message_handlers: Dict[str, Callable] = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Close HTTP client and WebSocket connections"""
        await self.client.aclose()
        if self.websocket:
            await self.websocket.close()

    # HTTP API Methods

    async def execute_command(self, command: str) -> Dict[str, Any]:
        """Execute a command via the backend API"""
        try:
            response = await self.client.post(
                "/api/commands/execute",
                json={"command": command}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            return {
                "success": False,
                "output": "",
                "error": f"HTTP error: {e}"
            }

    async def get_agents(self) -> List[Dict[str, Any]]:
        """Get list of active agents"""
        try:
            response = await self.client.get("/api/agents")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            print(f"Failed to get agents: {e}")
            return []

    async def create_agent(self, name: str, role: str, config: Optional[Dict] = None) -> Dict[str, Any]:
        """Create a new agent"""
        try:
            response = await self.client.post(
                "/api/agents",
                json={"name": name, "role": role, "config": config}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            return {"error": f"Failed to create agent: {e}"}

    async def get_projects(self) -> List[Dict[str, Any]]:
        """Get list of projects"""
        try:
            response = await self.client.get("/api/projects")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            print(f"Failed to get projects: {e}")
            return []

    async def get_config(self) -> Dict[str, Any]:
        """Get system configuration"""
        try:
            response = await self.client.get("/api/config")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            print(f"Failed to get config: {e}")
            return {}

    async def get_mcp_status(self) -> List[Dict[str, Any]]:
        """Get MCP server status"""
        try:
            response = await self.client.get("/api/mcp/status")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            print(f"Failed to get MCP status: {e}")
            return []

    # WebSocket Methods

    def on_message(self, message_type: str, handler: Callable):
        """Register a message handler for WebSocket messages"""
        self.message_handlers[message_type] = handler

    async def connect_websocket(self):
        """Connect to WebSocket for real-time updates"""
        try:
            websocket_url = self.base_url.replace('http://', 'ws://').replace('https://', 'wss://') + '/ws'
            self.websocket = await websockets.connect(websocket_url)

            # Send connection message
            await self.websocket.send(json.dumps({
                "type": "connect",
                "data": {"client": "tui"}
            }))

            # Start message handling loop
            asyncio.create_task(self._handle_websocket_messages())

        except Exception as e:
            print(f"Failed to connect to WebSocket: {e}")

    async def _handle_websocket_messages(self):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get('type')

                    if message_type in self.message_handlers:
                        handler = self.message_handlers[message_type]
                        await handler(data.get('data', {}))
                    else:
                        print(f"Unhandled WebSocket message: {message_type}")

                except json.JSONDecodeError:
                    print(f"Invalid WebSocket message: {message}")

        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")
        except Exception as e:
            print(f"WebSocket error: {e}")

    async def send_websocket_message(self, message_type: str, data: Any):
        """Send a message via WebSocket"""
        if self.websocket:
            try:
                await self.websocket.send(json.dumps({
                    "type": message_type,
                    "data": data
                }))
            except Exception as e:
                print(f"Failed to send WebSocket message: {e}")

    # Utility Methods

    async def health_check(self) -> bool:
        """Check if the backend is healthy"""
        try:
            response = await self.client.get("/api/config")
            return response.status_code == 200
        except:
            return False
