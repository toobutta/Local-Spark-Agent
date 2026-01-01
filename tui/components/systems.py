"""
Systems Infrastructure Components.

Provides DGX configuration, model configuration, tools grid, and MCP integrations.
"""

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import (
    Static, Button, TabbedContent, TabPane, Input, Switch, Label, Tree, ProgressBar
)
from textual.containers import Horizontal, Vertical, Grid
from components.memory_manager import MemoryManager
from components.orchestrator import AgentOrchestrator
from components.communicator import AgentCommunicator
from components.code_assistant import CodeAssistantPanel
import random
import time
import asyncio
import logging

logger = logging.getLogger(__name__)

# Try to import DGX integration
try:
    from ..integrations.dgx_spark import DGXSparkAPI, get_dgx_api, GPUMetrics, SystemMetrics
    HAS_DGX = True
except ImportError:
    HAS_DGX = False

# Try to import services
try:
    from ..services.event_bus import EventBus, EventType, get_event_bus
    from ..services.config_store import get_config_store
    HAS_SERVICES = True
except ImportError:
    HAS_SERVICES = False


class DGXConfigPanel(Widget):
    """
    DGX Configuration Panel with real GPU metrics.
    
    Uses pynvml/nvidia-smi when available, falls back to simulation.
    """
    
    DEFAULT_CSS = """
    DGXConfigPanel {
        height: auto;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dgx_api = None
        self._event_bus = None
        self._config_store = None
        self._simulation_mode = True
        
        # Metrics state
        self.gpu_utilization = 0.0
        self.memory_bandwidth = 273.0
        self.memory_used = 0
        self.memory_total = 128 * 1024  # 128 GB default
        self.temperature = 45
        self.power_draw = 100.0
        self.inference_tokens = 32
        self.active_agents = 0
        self.agent_memory_usage = 0.0
        
        # Initialize services
        if HAS_DGX:
            try:
                self._dgx_api = get_dgx_api()
            except Exception as e:
                logger.warning(f"Failed to initialize DGX API: {e}")
                self._dgx_api = None

        if HAS_SERVICES:
            try:
                self._event_bus = get_event_bus()
            except Exception as e:
                logger.warning(f"Failed to get event bus: {e}")
                self._event_bus = None

            try:
                self._config_store = get_config_store()
                # Check simulation mode preference
                if self._config_store:
                    self._simulation_mode = self._config_store.get("dgx_simulation_mode", True)
            except Exception as e:
                logger.warning(f"Failed to get config store: {e}")
                self._config_store = None

    def on_mount(self):
        """Start real-time monitoring when the widget is mounted."""
        # Get refresh interval from config
        interval = 1.0
        if self._config_store:
            interval = self._config_store.get("dgx_refresh_interval", 1.0)
        
        self.set_interval(interval, self.update_metrics)

    async def update_metrics(self):
        """Update GPU metrics in real-time."""
        if self._dgx_api and not self._simulation_mode:
            try:
                metrics = await self._dgx_api.get_metrics()
                
                if metrics.gpus:
                    gpu = metrics.gpus[0]
                    self.gpu_utilization = gpu.gpu_utilization
                    self.memory_bandwidth = gpu.memory_bandwidth
                    self.memory_used = gpu.memory_used
                    self.memory_total = gpu.memory_total
                    self.temperature = gpu.temperature
                    self.power_draw = gpu.power_draw
                    
                    # Estimate inference tokens based on utilization
                    self.inference_tokens = int(32 + (gpu.gpu_utilization / 100) * 20)
                    
                    # Update simulation mode indicator
                    self._simulation_mode = metrics.simulation_mode
            except Exception:
                # Fall back to simulation
                self._update_simulated_metrics()
        else:
            self._update_simulated_metrics()
        
        # Update the display
        self.refresh()
    
    def _update_simulated_metrics(self):
        """Update with simulated metrics."""
        current_time = time.time()
        
        # GPU utilization: varies between 40-95% based on load
        base_util = 60 + 25 * (0.5 + 0.5 * (current_time % 10) / 10)
        noise = random.uniform(-5, 5)
        self.gpu_utilization = max(40, min(95, base_util + noise))
        
        # Memory bandwidth: varies slightly around 273 GB/s
        self.memory_bandwidth = 273 + random.uniform(-10, 10)
        
        # Memory usage
        self.memory_used = int(self.memory_total * (0.3 + self.gpu_utilization / 100 * 0.4))
        
        # Temperature
        self.temperature = int(40 + self.gpu_utilization * 0.4 + random.randint(-2, 2))
        
        # Power
        self.power_draw = 50 + (self.gpu_utilization / 100) * 650 + random.uniform(-10, 10)
        
        # Inference tokens: varies based on model complexity
        self.inference_tokens = 32 + random.randint(-3, 6)
        
        # Agent activity simulation
        self.active_agents = random.randint(0, 4)
        self.agent_memory_usage = self.active_agents * random.uniform(2, 8)

    def compose(self) -> ComposeResult:
        with Vertical(classes="dgx-panel"):
            # Header with simulation indicator
            mode_indicator = " [SIM]" if self._simulation_mode else ""
            yield Static(f"DGX SPARK [NVIDIA LPDDR5x]{mode_indicator}", classes="dgx-header")

            with Grid(classes="dgx-grid"):
                with Vertical(classes="dgx-info-box"):
                    yield Label("NETWORK IDENTITY", classes="dgx-label")
                    yield Static("dgx-h200-node-01", classes="dgx-value")

                with Vertical(classes="dgx-info-box"):
                    yield Label("ALLOCATION", classes="dgx-label")
                    yield Static("DGX SPARK (ACTIVE)", classes="dgx-value active")

            with Horizontal(classes="dgx-actions"):
                yield Static("SSH ACCESS: CONFIGURED", classes="ssh-status")
                yield Button("MANAGE KEYS", variant="primary", classes="manage-keys-btn")
                yield Button("Toggle Sim", variant="outline", id="toggle-sim-btn")

            # Real-time Performance Metrics
            yield Label(f"GPU UTILIZATION: {self.gpu_utilization:.1f}%", classes="allocation-label", id="gpu-util-label")
            gpu_bar = "█" * int(self.gpu_utilization / 5) + "░" * (20 - int(self.gpu_utilization / 5))
            yield Static(gpu_bar[:20], classes="allocation-bar", id="gpu-util-bar")

            # Memory usage
            mem_percent = (self.memory_used / self.memory_total * 100) if self.memory_total > 0 else 0
            yield Label(f"MEMORY: {self.memory_used / 1024:.1f} GB / {self.memory_total / 1024:.0f} GB ({mem_percent:.1f}%)", 
                       classes="allocation-label", id="mem-label")
            
            yield Label(f"MEMORY BANDWIDTH: {self.memory_bandwidth:.1f} GB/s LPDDR5x", classes="allocation-label")
            yield Label(f"TEMPERATURE: {self.temperature}°C | POWER: {self.power_draw:.0f}W", classes="allocation-label")
            yield Label(f"INFERENCE: {self.inference_tokens}-{self.inference_tokens+6} tok/s (FP4 Large Models)", classes="allocation-label")

            # Agent Activity Metrics
            yield Label(f"ACTIVE AGENTS: {self.active_agents}", classes="allocation-label")
            yield Label(f"AGENT MEMORY: {self.agent_memory_usage:.1f} GB", classes="allocation-label")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "toggle-sim-btn":
            self._simulation_mode = not self._simulation_mode
            
            # Save preference
            if self._config_store:
                self._config_store.set("dgx_simulation_mode", self._simulation_mode)
            
            mode = "Simulation" if self._simulation_mode else "Real Hardware"
            self.app.notify(f"DGX Mode: {mode}", title="DGX Config")
            
            # Force refresh
            await self.update_metrics()


class ModelConfigPanel(Widget):
    def compose(self) -> ComposeResult:
        with Vertical(id="model-config-panel"):
            yield Label("SECURE VAULT", classes="secure-vault-label")
            with TabbedContent():
                with TabPane("Anthropic"):
                    yield Label("API Key:")
                    yield Horizontal(
                        Input(placeholder="sk-ant-...", password=True, classes="api-input"),
                        Button("Connect", variant="warning")
                    )
                with TabPane("OpenAI"):
                    yield Label("API Key:")
                    yield Horizontal(
                        Input(placeholder="sk-...", password=True, classes="api-input"),
                        Button("Connect", variant="warning")
                    )
                with TabPane("Gemini"):
                    yield Label("API Key:")
                    yield Horizontal(
                        Input(placeholder="AIza...", password=True, classes="api-input"),
                        Button("Connect", variant="warning")
                    )
            
            yield Horizontal(
                Label("Local Inference Override (Ollama/LocalAI)"),
                Switch(value=False, id="local-inference-switch")
            )


class ToolCard(Static):
    def __init__(self, label, status="off", **kwargs):
        super().__init__(**kwargs)
        self.label_text = label
        self.status = status

    def render(self):
        icon = "◉" if self.status == "on" else "○"
        return f"{self.label_text}\n[{icon}] {self.status.upper()}"

    def on_mount(self):
        if self.status == "on":
            self.add_class("active")
        self.add_class("tool-card")


class ToolsGrid(Widget):
    def compose(self) -> ComposeResult:
        yield Static("EXTERNAL TOOLS & APIs", classes="section-header")
        with Grid(classes="tools-grid"):
            yield ToolCard("GitHub", "off")
            yield ToolCard("PostgreSQL", "on")
            yield ToolCard("Pinecone", "on")
            yield ToolCard("Brave Search", "on")
            yield ToolCard("Slack", "off")
            yield ToolCard("Vercel", "off")
            yield ToolCard("Sentry", "off")
            yield ToolCard("Hugging Face", "off")
        yield Button("+ ADD CUSTOM TOOL / API", variant="primary", id="add-tool-btn")


class MCPIntegrationTree(Widget):
    def compose(self) -> ComposeResult:
        yield Static("MCP INTEGRATIONS", classes="section-header")
        tree = Tree("MCP Servers")
        tree.root.expand()
        tree.root.add("PostgreSQL Connector [CONFIG]", expand=True)
        tree.root.add("Filesystem Watcher [CONFIG]")
        tree.root.add("GitHub Repository [CONFIG]")
        tree.root.add("Memory Service [CONFIG]")
        tree.root.add("Google Drive [CONFIG]")
        yield tree
        yield Button("+ ADD MCP SERVER", variant="success")


class BuildPipelinePanel(Widget):
    def compose(self) -> ComposeResult:
        yield Static("BUILD PIPELINE", classes="section-header")
        with Vertical(classes="card"):
            yield Horizontal(
                Label("Auto-Deploy Agents"),
                Switch(value=True)
            )
            yield Horizontal(
                Label("Verbose Logging"),
                Switch(value=False)
            )


class SystemsContent(Widget):
    def compose(self) -> ComposeResult:
        with TabbedContent():
            with TabPane("Infrastructure", id="infra-tab"):
                with Vertical():
                    yield Static("SYSTEMS INFRASTRUCTURE", classes="section-header")
                    yield DGXConfigPanel()
                    yield ModelConfigPanel()
                    yield ToolsGrid()
                    yield Horizontal(
                        MCPIntegrationTree(classes="card"),
                        BuildPipelinePanel(classes="card")
                    )

            with TabPane("Agent Memory", id="memory-tab"):
                yield MemoryManager()

            with TabPane("Orchestration", id="orchestration-tab"):
                yield AgentOrchestrator()

            with TabPane("Communication", id="communication-tab"):
                yield AgentCommunicator()

            with TabPane("Code Assistant", id="code-tab"):
                yield CodeAssistantPanel()
