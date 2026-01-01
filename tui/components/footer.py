"""
Footer Widget with real-time metrics sparklines.

Displays GPU utilization, memory usage, and agent activity metrics.
"""

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static, Label
from textual.containers import Horizontal
from rich.text import Text
import random
import asyncio
from typing import List, Optional

# Try to import integrations
try:
    from ..integrations.dgx_spark import DGXSparkAPI, get_dgx_api, GPUMetrics
    HAS_DGX = True
except ImportError:
    HAS_DGX = False

# Try to import services
try:
    from ..services.event_bus import EventBus, EventType, get_event_bus
    from ..services.workspace_manager import get_workspace_manager
    HAS_SERVICES = True
except ImportError:
    HAS_SERVICES = False


class Sparkline(Static):
    """A sparkline widget for displaying metric history."""
    
    SPARKLINE_CHARS = " ▁▂▃▄▅▆▇█"
    
    def __init__(
        self,
        label: str = "",
        width: int = 20,
        color: str = "#00f5d4",
        **kwargs
    ):
        super().__init__(**kwargs)
        self._label = label
        self._width = width
        self._color = color
        self._values: List[float] = []
        self._min_val = 0.0
        self._max_val = 100.0
    
    def update_values(self, values: List[float], min_val: float = 0.0, max_val: float = 100.0):
        """Update the sparkline with new values."""
        self._values = values[-self._width:]  # Keep only last N values
        self._min_val = min_val
        self._max_val = max_val
        self.refresh()
    
    def render(self) -> Text:
        """Render the sparkline."""
        if not self._values:
            bars = "─" * self._width
        else:
            # Normalize values to 0-8 range for character selection
            range_val = self._max_val - self._min_val
            if range_val == 0:
                range_val = 1
            
            bars = ""
            for val in self._values:
                normalized = (val - self._min_val) / range_val
                normalized = max(0, min(1, normalized))  # Clamp to 0-1
                char_idx = int(normalized * 8)
                bars += self.SPARKLINE_CHARS[char_idx]
            
            # Pad to width if needed
            if len(bars) < self._width:
                bars = " " * (self._width - len(bars)) + bars
        
        # Get current value for display
        current = self._values[-1] if self._values else 0
        
        text = Text()
        text.append(f"{self._label}: ", style="bold")
        text.append(bars, style=self._color)
        text.append(f" {current:.0f}%", style=self._color)
        
        return text


class FooterWidget(Widget):
    """
    Enhanced footer widget with real-time metrics sparklines.
    
    Features:
    - GPU utilization sparkline
    - Memory usage sparkline
    - Agent activity indicator
    - Network I/O display
    - Temperature indicator
    """
    
    DEFAULT_CSS = """
    FooterWidget {
        height: 3;
        layout: horizontal;
        align-vertical: middle;
        padding: 0 1;
    }
    
    FooterWidget .footer-section {
        width: auto;
        padding: 0 2;
    }
    
    FooterWidget #gpu-sparkline {
        width: auto;
    }
    
    FooterWidget #mem-sparkline {
        width: auto;
    }
    
    FooterWidget #agent-status {
        color: #ff4bcb;
    }
    
    FooterWidget #temp-indicator {
        width: auto;
    }
    
    FooterWidget #equalizer {
        color: #00f5d4;
        width: 10;
        text-align: right;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dgx_api: Optional[DGXSparkAPI] = None
        self._event_bus: Optional[EventBus] = None
        self._workspace_manager = None
        
        # Metrics history
        self._gpu_history: List[float] = []
        self._mem_history: List[float] = []
        self._temp_history: List[int] = []
        
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
                self._workspace_manager = get_workspace_manager()
            except Exception as e:
                logger.warning(f"Failed to get workspace manager: {e}")
                self._workspace_manager = None
    
    def compose(self) -> ComposeResult:
        with Horizontal():
            # Keyboard shortcuts hint
            yield Static("F1-F5: Tabs | Ctrl+P: Commands", classes="footer-section subtitle")
            
            # GPU utilization sparkline
            yield Sparkline(
                label="GPU",
                width=15,
                color="#00f5d4",
                id="gpu-sparkline",
                classes="footer-section"
            )
            
            # Memory usage sparkline
            yield Sparkline(
                label="MEM",
                width=15,
                color="#ffb000",
                id="mem-sparkline",
                classes="footer-section"
            )
            
            # Temperature indicator
            yield Static("🌡️ --°C", id="temp-indicator", classes="footer-section")
            
            # Agent status
            yield Static("⚡ 0 agents", id="agent-status", classes="footer-section")
            
            # Activity equalizer
            yield Static("▂▃▄▅▆▇█", id="equalizer")
    
    def on_mount(self):
        """Start metrics collection when mounted."""
        # Update metrics every second
        self.set_interval(1.0, self._update_metrics)
        
        # Update equalizer animation faster
        self.set_interval(0.1, self._refresh_equalizer)
        
        # Subscribe to events
        if self._event_bus:
            self._event_bus.subscribe(EventType.GPU_METRICS_UPDATED, self._on_gpu_metrics)
            self._event_bus.subscribe(EventType.AGENT_DEPLOYED, self._on_agent_change)
            self._event_bus.subscribe(EventType.AGENT_STOPPED, self._on_agent_change)
    
    async def _update_metrics(self):
        """Update all metrics displays."""
        await self._update_gpu_metrics()
        self._update_agent_count()
    
    async def _update_gpu_metrics(self):
        """Update GPU metrics from DGX API."""
        if self._dgx_api:
            try:
                metrics = await self._dgx_api.get_metrics()
                
                if metrics.gpus:
                    gpu = metrics.gpus[0]
                    
                    # Update history
                    self._gpu_history.append(gpu.gpu_utilization)
                    self._mem_history.append(gpu.memory_used_percent)
                    self._temp_history.append(gpu.temperature)
                    
                    # Keep history limited
                    self._gpu_history = self._gpu_history[-60:]
                    self._mem_history = self._mem_history[-60:]
                    self._temp_history = self._temp_history[-60:]
                    
                    # Update sparklines
                    gpu_sparkline = self.query_one("#gpu-sparkline", Sparkline)
                    gpu_sparkline.update_values(self._gpu_history)
                    
                    mem_sparkline = self.query_one("#mem-sparkline", Sparkline)
                    mem_sparkline.update_values(self._mem_history)
                    
                    # Update temperature
                    temp_indicator = self.query_one("#temp-indicator", Static)
                    temp_color = self._get_temp_color(gpu.temperature)
                    temp_indicator.update(f"🌡️ {gpu.temperature}°C")
                    
                    # Emit event
                    if self._event_bus:
                        self._event_bus.emit(
                            EventType.GPU_METRICS_UPDATED,
                            data={
                                "utilization": gpu.gpu_utilization,
                                "memory_percent": gpu.memory_used_percent,
                                "temperature": gpu.temperature,
                            },
                            source="footer"
                        )
            
            except Exception as e:
                # Fall back to simulated data
                self._update_simulated_metrics()
        else:
            self._update_simulated_metrics()
    
    def _update_simulated_metrics(self):
        """Update with simulated metrics when DGX API unavailable."""
        # Simulate GPU utilization
        if self._gpu_history:
            last_gpu = self._gpu_history[-1]
            new_gpu = max(0, min(100, last_gpu + random.uniform(-5, 5)))
        else:
            new_gpu = random.uniform(40, 70)
        
        self._gpu_history.append(new_gpu)
        self._gpu_history = self._gpu_history[-60:]
        
        # Simulate memory
        if self._mem_history:
            last_mem = self._mem_history[-1]
            new_mem = max(0, min(100, last_mem + random.uniform(-2, 2)))
        else:
            new_mem = random.uniform(30, 50)
        
        self._mem_history.append(new_mem)
        self._mem_history = self._mem_history[-60:]
        
        # Simulate temperature
        temp = random.randint(45, 65)
        self._temp_history.append(temp)
        self._temp_history = self._temp_history[-60:]
        
        # Update displays
        try:
            gpu_sparkline = self.query_one("#gpu-sparkline", Sparkline)
            gpu_sparkline.update_values(self._gpu_history)
            
            mem_sparkline = self.query_one("#mem-sparkline", Sparkline)
            mem_sparkline.update_values(self._mem_history)
            
            temp_indicator = self.query_one("#temp-indicator", Static)
            temp_indicator.update(f"🌡️ {temp}°C")
        except Exception:
            pass
    
    def _get_temp_color(self, temp: int) -> str:
        """Get color based on temperature."""
        if temp < 50:
            return "#00f5d4"  # Teal - cool
        elif temp < 70:
            return "#ffb000"  # Amber - warm
        elif temp < 85:
            return "#ff8800"  # Orange - hot
        else:
            return "#ff0000"  # Red - critical
    
    def _update_agent_count(self):
        """Update agent count display."""
        try:
            agent_status = self.query_one("#agent-status", Static)
            
            if self._workspace_manager:
                agents = self._workspace_manager.get_active_agents()
                count = len(agents)
            else:
                count = random.randint(0, 3)  # Simulated
            
            agent_status.update(f"⚡ {count} agent{'s' if count != 1 else ''}")
        except Exception:
            pass
    
    def _refresh_equalizer(self):
        """Refresh the activity equalizer animation."""
        try:
            equalizer = self.query_one("#equalizer", Static)
            bars = " ▂▃▄▅▆▇█"
            
            # Generate animated bars based on GPU activity
            if self._gpu_history:
                activity = self._gpu_history[-1] / 100.0
            else:
                activity = 0.5
            
            # Create bars with activity-influenced randomness
            bar_str = ""
            for _ in range(8):
                # Higher activity = higher bars on average
                base = int(activity * 4)
                variation = random.randint(-2, 3)
                idx = max(0, min(8, base + variation))
                bar_str += bars[idx]
            
            equalizer.update(bar_str)
        except Exception:
            pass
    
    def _on_gpu_metrics(self, event):
        """Handle GPU metrics events."""
        # Metrics are already updated by _update_gpu_metrics
        pass
    
    def _on_agent_change(self, event):
        """Handle agent deployment/stop events."""
        self._update_agent_count()
    
    def get_gpu_history(self) -> List[float]:
        """Get GPU utilization history."""
        return self._gpu_history.copy()
    
    def get_memory_history(self) -> List[float]:
        """Get memory usage history."""
        return self._mem_history.copy()
    
    def get_temperature_history(self) -> List[int]:
        """Get temperature history."""
        return self._temp_history.copy()
