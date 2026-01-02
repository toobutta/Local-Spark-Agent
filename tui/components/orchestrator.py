from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static, Button, Input, Label, ListView, ListItem, ProgressBar
from textual.containers import Vertical, Horizontal, Grid
from typing import Dict, List, Optional, Any
import asyncio
from dataclasses import dataclass

@dataclass
class AgentState:
    """State representation for agent workflows"""
    agent_id: str
    status: str = "idle"
    current_task: Optional[str] = None
    progress: float = 0.0
    memory_usage: float = 0.0
    last_updated: Optional[str] = None

@dataclass
class WorkflowNode:
    """Node in the agent workflow graph"""
    id: str
    name: str
    agent_type: str
    status: str = "pending"
    dependencies: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class AgentOrchestrator(Widget):
    """Agent workflow orchestration using LangGraph"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.workflow_graph = None
        self.active_workflows: Dict[str, List[WorkflowNode]] = {}
        self.agent_states: Dict[str, AgentState] = {}
        self.initialize_orchestrator()

    def initialize_orchestrator(self):
        """Initialize LangGraph orchestrator"""
        try:
            from langgraph.graph import StateGraph
            # Define the agent state schema
            self.workflow_graph = StateGraph(AgentState)
            # Only notify if we have an app context
            try:
                self.notify("Orchestrator initialized", title="Orchestrator", severity="success")
            except:
                print("Orchestrator initialized (no app context)")
        except ImportError:
            try:
                self.notify("LangGraph not installed. Run: pip install langgraph", title="Orchestrator", severity="warning")
            except:
                print("LangGraph not installed. Run: pip install langgraph")
        except Exception as e:
            try:
                self.notify(f"Orchestrator initialization failed: {e}", title="Orchestrator", severity="error")
            except:
                print(f"Orchestrator initialization failed: {e}")

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("AGENT ORCHESTRATION", classes="section-header")

            # Workflow controls
            with Horizontal():
                yield Button("Create Workflow", variant="primary", id="create-workflow-btn")
                yield Button("Start Workflow", variant="success", id="start-workflow-btn")
                yield Button("Stop Workflow", variant="warning", id="stop-workflow-btn")
                yield Button("View Status", variant="outline", id="view-status-btn")

            # Workflow selection
            with Horizontal():
                yield Label("Active Workflows:")
                yield Input(placeholder="workflow-name", id="workflow-selector")
                yield Button("Load", variant="outline", id="load-workflow-btn")

            # Workflow visualization
            with Vertical(id="workflow-visualization"):
                yield Label("WORKFLOW GRAPH", classes="section-header")
                yield Static("No active workflow", id="workflow-display")

            # Agent status panel
            with Vertical(id="agent-status-panel"):
                yield Label("AGENT STATUS", classes="section-header")
                yield ListView(id="agent-status-list")

            # Progress tracking
            with Vertical(id="progress-panel"):
                yield Label("WORKFLOW PROGRESS", classes="section-header")
                yield ProgressBar(id="workflow-progress", total=100)
                yield Static("Progress: 0%", id="progress-text")

    def on_mount(self):
        """Initialize the orchestrator when mounted"""
        self.update_agent_status()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        button_id = event.button.id

        if button_id == "create-workflow-btn":
            await self.create_sample_workflow()
        elif button_id == "start-workflow-btn":
            await self.start_workflow()
        elif button_id == "stop-workflow-btn":
            await self.stop_workflow()
        elif button_id == "view-status-btn":
            self.update_agent_status()
        elif button_id == "load-workflow-btn":
            await self.load_workflow()

    async def create_sample_workflow(self):
        """Create a sample agent workflow"""
        workflow_name = "sample_research_workflow"

        # Define workflow nodes
        nodes = [
            WorkflowNode("researcher", "Research Agent", "research", dependencies=[]),
            WorkflowNode("writer", "Content Writer", "writing", dependencies=["researcher"]),
            WorkflowNode("reviewer", "Review Agent", "review", dependencies=["writer"]),
            WorkflowNode("publisher", "Publish Agent", "publish", dependencies=["reviewer"])
        ]

        self.active_workflows[workflow_name] = nodes
        self.update_workflow_display(workflow_name)
        self.notify(f"Workflow '{workflow_name}' created", title="Orchestrator", severity="success")

    async def start_workflow(self):
        """Start the selected workflow"""
        workflow_input = self.query_one("#workflow-selector", Input)
        workflow_name = workflow_input.value.strip()

        if not workflow_name or workflow_name not in self.active_workflows:
            self.notify("Please select a valid workflow", title="Orchestrator", severity="warning")
            return

        workflow = self.active_workflows[workflow_name]

        # Simulate workflow execution
        self.notify(f"Starting workflow: {workflow_name}", title="Orchestrator", severity="info")

        for i, node in enumerate(workflow):
            # Update node status
            node.status = "running"
            self.update_workflow_display(workflow_name)

            # Simulate processing time
            await asyncio.sleep(1)

            # Mark as completed
            node.status = "completed"
            self.update_workflow_display(workflow_name)

            # Update progress
            progress = (i + 1) / len(workflow) * 100
            await self.update_progress(progress)

        self.notify(f"Workflow '{workflow_name}' completed", title="Orchestrator", severity="success")

    async def stop_workflow(self):
        """Stop the current workflow"""
        workflow_input = self.query_one("#workflow-selector", Input)
        workflow_name = workflow_input.value.strip()

        if workflow_name in self.active_workflows:
            for node in self.active_workflows[workflow_name]:
                if node.status == "running":
                    node.status = "stopped"

            self.update_workflow_display(workflow_name)
            self.notify(f"Workflow '{workflow_name}' stopped", title="Orchestrator", severity="warning")

    async def load_workflow(self):
        """Load and display the selected workflow"""
        workflow_input = self.query_one("#workflow-selector", Input)
        workflow_name = workflow_input.value.strip()

        if workflow_name in self.active_workflows:
            self.update_workflow_display(workflow_name)
            self.notify(f"Workflow '{workflow_name}' loaded", title="Orchestrator", severity="info")
        else:
            self.notify(f"Workflow '{workflow_name}' not found", title="Orchestrator", severity="warning")

    def update_workflow_display(self, workflow_name: str):
        """Update the workflow visualization"""
        if workflow_name not in self.active_workflows:
            return

        workflow = self.active_workflows[workflow_name]
        display_lines = []

        for node in workflow:
            status_icon = {
                "pending": "○",
                "running": "●",
                "completed": "✓",
                "stopped": "✗"
            }.get(node.status, "?")

            # Show dependencies
            deps = f" ← {', '.join(node.dependencies)}" if node.dependencies else ""
            display_lines.append(f"{status_icon} {node.name} ({node.agent_type}){deps}")

        workflow_display = self.query_one("#workflow-display", Static)
        workflow_display.update("\n".join(display_lines))

    def update_agent_status(self):
        """Update the agent status list"""
        status_list = self.query_one("#agent-status-list", ListView)
        status_list.clear()

        # Sample agent statuses
        sample_agents = [
            AgentState("research-agent-01", "running", "Researching AI trends", 75.0, 2.3),
            AgentState("writer-agent-01", "idle", None, 0.0, 1.1),
            AgentState("review-agent-01", "completed", "Review completed", 100.0, 0.8)
        ]

        for agent in sample_agents:
            status_text = f"{agent.agent_id}: {agent.status.upper()}"
            if agent.current_task:
                status_text += f" - {agent.current_task}"
            status_text += f" ({agent.progress:.0f}%)"

            status_list.append(ListItem(Label(status_text)))

    async def update_progress(self, progress: float):
        """Update workflow progress display"""
        progress_bar = self.query_one("#workflow-progress", ProgressBar)
        progress_text = self.query_one("#progress-text", Static)

        progress_bar.progress = progress
        progress_text.update(f"Progress: {progress:.1f}%")
