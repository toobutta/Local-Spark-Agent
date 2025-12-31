from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static, Button, TabbedContent, TabPane, Input, Switch, Label, Tree
from textual.containers import Horizontal, Vertical, Grid

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
                Switch(value=False)
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
        with Vertical():
            yield ModelConfigPanel()
            yield ToolsGrid()
            yield Horizontal(
                MCPIntegrationTree(classes="card"),
                BuildPipelinePanel(classes="card")
            )
