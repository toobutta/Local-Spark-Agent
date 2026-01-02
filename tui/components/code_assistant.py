"""
Code Assistant Panel with Aider integration and model selector.

Provides async streaming for Aider responses and model switching between
cloud providers and local Ollama models.
"""

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import (
    Static, Button, Input, Label, TextArea, Tree, Select, Switch, ProgressBar
)
from textual.containers import Vertical, Horizontal, Grid
from textual.message import Message
from textual import events
import os
import subprocess
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Try to import git
try:
    import git
    HAS_GIT = True
except ImportError:
    HAS_GIT = False
    git = None

# Try to import integrations
try:
    from ..integrations.ollama import OllamaService, get_ollama_service
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
    OllamaService = None
    get_ollama_service = None

# Try to import services
try:
    from ..services.event_bus import EventBus, EventType, get_event_bus
    from ..services.config_store import get_config_store
    HAS_SERVICES = True
except ImportError:
    HAS_SERVICES = False
    EventBus = None
    EventType = None
    get_event_bus = None
    get_config_store = None


class ModelChanged(Message):
    """Message emitted when model changes."""
    def __init__(self, model_name: str, provider: str):
        self.model_name = model_name
        self.provider = provider
        super().__init__()


class CodeAssistantPanel(Widget):
    """
    Integrated Aider coding assistant for SparkPlug.
    
    Features:
    - Proper async streaming for Aider responses
    - Model selector (Aider cloud vs Ollama local)
    - Context window management
    - Code diff preview before applying changes
    """
    
    DEFAULT_CSS = """
    CodeAssistantPanel {
        height: 100%;
    }
    
    CodeAssistantPanel Grid {
        grid-size: 2;
        grid-columns: 1fr 2fr;
        height: 100%;
    }
    
    CodeAssistantPanel #file-panel {
        height: 100%;
        padding: 1;
    }
    
    CodeAssistantPanel #code-panel {
        height: 100%;
        padding: 1;
    }
    
    CodeAssistantPanel #model-selector-section {
        height: auto;
        padding: 1;
        border: solid #444;
        margin-bottom: 1;
    }
    
    CodeAssistantPanel #model-select {
        width: 100%;
    }
    
    CodeAssistantPanel #aider-section {
        height: auto;
        min-height: 20;
    }
    
    CodeAssistantPanel #aider-output {
        height: 15;
        border: solid #333;
    }
    
    CodeAssistantPanel #code-editor {
        height: 1fr;
        min-height: 10;
    }
    
    CodeAssistantPanel .streaming-indicator {
        color: #00f5d4;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_file: Optional[Path] = None
        self.project_root = Path.cwd()
        self.aider_process: Optional[asyncio.subprocess.Process] = None
        self.aider_output: List[str] = []
        self.git_repo = None
        self._ollama_service: Optional[Any] = None
        self._event_bus = None
        self._config_store = None
        self._streaming = False
        self._current_model = "gpt-4"
        self._current_provider = "openai"  # "openai", "anthropic", "ollama"
        self._ollama_check_task = None  # Store reference to prevent GC

        # Initialize services
        if HAS_OLLAMA and get_ollama_service is not None:
            self._ollama_service = get_ollama_service()

        if HAS_SERVICES and get_event_bus is not None and get_config_store is not None:
            self._event_bus = get_event_bus()
            self._config_store = get_config_store()
            
            # Load saved model preference
            if self._config_store:
                self._current_model = self._config_store.get("default_model", "gpt-4")
                self._current_provider = "ollama" if self._config_store.get("use_local_inference") else "openai"
        
        # Initialize git
        if HAS_GIT:
            self._init_git_repo()

    def _init_git_repo(self):
        """Initialize git repository connection."""
        if not HAS_GIT or git is None:
            self.git_repo = None
            return

        try:
            self.git_repo = git.Repo(self.project_root)
        except Exception:
            self.git_repo = None

    def compose(self) -> ComposeResult:
        with Grid():
            # Left panel - File browser and git status
            with Vertical(id="file-panel"):
                yield Label("PROJECT FILES", classes="section-header")

                # Git status
                with Horizontal(id="git-status"):
                    yield Button("Git Status", variant="default", id="git-status-btn")
                    yield Button("Git Diff", variant="default", id="git-diff-btn")
                    yield Button("Commit", variant="primary", id="git-commit-btn")

                # File tree
                yield Tree("Project Root", id="file-tree")

                # Quick file operations
                with Horizontal(id="file-ops"):
                    yield Button("Open File", variant="default", id="open-file-btn")
                    yield Button("New File", variant="default", id="new-file-btn")

            # Right panel - Code editor and Aider interface
            with Vertical(id="code-panel"):
                # Model selector section
                with Vertical(id="model-selector-section"):
                    yield Label("AI MODEL", classes="section-header")
                    
                    with Horizontal():
                        yield Select(
                            options=self._get_model_options(),
                            value=self._current_model,
                            id="model-select"
                        )
                        yield Switch(
                            value=self._current_provider == "ollama",
                            id="local-inference-switch"
                        )
                        yield Label("Local (Ollama)", id="local-label")
                    
                    # Model status
                    yield Static(self._get_model_status(), id="model-status")
                
                # Current file header
                with Horizontal(id="file-header"):
                    yield Label("CURRENT FILE: None", id="current-file-label")
                    yield Button("Save", variant="success", id="save-file-btn")

                # Code editor
                yield TextArea("", id="code-editor", read_only=False)

                # Aider chat interface
                with Vertical(id="aider-section"):
                    yield Label("AI CODE ASSISTANT", classes="section-header")

                    # Aider status with streaming indicator
                    with Horizontal():
                        yield Static("Status: Ready", id="aider-status")
                        yield Static("", id="streaming-indicator", classes="streaming-indicator")

                    # Aider input
                    with Horizontal():
                        yield Input(
                            placeholder="Ask the AI to help with code...",
                            id="aider-input"
                        )
                        yield Button("Send", variant="primary", id="send-aider-btn")

                    # Aider output with streaming support
                    yield TextArea("", id="aider-output", read_only=True)

                    # Aider controls
                    with Horizontal(id="aider-controls"):
                        yield Button("Start Aider", variant="success", id="start-aider-btn")
                        yield Button("Stop Aider", variant="warning", id="stop-aider-btn")
                        yield Button("Clear Chat", variant="default", id="clear-chat-btn")
                        yield Button("Preview Diff", variant="default", id="preview-diff-btn")

    def _get_model_options(self) -> List[tuple]:
        """Get available model options."""
        options = [
            # Cloud models
            ("gpt-4", "GPT-4 (OpenAI)"),
            ("gpt-3.5-turbo", "GPT-3.5 Turbo (OpenAI)"),
            ("claude-3-opus", "Claude 3 Opus (Anthropic)"),
            ("claude-3-sonnet", "Claude 3 Sonnet (Anthropic)"),
        ]
        
        # Add Ollama models if available
        if self._ollama_service and HAS_OLLAMA:
            # These will be populated when Ollama is checked
            options.extend([
                ("ollama:llama2", "Llama 2 (Local)"),
                ("ollama:codellama", "Code Llama (Local)"),
                ("ollama:mistral", "Mistral (Local)"),
                ("ollama:deepseek-coder", "DeepSeek Coder (Local)"),
            ])
        
        return options
    
    def _get_model_status(self) -> str:
        """Get current model status text."""
        if self._current_provider == "ollama":
            return f"🖥️ Local: {self._current_model}"
        else:
            return f"☁️ Cloud: {self._current_model}"

    def on_mount(self):
        """Initialize the code assistant when mounted."""
        self.load_file_tree()
        self.update_git_status()
        
        # Check Ollama availability (schedule after widget is ready)
        if self._ollama_service:
            self._ollama_check_task = asyncio.create_task(self._check_ollama_models())

    async def _check_ollama_models(self):
        """Check available Ollama models and update selector."""
        if not self._ollama_service:
            return

        # Wait a bit to ensure widgets are ready
        await asyncio.sleep(0.1)

        try:
            if await self._ollama_service.is_available():
                models = await self._ollama_service.list_models()

                # Update model selector with actual Ollama models
                try:
                    select = self.query_one("#model-select", Select)
                    current_options = list(select._options)

                    # Add discovered Ollama models
                    for model in models:
                        option = (f"ollama:{model.name}", f"{model.display_name} (Local)")
                        if option not in current_options:
                            current_options.append(option)

                    select.set_options(current_options)
                except Exception as e:
                    logger.warning(f"Failed to update model selector: {e}")
        except Exception as e:
            logger.warning(f"Failed to check Ollama models: {e}")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "git-status-btn":
            await self.show_git_status()
        elif button_id == "git-diff-btn":
            await self.show_git_diff()
        elif button_id == "git-commit-btn":
            await self.git_commit()
        elif button_id == "open-file-btn":
            await self.open_selected_file()
        elif button_id == "new-file-btn":
            await self.create_new_file()
        elif button_id == "save-file-btn":
            await self.save_current_file()
        elif button_id == "send-aider-btn":
            await self.send_to_aider()
        elif button_id == "start-aider-btn":
            await self.start_aider()
        elif button_id == "stop-aider-btn":
            await self.stop_aider()
        elif button_id == "clear-chat-btn":
            await self.clear_aider_chat()
        elif button_id == "preview-diff-btn":
            await self.preview_diff()

    async def on_select_changed(self, event: Select.Changed) -> None:
        """Handle model selection change."""
        if event.select.id == "model-select":
            model_value = str(event.value)
            
            if model_value.startswith("ollama:"):
                self._current_provider = "ollama"
                self._current_model = model_value.replace("ollama:", "")
            else:
                self._current_provider = "openai" if "gpt" in model_value else "anthropic"
                self._current_model = model_value
            
            # Update status display
            status = self.query_one("#model-status", Static)
            status.update(self._get_model_status())
            
            # Save preference
            if self._config_store:
                self._config_store.set("default_model", self._current_model)
                self._config_store.set("use_local_inference", self._current_provider == "ollama")
            
            # Post message
            self.post_message(ModelChanged(self._current_model, self._current_provider))
            self.app.notify(f"Model changed to: {self._current_model}", title="Model")

    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle local inference switch."""
        if event.switch.id == "local-inference-switch":
            if event.value:
                self._current_provider = "ollama"
                # Switch to first available Ollama model
                select = self.query_one("#model-select", Select)
                for value, _ in select._options:
                    if str(value).startswith("ollama:"):
                        select.value = value
                        break
            else:
                self._current_provider = "openai"
                select = self.query_one("#model-select", Select)
                select.value = "gpt-4"
            
            # Update status
            status = self.query_one("#model-status", Static)
            status.update(self._get_model_status())

    async def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle file selection from tree."""
        if hasattr(event.node, 'data') and event.node.data:
            file_path = event.node.data
            if file_path.is_file():
                await self.load_file(file_path)

    def load_file_tree(self):
        """Load the project file tree."""
        tree = self.query_one("#file-tree", Tree)
        tree.clear()

        root_node = tree.root
        root_node.label = self.project_root.name
        root_node.data = self.project_root

        self._load_directory(root_node, self.project_root)

    def _load_directory(self, parent_node, directory: Path):
        """Recursively load directory structure."""
        try:
            for item in sorted(directory.iterdir()):
                if item.name.startswith('.') or item.name in ['__pycache__', 'node_modules', '.git', 'venv', '.venv']:
                    continue

                if item.is_file():
                    if item.suffix in ['.py', '.js', '.ts', '.tsx', '.md', '.txt', '.json', '.yaml', '.yml', '.toml', '.css', '.html']:
                        node = parent_node.add(item.name)
                        node.data = item
                elif item.is_dir():
                    dir_node = parent_node.add(f"📁 {item.name}")
                    dir_node.data = item
                    if len(str(item.relative_to(self.project_root)).split(os.sep)) <= 3:
                        self._load_directory(dir_node, item)
        except PermissionError:
            pass

    async def load_file(self, file_path: Path):
        """Load a file into the code editor."""
        try:
            content = file_path.read_text(encoding='utf-8')
            editor = self.query_one("#code-editor", TextArea)
            editor.load_text(content)

            label = self.query_one("#current-file-label", Label)
            label.update(f"CURRENT FILE: {file_path.relative_to(self.project_root)}")

            self.current_file = file_path
            self.app.notify(f"Loaded {file_path.name}", title="File", severity="information")

            # Emit event
            if self._event_bus and EventType is not None:
                await self._event_bus.emit_async(
                    EventType.FILE_OPENED,
                    data={"file": str(file_path)},
                    source="code_assistant"
                )

        except Exception as e:
            self.app.notify(f"Failed to load file: {e}", title="File", severity="error")

    async def save_current_file(self):
        """Save the current file."""
        if not self.current_file:
            self.app.notify("No file currently open", title="File", severity="warning")
            return

        try:
            editor = self.query_one("#code-editor", TextArea)
            content = editor.text

            self.current_file.write_text(content, encoding='utf-8')
            self.app.notify(f"Saved {self.current_file.name}", title="File", severity="information")

            self.update_git_status()
            
            # Emit event
            if self._event_bus and EventType is not None:
                await self._event_bus.emit_async(
                    EventType.FILE_SAVED,
                    data={"file": str(self.current_file)},
                    source="code_assistant"
                )

        except Exception as e:
            self.app.notify(f"Failed to save file: {e}", title="File", severity="error")

    async def open_selected_file(self):
        """Open the currently selected file in tree."""
        tree = self.query_one("#file-tree", Tree)
        if tree.cursor_node and hasattr(tree.cursor_node, 'data'):
            file_path = tree.cursor_node.data
            if file_path and file_path.is_file():
                await self.load_file(file_path)

    async def create_new_file(self):
        """Create a new file."""
        self.app.notify("New file creation - feature coming soon!", title="File", severity="information")

    def update_git_status(self):
        """Update git status display."""
        if not self.git_repo:
            return

        try:
            status = self.git_repo.git.status('--porcelain')
            if status:
                self.app.notify(f"Git changes: {len(status.splitlines())} files", title="Git", severity="information")
        except Exception:
            pass

    async def show_git_status(self):
        """Show detailed git status."""
        if not self.git_repo:
            self.app.notify("Not a git repository", title="Git", severity="warning")
            return

        try:
            status = self.git_repo.git.status()
            self.app.notify(status, title="Git Status", severity="information")
        except Exception as e:
            self.app.notify(f"Git status error: {e}", title="Git", severity="error")

    async def show_git_diff(self):
        """Show git diff."""
        if not self.git_repo:
            self.app.notify("Not a git repository", title="Git", severity="warning")
            return

        try:
            diff = self.git_repo.git.diff()
            if diff:
                self.app.notify(diff[:500] + "..." if len(diff) > 500 else diff,
                              title="Git Diff", severity="information")
            else:
                self.app.notify("No changes to show", title="Git Diff", severity="information")
        except Exception as e:
            self.app.notify(f"Git diff error: {e}", title="Git", severity="error")

    async def git_commit(self):
        """Commit changes to git."""
        if not self.git_repo:
            self.app.notify("Not a git repository", title="Git", severity="warning")
            return
        self.app.notify("Git commit - feature coming soon!", title="Git", severity="information")

    async def start_aider(self):
        """Start the Aider process with proper async streaming."""
        if self.aider_process and self.aider_process.returncode is None:
            self.app.notify("Aider is already running", title="Aider", severity="warning")
            return

        try:
            # Determine model flag based on current selection
            model_flag = []
            if self._current_provider == "ollama":
                model_flag = ["--model", f"ollama/{self._current_model}"]
            elif self._current_provider == "anthropic":
                model_flag = ["--model", self._current_model]
            else:
                model_flag = ["--model", self._current_model]
            
            # Start Aider process with async subprocess
            self.aider_process = await asyncio.create_subprocess_exec(
                'aider',
                *model_flag,
                '--no-auto-commits',
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root
            )

            # Update status
            status_label = self.query_one("#aider-status", Static)
            status_label.update(f"Status: Running ({self._current_model})")

            self.app.notify("Aider started successfully", title="Aider", severity="information")
            
            # Emit event
            if self._event_bus and EventType is not None:
                await self._event_bus.emit_async(
                    EventType.AIDER_STARTED,
                    data={"model": self._current_model},
                    source="code_assistant"
                )

            # Start reading output in background
            asyncio.create_task(self._read_aider_output())

        except FileNotFoundError:
            self.app.notify("Aider not found. Please install with: pip install aider-chat",
                           title="Aider", severity="error")
        except Exception as e:
            self.app.notify(f"Failed to start Aider: {e}", title="Aider", severity="error")

    async def _read_aider_output(self):
        """Read Aider output asynchronously with streaming."""
        if not self.aider_process or not self.aider_process.stdout:
            return
        
        output_area = self.query_one("#aider-output", TextArea)
        streaming_indicator = self.query_one("#streaming-indicator", Static)
        
        try:
            while True:
                # Read a chunk of data
                chunk = await self.aider_process.stdout.read(100)
                
                if not chunk:
                    break
                
                # Decode and append to output
                text = chunk.decode('utf-8', errors='replace')
                self.aider_output.append(text)
                
                # Update display with streaming indicator
                self._streaming = True
                streaming_indicator.update("⚡ Streaming...")
                
                current_output = output_area.text
                output_area.load_text(current_output + text)
                
                # Scroll to bottom
                output_area.scroll_end()
                
                # Emit event for streaming response
                if self._event_bus and EventType is not None:
                    self._event_bus.emit(
                        EventType.AIDER_RESPONSE,
                        data={"chunk": text, "streaming": True},
                        source="code_assistant"
                    )
        
        except Exception as e:
            logger.error(f"Error reading Aider output: {e}")
        
        finally:
            self._streaming = False
            streaming_indicator.update("")

    async def stop_aider(self):
        """Stop the Aider process."""
        if self.aider_process and self.aider_process.returncode is None:
            self.aider_process.terminate()
            await self.aider_process.wait()

            status_label = self.query_one("#aider-status", Static)
            status_label.update("Status: Stopped")

            self.app.notify("Aider stopped", title="Aider", severity="information")
            
            # Emit event
            if self._event_bus and EventType is not None:
                await self._event_bus.emit_async(
                    EventType.AIDER_STOPPED,
                    data={},
                    source="code_assistant"
                )
        else:
            self.app.notify("Aider is not running", title="Aider", severity="warning")

    async def send_to_aider(self):
        """Send a message to Aider with streaming response."""
        input_field = self.query_one("#aider-input", Input)
        message = input_field.value.strip()

        if not message:
            self.app.notify("Please enter a message", title="Aider", severity="warning")
            return

        # If Aider process is running, send to it
        if self.aider_process and self.aider_process.returncode is None and self.aider_process.stdin is not None:
            try:
                self.aider_process.stdin.write((message + '\n').encode())
                await self.aider_process.stdin.drain()

                output_area = self.query_one("#aider-output", TextArea)
                current_output = output_area.text
                output_area.load_text(f"{current_output}\n> {message}\n")

                input_field.value = ""
                self.app.notify(f"Sent to Aider", title="Aider", severity="information")

            except Exception as e:
                self.app.notify(f"Failed to send to Aider: {e}", title="Aider", severity="error")
        
        # If using Ollama directly (without Aider)
        elif self._current_provider == "ollama" and self._ollama_service:
            await self._chat_with_ollama(message)
            input_field.value = ""
        
        else:
            self.app.notify("Aider is not running. Start Aider first or use local Ollama.", 
                          title="Aider", severity="warning")

    async def _chat_with_ollama(self, message: str):
        """Chat directly with Ollama for code assistance."""
        if not self._ollama_service:
            return
        
        output_area = self.query_one("#aider-output", TextArea)
        streaming_indicator = self.query_one("#streaming-indicator", Static)
        
        # Add user message to output
        current_output = output_area.text
        output_area.load_text(f"{current_output}\n> {message}\n")
        
        # Build context with current file if open
        system_prompt = """You are a helpful coding assistant. Analyze code, suggest improvements, 
        fix bugs, and help with programming tasks. Be concise and provide code examples when helpful."""
        
        if self.current_file:
            editor = self.query_one("#code-editor", TextArea)
            code_context = editor.text[:2000]  # Limit context size
            system_prompt += f"\n\nCurrent file ({self.current_file.name}):\n```\n{code_context}\n```"
        
        # Stream response
        self._streaming = True
        streaming_indicator.update("⚡ Streaming...")
        
        try:
            response_text = ""
            async for chunk in self._ollama_service.chat_stream(
                self._current_model,
                message,
                system=system_prompt
            ):
                response_text += chunk
                current = output_area.text
                output_area.load_text(current + chunk)
                output_area.scroll_end()
            
            # Add newline after response
            output_area.load_text(output_area.text + "\n")
            
        except Exception as e:
            self.app.notify(f"Ollama error: {e}", title="Error", severity="error")
        
        finally:
            self._streaming = False
            streaming_indicator.update("")

    async def clear_aider_chat(self):
        """Clear the Aider chat history."""
        output_area = self.query_one("#aider-output", TextArea)
        output_area.load_text("")
        self.aider_output.clear()
        
        # Clear Ollama history too
        if self._ollama_service:
            self._ollama_service.clear_chat_history()
        
        self.app.notify("Chat cleared", title="Aider", severity="information")

    async def preview_diff(self):
        """Preview code diff before applying changes."""
        if not self.current_file:
            self.app.notify("No file open to diff", title="Diff", severity="warning")
            return
        
        if not self.git_repo:
            self.app.notify("Git not available for diff", title="Diff", severity="warning")
            return
        
        try:
            # Get diff for current file
            diff = self.git_repo.git.diff(str(self.current_file))
            
            if diff:
                # Show diff in output area
                output_area = self.query_one("#aider-output", TextArea)
                output_area.load_text(f"=== DIFF PREVIEW ===\n{diff}\n===================")
                self.app.notify("Diff preview shown", title="Diff", severity="information")
            else:
                self.app.notify("No changes to preview", title="Diff", severity="information")
        
        except Exception as e:
            self.app.notify(f"Diff error: {e}", title="Diff", severity="error")
