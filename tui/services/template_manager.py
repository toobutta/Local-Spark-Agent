"""
Template Manager Service.

Provides project templates for quick workspace setup.
"""

import json
import yaml
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid
import logging

from .config_store import get_config_store
from .event_bus import get_event_bus, EventType

logger = logging.getLogger(__name__)


@dataclass
class AgentTemplate:
    """Template for an agent configuration."""
    name: str
    type: str  # e.g., "research", "coding", "orchestrator"
    model: str = "gpt-4"
    tools: List[str] = field(default_factory=list)
    memory_enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPServerTemplate:
    """Template for an MCP server configuration."""
    name: str
    type: str  # e.g., "filesystem", "database", "api"
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectTemplate:
    """
    Project template for workspace initialization.
    
    Contains pre-configured agents, MCP servers, and settings.
    """
    id: str
    name: str
    description: str
    category: str  # e.g., "research", "development", "data", "custom"
    icon: str = "📁"
    
    # Agent configurations
    agents: List[AgentTemplate] = field(default_factory=list)
    
    # MCP server configurations
    mcp_servers: List[MCPServerTemplate] = field(default_factory=list)
    
    # Default tools to enable
    tools: List[str] = field(default_factory=list)
    
    # Workspace settings
    settings: Dict[str, Any] = field(default_factory=dict)
    
    # Template metadata
    version: str = "1.0.0"
    author: str = "SparkPlug"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    is_builtin: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "icon": self.icon,
            "agents": [asdict(a) for a in self.agents],
            "mcp_servers": [asdict(m) for m in self.mcp_servers],
            "tools": self.tools,
            "settings": self.settings,
            "version": self.version,
            "author": self.author,
            "created_at": self.created_at,
            "is_builtin": self.is_builtin,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectTemplate':
        """Create from dictionary."""
        if "name" not in data:
            raise ValueError("ProjectTemplate.from_dict missing required field 'name'")

        agents = [AgentTemplate(**a) for a in data.get("agents", [])]
        mcp_servers = [MCPServerTemplate(**m) for m in data.get("mcp_servers", [])]

        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            name=data["name"],
            description=data.get("description", ""),
            category=data.get("category", "custom"),
            icon=data.get("icon", "📁"),
            agents=agents,
            mcp_servers=mcp_servers,
            tools=data.get("tools", []),
            settings=data.get("settings", {}),
            version=data.get("version", "1.0.0"),
            author=data.get("author", "User"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            is_builtin=data.get("is_builtin", False),
        )


# ==================== Built-in Templates ====================

BUILTIN_TEMPLATES = [
    ProjectTemplate(
        id="research-agent",
        name="Research Agent",
        description="Memory-enabled research agent with web search and document analysis capabilities.",
        category="research",
        icon="🔬",
        agents=[
            AgentTemplate(
                name="Research Assistant",
                type="research",
                model="gpt-4",
                tools=["web_search", "document_reader", "note_taker"],
                memory_enabled=True,
                config={
                    "memory_type": "mem0",
                    "search_provider": "brave",
                    "max_results": 10,
                }
            ),
        ],
        mcp_servers=[
            MCPServerTemplate(
                name="Memory Service",
                type="memory",
                config={"backend": "mem0"}
            ),
        ],
        tools=["brave_search", "document_reader", "note_taker", "summarizer"],
        settings={
            "auto_save_notes": True,
            "memory_retention_days": 30,
        },
        is_builtin=True,
    ),
    
    ProjectTemplate(
        id="code-assistant",
        name="Code Assistant",
        description="Aider-powered coding agent with git integration and file operations.",
        category="development",
        icon="💻",
        agents=[
            AgentTemplate(
                name="Aider Coder",
                type="coding",
                model="gpt-4",
                tools=["aider", "git", "file_operations"],
                memory_enabled=True,
                config={
                    "aider_mode": "architect",
                    "auto_commit": False,
                    "diff_preview": True,
                }
            ),
        ],
        mcp_servers=[
            MCPServerTemplate(
                name="Filesystem",
                type="filesystem",
                config={"watch_changes": True}
            ),
            MCPServerTemplate(
                name="GitHub",
                type="github",
                config={"auto_pr": False}
            ),
        ],
        tools=["aider", "git", "file_read", "file_write", "grep", "tree"],
        settings={
            "auto_format": True,
            "lint_on_save": True,
            "git_auto_stage": False,
        },
        is_builtin=True,
    ),
    
    ProjectTemplate(
        id="data-pipeline",
        name="Data Pipeline",
        description="Orchestrated data processing with monitoring and communication.",
        category="data",
        icon="📊",
        agents=[
            AgentTemplate(
                name="Pipeline Orchestrator",
                type="orchestrator",
                model="gpt-4",
                tools=["workflow_engine", "scheduler", "monitor"],
                memory_enabled=True,
                config={
                    "orchestrator": "langgraph",
                    "retry_policy": "exponential",
                    "max_retries": 3,
                }
            ),
            AgentTemplate(
                name="Data Processor",
                type="worker",
                model="gpt-3.5-turbo",
                tools=["pandas", "numpy", "sql"],
                memory_enabled=False,
                config={
                    "batch_size": 1000,
                    "parallel_workers": 4,
                }
            ),
        ],
        mcp_servers=[
            MCPServerTemplate(
                name="PostgreSQL",
                type="database",
                config={"pool_size": 10}
            ),
            MCPServerTemplate(
                name="Redis",
                type="cache",
                config={"ttl": 3600}
            ),
        ],
        tools=["sql_query", "data_transform", "chart_generator", "export"],
        settings={
            "log_level": "INFO",
            "metrics_enabled": True,
            "alert_on_failure": True,
        },
        is_builtin=True,
    ),
    
    ProjectTemplate(
        id="multi-agent-crew",
        name="Multi-Agent Crew",
        description="CrewAI-powered team of specialized agents working together.",
        category="research",
        icon="👥",
        agents=[
            AgentTemplate(
                name="Project Manager",
                type="manager",
                model="gpt-4",
                tools=["task_planner", "delegation", "status_tracker"],
                memory_enabled=True,
                config={
                    "framework": "crewai",
                    "role": "manager",
                }
            ),
            AgentTemplate(
                name="Researcher",
                type="research",
                model="gpt-4",
                tools=["web_search", "document_reader"],
                memory_enabled=True,
                config={
                    "framework": "crewai",
                    "role": "researcher",
                }
            ),
            AgentTemplate(
                name="Writer",
                type="content",
                model="gpt-4",
                tools=["text_generator", "editor"],
                memory_enabled=True,
                config={
                    "framework": "crewai",
                    "role": "writer",
                }
            ),
        ],
        mcp_servers=[
            MCPServerTemplate(
                name="Memory Service",
                type="memory",
                config={"shared": True}
            ),
        ],
        tools=["web_search", "document_reader", "text_generator", "task_planner"],
        settings={
            "crew_verbose": True,
            "shared_memory": True,
            "max_iterations": 10,
        },
        is_builtin=True,
    ),
    
    ProjectTemplate(
        id="local-inference",
        name="Local Inference",
        description="Ollama-powered local model inference with GPU acceleration.",
        category="development",
        icon="🖥️",
        agents=[
            AgentTemplate(
                name="Local Assistant",
                type="chat",
                model="llama2",
                tools=["chat", "code_completion"],
                memory_enabled=True,
                config={
                    "provider": "ollama",
                    "host": "http://localhost:11434",
                    "context_window": 4096,
                }
            ),
        ],
        mcp_servers=[],
        tools=["chat", "code_completion", "summarizer"],
        settings={
            "gpu_layers": -1,  # Use all available GPU layers
            "temperature": 0.7,
            "max_tokens": 2048,
        },
        is_builtin=True,
    ),
    
    ProjectTemplate(
        id="blank",
        name="Blank Project",
        description="Empty workspace for custom configuration.",
        category="custom",
        icon="📄",
        agents=[],
        mcp_servers=[],
        tools=[],
        settings={},
        is_builtin=True,
    ),
]


class TemplateManager:
    """
    Manages project templates for workspace initialization.
    
    Features:
    - Built-in templates for common use cases
    - Custom template creation and management
    - Template import/export (JSON/YAML)
    - Template versioning
    
    Usage:
        manager = TemplateManager()
        
        # List templates
        templates = manager.list_templates()
        
        # Get template
        template = manager.get_template("research-agent")
        
        # Apply template to workspace
        await manager.apply_template("research-agent", workspace_id)
    """
    
    _instance: Optional['TemplateManager'] = None
    
    TEMPLATES_DIR = Path.home() / ".sparkplug" / "templates"
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._templates: Dict[str, ProjectTemplate] = {}
        self._config_store = get_config_store()
        self._event_bus = get_event_bus()
        
        # Ensure templates directory exists
        self.TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load templates
        self._load_builtin_templates()
        self._load_custom_templates()
    
    def _load_builtin_templates(self):
        """Load built-in templates."""
        for template in BUILTIN_TEMPLATES:
            self._templates[template.id] = template
    
    def _load_custom_templates(self):
        """Load custom templates from disk."""
        for file_path in self.TEMPLATES_DIR.glob("*.yaml"):
            try:
                with open(file_path, 'r') as f:
                    data = yaml.safe_load(f)

                if isinstance(data, dict):
                    template = ProjectTemplate.from_dict(data)
                else:
                    logger.error(f"Template file {file_path} does not contain a valid dictionary")
                    continue
                self._templates[template.id] = template
                logger.info(f"Loaded custom template: {template.name}")
            except Exception as e:
                logger.error(f"Failed to load template {file_path}: {e}")
        
        for file_path in self.TEMPLATES_DIR.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                if isinstance(data, dict):
                    template = ProjectTemplate.from_dict(data)
                else:
                    logger.error(f"Template file {file_path} does not contain a valid dictionary")
                    continue
                self._templates[template.id] = template
                logger.info(f"Loaded custom template: {template.name}")
            except Exception as e:
                logger.error(f"Failed to load template {file_path}: {e}")
    
    # ==================== Template Queries ====================
    
    def list_templates(self, category: Optional[str] = None) -> List[ProjectTemplate]:
        """
        List all available templates.
        
        Args:
            category: Filter by category (optional)
            
        Returns:
            List of templates
        """
        templates = list(self._templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        # Sort: built-in first, then by name
        templates.sort(key=lambda t: (not t.is_builtin, t.name))
        
        return templates
    
    def get_template(self, template_id: str) -> Optional[ProjectTemplate]:
        """Get a template by ID."""
        return self._templates.get(template_id)
    
    def get_categories(self) -> List[str]:
        """Get all unique template categories."""
        categories = set(t.category for t in self._templates.values())
        return sorted(categories)
    
    def search_templates(self, query: str) -> List[ProjectTemplate]:
        """
        Search templates by name or description.
        
        Args:
            query: Search query
            
        Returns:
            List of matching templates
        """
        query = query.lower()
        results = []
        
        for template in self._templates.values():
            if query in template.name.lower() or query in template.description.lower():
                results.append(template)
        
        return results
    
    # ==================== Template Management ====================
    
    def create_template(
        self,
        name: str,
        description: str,
        category: str = "custom",
        **kwargs
    ) -> ProjectTemplate:
        """
        Create a new custom template.
        
        Args:
            name: Template name
            description: Template description
            category: Template category
            **kwargs: Additional template properties
            
        Returns:
            The created template
        """
        template_id = str(uuid.uuid4())[:8]
        
        template = ProjectTemplate(
            id=template_id,
            name=name,
            description=description,
            category=category,
            author="User",
            is_builtin=False,
            **kwargs
        )
        
        self._templates[template_id] = template
        self._save_template(template)
        
        logger.info(f"Created template: {name}")
        return template
    
    def save_workspace_as_template(
        self,
        workspace_id: str,
        name: str,
        description: str = ""
    ) -> Optional[ProjectTemplate]:
        """
        Save current workspace configuration as a template.
        
        Args:
            workspace_id: The workspace to save
            name: Template name
            description: Template description
            
        Returns:
            The created template, or None if workspace not found
        """
        # Get workspace configuration
        workspace = self._config_store.get_workspace(workspace_id)
        if not workspace:
            return None
        
        # Create template from workspace
        template = ProjectTemplate(
            id=str(uuid.uuid4())[:8],
            name=name,
            description=description or f"Template from {workspace.name}",
            category="custom",
            agents=[],  # Would need to extract from workspace state
            mcp_servers=[],  # Would need to extract from workspace state
            tools=[],
            settings={},
            author="User",
            is_builtin=False,
        )
        
        self._templates[template.id] = template
        self._save_template(template)
        
        return template
    
    def delete_template(self, template_id: str) -> bool:
        """
        Delete a custom template.
        
        Args:
            template_id: The template ID to delete
            
        Returns:
            True if deleted successfully
        """
        if template_id not in self._templates:
            return False
        
        template = self._templates[template_id]
        
        # Can't delete built-in templates
        if template.is_builtin:
            logger.warning(f"Cannot delete built-in template: {template.name}")
            return False
        
        # Remove from memory
        del self._templates[template_id]
        
        # Remove from disk
        for ext in [".yaml", ".json"]:
            file_path = self.TEMPLATES_DIR / f"{template_id}{ext}"
            if file_path.exists():
                file_path.unlink()
        
        logger.info(f"Deleted template: {template.name}")
        return True
    
    def _save_template(self, template: ProjectTemplate):
        """Save a template to disk."""
        file_path = self.TEMPLATES_DIR / f"{template.id}.yaml"
        
        with open(file_path, 'w') as f:
            yaml.dump(template.to_dict(), f, default_flow_style=False)
    
    # ==================== Template Application ====================
    
    async def apply_template(
        self,
        template_id: str,
        workspace_id: str
    ) -> bool:
        """
        Apply a template to a workspace.
        
        Args:
            template_id: The template to apply
            workspace_id: The target workspace
            
        Returns:
            True if applied successfully
        """
        template = self.get_template(template_id)
        if not template:
            logger.warning(f"Template not found: {template_id}")
            return False
        
        workspace = self._config_store.get_workspace(workspace_id)
        if not workspace:
            logger.warning(f"Workspace not found: {workspace_id}")
            return False
        
        # Apply template settings to workspace
        # In a full implementation, this would:
        # 1. Configure agents based on template.agents
        # 2. Set up MCP servers based on template.mcp_servers
        # 3. Enable tools based on template.tools
        # 4. Apply settings based on template.settings
        
        # For now, just update workspace with template info
        self._config_store.update_workspace(
            workspace_id,
            active_agents=[a.name for a in template.agents],
            mcp_servers=[m.name for m in template.mcp_servers],
        )
        
        logger.info(f"Applied template '{template.name}' to workspace '{workspace.name}'")
        return True
    
    # ==================== Import/Export ====================
    
    def export_template(self, template_id: str, format: str = "yaml") -> Optional[str]:
        """
        Export a template as YAML or JSON string.
        
        Args:
            template_id: The template to export
            format: "yaml" or "json"
            
        Returns:
            Exported string, or None if not found
        """
        template = self.get_template(template_id)
        if not template:
            return None
        
        data = template.to_dict()
        
        if format == "json":
            return json.dumps(data, indent=2)
        else:
            result = yaml.dump(data, default_flow_style=False)
            return result if isinstance(result, str) else str(result)
    
    def import_template(self, content: str, format: str = "yaml") -> Optional[ProjectTemplate]:
        """
        Import a template from YAML or JSON string.
        
        Args:
            content: Template content
            format: "yaml" or "json"
            
        Returns:
            Imported template, or None if invalid
        """
        try:
            if format == "json":
                data = json.loads(content)
            else:
                data = yaml.safe_load(content)

            if not isinstance(data, dict):
                logger.error("Template content does not contain a valid dictionary")
                return None

            # Generate new ID to avoid conflicts
            data["id"] = str(uuid.uuid4())[:8]
            data["is_builtin"] = False
            data["author"] = "Imported"

            template = ProjectTemplate.from_dict(data)
            self._templates[template.id] = template
            self._save_template(template)
            
            logger.info(f"Imported template: {template.name}")
            return template
            
        except Exception as e:
            logger.error(f"Failed to import template: {e}")
            return None


# Global singleton accessor
def get_template_manager() -> TemplateManager:
    """Get the global TemplateManager instance."""
    return TemplateManager()


