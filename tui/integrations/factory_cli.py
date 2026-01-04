"""
Factory CLI Integration Adapter.

Provides integration with Factory CLI for autonomous agent workflows.
"""

import asyncio
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, AsyncIterator
import logging

logger = logging.getLogger(__name__)

# Try to import services
try:
    from ..services.event_bus import EventBus, EventType, get_event_bus
    HAS_EVENT_BUS = True
except ImportError:
    HAS_EVENT_BUS = False
    EventBus = None
    EventType = None
    get_event_bus = None


@dataclass
class WorkflowStep:
    """A step in a Factory CLI workflow."""
    name: str
    type: str  # e.g., "agent", "tool", "condition"
    config: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)


@dataclass
class Workflow:
    """A Factory CLI workflow definition."""
    id: str
    name: str
    description: str = ""
    steps: List[WorkflowStep] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": [
                {
                    "name": s.name,
                    "type": s.type,
                    "config": s.config,
                    "depends_on": s.depends_on,
                }
                for s in self.steps
            ],
            "variables": self.variables,
            "created_at": self.created_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Workflow':
        """Create from dictionary."""
        steps = [
            WorkflowStep(
                name=s["name"],
                type=s["type"],
                config=s.get("config", {}),
                depends_on=s.get("depends_on", []),
            )
            for s in data.get("steps", [])
        ]
        
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            steps=steps,
            variables=data.get("variables", {}),
            created_at=data.get("created_at", datetime.now().isoformat()),
        )


@dataclass
class WorkflowRun:
    """Status of a workflow run."""
    workflow_id: str
    run_id: str
    status: str  # "pending", "running", "completed", "failed", "cancelled"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    current_step: Optional[str] = None
    step_statuses: Dict[str, str] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class FactoryCLIAdapter:
    """
    Adapter for Factory CLI integration.
    
    Features:
    - Detect Factory CLI installation
    - Parse Factory CLI configuration
    - Execute Factory CLI commands
    - Stream Factory CLI output
    - Sync workflow states
    
    Usage:
        adapter = FactoryCLIAdapter()
        
        # Check if Factory CLI is available
        if adapter.is_available():
            # List workflows
            workflows = await adapter.list_workflows()
            
            # Run a workflow
            run = await adapter.run_workflow("my-workflow")
            
            # Stream logs
            async for line in adapter.stream_logs("my-workflow"):
                print(line)
    """
    
    _instance: Optional['FactoryCLIAdapter'] = None
    
    # Factory CLI binary names to search for
    FACTORY_BINARIES = ["factory", "factory-cli", "fcli"]
    
    # Default config locations
    CONFIG_LOCATIONS = [
        Path.home() / ".factory" / "config.yaml",
        Path.home() / ".factory" / "config.json",
        Path.cwd() / ".factory.yaml",
        Path.cwd() / ".factory.json",
    ]
    
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
        self._factory_path: Optional[Path] = None
        self._config: Dict[str, Any] = {}
        self._workflows = {}
        self._active_runs = {}
        self._event_bus = None
        self._output_callbacks = []
        
        if HAS_EVENT_BUS and get_event_bus is not None:
            self._event_bus = get_event_bus()
        
        # Detect Factory CLI
        self._detect_factory_cli()
        self._load_config()
    
    def _detect_factory_cli(self):
        """Detect Factory CLI installation."""
        for binary in self.FACTORY_BINARIES:
            path = shutil.which(binary)
            if path:
                self._factory_path = Path(path)
                logger.info(f"Found Factory CLI at: {path}")
                return
        
        logger.info("Factory CLI not found in PATH")
    
    def _load_config(self):
        """Load Factory CLI configuration."""
        import yaml
        
        for config_path in self.CONFIG_LOCATIONS:
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        if config_path.suffix == ".json":
                            loaded_config = json.load(f)
                        else:
                            loaded_config = yaml.safe_load(f) or {}

                        # Ensure we have a dictionary
                        if isinstance(loaded_config, dict):
                            self._config = loaded_config
                        else:
                            logger.warning(f"Config file {config_path} does not contain a valid dictionary")
                            self._config = {}
                    logger.info(f"Loaded Factory CLI config from: {config_path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load config from {config_path}: {e}")
    
    # ==================== Status Methods ====================
    
    def is_available(self) -> bool:
        """Check if Factory CLI is available."""
        return self._factory_path is not None
    
    def get_version(self) -> Optional[str]:
        """Get Factory CLI version."""
        if not self.is_available():
            return None
        
        try:
            result = subprocess.run(
                [str(self._factory_path), "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        
        return None
    
    async def check_status(self) -> Dict[str, Any]:
        """
        Check Factory CLI status.
        
        Returns:
            Status information dict
        """
        status = {
            "available": self.is_available(),
            "version": self.get_version(),
            "config_loaded": bool(self._config),
            "active_runs": len(self._active_runs),
        }
        
        if self.is_available():
            try:
                result = await self._run_command(["status", "--json"])
                if result["success"]:
                    status["factory_status"] = json.loads(result["stdout"])
            except Exception:
                pass
        
        return status
    
    # ==================== Workflow Management ====================
    
    async def list_workflows(self) -> List[Workflow]:
        """
        List available workflows.
        
        Returns:
            List of Workflow objects
        """
        if not self.is_available():
            return list(self._workflows.values())
        
        try:
            result = await self._run_command(["workflow", "list", "--json"])
            if result["success"]:
                data = json.loads(result["stdout"])
                workflows = []
                for wf_data in data.get("workflows", []):
                    wf = Workflow.from_dict(wf_data)
                    self._workflows[wf.id] = wf
                    workflows.append(wf)
                return workflows
        except Exception as e:
            logger.error(f"Failed to list workflows: {e}")
        
        return list(self._workflows.values())
    
    async def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get a specific workflow."""
        if workflow_id in self._workflows:
            return self._workflows[workflow_id]
        
        if not self.is_available():
            return None
        
        try:
            result = await self._run_command(["workflow", "get", workflow_id, "--json"])
            if result["success"]:
                data = json.loads(result["stdout"])
                wf = Workflow.from_dict(data)
                self._workflows[wf.id] = wf
                return wf
        except Exception as e:
            logger.error(f"Failed to get workflow {workflow_id}: {e}")
        
        return None
    
    async def create_workflow(self, workflow: Workflow) -> bool:
        """
        Create a new workflow.
        
        Args:
            workflow: The workflow to create
            
        Returns:
            True if created successfully
        """
        self._workflows[workflow.id] = workflow
        
        if not self.is_available():
            return True
        
        temp_path = None
        try:
            # Write workflow definition to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(workflow.to_dict(), f)
                temp_path = f.name

            result = await self._run_command(["workflow", "create", "-f", temp_path])

            return result["success"]
        except Exception as e:
            logger.error(f"Failed to create workflow: {e}")
            return False
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass  # Ignore cleanup errors
    
    async def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow."""
        if workflow_id in self._workflows:
            del self._workflows[workflow_id]
        
        if not self.is_available():
            return True
        
        try:
            result = await self._run_command(["workflow", "delete", workflow_id])
            return result["success"]
        except Exception as e:
            logger.error(f"Failed to delete workflow {workflow_id}: {e}")
            return False
    
    # ==================== Workflow Execution ====================
    
    async def run_workflow(
        self,
        workflow_id: str,
        variables: Optional[Dict[str, Any]] = None
    ) -> Optional[WorkflowRun]:
        """
        Run a workflow.
        
        Args:
            workflow_id: The workflow to run
            variables: Optional variables to pass
            
        Returns:
            WorkflowRun object, or None if failed
        """
        import uuid
        
        run_id = str(uuid.uuid4())[:8]
        run = WorkflowRun(
            workflow_id=workflow_id,
            run_id=run_id,
            status="pending",
            started_at=datetime.now().isoformat(),
        )
        
        self._active_runs[run_id] = run
        
        # Emit event
        if self._event_bus and EventType is not None:
            await self._event_bus.emit_async(
                EventType.FACTORY_WORKFLOW_STARTED,
                data={"workflow_id": workflow_id, "run_id": run_id},
                source="factory_cli"
            )
        
        if not self.is_available():
            # Simulate workflow run
            run.status = "running"
            asyncio.create_task(self._simulate_workflow_run(run))
            return run
        
        try:
            cmd = ["workflow", "run", workflow_id]
            
            if variables:
                for key, value in variables.items():
                    cmd.extend(["--var", f"{key}={value}"])
            
            # Start the workflow in background
            asyncio.create_task(self._execute_workflow(run, cmd))
            
            return run
        
        except Exception as e:
            logger.error(f"Failed to run workflow {workflow_id}: {e}")
            run.status = "failed"
            run.error = str(e)
            return run
    
    async def _execute_workflow(self, run: WorkflowRun, cmd: List[str]):
        """Execute a workflow and track its progress."""
        run.status = "running"
        
        try:
            result = await self._run_command(cmd, stream=True)
            
            if result["success"]:
                run.status = "completed"
                run.output = {"stdout": result["stdout"]}
            else:
                run.status = "failed"
                run.error = result["stderr"]
        
        except Exception as e:
            run.status = "failed"
            run.error = str(e)
        
        run.completed_at = datetime.now().isoformat()
        
        # Emit event
        if self._event_bus and hasattr(EventType, "FACTORY_WORKFLOW_COMPLETED") and hasattr(EventType, "FACTORY_WORKFLOW_ERROR"):
            event_type = (
                getattr(EventType, "FACTORY_WORKFLOW_COMPLETED")
                if run.status == "completed"
                else getattr(EventType, "FACTORY_WORKFLOW_ERROR")
            )
            await self._event_bus.emit_async(
                event_type,
                data={"workflow_id": run.workflow_id, "run_id": run.run_id, "status": run.status},
                source="factory_cli"
            )
    
    async def _simulate_workflow_run(self, run: WorkflowRun):
        """Simulate a workflow run for testing."""
        run.status = "running"
        
        workflow = self._workflows.get(run.workflow_id)
        if workflow:
            for step in workflow.steps:
                run.current_step = step.name
                run.step_statuses[step.name] = "running"
                await asyncio.sleep(1)  # Simulate step execution
                run.step_statuses[step.name] = "completed"
        
        await asyncio.sleep(0.5)
        run.status = "completed"
        run.completed_at = datetime.now().isoformat()
        
        # Emit event
        if self._event_bus and EventType is not None:
            await self._event_bus.emit_async(
                EventType.FACTORY_WORKFLOW_COMPLETED,
                data={"workflow_id": run.workflow_id, "run_id": run.run_id},
                source="factory_cli"
            )
    
    async def stop_workflow(self, run_id: str) -> bool:
        """Stop a running workflow."""
        if run_id not in self._active_runs:
            return False
        
        run = self._active_runs[run_id]
        
        if run.status != "running":
            return False
        
        if not self.is_available():
            run.status = "cancelled"
            run.completed_at = datetime.now().isoformat()
            return True
        
        try:
            result = await self._run_command(["workflow", "stop", run_id])
            if result["success"]:
                run.status = "cancelled"
                run.completed_at = datetime.now().isoformat()
                return True
        except Exception as e:
            logger.error(f"Failed to stop workflow run {run_id}: {e}")
        
        return False
    
    def get_run(self, run_id: str) -> Optional[WorkflowRun]:
        """Get a workflow run by ID."""
        return self._active_runs.get(run_id)
    
    def get_active_runs(self) -> List[WorkflowRun]:
        """Get all active workflow runs."""
        return [r for r in self._active_runs.values() if r.status == "running"]
    
    # ==================== Logging ====================
    
    async def get_logs(
        self,
        workflow_id: Optional[str] = None,
        run_id: Optional[str] = None,
        lines: int = 100
    ) -> List[str]:
        """
        Get workflow logs.
        
        Args:
            workflow_id: Filter by workflow ID
            run_id: Filter by run ID
            lines: Number of lines to return
            
        Returns:
            List of log lines
        """
        if not self.is_available():
            return [f"[Simulated] Workflow {workflow_id or 'all'} logs"]
        
        try:
            cmd = ["logs"]
            if workflow_id:
                cmd.extend(["--workflow", workflow_id])
            if run_id:
                cmd.extend(["--run", run_id])
            cmd.extend(["--lines", str(lines)])
            
            result = await self._run_command(cmd)
            if result["success"]:
                return result["stdout"].splitlines()
        except Exception as e:
            logger.error(f"Failed to get logs: {e}")
        
        return []
    
    async def stream_logs(
        self,
        workflow_id: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Stream workflow logs.
        
        Args:
            workflow_id: Filter by workflow ID
            run_id: Filter by run ID
            
        Yields:
            Log lines as they arrive
        """
        if not self.is_available():
            yield f"[Simulated] Streaming logs for {workflow_id or 'all'}"
            return
        
        cmd = [str(self._factory_path), "logs", "--follow"]
        if workflow_id:
            cmd.extend(["--workflow", workflow_id])
        if run_id:
            cmd.extend(["--run", run_id])
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            if process.stdout is not None:
                async for line in process.stdout:
                    yield line.decode().rstrip()
        
        except Exception as e:
            logger.error(f"Failed to stream logs: {e}")
            yield f"Error streaming logs: {e}"
    
    # ==================== Command Execution ====================
    
    async def _run_command(
        self,
        args: List[str],
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Run a Factory CLI command.
        
        Args:
            args: Command arguments
            stream: Whether to stream output
            
        Returns:
            Dict with success, stdout, stderr
        """
        if not self._factory_path:
            return {"success": False, "stdout": "", "stderr": "Factory CLI not available"}
        
        cmd = [str(self._factory_path)] + args
        
        try:
            if stream:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout_lines = []
                if process.stdout is not None:
                    async for line in process.stdout:
                        decoded = line.decode().rstrip()
                        stdout_lines.append(decoded)
                        for callback in self._output_callbacks:
                            callback(decoded)

                await process.wait()

                stderr_data = ""
                if process.stderr is not None:
                    stderr_data = (await process.stderr.read()).decode()

                return {
                    "success": process.returncode == 0,
                    "stdout": "\n".join(stdout_lines),
                    "stderr": stderr_data,
                }
            else:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                return {
                    "success": process.returncode == 0,
                    "stdout": stdout.decode(),
                    "stderr": stderr.decode(),
                }
        
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e)}
    
    def add_output_callback(self, callback: Callable[[str], None]):
        """Add a callback for command output."""
        self._output_callbacks.append(callback)
    
    def remove_output_callback(self, callback: Callable[[str], None]):
        """Remove an output callback."""
        if callback in self._output_callbacks:
            self._output_callbacks.remove(callback)


# Global singleton accessor
def get_factory_cli() -> FactoryCLIAdapter:
    """Get the global FactoryCLIAdapter instance."""
    return FactoryCLIAdapter()


