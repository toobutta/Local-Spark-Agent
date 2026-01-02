"""
Ollama Integration Service.

Provides local model management and chat capabilities using Ollama.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Callable, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import httpx for async HTTP
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    httpx = None
    logger.warning("httpx not available, Ollama integration will be limited")

# Try to import ollama library
try:
    import ollama as ollama_lib
    HAS_OLLAMA_LIB = True
except ImportError:
    HAS_OLLAMA_LIB = False

# Try to import event bus
try:
    from ..services.event_bus import EventBus, EventType, get_event_bus
    HAS_EVENT_BUS = True
except ImportError:
    HAS_EVENT_BUS = False
    EventBus = None
    EventType = None
    get_event_bus = None


@dataclass
class OllamaModel:
    """Information about an Ollama model."""
    name: str
    size: int  # Size in bytes
    digest: str
    modified_at: str
    
    # Model details (populated when model is loaded)
    parameters: Optional[Dict[str, Any]] = None
    template: Optional[str] = None
    system: Optional[str] = None
    
    @property
    def size_gb(self) -> float:
        """Size in gigabytes."""
        return self.size / (1024 ** 3)
    
    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        # Extract base name and tag
        if ":" in self.name:
            base, tag = self.name.rsplit(":", 1)
            return f"{base} ({tag})"
        return self.name


@dataclass
class ChatMessage:
    """A chat message."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to API format."""
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResponse:
    """Response from a chat completion."""
    message: ChatMessage
    model: str
    done: bool
    total_duration: Optional[int] = None  # nanoseconds
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None
    
    @property
    def tokens_per_second(self) -> Optional[float]:
        """Calculate tokens per second."""
        if self.eval_count and self.eval_duration:
            return self.eval_count / (self.eval_duration / 1e9)
        return None


class OllamaService:
    """
    Ollama integration service for local model inference.
    
    Features:
    - Model listing, pulling, and deletion
    - Chat interface with streaming responses
    - Model switching
    - GPU memory allocation management
    
    Usage:
        service = OllamaService()
        
        # Check availability
        if await service.is_available():
            # List models
            models = await service.list_models()
            
            # Chat with streaming
            async for chunk in service.chat_stream("llama2", "Hello!"):
                print(chunk, end="")
    """
    
    _instance: Optional['OllamaService'] = None
    
    DEFAULT_HOST = "http://localhost:11434"
    
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
        self._host = self.DEFAULT_HOST
        self._client: Optional[Any] = None
        self._models: Dict[str, OllamaModel] = {}
        self._current_model: Optional[str] = None
        self._chat_history: List[ChatMessage] = []
        self._event_bus: Optional[Any] = None
        self._stream_callbacks: List[Callable[[str], None]] = []
        
        if HAS_EVENT_BUS and get_event_bus is not None:
            self._event_bus = get_event_bus()
    
    async def _get_client(self) -> Optional[Any]:
        """Get or create HTTP client."""
        if not HAS_HTTPX or httpx is None:
            logger.warning("httpx not available for Ollama integration")
            return None

        if self._client is None:
            try:
                self._client = httpx.AsyncClient(
                    base_url=self._host,
                    timeout=httpx.Timeout(60.0, connect=5.0)
                )
            except Exception as e:
                logger.error(f"Failed to create HTTP client: {e}")
                return None

        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def set_host(self, host: str):
        """Set the Ollama host URL."""
        self._host = host
        # Reset client to use new host
        if self._client:
            asyncio.create_task(self.close())
    
    # ==================== Status Methods ====================
    
    async def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            client = await self._get_client()
            if not client:
                return False
            
            response = await client.get("/")
            return response.status_code == 200
        except Exception:
            return False
    
    async def get_version(self) -> Optional[str]:
        """Get Ollama version."""
        try:
            client = await self._get_client()
            if not client:
                return None
            
            response = await client.get("/api/version")
            if response.status_code == 200:
                data = response.json()
                return data.get("version")
        except Exception:
            pass
        
        return None
    
    # ==================== Model Management ====================
    
    async def list_models(self) -> List[OllamaModel]:
        """
        List available models.
        
        Returns:
            List of OllamaModel objects
        """
        try:
            client = await self._get_client()
            if not client:
                return list(self._models.values())
            
            response = await client.get("/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = []
                
                for model_data in data.get("models", []):
                    model = OllamaModel(
                        name=model_data["name"],
                        size=model_data.get("size", 0),
                        digest=model_data.get("digest", ""),
                        modified_at=model_data.get("modified_at", ""),
                    )
                    self._models[model.name] = model
                    models.append(model)
                
                return models
        
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
        
        return list(self._models.values())
    
    async def get_model(self, model_name: str) -> Optional[OllamaModel]:
        """Get model information."""
        if model_name in self._models:
            return self._models[model_name]
        
        # Try to fetch model info
        try:
            client = await self._get_client()
            if not client:
                return None
            
            response = await client.post("/api/show", json={"name": model_name})
            if response.status_code == 200:
                data = response.json()
                model = OllamaModel(
                    name=model_name,
                    size=data.get("size", 0),
                    digest=data.get("digest", ""),
                    modified_at=data.get("modified_at", ""),
                    parameters=data.get("parameters"),
                    template=data.get("template"),
                    system=data.get("system"),
                )
                self._models[model_name] = model
                return model
        
        except Exception as e:
            logger.error(f"Failed to get model {model_name}: {e}")
        
        return None
    
    async def pull_model(
        self,
        model_name: str,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> bool:
        """
        Pull a model from Ollama library.
        
        Args:
            model_name: Name of the model to pull
            progress_callback: Optional callback for progress updates (status, percent)
            
        Returns:
            True if successful
        """
        try:
            client = await self._get_client()
            if not client:
                return False
            
            async with client.stream(
                "POST",
                "/api/pull",
                json={"name": model_name},
                timeout=None  # No timeout for large downloads
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            status = data.get("status", "")
                            
                            # Calculate progress
                            total = data.get("total", 0)
                            completed = data.get("completed", 0)
                            percent = (completed / total * 100) if total > 0 else 0
                            
                            if progress_callback:
                                progress_callback(status, percent)
                            
                            if data.get("error"):
                                logger.error(f"Pull error: {data['error']}")
                                return False
                        
                        except json.JSONDecodeError:
                            pass
            
            # Refresh model list
            await self.list_models()
            
            logger.info(f"Successfully pulled model: {model_name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    async def delete_model(self, model_name: str) -> bool:
        """
        Delete a model.
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            True if successful
        """
        try:
            client = await self._get_client()
            if not client:
                return False
            
            response = await client.delete("/api/delete", content=json.dumps({"name": model_name}), headers={"Content-Type": "application/json"})
            
            if response.status_code == 200:
                if model_name in self._models:
                    del self._models[model_name]
                
                if self._current_model == model_name:
                    self._current_model = None
                
                logger.info(f"Deleted model: {model_name}")
                return True
        
        except Exception as e:
            logger.error(f"Failed to delete model {model_name}: {e}")
        
        return False
    
    # ==================== Chat Interface ====================
    
    async def chat(
        self,
        model: str,
        message: str,
        system: Optional[str] = None,
        context: Optional[List[ChatMessage]] = None
    ) -> Optional[ChatResponse]:
        """
        Send a chat message and get a response.
        
        Args:
            model: Model name to use
            message: User message
            system: Optional system prompt
            context: Optional conversation context
            
        Returns:
            ChatResponse object, or None if failed
        """
        try:
            client = await self._get_client()
            if not client:
                return None
            
            # Build messages
            messages = []
            
            if system:
                messages.append({"role": "system", "content": system})
            
            if context:
                messages.extend([m.to_dict() for m in context])
            
            messages.append({"role": "user", "content": message})
            
            # Emit event
            if self._event_bus and EventType is not None:
                await self._event_bus.emit_async(
                    EventType.INFERENCE_STARTED,
                    data={"model": model, "message": message[:100]},
                    source="ollama"
                )
            
            response = await client.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                
                chat_response = ChatResponse(
                    message=ChatMessage(
                        role="assistant",
                        content=data["message"]["content"]
                    ),
                    model=model,
                    done=data.get("done", True),
                    total_duration=data.get("total_duration"),
                    load_duration=data.get("load_duration"),
                    prompt_eval_count=data.get("prompt_eval_count"),
                    eval_count=data.get("eval_count"),
                    eval_duration=data.get("eval_duration"),
                )
                
                # Add to history
                self._chat_history.append(ChatMessage(role="user", content=message))
                self._chat_history.append(chat_response.message)
                
                # Emit event
                if self._event_bus and EventType is not None:
                    await self._event_bus.emit_async(
                        EventType.INFERENCE_COMPLETED,
                        data={
                            "model": model,
                            "tokens": chat_response.eval_count,
                            "duration": chat_response.total_duration,
                        },
                        source="ollama"
                    )
                
                return chat_response
        
        except Exception as e:
            logger.error(f"Chat error: {e}")
        
        return None
    
    async def chat_stream(
        self,
        model: str,
        message: str,
        system: Optional[str] = None,
        context: Optional[List[ChatMessage]] = None
    ) -> AsyncIterator[str]:
        """
        Send a chat message and stream the response.
        
        Args:
            model: Model name to use
            message: User message
            system: Optional system prompt
            context: Optional conversation context
            
        Yields:
            Response chunks as they arrive
        """
        try:
            client = await self._get_client()
            if not client:
                yield "Error: HTTP client not available"
                return
            
            # Build messages
            messages = []
            
            if system:
                messages.append({"role": "system", "content": system})
            
            if context:
                messages.extend([m.to_dict() for m in context])
            
            messages.append({"role": "user", "content": message})
            
            # Emit event
            if self._event_bus and EventType is not None:
                await self._event_bus.emit_async(
                    EventType.INFERENCE_STARTED,
                    data={"model": model, "message": message[:100], "streaming": True},
                    source="ollama"
                )
            
            full_response = []
            
            async with client.stream(
                "POST",
                "/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": True,
                }
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            
                            if "message" in data:
                                content = data["message"].get("content", "")
                                if content:
                                    full_response.append(content)
                                    yield content
                                    
                                    # Call stream callbacks
                                    for callback in self._stream_callbacks:
                                        callback(content)
                            
                            if data.get("done"):
                                # Add to history
                                self._chat_history.append(
                                    ChatMessage(role="user", content=message)
                                )
                                self._chat_history.append(
                                    ChatMessage(
                                        role="assistant",
                                        content="".join(full_response)
                                    )
                                )
                                
                                # Emit event
                                if self._event_bus and EventType is not None:
                                    await self._event_bus.emit_async(
                                        EventType.INFERENCE_COMPLETED,
                                        data={
                                            "model": model,
                                            "tokens": data.get("eval_count"),
                                            "duration": data.get("total_duration"),
                                        },
                                        source="ollama"
                                    )
                        
                        except json.JSONDecodeError:
                            pass
        
        except Exception as e:
            logger.error(f"Stream chat error: {e}")
            yield f"Error: {e}"
    
    # ==================== Generation ====================
    
    async def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Generate text completion.
        
        Args:
            model: Model name
            prompt: Input prompt
            system: Optional system prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text, or None if failed
        """
        try:
            client = await self._get_client()
            if not client:
                return None
            
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
            }
            
            if system:
                payload["system"] = system
            
            payload.update(kwargs)
            
            response = await client.post("/api/generate", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("response")
        
        except Exception as e:
            logger.error(f"Generate error: {e}")
        
        return None
    
    # ==================== History Management ====================
    
    def get_chat_history(self, limit: int = 50) -> List[ChatMessage]:
        """Get chat history."""
        return self._chat_history[-limit:]
    
    def clear_chat_history(self):
        """Clear chat history."""
        self._chat_history.clear()
    
    def set_current_model(self, model_name: str):
        """Set the current model for chat."""
        self._current_model = model_name
        
        if self._event_bus and EventType is not None:
            self._event_bus.emit(
                EventType.MODEL_LOADED,
                data={"model": model_name},
                source="ollama"
            )
    
    def get_current_model(self) -> Optional[str]:
        """Get the current model."""
        return self._current_model
    
    # ==================== Callbacks ====================
    
    def add_stream_callback(self, callback: Callable[[str], None]):
        """Add a callback for streaming responses."""
        self._stream_callbacks.append(callback)
    
    def remove_stream_callback(self, callback: Callable[[str], None]):
        """Remove a stream callback."""
        if callback in self._stream_callbacks:
            self._stream_callbacks.remove(callback)


# Global singleton accessor
def get_ollama_service() -> OllamaService:
    """Get the global OllamaService instance."""
    return OllamaService()

