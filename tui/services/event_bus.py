"""
Event Bus for decoupled component communication.

Provides a pub/sub pattern for TUI components to communicate without direct dependencies.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Union
from weakref import WeakSet
import logging

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Standard event types for SparkPlug TUI."""
    
    # Workspace events
    WORKSPACE_CREATED = auto()
    WORKSPACE_DELETED = auto()
    WORKSPACE_SWITCHED = auto()
    WORKSPACE_UPDATED = auto()
    
    # Agent events
    AGENT_DEPLOYED = auto()
    AGENT_STOPPED = auto()
    AGENT_STATUS_CHANGED = auto()
    AGENT_MEMORY_UPDATED = auto()
    
    # System events
    METRICS_UPDATED = auto()
    CONFIG_SAVED = auto()
    CONFIG_LOADED = auto()
    
    # DGX/GPU events
    GPU_METRICS_UPDATED = auto()
    GPU_ERROR = auto()
    
    # Plugin events
    PLUGIN_LOADED = auto()
    PLUGIN_UNLOADED = auto()
    PLUGIN_ERROR = auto()
    
    # UI events
    TAB_SWITCHED = auto()
    SIDEBAR_TOGGLED = auto()
    THEME_CHANGED = auto()
    NOTIFICATION_SHOWN = auto()
    
    # Communication events
    MESSAGE_RECEIVED = auto()
    MESSAGE_SENT = auto()
    CONNECTION_ESTABLISHED = auto()
    CONNECTION_LOST = auto()
    
    # Code Assistant events
    FILE_OPENED = auto()
    FILE_SAVED = auto()
    AIDER_STARTED = auto()
    AIDER_STOPPED = auto()
    AIDER_RESPONSE = auto()
    
    # Ollama events
    MODEL_LOADED = auto()
    MODEL_UNLOADED = auto()
    INFERENCE_STARTED = auto()
    INFERENCE_COMPLETED = auto()
    
    # Factory CLI events
    FACTORY_WORKFLOW_STARTED = auto()
    FACTORY_WORKFLOW_COMPLETED = auto()
    FACTORY_WORKFLOW_ERROR = auto()
    
    # Custom event (for plugins)
    CUSTOM = auto()


@dataclass
class Event:
    """Event data container."""
    type: Union[EventType, str]
    data: Any = None
    source: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if isinstance(self.type, str):
            # Allow string event types for custom events
            pass


# Type alias for event handlers
EventHandler = Callable[[Event], None]
AsyncEventHandler = Callable[[Event], Any]  # Can be async


class EventBus:
    """
    Central event bus for component communication.
    
    Supports both sync and async handlers, with optional filtering by source.
    
    Usage:
        bus = EventBus()
        
        # Subscribe to events
        def on_workspace_change(event: Event):
            print(f"Workspace changed: {event.data}")
        
        bus.subscribe(EventType.WORKSPACE_SWITCHED, on_workspace_change)
        
        # Publish events
        bus.emit(EventType.WORKSPACE_SWITCHED, data={"workspace_id": "ws1"})
        
        # Async usage
        await bus.emit_async(EventType.METRICS_UPDATED, data=metrics)
    """
    
    _instance: Optional['EventBus'] = None
    
    def __new__(cls):
        """Singleton pattern for global event bus access."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._handlers: Dict[Union[EventType, str], List[EventHandler]] = {}
        self._async_handlers: Dict[Union[EventType, str], List[AsyncEventHandler]] = {}
        self._global_handlers: List[EventHandler] = []
        self._async_global_handlers: List[AsyncEventHandler] = []
        self._event_history: List[Event] = []
        self._history_limit = 100
        self._paused = False
        self._queued_events: List[Event] = []
    
    def subscribe(
        self,
        event_type: Union[EventType, str],
        handler: EventHandler,
        *,
        is_async: bool = False
    ) -> Callable[[], None]:
        """
        Subscribe to an event type.
        
        Args:
            event_type: The event type to subscribe to
            handler: The callback function
            is_async: Whether the handler is async
            
        Returns:
            Unsubscribe function
        """
        handlers_dict = self._async_handlers if is_async else self._handlers
        
        if event_type not in handlers_dict:
            handlers_dict[event_type] = []
        
        handlers_dict[event_type].append(handler)
        
        # Return unsubscribe function
        def unsubscribe():
            if event_type in handlers_dict and handler in handlers_dict[event_type]:
                handlers_dict[event_type].remove(handler)
        
        return unsubscribe
    
    def subscribe_all(
        self,
        handler: EventHandler,
        *,
        is_async: bool = False
    ) -> Callable[[], None]:
        """
        Subscribe to all events.
        
        Args:
            handler: The callback function
            is_async: Whether the handler is async
            
        Returns:
            Unsubscribe function
        """
        if is_async:
            self._async_global_handlers.append(handler)
            return lambda: self._async_global_handlers.remove(handler) if handler in self._async_global_handlers else None
        else:
            self._global_handlers.append(handler)
            return lambda: self._global_handlers.remove(handler) if handler in self._global_handlers else None
    
    def unsubscribe(
        self,
        event_type: Union[EventType, str],
        handler: EventHandler,
        *,
        is_async: bool = False
    ):
        """Unsubscribe a handler from an event type."""
        handlers_dict = self._async_handlers if is_async else self._handlers
        
        if event_type in handlers_dict and handler in handlers_dict[event_type]:
            handlers_dict[event_type].remove(handler)
    
    def emit(
        self,
        event_type: Union[EventType, str],
        data: Any = None,
        source: Optional[str] = None
    ) -> Event:
        """
        Emit an event synchronously.
        
        Args:
            event_type: The event type
            data: Event data payload
            source: Source component identifier
            
        Returns:
            The emitted Event object
        """
        event = Event(type=event_type, data=data, source=source)
        
        if self._paused:
            self._queued_events.append(event)
            return event
        
        self._dispatch_sync(event)
        return event
    
    async def emit_async(
        self,
        event_type: Union[EventType, str],
        data: Any = None,
        source: Optional[str] = None
    ) -> Event:
        """
        Emit an event asynchronously.
        
        Args:
            event_type: The event type
            data: Event data payload
            source: Source component identifier
            
        Returns:
            The emitted Event object
        """
        event = Event(type=event_type, data=data, source=source)
        
        if self._paused:
            self._queued_events.append(event)
            return event
        
        await self._dispatch_async(event)
        return event
    
    def _dispatch_sync(self, event: Event):
        """Dispatch event to sync handlers."""
        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._history_limit:
            self._event_history.pop(0)
        
        # Call specific handlers
        if event.type in self._handlers:
            for handler in self._handlers[event.type]:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Error in event handler for {event.type}: {e}")
        
        # Call global handlers
        for handler in self._global_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in global event handler: {e}")
    
    async def _dispatch_async(self, event: Event):
        """Dispatch event to both sync and async handlers."""
        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._history_limit:
            self._event_history.pop(0)
        
        # Call sync handlers first
        self._dispatch_sync(event)
        
        # Call async specific handlers
        if event.type in self._async_handlers:
            tasks = []
            for handler in self._async_handlers[event.type]:
                try:
                    result = handler(event)
                    if asyncio.iscoroutine(result):
                        tasks.append(result)
                except Exception as e:
                    logger.error(f"Error in async event handler for {event.type}: {e}")
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        
        # Call async global handlers
        tasks = []
        for handler in self._async_global_handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    tasks.append(result)
            except Exception as e:
                logger.error(f"Error in async global event handler: {e}")
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def pause(self):
        """Pause event dispatching (events will be queued)."""
        self._paused = True
    
    def resume(self):
        """Resume event dispatching and process queued events."""
        self._paused = False
        
        # Process queued events
        queued = self._queued_events.copy()
        self._queued_events.clear()
        
        for event in queued:
            self._dispatch_sync(event)
    
    async def resume_async(self):
        """Resume event dispatching asynchronously."""
        self._paused = False
        
        # Process queued events
        queued = self._queued_events.copy()
        self._queued_events.clear()
        
        for event in queued:
            await self._dispatch_async(event)
    
    def get_history(
        self,
        event_type: Optional[Union[EventType, str]] = None,
        limit: int = 50
    ) -> List[Event]:
        """
        Get event history.
        
        Args:
            event_type: Filter by event type (optional)
            limit: Maximum number of events to return
            
        Returns:
            List of events (newest first)
        """
        if event_type:
            filtered = [e for e in self._event_history if e.type == event_type]
        else:
            filtered = self._event_history.copy()
        
        return list(reversed(filtered[-limit:]))
    
    def clear_history(self):
        """Clear event history."""
        self._event_history.clear()
    
    def clear_handlers(self, event_type: Optional[Union[EventType, str]] = None):
        """
        Clear handlers.
        
        Args:
            event_type: Clear handlers for specific type, or all if None
        """
        if event_type:
            self._handlers.pop(event_type, None)
            self._async_handlers.pop(event_type, None)
        else:
            self._handlers.clear()
            self._async_handlers.clear()
            self._global_handlers.clear()
            self._async_global_handlers.clear()


# Global singleton accessor
def get_event_bus() -> EventBus:
    """Get the global EventBus instance."""
    return EventBus()


# Decorator for event handlers
def on_event(event_type: Union[EventType, str], is_async: bool = False):
    """
    Decorator to register a function as an event handler.
    
    Usage:
        @on_event(EventType.WORKSPACE_SWITCHED)
        def handle_workspace_switch(event: Event):
            print(f"Switched to {event.data}")
    """
    def decorator(func: EventHandler):
        get_event_bus().subscribe(event_type, func, is_async=is_async)
        return func
    return decorator

