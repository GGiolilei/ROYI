from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Union


@dataclass
class Event:
    type: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class EventBus:

    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Event], None]]] = {}

    def subscribe(self, event_type: str, callback: Callable[[Event], None]):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    def publish(self, event: Union[Event, Dict[str, Any]]):
        """Publishes events cleanly whether passed as an Event object or a dictionary."""
        # Normalize dictionary input to Event dataclass
        if isinstance(event, dict):
            event_type = event.get("type")
            if not event_type:
                return
            event_data = {k: v for k, v in event.items() if k != "type"}
            event_obj = Event(type=event_type, data=event_data)
        else:
            event_obj = event

        if event_obj.type in self._subscribers:
            for callback in self._subscribers[event_obj.type]:
                try:
                    callback(event_obj)
                except Exception as e:
                    print(
                        f"[EventBus Error]: Callback failed for '{event_obj.type}': {e}"
                    )