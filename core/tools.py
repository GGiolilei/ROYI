from typing import Callable, Dict, Any, Tuple

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable):
        self._tools[name] = func

    def execute(self, name: str, **kwargs) -> Tuple[bool, Any]:
        if name not in self._tools:
            return False, f"Tool '{name}' not found."
        try:
            result = self._tools[name](**kwargs)
            return True, result
        except Exception as e:
            return False, str(e)

    def get_available_tools(self) -> list:
        return list(self._tools.keys())