import logging
from core.events import EventBus, Event
from core.tools import ToolRegistry
from ai.brain import Brain
from memory.database import init_db, add_note
from pc.applications import open_notepad

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ROYICore:
    def __init__(self):
        self.bus = EventBus()
        self.tools = ToolRegistry()
        self.brain = Brain()
        
        # Initialize Core Systems
        init_db()
        self._register_default_tools()
        self._setup_event_listeners()

    def _register_default_tools(self):
        self.tools.register("open_notepad", open_notepad)
        self.tools.register("create_note", add_note)

    def _setup_event_listeners(self):
        self.bus.subscribe("USER_COMMAND", self._handle_command)

    def _handle_command(self, event: Event):
        command_text = event.data.get("text", "")
        logging.info(f"Processing command: {command_text}")
        
        intent = self.brain.parse_intent(command_text)
        tool_name = intent.get("tool")
        args = intent.get("args", {})

        if tool_name != "none":
            success, result = self.tools.execute(tool_name, **args)
            logging.info(f"Execution Result [{tool_name}]: {result}")
        else:
            logging.info(f"No actionable tool found. LLM output: {args.get('response')}")

    def process_input(self, text: str):
        self.bus.publish(Event(type="USER_COMMAND", data={"text": text}))