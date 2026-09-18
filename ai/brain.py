import json
import ollama
from config import OLLAMA_MODEL

SYSTEM_PROMPT = """You are ROYI's decision engine. Convert user requests into structured JSON tool calls.
Available tools:
1. open_notepad() -> Opens the default text editor.
2. create_note(content: str) -> Saves a text note.

Output MUST be strictly valid JSON with keys "tool" and "args". If no tool matches, set "tool" to "none".
Example: {"tool": "open_notepad", "args": {}}
"""

class Brain:
    def parse_intent(self, user_input: str) -> dict:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ]
        )
        content = response['message']['content'].strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"tool": "none", "args": {"response": content}}