import subprocess
import sys

def open_notepad() -> str:
    if sys.platform == "win32":
        subprocess.Popen(["notepad.exe"])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-a", "TextEdit"])
    else:
        subprocess.Popen(["gedit"])
    return "Opened text editor."