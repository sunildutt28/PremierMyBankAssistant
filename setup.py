# setup.py
from cx_Freeze import setup, Executable
import sys

base = None
if sys.platform == "win32":
    base = "Win32GUI"  # For GUI application

build_options = {
    "packages": ["tkinter", "PIL"],
    "excludes": [],
    "include_files": ["assets/"]
}

setup(
    name="PremierMyBankAssistant",
    version="1.0",
    description="Premier MyBank Chatbot Application",
    options={"build_exe": build_options},
    executables=[Executable("banking_chatbot.py", base=base, 
                          icon="assets/icon.ico",
                          target_name="MyBankAssistant.exe")]
)