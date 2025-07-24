import sys
sys.setrecursionlimit(2000)  # Raise recursion limit in your entry script

# Also in your setup.py or build script:
from cx_Freeze import setup, Executable

build_exe_options = {
    "packages": ["numpy", "pandas", "torch", "PyQt6", "roifile", "cv2", "skimage", "cellpose", "PIL"],
    "excludes": ["tkinter"],  # if you don't use tkinter
}

setup(
    name="Toggle-Untoggle",
    version="1.0",
    description="My PyQt6 App",
    options={"build_exe": build_exe_options},
    executables=[Executable("main.py", base="Win32GUI" if sys.platform=="win32" else None, icon="icon.png")],
)

from cx_Freeze import setup, Executable

import sys

# Dependencies are automatically detected, but it might need help.
build_exe_options = {
    "packages": ["PyQt6"],
    "include_files": ["pen.png", "eraser.png", "icon.png"],  # add any resources here
}

# Base for GUI apps (no console window)
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="Toggle-Untoggle",
    version="1.0",
    description="My PyQt6 application",
    options={"build_exe": build_exe_options},
    executables=[Executable("main.py", base=base, icon="icon.png")],
)
