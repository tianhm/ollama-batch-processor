"""
ollama-batch-processor - configuration and constants
"""
import os
import sys

APP_NAME = "ollama-batch-processor"
APP_VERSION = "2.0.1"
WINDOW_MIN_WIDTH = 900
WINDOW_MIN_HEIGHT = 600

TEXT_EXTENSIONS = [".txt", ".md", ".srt", ".vtt", ".html", ".htm", ".csv", ".json", ".tex", ".rst"]

CHUNK_PRESETS = {
    "Fast (2000 / 150)": (2000, 150),
    "Balanced (2500 / 200)": (2500, 200),
    "High context (3000 / 250)": (3000, 250),
    "Large (4000 / 300)": (4000, 300),
    "Extra large (6000 / 400)": (6000, 400),
}

DEFAULT_SETTINGS = {
    # Server
    "host": "http://localhost:11434",
    "timeout": 600,              # seconds per request (0 = none)
    "keep_alive": "10m",
    "num_ctx": 0,                # 0 = auto from chunk size
    "num_predict": -1,
    "top_p": 0.9,
    "strip_thinking": True,
    # Pipeline
    "pipeline": ["translation", "audiobook", "debookify", "paraphrase"],
    "enabled": ["translation"],
    "chunk_size": 2500,
    "overlap": 200,
    "whole_file": False,
    "deduplicate": True,
    # Output
    "output_mode": "same",       # same | custom
    "output_dir": "",
    "suffix": "_processed",
    "save_steps": True,
    "overwrite": False,
    # Per-operation values live under "ops": {op_id: {option_id: value, "model": "..."}}
    "ops": {},
}


def app_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.argv[0]))
    return os.path.dirname(os.path.abspath(__file__))


def resource_dir() -> str:
    """Folder with bundled read-only files (config.json, icon) - _MEIPASS when frozen"""
    return getattr(sys, "_MEIPASS", app_dir())
