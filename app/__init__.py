"""Redireccion hacia app/main.py en caso de que Streamlit apunte a app/__init__.py."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runpy.run_path(str(ROOT / "app" / "main.py"), run_name="__main__")
