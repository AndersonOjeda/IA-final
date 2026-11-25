"""Punto de entrada raiz para Streamlit Community Cloud."""
import runpy
import sys
from pathlib import Path

# Agregar directorio raiz al sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    runpy.run_path(str(ROOT / "app" / "main.py"), run_name="__main__")
