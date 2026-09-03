import sys
from pathlib import Path

# scripts live at the repo root and import each other by bare name
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
