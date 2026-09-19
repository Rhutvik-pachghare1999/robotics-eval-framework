import sys
from pathlib import Path

# Make the repo root importable (adapters/, scripts/) without requiring an install.
sys.path.insert(0, str(Path(__file__).resolve().parent))
