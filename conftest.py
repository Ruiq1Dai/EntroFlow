"""Make the src-layout package importable with legacy pytest versions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
