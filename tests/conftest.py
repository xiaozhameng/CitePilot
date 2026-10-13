import sys
from pathlib import Path

# 让 tests 能直接 import citepilot
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
