"""FastAPI scoring service package (docs/Plan.md 8).

Makes ``src/`` importable when launched as ``uvicorn api.main:app`` from the
project root, regardless of PYTHONPATH.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
