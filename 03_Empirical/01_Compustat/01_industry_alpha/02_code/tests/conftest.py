"""Put the pipeline code dir (02_code) on sys.path so stage modules and the
shared ``utils`` module import cleanly from tests."""

from __future__ import annotations

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
