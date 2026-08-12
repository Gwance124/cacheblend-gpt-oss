# SPDX-License-Identifier: Apache-2.0
"""Make source-checkout CLI execution match the editable-install layout."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_source_path() -> None:
    """Add the repository's ``src`` directory when a script runs directly."""

    source_root = str(Path(__file__).resolve().parents[1] / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
