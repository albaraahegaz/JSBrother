"""
JSBrother — Application tracker CSV (M7)
Columns: timestamp, mode, title, company, score, verdict, draft_path, email_status, recipient
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import config

logger = logging.getLogger("jsbrother.tracker")

TRACKER_FIELDS = [
    "timestamp",
    "mode",
    "title",
    "company",
    "score",
    "verdict",
    "draft_path",
    "email_status",
    "recipient",
]


def _ensure_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRACKER_FIELDS)
            writer.writeheader()


def append_tracker_row(
    mode: str,
    title: str,
    company: str,
    score: float,
    verdict: str,
    draft_path: str = "",
    email_status: str = "",
    recipient: str = "",
    tracker_path: Optional[Path] = None,
) -> Path:
    path = Path(tracker_path) if tracker_path else config.TRACKER_PATH
    _ensure_header(path)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "title": title,
        "company": company,
        "score": f"{score:.1f}",
        "verdict": verdict,
        "draft_path": draft_path,
        "email_status": email_status,
        "recipient": recipient,
    }
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRACKER_FIELDS)
        writer.writerow(row)
    logger.info(f"Tracker row appended → {path}")
    return path


def read_tracker(tracker_path: Optional[Path] = None) -> List[Dict[str, str]]:
    path = Path(tracker_path) if tracker_path else config.TRACKER_PATH
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
