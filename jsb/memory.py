"""
JSBrother — Session Memory (M6)
Buffer memory for conversation turns, persisted profile.json, and last posting/draft
for follow-up refine. Optional 3-line summary every 4 turns.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from jsb.schemas import ProfileJSON, PostingJSON, Draft, MatchResult

logger = logging.getLogger("jsbrother.memory")


@dataclass
class Turn:
    user: str
    assistant: str


class BufferMemory:
    """Full-session list of {user, assistant} turns with reset()."""

    def __init__(self) -> None:
        self.turns: List[Turn] = []
        self.summary: Optional[str] = None

    def add(self, user: str, assistant: str) -> None:
        self.turns.append(Turn(user=user, assistant=assistant))
        if len(self.turns) % 4 == 0:
            self._maybe_summarize()

    def reset(self) -> None:
        self.turns.clear()
        self.summary = None

    def as_dicts(self) -> List[Dict[str, str]]:
        return [{"user": t.user, "assistant": t.assistant} for t in self.turns]

    def _maybe_summarize(self) -> None:
        """Lightweight local summary (no LLM required for reliability in mock)."""
        n = len(self.turns)
        last = self.turns[-4:]
        lines = [f"Turn {n - 3 + i}: user={t.user[:80]!r}" for i, t in enumerate(last)]
        self.summary = "\n".join(lines[:3])


class SessionMemory:
    """
    Session-scoped memory used by the orchestrator and Streamlit st.session_state.
    Persists profile to data/profile.json; keeps last_posting / last_draft / last_match.
    """

    def __init__(self, profile_path: Optional[Path] = None) -> None:
        self.profile_path = Path(profile_path) if profile_path else config.PROFILE_PATH
        self.buffer = BufferMemory()
        self.profile: Optional[ProfileJSON] = None
        self.last_posting: Optional[PostingJSON] = None
        self.last_draft: Optional[Draft] = None
        self.last_match: Optional[MatchResult] = None
        self.last_trace: Dict[str, Any] = {}
        self.load_profile()

    def load_profile(self) -> Optional[ProfileJSON]:
        if not self.profile_path.exists():
            return None
        try:
            data = json.loads(self.profile_path.read_text(encoding="utf-8"))
            self.profile = ProfileJSON(**data)
            return self.profile
        except Exception as e:
            logger.warning(f"Failed to load profile from {self.profile_path}: {e}")
            return None

    def save_profile(self, profile: Optional[ProfileJSON] = None) -> Path:
        if profile is not None:
            self.profile = profile
        if self.profile is None:
            raise ValueError("No profile to save")
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.profile.model_dump() if hasattr(self.profile, "model_dump") else self.profile.dict()
        self.profile_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return self.profile_path

    def remember_run(
        self,
        posting: PostingJSON,
        match: MatchResult,
        draft: Optional[Draft],
        trace: Optional[Dict[str, Any]] = None,
        user_note: str = "",
    ) -> None:
        self.last_posting = posting
        self.last_match = match
        if draft is not None:
            self.last_draft = draft
        self.last_trace = trace or {}
        verdict = match.verdict
        draft_note = f"draft={draft.kind}" if draft else "no draft"
        self.buffer.add(
            user=user_note or f"Analyze: {posting.title}",
            assistant=f"verdict={verdict} score={match.score:.1f} {draft_note}",
        )

    def reset(self) -> None:
        self.buffer.reset()
        self.last_posting = None
        self.last_draft = None
        self.last_match = None
        self.last_trace = {}
        # Keep persisted profile; only clear session artifacts
