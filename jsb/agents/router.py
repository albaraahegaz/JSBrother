"""
JSBrother — Router Agent (M6)
Detects job vs project mode from explicit mode, posting type, or keyword heuristics.
Optional short LLM fallback when ambiguous.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional, Union

from jsb.llm_client import LLMClient, get_client
from jsb.schemas import PostingJSON, ProfileJSON

logger = logging.getLogger("jsbrother.router")

JOB_KEYWORDS = (
    "full-time", "full time", "part-time", "salary", "benefits", "hiring",
    "job description", "responsibilities", "requirements", "equal opportunity",
    "apply now", "years of experience", "team member", "employee",
)
PROJECT_KEYWORDS = (
    "freelance", "freelancer", "upwork", "fixed price", "hourly rate",
    "budget", "deliverable", "milestone", "proposal", "client", "gig",
    "contract", "nda", "turnaround", "scope of work",
)


def detect_mode(
    posting_text: str = "",
    posting: Optional[PostingJSON] = None,
    mode: Optional[str] = None,
    client: Optional[LLMClient] = None,
) -> str:
    """
    Returns 'job' or 'project'.
    Priority: explicit mode → posting.type → keyword scores → LLM fallback → 'job'.
    """
    if mode:
        m = mode.strip().lower()
        if m in ("job", "job_seeker", "jobseeker", "seeker"):
            return "job"
        if m in ("project", "freelance", "freelancer", "gig"):
            return "project"

    if posting is not None and posting.type in ("job", "project"):
        return posting.type

    text = (posting_text or "").lower()
    if not text and posting is not None:
        text = f"{posting.title} {posting.summary} {' '.join(posting.must_have)}".lower()

    job_hits = sum(1 for k in JOB_KEYWORDS if k in text)
    project_hits = sum(1 for k in PROJECT_KEYWORDS if k in text)

    if project_hits > job_hits and project_hits > 0:
        return "project"
    if job_hits > project_hits and job_hits > 0:
        return "job"

    # Ambiguous — short LLM fallback (skipped in mock if no text)
    if text and len(text) > 40:
        try:
            client = client or get_client()
            resp = client.generate(
                prompt=(
                    "Classify this posting as exactly one word: job OR project.\n"
                    f"Text:\n{text[:1500]}"
                ),
                system="Reply with only 'job' or 'project'.",
                max_new_tokens=8,
                temperature=0.0,
            )
            token = re.findall(r"(job|project)", resp.lower())
            if token:
                return token[0]
        except Exception as e:
            logger.warning(f"Router LLM fallback failed: {e}")

    return "job"


def validate_inputs(
    profile: Optional[Union[ProfileJSON, str]] = None,
    posting: Optional[Union[PostingJSON, str]] = None,
) -> Dict[str, Any]:
    """Basic input validation before the pipeline runs."""
    errors = []
    if profile is None or (isinstance(profile, str) and not profile.strip()):
        errors.append("Profile is missing (upload a CV or reuse saved profile).")
    if posting is None or (isinstance(posting, str) and not posting.strip()):
        errors.append("Posting text is empty.")
    return {"ok": len(errors) == 0, "errors": errors}
