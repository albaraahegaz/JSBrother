"""
JSBrother — Action Agent (M7)
Approve & Send: save draft, email (optional), append tracker.
Freelancer path: save/download copy + log (email optional).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import config
from jsb.schemas import Draft, MatchResult, PostingJSON, ProfileJSON
from jsb.tools.email_sender import EmailResult, send_email
from jsb.tools.tracker import append_tracker_row

logger = logging.getLogger("jsbrother.action_agent")


def _safe_slug(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", (text or "draft").strip())[:max_len]
    return slug.strip("_") or "draft"


def save_draft(
    draft: Draft,
    posting: Optional[PostingJSON] = None,
    drafts_dir: Optional[Path] = None,
) -> Path:
    """Persist draft body to data/drafts/ and return path."""
    out_dir = Path(drafts_dir) if drafts_dir else config.DRAFTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    title = posting.title if posting else draft.kind
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{stamp}_{draft.kind}_{_safe_slug(title)}.txt"
    content = f"Subject: {draft.subject}\n\n{draft.body}\n"
    path.write_text(content, encoding="utf-8")
    logger.info(f"Draft saved → {path}")
    return path


def approve_and_send(
    profile: ProfileJSON,
    posting: PostingJSON,
    match: MatchResult,
    draft: Draft,
    recipient: Optional[str] = None,
    attach_cv: Optional[Path] = None,
    extra_attachment: Optional[Path] = None,
    dry_run: Optional[bool] = None,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Job-seeker path: save draft, email application, log tracker.
    Freelancer path: save draft (download/copy), optional email, log tracker.
    """
    resolved_mode = mode or ("project" if posting.type == "project" else "job")
    draft_path = save_draft(draft, posting)

    to_addr = recipient or posting.contact_email or profile.email or ""
    email_result: Optional[EmailResult] = None

    if resolved_mode == "project" and not to_addr:
        # Freelancer download/copy path — email optional
        email_status = "not_requested"
    else:
        email_result = send_email(
            to_addr=to_addr or "undeliverable@example.com",
            subject=draft.subject or f"Application: {posting.title}",
            body=draft.body,
            attachment_path=attach_cv,
            extra_attachment_path=extra_attachment,
            dry_run=dry_run,
        )
        email_status = email_result.status
        if not to_addr:
            email_status = "error_missing_recipient"

    append_tracker_row(
        mode=resolved_mode,
        title=posting.title,
        company=posting.company_or_client or "",
        score=match.score,
        verdict=match.verdict,
        draft_path=str(draft_path),
        email_status=email_status,
        recipient=to_addr,
    )

    return {
        "draft_path": str(draft_path),
        "email_status": email_status,
        "recipient": to_addr,
        "email": email_result.__dict__ if email_result else None,
        "mode": resolved_mode,
    }


def freelancer_save_and_log(
    profile: ProfileJSON,
    posting: PostingJSON,
    match: MatchResult,
    draft: Draft,
    email_to: Optional[str] = None,
    dry_run: Optional[bool] = None,
) -> Dict[str, Any]:
    """Freelancer convenience: always save draft + tracker; email only if email_to set."""
    return approve_and_send(
        profile=profile,
        posting=posting,
        match=match,
        draft=draft,
        recipient=email_to,
        dry_run=dry_run,
        mode="project",
    )
