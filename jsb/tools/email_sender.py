"""
JSBrother — Gmail SMTP email sender (M7)
Dry-run ON by default. Optional CV attachment. Never logs secrets.
"""

from __future__ import annotations

import logging
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger("jsbrother.email_sender")


@dataclass
class EmailResult:
    status: str  # dry_run | sent | error | skipped_no_credentials
    recipient: str
    subject: str
    dry_run: bool
    detail: str = ""


def send_email(
    to_addr: str,
    subject: str,
    body: str,
    from_addr: Optional[str] = None,
    attachment_path: Optional[Path] = None,
    extra_attachment_path: Optional[Path] = None,
    dry_run: Optional[bool] = None,
) -> EmailResult:
    """
    Send via Gmail SMTP SSL (port 465).
    If dry_run is True (default from EMAIL_DRY_RUN), do not connect — just log and return.
    """
    if dry_run is None:
        dry_run = config.EMAIL_DRY_RUN
    sender = from_addr or config.SMTP_USER or "jsbrother@localhost"
    recipient = (to_addr or "").strip()
    if not recipient:
        return EmailResult(
            status="error",
            recipient="",
            subject=subject,
            dry_run=bool(dry_run),
            detail="Missing recipient address",
        )

    if dry_run:
        logger.info(f"[DRY-RUN] Would email {recipient!r} subject={subject!r}")
        return EmailResult(
            status="dry_run",
            recipient=recipient,
            subject=subject,
            dry_run=True,
            detail="Email not sent (EMAIL_DRY_RUN=true)",
        )

    user = config.SMTP_USER
    password = config.SMTP_APP_PASSWORD
    if not user or not password:
        return EmailResult(
            status="skipped_no_credentials",
            recipient=recipient,
            subject=subject,
            dry_run=False,
            detail="SMTP_USER / SMTP_APP_PASSWORD not set",
        )

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)

    for att in (attachment_path, extra_attachment_path):
        if not att:
            continue
        path = Path(att)
        if path.exists() and path.is_file():
            data = path.read_bytes()
            msg.add_attachment(
                data,
                maintype="application",
                subtype="octet-stream",
                filename=path.name,
            )

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(config.SMTP_HOST, config.SMTP_PORT, context=context) as server:
            server.login(user, password)
            server.send_message(msg)
        logger.info(f"Email sent to {recipient}")
        return EmailResult(
            status="sent",
            recipient=recipient,
            subject=subject,
            dry_run=False,
            detail="OK",
        )
    except Exception as e:
        logger.error(f"SMTP send failed: {e}")
        return EmailResult(
            status="error",
            recipient=recipient,
            subject=subject,
            dry_run=False,
            detail=str(e),
        )
