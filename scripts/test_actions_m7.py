"""M7 dry-run actions test + optional real email if SMTP creds exist."""
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Prefer env already loaded via config; do not print secrets
os.environ.setdefault("LLM_MODE", "mock")

import config
from jsb.schemas import ProfileJSON, PostingJSON, MatchResult, Draft
from jsb.agents.action_agent import save_draft, approve_and_send, freelancer_save_and_log
from jsb.tools.email_sender import send_email
from jsb.tools.tracker import read_tracker


def main():
    profile = ProfileJSON(**json.loads(config.PROFILE_PATH.read_text(encoding="utf-8")))
    posting = PostingJSON(
        title="Test Role",
        company_or_client="TestCo",
        type="job",
        must_have=["python"],
        summary="Test posting for M7.",
        contact_email=profile.email or "test@example.com",
    )
    match = MatchResult(
        score=80.0,
        must_coverage=1.0,
        nice_coverage=0.0,
        matched=["python"],
        missing_must=[],
        missing_nice=[],
        deal_breaker_hits=[],
        verdict="APPLY",
        reasons=["r1", "r2", "r3"],
        advice="Apply.",
    )
    draft = Draft(
        kind="cover_letter",
        subject="Application: Test Role",
        body="Dear Team,\n\nThis is a dry-run test draft for JSBrother M7.\n\nSincerely,\n" + (profile.name or "Candidate"),
        claims_ok=True,
        flagged_claims=[],
    )

    # 1) Dry-run send (must work without caring about credentials)
    dry = send_email(
        to_addr=posting.contact_email,
        subject=draft.subject,
        body=draft.body,
        dry_run=True,
    )
    assert dry.status == "dry_run", dry
    print("dry-run email OK:", dry.status)

    # 2) Approve & send dry-run + tracker
    result = approve_and_send(
        profile, posting, match, draft, dry_run=True
    )
    assert Path(result["draft_path"]).exists()
    assert result["email_status"] == "dry_run"
    rows = read_tracker()
    assert rows and rows[-1]["title"] == "Test Role"
    print("approve_and_send dry-run OK:", result["draft_path"])

    # 3) Freelancer path
    proj = posting.model_copy(update={"type": "project", "title": "Freelance Gig"})
    prop = draft.model_copy(update={"kind": "proposal", "subject": "Proposal: Gig"})
    fr = freelancer_save_and_log(profile, proj, match, prop, email_to=None, dry_run=True)
    assert Path(fr["draft_path"]).exists()
    print("freelancer save/log OK:", fr["email_status"])

    # 4) Real email only if credentials present
    has_creds = bool(config.SMTP_USER and config.SMTP_APP_PASSWORD)
    real_log = PROJECT_ROOT / "logs" / "m7_real_email.txt"
    if has_creds:
        # Temporarily force non-dry-run for one test to self
        real = send_email(
            to_addr=config.SMTP_USER,  # send to self
            subject="[JSBrother M7] Real SMTP test",
            body=(
                "This is an automated JSBrother Milestone 7 SMTP connectivity test.\n"
                "If you received this, Gmail app-password sending works.\n"
            ),
            dry_run=False,
        )
        real_log.write_text(
            f"status={real.status}\nrecipient={real.recipient}\ndetail={real.detail}\n",
            encoding="utf-8",
        )
        print("real email result:", real.status, real.detail)
        if real.status != "sent":
            print("PENDING/FAILED real email — see logs/m7_real_email.txt")
        else:
            # Also log a tracker row for the real send
            approve_and_send(
                profile,
                posting.model_copy(update={"title": "M7 Real SMTP Test"}),
                match,
                draft.model_copy(update={"subject": "[JSBrother M7] Real SMTP test", "body": real.detail}),
                recipient=config.SMTP_USER,
                dry_run=False,
            )
    else:
        msg = "PENDING: real test email needs credentials (SMTP_USER / SMTP_APP_PASSWORD)"
        real_log.write_text(msg + "\n", encoding="utf-8")
        print(msg)

    print("M7 ACTION CHECKS COMPLETE")


if __name__ == "__main__":
    main()
