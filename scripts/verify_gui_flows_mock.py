"""
M8 verification without driving the browser:
Simulates what the GUI calls for Job Seeker (real_job_1) and Freelancer (real_project_1)
in mock mode with dry-run actions.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["LLM_MODE"] = "mock"

import config
from jsb import llm_client as llm_mod
from jsb.agents.action_agent import approve_and_send, freelancer_save_and_log
from jsb.agents.writer import refine_draft
from jsb.gui_helpers import friendly_error, score_gauge_svg, tracker_metrics
from jsb.llm_client import LLMClient
from jsb.orchestrator import run
from jsb.schemas import ProfileJSON
from jsb.tools.tracker import read_tracker

llm_mod._default_client = LLMClient(mode="mock")


def main() -> None:
    out = {"job": {}, "freelance": {}, "helpers": {}}
    profile = ProfileJSON(**json.loads(config.PROFILE_PATH.read_text(encoding="utf-8")))
    client = LLMClient(mode="mock")

    job_text = (config.SAMPLES_POSTINGS_DIR / "real_job_1.txt").read_text(encoding="utf-8")
    r_job = run(profile=profile, posting=job_text, mode="job", client=client, force=False)
    out["job"] = {
        "verdict": r_job.match.verdict,
        "score": r_job.match.score,
        "skipped": r_job.skipped,
        "has_draft": r_job.draft is not None,
        "kind": r_job.draft.kind if r_job.draft else None,
    }
    assert r_job.match is not None
    if r_job.draft:
        refined = refine_draft(r_job.draft, "shorter", profile, r_job.posting, r_job.match, client)
        assert refined.body
        act = approve_and_send(
            profile, r_job.posting, r_job.match, refined,
            recipient=profile.email or "test@example.com",
            dry_run=True,
            mode="job",
        )
        out["job"]["email_status"] = act["email_status"]
        out["job"]["draft_path"] = act["draft_path"]

    proj_text = (config.SAMPLES_POSTINGS_DIR / "real_project_1.txt").read_text(encoding="utf-8")
    r_fl = run(profile=profile, posting=proj_text, mode="project", client=client, force=False)
    out["freelance"] = {
        "verdict": r_fl.match.verdict,
        "score": r_fl.match.score,
        "skipped": r_fl.skipped,
        "has_draft": r_fl.draft is not None,
        "kind": r_fl.draft.kind if r_fl.draft else None,
    }
    if r_fl.skipped and r_fl.draft is None:
        r_fl = run(profile=profile, posting=proj_text, mode="project", client=client, force=True)
        out["freelance"]["forced"] = True
        out["freelance"]["has_draft"] = r_fl.draft is not None
    if r_fl.draft:
        act2 = freelancer_save_and_log(
            profile, r_fl.posting, r_fl.match, r_fl.draft, email_to=None, dry_run=True
        )
        out["freelance"]["email_status"] = act2["email_status"]
        out["freelance"]["draft_path"] = act2["draft_path"]

    svg = score_gauge_svg(r_job.match.score, r_job.match.verdict)
    out["helpers"]["gauge_ok"] = "jsb-gauge" in svg
    out["helpers"]["friendly"] = friendly_error(RuntimeError("connection refused"))
    out["helpers"]["tracker"] = tracker_metrics(read_tracker())

    path = ROOT / "logs" / "m8_gui_verify_mock.json"
    try:
        import app as app_mod  # noqa: F401
        out["helpers"]["app_import"] = True
    except Exception as e:
        out["helpers"]["app_import"] = False
        out["helpers"]["app_import_error"] = type(e).__name__ + ": " + str(e)[:120]
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        raise

    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print("VERIFY OK ->", path)


if __name__ == "__main__":
    main()
