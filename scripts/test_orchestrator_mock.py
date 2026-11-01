"""Smoke-test M6 orchestrator in mock mode (no remote LLM required)."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from jsb.schemas import ProfileJSON, PostingJSON, BudgetInfo
from jsb.orchestrator import run
from jsb.memory import SessionMemory
from jsb.llm_client import LLMClient


def main():
    client = LLMClient(mode="mock")
    profile = ProfileJSON(**json.loads(config.PROFILE_PATH.read_text(encoding="utf-8")))
    posting = PostingJSON(
        title="ML Engineer",
        company_or_client="Acme AI",
        type="job",
        must_have=["python", "pytorch", "machine learning"],
        nice_to_have=["docker"],
        min_years=2,
        deal_breakers=[],
        summary="Build ML models with Python and PyTorch.",
    )
    mem = SessionMemory()
    result = run(
        profile=profile, posting=posting, mode="job", memory=mem,
        client=client, skip_analyzers=True,
    )
    assert result.match.verdict in ("APPLY", "CAUTION", "SKIP")
    assert result.draft is not None or result.skipped
    assert "steps" in result.trace
    assert result.trace["llm_mode"] == "mock"
    print("mock job OK", result.match.verdict, result.match.score, "draft=", bool(result.draft))

    # SKIP gate + force
    skip_posting = PostingJSON(
        title="Impossible Role",
        company_or_client="X",
        type="job",
        must_have=["cobol", "fortran", "mainframe", "sap abap"],
        nice_to_have=[],
        min_years=20,
        deal_breakers=["Must have 20 years COBOL"],
        summary="Legacy enterprise.",
    )
    r_skip = run(
        profile=profile, posting=skip_posting, mode="job", force=False,
        client=client, skip_analyzers=True,
    )
    assert r_skip.skipped or r_skip.match.verdict == "SKIP"
    print("skip gate OK", r_skip.match.verdict, "skipped=", r_skip.skipped)

    r_force = run(
        profile=profile, posting=skip_posting, mode="job", force=True,
        client=client, skip_analyzers=True,
    )
    assert r_force.draft is not None
    print("force draft OK", r_force.match.verdict, "words=", len(r_force.draft.body.split()))

    # Freelancer path
    proj = PostingJSON(
        title="RAG Notebook",
        company_or_client="Client",
        type="project",
        must_have=["python", "pytorch"],
        nice_to_have=["fastapi"],
        budget=BudgetInfo(amount=500, kind="fixed"),
        deadline_days=14,
        summary="Build a classification notebook.",
    )
    r_proj = run(
        profile=profile, posting=proj, mode="project",
        client=client, skip_analyzers=True,
    )
    assert r_proj.mode == "project"
    if r_proj.draft:
        assert r_proj.draft.kind == "proposal"
    print("mock project OK", r_proj.match.verdict, "kind=", getattr(r_proj.draft, "kind", None))
    print("ALL MOCK ORCHESTRATOR CHECKS PASSED")


if __name__ == "__main__":
    main()
