"""M6 remote smoke: structured profile+posting through orchestrator (skip analyzers)."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from jsb.schemas import ProfileJSON, PostingJSON
from jsb.orchestrator import run
from jsb.llm_client import get_client


def main():
    client = get_client()
    print(f"LLM mode={client.mode} url={client.base_url}")
    if client.mode != "remote":
        print("WARNING: not remote; set LLM_MODE=remote in .env")

    profile = ProfileJSON(**json.loads(config.PROFILE_PATH.read_text(encoding="utf-8")))
    text = (config.SAMPLES_POSTINGS_DIR / "real_job_1.txt").read_text(encoding="utf-8")
    # Minimal structured posting from hand labels (avoid re-extract)
    from scripts.eval_drafts import load_real_postings
    posting = load_real_postings()["real_job_1.txt"]

    result = run(
        profile=profile,
        posting=posting,
        mode="job",
        force=False,
        client=client,
        skip_analyzers=True,
    )
    out = {
        "verdict": result.match.verdict,
        "score": result.match.score,
        "skipped": result.skipped,
        "has_draft": result.draft is not None,
        "draft_kind": result.draft.kind if result.draft else None,
        "claims_ok": result.draft.claims_ok if result.draft else None,
        "word_count": len(result.draft.body.split()) if result.draft else 0,
        "trace_steps": [s["step"] for s in result.trace.get("steps", [])],
        "total_ms": result.trace.get("total_ms"),
        "llm_mode": result.trace.get("llm_mode"),
    }
    path = PROJECT_ROOT / "logs" / "m6_orchestrator_remote.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    assert result.match.verdict in ("APPLY", "CAUTION", "SKIP")
    if not result.skipped:
        assert result.draft is not None
    print("REMOTE ORCHESTRATOR OK ->", path)


if __name__ == "__main__":
    main()
