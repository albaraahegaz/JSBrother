"""Re-evaluate a single posting for M5 length/structure/claims checks."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_drafts import (
    load_profile,
    load_real_postings,
    load_synthetic_postings,
    check_cover_letter_structure,
    check_proposal_structure,
)
from jsb.agents.matcher import match
from jsb.agents.writer import write_draft
from jsb.llm_client import get_client


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "synthetic_job_1.txt"
    profile = load_profile()
    reals = load_real_postings()
    synthetics = load_synthetic_postings()
    pool = {**reals, **synthetics}
    if name not in pool:
        raise SystemExit(f"Unknown posting: {name}. Keys: {list(pool)}")

    posting = pool[name]
    client = get_client()
    match_res = match(profile, posting)
    draft, trace = write_draft(profile, posting, match_res, client=client, return_trace=True)

    words = len(draft.body.split())
    is_freelance = posting.type == "project"
    min_w, max_w = (150, 250) if is_freelance else (220, 300)
    struct = (
        check_proposal_structure(draft.body)
        if is_freelance
        else check_cover_letter_structure(draft.body)
    )
    length_ok = min_w <= words <= max_w
    result = {
        "name": name.replace(".txt", ""),
        "word_count": words,
        "length_ok": length_ok,
        "struct_ok": struct["passed"],
        "struct_details": struct,
        "claims_ok": draft.claims_ok,
        "final_flagged": trace.get("final_flagged", []),
        "regenerated": trace.get("regenerated"),
        "subject": draft.subject,
        "body": draft.body,
    }
    out = PROJECT_ROOT / "logs" / f"m5_eval_one_{result['name']}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "body"}, indent=2))
    print(f"PASS={length_ok and struct['passed'] and draft.claims_ok} saved={out}")


if __name__ == "__main__":
    main()
