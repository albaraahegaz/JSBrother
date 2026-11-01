"""
M4 Eval: Run matcher against all 11 postings in data/samples/postings/.
Builds PostingJSON from labels.json (synthetics) and real_labels.txt (reals) — no LLM calls.
Prints a table: posting, score, verdict, matched, missing must, deal-breakers hit.

Usage:
    python scripts/eval_postings.py
"""
import sys
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os
os.environ["LLM_MODE"] = "mock"

import config
from jsb.schemas import ProfileJSON, PostingJSON, BudgetInfo
from jsb.agents.matcher import compute_match_score_and_verdict, evaluate_deal_breakers_with_warnings

POSTINGS_DIR = config.SAMPLES_POSTINGS_DIR
PROFILE_PATH = config.PROFILE_PATH


def load_profile() -> ProfileJSON:
    with open(PROFILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ProfileJSON(**data)


def load_synthetic_postings() -> dict:
    """Load synthetic postings from labels.json."""
    labels_path = POSTINGS_DIR / "labels.json"
    with open(labels_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    postings = {}
    for fname, d in raw.items():
        budget = None
        if d.get("budget"):
            b = d["budget"]
            budget = BudgetInfo(amount=b.get("amount", 0), kind=b.get("kind", "fixed"))
        p = PostingJSON(
            title=d.get("title", fname),
            company_or_client=d.get("company_or_client", ""),
            type=d.get("type", "job"),
            must_have=d.get("must_have", []),
            nice_to_have=d.get("nice_to_have", []),
            min_years=d.get("min_years"),
            deal_breakers=d.get("deal_breakers", []),
            budget=budget,
            deadline_days=d.get("deadline_days"),
            vague_scope=d.get("vague_scope", False),
        )
        postings[fname] = p
    return postings


def load_real_postings() -> dict:
    """Parse real_labels.txt into PostingJSON objects."""
    labels_path = POSTINGS_DIR / "real_labels.txt"
    text = labels_path.read_text(encoding="utf-8")

    postings = {}
    # Split by [filename] blocks
    blocks = re.split(r"\[(\w+\.txt)\]", text)
    # blocks[0] = preamble, then alternating name, content
    for i in range(1, len(blocks), 2):
        fname = blocks[i]
        content = blocks[i + 1]

        must, nice, deal_breakers = [], [], []
        ptype = "job"
        budget = None
        deadline_days = None
        vague_scope = False

        section = None
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                # detect section headers from comment
                lower = line.lower()
                if "# must" in lower:
                    section = "must"
                elif "# nice" in lower:
                    section = "nice"
                elif "# deal" in lower:
                    section = "deal"
                elif "# meta" in lower:
                    section = "meta"
                continue
            if section == "meta":
                if line.startswith("type:"):
                    ptype = line.split(":", 1)[1].strip()
                elif line.startswith("budget:") and "none" not in line.lower():
                    # Try to extract a number
                    nums = re.findall(r"[\d,]+", line)
                    if nums:
                        amt = float(nums[0].replace(",", ""))
                        budget = BudgetInfo(amount=amt, kind="fixed")
                elif line.startswith("deadline:") and "none" not in line.lower():
                    days_m = re.search(r"(\d+)\s*days?", line, re.IGNORECASE)
                    if days_m:
                        deadline_days = int(days_m.group(1))
                elif line.startswith("red_flags:") and "none" not in line.lower():
                    vague_scope = True
            elif section == "must" and line.lower() != "none":
                must.append(line)
            elif section == "nice" and line.lower() != "none":
                nice.append(line)
            elif section == "deal" and line.lower() != "none":
                deal_breakers.append(line)

        p = PostingJSON(
            title=fname.replace(".txt", "").replace("_", " ").title(),
            type="project" if "project" in fname else ptype,
            must_have=must,
            nice_to_have=nice,
            deal_breakers=deal_breakers,
            budget=budget,
            deadline_days=deadline_days,
            vague_scope=vague_scope,
        )
        postings[fname] = p
    return postings


def fmt_list(items, n=3):
    if not items:
        return "-"
    s = ", ".join(str(x) for x in items[:n])
    if len(items) > n:
        s += f" +{len(items)-n}"
    return s


def main():
    profile = load_profile()
    print(f"\nCandidate: {profile.name} | {profile.years_experience}y exp | {len(profile.skills)} skills\n")

    synthetics = load_synthetic_postings()
    reals = load_real_postings()

    all_postings = {}
    # Order: real first, then synthetic
    for fname in sorted([
        "real_job_1.txt", "real_job_2.txt", "real_job_3.txt",
        "real_project_1.txt", "real_project_2.txt",
    ]):
        if fname in reals:
            all_postings[fname] = reals[fname]
    for fname in sorted([
        "synthetic_job_1.txt", "synthetic_job_2.txt", "synthetic_job_3.txt",
        "synthetic_project_1.txt", "synthetic_project_2.txt", "synthetic_project_3.txt",
    ]):
        if fname in synthetics:
            all_postings[fname] = synthetics[fname]

    rows = []
    for fname, posting in all_postings.items():
        score, must_cov, nice_cov, matched, miss_must, miss_nice, db_hits, verdict, fl = \
            compute_match_score_and_verdict(profile, posting)
        _, warnings = evaluate_deal_breakers_with_warnings(posting, profile)
        rows.append({
            "name": fname.replace(".txt", ""),
            "score": score,
            "verdict": verdict,
            "matched": f"{len(matched)}/{len(posting.must_have)+len(posting.nice_to_have)}",
            "miss_must": fmt_list(miss_must) if miss_must else "-",
            "db_hits": fmt_list(db_hits) if db_hits else "-",
            "warnings": fmt_list(warnings) if warnings else "-",
        })

    # Print table
    print(f"{'Posting':<23} {'Score':>5} {'Verdict':>7}  {'Matched':>7}  {'Missing Must':<30}  {'Deal-breakers':<30}  {'Warnings'}")
    print("-" * 135)
    for r in rows:
        print(f"{r['name']:<23} {r['score']:>5.1f} {r['verdict']:>7}  {r['matched']:>7}  {r['miss_must']:<30}  {r['db_hits']:<30}  {r['warnings']}")

    print(f"\n{len(rows)} postings evaluated.\n")

    # Sanity checks
    print("--- Sanity checks ---")
    flags = []
    for r in rows:
        name = r["name"]
        verdict = r["verdict"]
        score = r["score"]
        if "real_job_3" in name and verdict != "SKIP":
            flags.append(f"  [FAIL] {name}: expected SKIP (3 deal-breakers), got {verdict}")
        if verdict == "ERROR":
            flags.append(f"  [ERROR] {name}: failed to compute")
        # Flag if score > 85 but verdict is SKIP (likely deal-breaker logic ok, but check)
        if score >= 85 and verdict == "SKIP":
            flags.append(f"  [INFO]  {name}: score={score:.1f} but SKIP — deal-breaker triggered (expected)")
        # Flag suspiciously high scores for postings where candidate clearly shouldn't qualify
        if "synthetic_job_3" in name and verdict not in ("SKIP", "CAUTION"):
            flags.append(f"  [ODD]   {name}: full-stack TypeScript job, expected SKIP/CAUTION, got {verdict}")
        if "synthetic_project_3" in name and verdict != "SKIP":
            flags.append(f"  [INFO]  {name}: vague+low-budget project — check downgrade rule. Got {verdict}")

    if flags:
        for f in flags:
            print(f)
    else:
        print("  All sanity checks passed.")
    print()


if __name__ == "__main__":
    main()
