"""
Evaluation script for Milestone 5 (Playbook RAG and Writer Agent).
Generates 5 drafts with the real remote LLM:
1. real_job_1 (Job mode, expected APPLY)
2. real_project_1 (Freelance mode, expected CAUTION)
3. real_project_2 (Freelance mode, expected CAUTION)
4. synthetic_job_1 (Job mode, force mode bypass SKIP)
5. synthetic_project_1 (Freelance mode, force mode bypass SKIP)

Reports per draft:
- Word count vs length rule (Cover letter: 220-300 words; Proposal: 150-250 words)
- Structure check (Cover letter: 4 sections; Proposal: 6 sections)
- Flagged claims before and after regeneration
"""

import sys
import json
import re
from pathlib import Path

# Enable unbuffered / line-buffered stdout for clear logging
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from jsb.schemas import ProfileJSON, PostingJSON, BudgetInfo
from jsb.agents.matcher import match
from jsb.agents.writer import write_draft
from jsb.llm_client import get_client


def load_profile() -> ProfileJSON:
    with open(config.PROFILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ProfileJSON(**data)


def load_synthetic_postings() -> dict:
    """Load synthetic postings from labels.json."""
    labels_path = config.SAMPLES_POSTINGS_DIR / "labels.json"
    with open(labels_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    postings = {}
    for fname, d in raw.items():
        budget = None
        if d.get("budget"):
            b = d["budget"]
            budget = BudgetInfo(amount=b.get("amount", 0), kind=b.get("kind", "fixed"))
        
        # Read raw text if available for rich summary
        raw_text_file = config.SAMPLES_POSTINGS_DIR / fname
        summary_text = ""
        if raw_text_file.exists():
            summary_text = raw_text_file.read_text(encoding="utf-8")[:1000]

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
            summary=summary_text or d.get("summary", ""),
        )
        postings[fname] = p
    return postings


def load_real_postings() -> dict:
    """Parse real_labels.txt into PostingJSON objects."""
    labels_path = config.SAMPLES_POSTINGS_DIR / "real_labels.txt"
    text = labels_path.read_text(encoding="utf-8")

    postings = {}
    blocks = re.split(r"\[(\w+\.txt)\]", text)
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

        # Read raw text if available
        raw_text_file = config.SAMPLES_POSTINGS_DIR / fname
        summary_text = ""
        company = ""
        if raw_text_file.exists():
            raw_text = raw_text_file.read_text(encoding="utf-8")
            summary_text = raw_text[:1000]
            for ln in raw_text.splitlines()[:5]:
                if ln.startswith("Company:"):
                    company = ln.split(":", 1)[1].strip()
                elif ln.startswith("Platform:"):
                    company = ln.split(":", 1)[1].strip() + " Client"

        p = PostingJSON(
            title=fname.replace(".txt", "").replace("_", " ").title(),
            company_or_client=company or "Hiring Team",
            type="project" if "project" in fname else ptype,
            must_have=must,
            nice_to_have=nice,
            deal_breakers=deal_breakers,
            budget=budget,
            deadline_days=deadline_days,
            vague_scope=vague_scope,
            summary=summary_text,
        )
        postings[fname] = p
    return postings


def check_cover_letter_structure(body: str) -> dict:
    """
    Checks the 4 required sections of a cover letter per §7:
    1. Hook tied to company/role
    2. 2-3 achievements matching must-haves
    3. Gap handling / eagerness to learn
    4. Call to action
    """
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    lower_body = body.lower()
    
    # 1. Hook
    hook_keywords = ["excited to apply", "application for", "admire", "mission", "interest in", "position of", "role of", "opportunity", "team", "engineer"]
    has_hook = any(k in lower_body for k in hook_keywords)
    
    # 2. Achievements (quantified numbers, CV projects or roles)
    achievement_keywords = ["engineered", "developed", "built", "spearheaded", "deployed", "implemented", "rag", "pytorch", "model", "pipeline", "creativa", "neuromatch"]
    has_achievements = any(k in lower_body for k in achievement_keywords)
    
    # 3. Gap handling or proactive learning
    gap_keywords = ["gap", "deepen", "eager", "learn", "rapidly master", "expand", "growth", "foundation", "proactive", "adapt", "strengthen", "familiarize"]
    has_gap_handling = any(k in lower_body for k in gap_keywords)
    
    # 4. Call to Action
    cta_keywords = ["discuss", "interview", "conversation", "look forward", "thank you", "opportunity to speak", "contact", "sincerely", "regards"]
    has_cta = any(k in lower_body for k in cta_keywords)
    
    all_ok = has_hook and has_achievements and has_gap_handling and has_cta
    return {
        "passed": all_ok,
        "paragraphs_count": len(paragraphs),
        "hook": has_hook,
        "achievements": has_achievements,
        "gap_handling": has_gap_handling,
        "cta": has_cta,
    }


def check_proposal_structure(body: str) -> dict:
    """
    Checks the 6 required sections of a proposal per §7:
    1. Restate problem in client's words
    2. 3-step technical approach
    3. Timeline
    4. Price
    5. Proof point
    6. Clarifying question
    """
    lower_body = body.lower()
    
    # 1. Problem
    problem_keywords = ["understand", "goal", "challenge", "objective", "requirement", "problem", "need", "looking for", "project"]
    has_problem = any(k in lower_body for k in problem_keywords)
    
    # 2. 3-step approach
    has_step_1 = bool(re.search(r"(step 1|phase 1|1[\.\)]|first)", lower_body))
    has_step_2 = bool(re.search(r"(step 2|phase 2|2[\.\)]|second)", lower_body))
    has_step_3 = bool(re.search(r"(step 3|phase 3|3[\.\)]|third)", lower_body))
    has_3_steps = has_step_1 and has_step_2 and has_step_3
    
    # 3. Timeline
    timeline_keywords = ["day", "week", "timeline", "schedule", "turnaround", "delivery"]
    has_timeline = any(k in lower_body for k in timeline_keywords)
    
    # 4. Price
    has_price = bool(re.search(r"(\$|₹|price|budget|rate|fixed|fee|quote)", lower_body))
    
    # 5. Proof point
    proof_keywords = ["proof point", "past project", "previously", "experience", "built", "developed", "portfolio", "engineered", "completed", "in a past"]
    has_proof = any(k in lower_body for k in proof_keywords)
    
    # 6. Question
    has_question = "?" in body
    
    all_ok = has_problem and has_3_steps and has_timeline and has_price and has_proof and has_question
    return {
        "passed": all_ok,
        "problem": has_problem,
        "3_steps": has_3_steps,
        "timeline": has_timeline,
        "price": has_price,
        "proof_point": has_proof,
        "question": has_question,
    }


def main():
    profile = load_profile()
    reals = load_real_postings()
    synthetics = load_synthetic_postings()

    postings_to_eval = [
        ("real_job_1.txt", reals["real_job_1.txt"], False),
        ("real_project_1.txt", reals["real_project_1.txt"], False),
        ("real_project_2.txt", reals["real_project_2.txt"], False),
        ("synthetic_job_1.txt", synthetics["synthetic_job_1.txt"], True),
        ("synthetic_project_1.txt", synthetics["synthetic_project_1.txt"], True),
    ]

    client = get_client()
    print(f"=== M5 EVALUATION (LLM Mode: {client.mode}, Base URL: {client.base_url}) ===")
    print(f"Candidate: {profile.name} ({profile.years_experience} years exp)\n")

    results = []

    for fname, posting, force_mode in postings_to_eval:
        match_res = match(profile, posting)
        name = fname.replace(".txt", "")
        print(f"--- Evaluating {name} ---")
        print(f"Type: {posting.type} | Match Score: {match_res.score:.1f} | Verdict: {match_res.verdict}")
        if match_res.verdict == "SKIP" and force_mode:
            print("  [Force Mode]: Bypassing SKIP gate for test evaluation as requested.")

        draft, trace = write_draft(profile, posting, match_res, client=client, return_trace=True)

        word_count = len(draft.body.split())
        is_freelance = (posting.type == "project")

        if is_freelance:
            min_w, max_w = 150, 250
            rule_str = "150-250 words"
            struct_check = check_proposal_structure(draft.body)
        else:
            min_w, max_w = 220, 300
            rule_str = "220-300 words"
            struct_check = check_cover_letter_structure(draft.body)

        length_ok = (min_w <= word_count <= max_w)
        struct_ok = struct_check["passed"]

        init_flagged = trace.get("initial_flagged", [])
        final_flagged = trace.get("final_flagged", [])
        regenerated = trace.get("regenerated", False)

        print(f"  Draft Kind: {draft.kind}")
        print(f"  Subject: {draft.subject}")
        print(f"  Word Count: {word_count} (Rule: {rule_str}) -> {'PASS' if length_ok else 'FAIL'}")
        print(f"  Structure Check: {'PASS' if struct_ok else 'FAIL'} (Details: {struct_check})")
        print(f"  Flagged claims before regen: {len(init_flagged)} {init_flagged}")
        if regenerated:
            print(f"  Auto-regenerated once: YES")
        print(f"  Flagged claims after regen: {len(final_flagged)} {final_flagged}")
        print(f"  Final claims OK: {draft.claims_ok}\n")

        results.append({
            "name": name,
            "kind": draft.kind,
            "verdict": match_res.verdict,
            "score": match_res.score,
            "word_count": word_count,
            "rule": rule_str,
            "length_ok": length_ok,
            "struct_ok": struct_ok,
            "struct_details": struct_check,
            "init_flagged": len(init_flagged),
            "final_flagged": len(final_flagged),
            "regenerated": regenerated,
            "subject": draft.subject,
            "body": draft.body,
        })

    # Summary Table
    print("\n" + "=" * 95)
    print(f"{'Posting':<22} | {'Kind':<13} | {'Verdict':<8} | {'Words (Rule)':<16} | {'Structure':<10} | {'Flagged (Pre->Post)':<20}")
    print("-" * 95)
    all_clean = True
    for r in results:
        len_str = f"{r['word_count']} ({r['rule']})"
        flag_str = f"{r['init_flagged']} -> {r['final_flagged']}" + (" (regen)" if r['regenerated'] else "")
        struct_str = "PASS" if r['struct_ok'] else "FAIL"
        print(f"{r['name']:<22} | {r['kind']:<13} | {r['verdict']:<8} | {len_str:<16} | {struct_str:<10} | {flag_str:<20}")
        if r['final_flagged'] > 0 or not r['length_ok'] or not r['struct_ok']:
            all_clean = False
    print("=" * 95)

    if all_clean:
        print("\nALL 5 DRAFTS PASSED: No flagged claims, word count inside rule bands, and structure verified!")
    else:
        print("\nSOME DRAFTS NEED TUNING: Review details above.")

    # Save results to a json file for inspection
    out_file = config.DRAFTS_DIR / "m5_evaluation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation results to {out_file}")


if __name__ == "__main__":
    main()
