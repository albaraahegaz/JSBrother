"""
Re-evaluate only synthetic_project_3.txt against the updated labels.json.
Quick fix verification before running full M3 evaluation.
"""

import sys
import re
import json
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
load_dotenv(override=True)

from jsb.agents.posting_analyzer import analyze_posting


def normalize_skill(skill: str) -> str:
    cleaned = "".join(c.lower() for c in skill.strip() if c.isalnum() or c in (" ", "-", ".", "/", "+"))
    return re.sub(r"\s+", " ", cleaned).strip()


def check_skill_match(ground_truth_item: str, extracted_norm: set) -> bool:
    for alias in [normalize_skill(a) for a in ground_truth_item.split("|") if a.strip()]:
        if alias in extracted_norm:
            return True
        for ext in extracted_norm:
            if len(alias) >= 3 and (alias in ext or ext in alias):
                return True
    return False


labels = json.loads(Path("data/samples/postings/labels.json").read_text(encoding="utf-8"))
target = "synthetic_project_3.txt"
expected = labels[target]

print(f"Re-evaluating {target}...")
posting = analyze_posting(Path("data/samples/postings") / target)

all_norm = {normalize_skill(s) for s in posting.must_have + posting.nice_to_have}
print(f"Extracted must_have:  {posting.must_have}")

matched = [s for s in expected["must_have"] if check_skill_match(s, all_norm)]
missing = [s for s in expected["must_have"] if not check_skill_match(s, all_norm)]
recall = len(matched) / len(expected["must_have"]) * 100

print(f"MUST-HAVE: {len(matched)}/{len(expected['must_have'])} ({recall:.1f}%)")
if missing:
    print(f"Missing: {missing}")
if recall >= 80:
    print("✅ PASSED 80% target!")
else:
    print(f"❌ STILL FAILED: {recall:.1f}%")
