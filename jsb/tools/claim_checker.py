"""
JSBrother — Claim Checker Module
Validates generated application drafts (cover letters, proposals) against the candidate's ProfileJSON.
Flags unsupported or fabricated claims per §7:
1. Inflated years of experience (e.g. claiming 8 years when candidate has 2.5)
2. Invented employers or companies (e.g. claiming to have worked at Google or Amazon)
3. False skill claims, especially claiming skills listed in missing_must
4. Fabricated metrics or numbers not grounded in profile highlights/projects
"""

import re
import logging
from typing import List, Tuple, Set, Optional

from jsb.schemas import ProfileJSON, PostingJSON, MatchResult
from jsb.skills import normalize_skill, extract_all_candidate_skills

logger = logging.getLogger("jsbrother.claim_checker")


def check_claims(
    draft_text: str,
    profile: ProfileJSON,
    posting: Optional[PostingJSON] = None,
    match_result: Optional[MatchResult] = None,
) -> Tuple[bool, List[str]]:
    """
    Scans draft_text for claims not supported by the candidate profile or posting context.
    Returns:
        (claims_ok, flagged_claims)
    """
    flagged = []
    if not draft_text or not draft_text.strip():
        return True, []

    # 1. Build ground-truth corpora
    valid_companies: Set[str] = set()
    for role in profile.roles:
        if role.company:
            valid_companies.add(role.company.lower().strip())
            # Add simple tokens from company names
            for word in re.findall(r"[a-zA-Z]{3,}", role.company.lower()):
                valid_companies.add(word)

    if posting and posting.company_or_client:
        valid_companies.add(posting.company_or_client.lower().strip())
        for word in re.findall(r"[a-zA-Z]{3,}", posting.company_or_client.lower()):
            valid_companies.add(word)

    # All verified candidate skills
    cand_skills = {normalize_skill(s) for s in extract_all_candidate_skills(profile)}

    # Full profile and posting text for metric/number cross-checking
    profile_corpus = (
        f"{profile.headline} {' '.join(profile.skills)} "
        f"{' '.join(r.title + ' ' + r.company + ' ' + ' '.join(r.highlights) for r in profile.roles)} "
        f"{' '.join(profile.projects)} {' '.join(profile.education)}"
    ).lower()

    posting_corpus = ""
    if posting:
        posting_corpus = (
            f"{posting.title} {posting.company_or_client or ''} {posting.summary or ''} "
            f"{str(int(posting.budget.amount) if posting.budget and posting.budget.amount else '')} "
            f"{str(posting.deadline_days or '')} "
            f"{' '.join(posting.must_have)} {' '.join(posting.nice_to_have)}"
        ).lower()

    combined_corpus = profile_corpus + " " + posting_corpus

    # 2. Check Years of Experience Claims
    # Matches "X years of experience", "X+ years", "over X years"
    year_matches = re.finditer(r"\b(?:over\s+|more\s+than\s+)?(\d+(?:\.\d+)?)\+?\s*years?(?:\s+of)?(?:\s+(?:professional\s+)?experience)?\b", draft_text, re.IGNORECASE)
    cand_years = profile.years_experience or 0.0
    for ym in year_matches:
        claimed_years = float(ym.group(1))
        # Allow small tolerance (e.g., rounding 2.5 to 3 is ok, but claiming 5+ or 10+ is a violation)
        if claimed_years > max(cand_years + 0.9, 1.0) and claimed_years >= 3.5:
            flagged.append(f"Inflated experience claim: '{ym.group(0).strip()}' (candidate has {cand_years} years)")

    # 3. Check for Invented Famous Employers
    # High-profile companies frequently hallucinated by LLMs
    hallucination_employers = [
        "google", "meta", "facebook", "amazon", "apple", "microsoft", "netflix",
        "openai", "anthropic", "uber", "twitter", "mckinsey", "ibm", "salesforce"
    ]
    for emp in hallucination_employers:
        if emp not in valid_companies:
            pattern = re.compile(rf"\b(?:at|for|with)\s+{emp}\b", re.IGNORECASE)
            if pattern.search(draft_text):
                flagged.append(f"Invented employer claim: worked at/for '{emp.title()}'")

    # 4. Check for Claims of Missing Must-Have Skills
    # Per §7: 'never claim a skill listed in missing_must'
    if match_result and match_result.missing_must:
        for missing in match_result.missing_must:
            # Check if draft asserts expertise in this missing skill
            # Pattern: "experienced in X", "expert in X", "using X", "skills in X"
            # Ignore honest handling like "eager to deepen X" or "learning X"
            norm_miss = normalize_skill(missing)
            if not norm_miss or len(norm_miss) < 3:
                continue

            # Look for positive assertions of possessing this skill
            patterns = [
                rf"\b(?:extensive|hands-on|deep|solid|strong|proven)\s+(?:experience|background|expertise)\s+(?:in|with)\s+[^,.]*?\b{re.escape(norm_miss)}\b",
                rf"\b(?:I\s+have|I\s+bring|my\s+skills\s+in|specializing\s+in)\s+[^,.]*?\b{re.escape(norm_miss)}\b",
            ]
            for pat in patterns:
                m = re.search(pat, draft_text, re.IGNORECASE)
                if m:
                    # Check if it was framed as a gap / eager to learn
                    surrounding = draft_text[max(0, m.start()-30):min(len(draft_text), m.end()+30)].lower()
                    if not any(safe_word in surrounding for safe_word in ["eager", "learn", "deepen", "familiarize", "expand"]):
                        flagged.append(f"Claimed missing must-have skill: '{missing}'")
                        break

    # 5. Check for Extreme Unsubstantiated Quantified Metrics
    # e.g., "generated $5M in revenue", "increased sales by 85%"
    metric_matches = re.finditer(r"(\$\d+[\d,]*\s*(?:million|m|k)?|\b\d{2,3}%\s*(?:increase|reduction|improvement|accuracy|growth|boost)\b)", draft_text, re.IGNORECASE)
    for mm in metric_matches:
        metric_str = mm.group(1).lower()
        # If the metric is a proposal price quote (e.g. $1,200), verify against budget or allow
        num_part = re.search(r"\d+", metric_str.replace(",", ""))
        if num_part and num_part.group(0) not in combined_corpus:
            # Allow common small numbers or standard milestones
            if num_part.group(0) not in ("100", "24", "7", "3", "1", "2"):
                flagged.append(f"Unsupported metric claim: '{metric_str}' not found in candidate profile or posting")

    # Deduplicate
    flagged = list(dict.fromkeys(flagged))
    claims_ok = len(flagged) == 0
    return claims_ok, flagged
