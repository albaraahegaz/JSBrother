"""
JSBrother — Matcher & Verdict Agent
Computes deterministic match scores and verdicts in pure Python per §6.
The LLM never changes the score or verdict; it only phrases reasons (3) and advice (1).

Scoring formulas:
- Job: 100 * (0.60 * must_cov + 0.20 * nice_cov + 0.20 * exp_fit)
  exp_fit = min(1.0, user_years / min_years), or 1.0 if no min_years
- Freelancer: 100 * (0.55 * must_cov + 0.15 * nice_cov + 0.15 * budget_fit + 0.15 * deadline_fit)
  budget_fit: 1.0 if budget >= min_budget; 0.5 if within 20% below; 0.0 if lower; 0.7 if budget unknown
  deadline_fit: 1.0 if no deadline or deadline_days >= min_lead_days; 0.5 if >= half of it; else 0.0

Verdict rules (applied in this exact order):
1. any deal_breaker_hits -> SKIP
2. score < 50 or len(missing_must) >= 3 -> SKIP
3. score >= 75 and len(missing_must) == 0 -> APPLY
4. otherwise -> CAUTION
5. Freelancer only: if vague_scope or budget_fit == 0, downgrade APPLY -> CAUTION and add a red flag.
"""

import re
import json
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field
import config
from jsb.schemas import ProfileJSON, PostingJSON, MatchResult, FreelancerMatch
from jsb.skills import match_skills, normalize_skill, match_skill
from jsb.json_utils import extract_json
from jsb.llm_client import get_client, LLMClient

logger = logging.getLogger("jsbrother.matcher")

REASONS_SYSTEM_PROMPT = """You are a career coach and hiring intelligence assistant.
Your task is to summarize the matching evaluation between a candidate's profile and a job/project posting.
The match score and verdict have ALREADY been computed deterministically and MUST NOT be changed.

Return ONLY a JSON object with two fields:
{
  "reasons": [
    "First concise reason explaining key match factors (strengths/score)",
    "Second concise reason detailing any gaps or missing must-haves",
    "Third concise reason assessing experience, budget, or timeline fit"
  ],
  "advice": "One short, actionable line of advice for the candidate."
}

Rules:
- Exactly 3 bullet points in 'reasons'.
- Exactly 1 actionable sentence in 'advice'.
- Ground all reasons in the provided matched skills, missing skills, and deal-breakers.
- Do NOT output any markdown, commentary, or text outside the JSON object.
"""


from jsb.skills import match_skills, normalize_skill, match_skill, extract_all_candidate_skills


def evaluate_deal_breakers_with_warnings(
    posting: PostingJSON,
    profile: ProfileJSON,
) -> Tuple[List[str], List[str]]:
    """
    Checks deal-breakers per §6:
    - Counts as a 'hit' ONLY if the candidate demonstrably fails it
      (e.g., required years > profile years; or required tool months when tool is missing).
    - Unverifiable items (NDA, work authorization, payment terms, deadlines, scraper examples)
      must NOT auto-SKIP: surfaced as warnings/red flags instead.
    Returns:
        (deal_breaker_hits, unverified_warnings)
    """
    hits = []
    warnings = []
    user_years = profile.years_experience or 0.0

    role_texts = " ".join(
        f"{r.title} {r.company} {' '.join(r.highlights)}" for r in profile.roles
    ).lower()
    project_texts = " ".join(profile.projects).lower()
    profile_corpus = f"{' '.join(profile.skills).lower()} {role_texts} {project_texts}"

    unverifiable_patterns = [
        r"(?:work\s+authoriz\w*|authorization|visa|us-based|us\s+based|authorized\s+to\s+work|citizenship|fluent\s+in\s+english)",
        r"(?:nda|non-disclosure|sign\s+nda|agreement|background\s+check)",
        r"(?:no\s+upfront\s+payment|payment\s+until\s+demo|escrow|payment\s+terms)",
        r"(?:deliver\s+within|complete\s+in\s+\d+\s*days?|weekly\s+reviews|git\s+commits|turnaround)",
        r"(?:code\s+examples|examples\s+of|portfolio|work\s+samples)",
    ]

    for db in posting.deal_breakers:
        db_clean = db.strip()
        if not db_clean or db_clean.lower() == "none":
            continue

        # Check if unverifiable
        is_unverifiable = any(re.search(pat, db_clean, re.IGNORECASE) for pat in unverifiable_patterns)
        if is_unverifiable:
            warnings.append(db_clean)
            continue

        # 1. Years of experience check (e.g. "4+ years professional python", "2+ years experience")
        m_years = re.search(r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs)", db_clean, re.IGNORECASE)
        if m_years:
            req_years = float(m_years.group(1))
            if user_years < req_years:
                hits.append(db_clean)
            continue

        # 2. Months of experience check (e.g. "6+ months claude code")
        m_months = re.search(r"(\d+)\+?\s*months?", db_clean, re.IGNORECASE)
        if m_months:
            tool_part = re.sub(r"\d+\+?\s*months?", "", db_clean, flags=re.IGNORECASE).strip()
            tool_norm = normalize_skill(tool_part)
            if tool_norm and tool_norm not in profile_corpus:
                hits.append(db_clean)
            continue

        # 3. Explicit degree check (e.g. "phd required", "must have phd")
        m_phd = re.search(r"\bphd\b|\bdoctorate\b", db_clean, re.IGNORECASE)
        if m_phd:
            edu_text = " ".join(profile.education).lower()
            if "phd" not in edu_text and "doctorate" not in edu_text:
                hits.append(db_clean)
            continue

        # 4. Verifiable hard tool / technology requirement
        norm_db = normalize_skill(db_clean)
        is_hit, _, _, _ = match_skill(norm_db, profile.skills)
        if not is_hit and norm_db not in profile_corpus:
            hits.append(db_clean)

    return hits, warnings


def evaluate_deal_breakers(
    posting: PostingJSON,
    profile: ProfileJSON,
) -> List[str]:
    """Preserves backwards compatibility, returning only demonstrable hits."""
    hits, _ = evaluate_deal_breakers_with_warnings(posting, profile)
    return hits


def compute_freelancer_fit(
    posting: PostingJSON,
    profile: ProfileJSON,
) -> Tuple[float, float, List[str]]:
    """
    Computes budget_fit, deadline_fit, and red_flags for freelancer mode per §6:
    - budget_fit:
        1.0 if budget >= user's min_budget
        0.5 if within 20% below min_budget (>= 0.8 * min_budget)
        0.0 if lower (< 0.8 * min_budget)
        0.7 if budget unknown
    - deadline_fit:
        1.0 if no deadline or deadline_days >= min_lead_days
        0.5 if >= half of min_lead_days
        0.0 otherwise
    """
    user_min_budget = profile.min_budget
    min_lead_days = profile.min_lead_days if (profile.min_lead_days and profile.min_lead_days > 0) else config.DEFAULT_MIN_LEAD_DAYS

    red_flags: List[str] = []

    # Budget fit
    if user_min_budget is None or user_min_budget <= 0:
        # User has no minimum budget requirement
        budget_fit = 1.0 if (posting.budget and posting.budget.amount > 0) else 0.7
    else:
        if posting.budget is None or posting.budget.amount <= 0:
            budget_fit = 0.7  # budget unknown
        else:
            amount = posting.budget.amount
            if amount >= user_min_budget:
                budget_fit = 1.0
            elif amount >= 0.8 * user_min_budget:
                budget_fit = 0.5
            else:
                budget_fit = 0.0

    # Deadline fit
    if posting.deadline_days is None:
        deadline_fit = 1.0  # no deadline
    else:
        if posting.deadline_days >= min_lead_days:
            deadline_fit = 1.0
        elif posting.deadline_days >= 0.5 * min_lead_days:
            deadline_fit = 0.5
        else:
            deadline_fit = 0.0

    # Red flags
    if posting.vague_scope:
        red_flags.append("Vague scope: requirements are ambiguous or poorly specified.")
    if budget_fit == 0.0 and user_min_budget:
        amt = posting.budget.amount if posting.budget else 0.0
        red_flags.append(f"Low budget: project offer (${amt:.0f}) is below minimum requirement (${user_min_budget:.0f}).")
    if deadline_fit == 0.0 and posting.deadline_days is not None:
        red_flags.append(f"Tight turnaround: {posting.deadline_days} days is less than half your lead time ({min_lead_days} days).")

    return budget_fit, deadline_fit, red_flags


def compute_match_score_and_verdict(
    profile: ProfileJSON,
    posting: PostingJSON,
    mode: Optional[str] = None,
) -> Tuple[float, float, float, List[str], List[str], List[str], List[str], str, Optional[FreelancerMatch]]:
    """
    Pure Python deterministic calculation of score and verdict per §6.
    Returns:
        (score, must_coverage, nice_coverage, matched, missing_must, missing_nice,
         deal_breaker_hits, verdict, freelancer_match)
    """
    active_mode = mode or posting.type or "job"
    is_freelancer = (active_mode == "project" or active_mode == "freelancer")

    # 1. Skill matching using all explicit skills across candidate profile
    candidate_skills = extract_all_candidate_skills(profile)
    must_matched, must_missing, _ = match_skills(posting.must_have, candidate_skills)
    nice_matched, nice_missing, _ = match_skills(posting.nice_to_have, candidate_skills)
    all_matched = list(dict.fromkeys(must_matched + nice_matched))

    # Empty must_have must NOT inflate score (guard; formulas unchanged when list is non-empty)
    must_cov = len(must_matched) / len(posting.must_have) if posting.must_have else 0.0
    nice_cov = len(nice_matched) / len(posting.nice_to_have) if posting.nice_to_have else 1.0

    # 2. Deal-breakers check (demonstrable failures vs unverified warnings)
    deal_breaker_hits, deal_breaker_warnings = evaluate_deal_breakers_with_warnings(posting, profile)

    # 3. Mode-specific score calculation
    freelancer_match = None
    if not is_freelancer:
        # Job score: 100 * (0.60 * must_cov + 0.20 * nice_cov + 0.20 * exp_fit)
        user_years = profile.years_experience or 0.0
        min_years = posting.min_years
        if min_years is not None and min_years > 0:
            exp_fit = min(1.0, user_years / min_years)
        else:
            exp_fit = 1.0

        score = 100.0 * (
            config.JOB_MUST_WEIGHT * must_cov
            + config.JOB_NICE_WEIGHT * nice_cov
            + config.JOB_EXP_WEIGHT * exp_fit
        )
    else:
        # Freelancer score: 100 * (0.55 * must_cov + 0.15 * nice_cov + 0.15 * budget_fit + 0.15 * deadline_fit)
        budget_fit, deadline_fit, red_flags = compute_freelancer_fit(posting, profile)
        # Add unverified deal-breakers as red flags for freelancer visibility
        for w in deal_breaker_warnings:
            if w not in red_flags and not any(w.lower() in rf.lower() for rf in red_flags):
                red_flags.append(f"Unverified requirement: {w}")

        freelancer_match = FreelancerMatch(
            budget_fit=budget_fit,
            deadline_fit=deadline_fit,
            red_flags=red_flags,
        )

        score = 100.0 * (
            config.FL_MUST_WEIGHT * must_cov
            + config.FL_NICE_WEIGHT * nice_cov
            + config.FL_BUDGET_WEIGHT * budget_fit
            + config.FL_DEADLINE_WEIGHT * deadline_fit
        )

    score = round(score, 1)

    # 4. Verdict determination (in exact order per §6)
    # Rule 1: any deal_breaker_hits -> SKIP
    if len(deal_breaker_hits) > 0:
        verdict = "SKIP"
    # Rule 2: score < 50 or len(missing_must) >= 3 -> SKIP
    elif score < config.VERDICT_SKIP_SCORE_THRESHOLD or len(must_missing) >= config.VERDICT_SKIP_MISSING_MUST_COUNT:
        verdict = "SKIP"
    # Rule 3: score >= 75 and len(missing_must) == 0 -> APPLY
    elif score >= config.VERDICT_APPLY_SCORE_THRESHOLD and len(must_missing) == config.VERDICT_APPLY_MISSING_MUST_COUNT:
        verdict = "APPLY"
    # Rule 4: otherwise -> CAUTION
    else:
        verdict = "CAUTION"

    # Rule 5 (Freelancer only): if vague_scope or budget_fit == 0, downgrade APPLY -> CAUTION and add a red flag
    if is_freelancer and freelancer_match is not None:
        downgrade_needed = posting.vague_scope or freelancer_match.budget_fit == 0.0
        if verdict == "APPLY" and downgrade_needed:
            verdict = "CAUTION"

    return (
        score,
        round(must_cov, 3),
        round(nice_cov, 3),
        all_matched,
        must_missing,
        nice_missing,
        deal_breaker_hits,
        verdict,
        freelancer_match,
    )


def build_default_explanations(
    score: float,
    verdict: str,
    matched: List[str],
    missing_must: List[str],
    missing_nice: List[str],
    deal_breaker_hits: List[str],
    is_freelancer: bool = False,
    freelancer_match: Optional[FreelancerMatch] = None,
) -> Tuple[List[str], str]:
    """
    Deterministic fallback for 3 reasons and 1 advice line when LLM is offline or in tests.
    """
    reasons = []

    # Bullet 1: Match score & coverage overview
    if verdict == "APPLY":
        reasons.append(
            f"Strong profile fit with an overall match score of {score:.1f}/100 and full coverage of mandatory skills."
        )
    elif verdict == "CAUTION":
        reasons.append(
            f"Moderate profile fit with an overall score of {score:.1f}/100; candidate meets core requirements but has notable gaps."
        )
    else:
        reasons.append(
            f"Low compatibility fit with an overall score of {score:.1f}/100, falling below the direct application threshold."
        )

    # Bullet 2: Skills breakdown or deal-breakers
    if deal_breaker_hits:
        reasons.append(
            f"Triggered deal-breaker constraints: {', '.join(deal_breaker_hits)}."
        )
    elif missing_must:
        reasons.append(
            f"Missing {len(missing_must)} must-have skill(s): {', '.join(missing_must[:4])}."
        )
    elif matched:
        reasons.append(
            f"Successfully matched key technical competencies: {', '.join(matched[:5])}."
        )
    else:
        reasons.append("No direct skill matches were established against the mandatory requirements.")

    # Bullet 3: Experience or Freelancer timeline/budget
    if is_freelancer and freelancer_match:
        if freelancer_match.red_flags:
            reasons.append(f"Project constraints note: {freelancer_match.red_flags[0]}")
        else:
            reasons.append("Budget and timeline parameters align comfortably with your stated freelance preferences.")
    else:
        if missing_nice:
            reasons.append(
                f"Optional/nice-to-have skill gaps: {', '.join(missing_nice[:4])}."
            )
        else:
            reasons.append("Experience level and technical background provide solid qualification for the role.")

    # Advice (1 line)
    if verdict == "APPLY":
        advice = "Proceed with application and highlight matched technical achievements in your cover letter."
    elif verdict == "CAUTION":
        if is_freelancer:
            advice = "Submit a proposal with careful scope clarification and milestone estimates to mitigate risks."
        else:
            advice = "Apply with caution; address your skill gaps proactively in your introduction."
    else:
        advice = "Skip this posting or address deal-breakers before dedicating time to an application."

    return reasons[:3], advice


class MatchExplanation(BaseModel):
    """Schema for LLM-generated matching explanation."""
    reasons: List[str] = Field(default_factory=list)
    advice: str = ""


def generate_llm_explanations(
    profile: ProfileJSON,
    posting: PostingJSON,
    score: float,
    verdict: str,
    matched: List[str],
    missing_must: List[str],
    missing_nice: List[str],
    deal_breaker_hits: List[str],
    freelancer_match: Optional[FreelancerMatch] = None,
    client: Optional[LLMClient] = None,
) -> Tuple[List[str], str]:
    """
    Prompts the LLM to phrase exactly 3 reasons and 1 advice line.
    Score and verdict are locked and strictly preserved.
    """
    if client is None:
        client = get_client()

    # If mock mode, return high quality deterministic explanations directly
    if client.mode == "mock":
        is_fl = posting.type == "project"
        return build_default_explanations(
            score, verdict, matched, missing_must, missing_nice, deal_breaker_hits, is_fl, freelancer_match
        )

    prompt = f"""Posting Title: {posting.title}
Company / Client: {posting.company_or_client}
Type: {posting.type}
Candidate: {profile.name} ({profile.headline}), {profile.years_experience} years exp

LOCKED MATCH RESULTS (Do not alter):
- Match Score: {score:.1f} / 100
- Computed Verdict: {verdict}
- Matched Skills: {matched}
- Missing Must-Have Skills: {missing_must}
- Missing Nice-to-Have Skills: {missing_nice}
- Deal-Breakers Triggered: {deal_breaker_hits}
"""
    if freelancer_match:
        prompt += f"- Budget Fit: {freelancer_match.budget_fit}\n- Deadline Fit: {freelancer_match.deadline_fit}\n- Red Flags: {freelancer_match.red_flags}\n"

    prompt += "\nProvide the JSON with 'reasons' (list of 3 strings) and 'advice' (1 string)."

    try:
        response_text = client.generate(
            prompt=prompt,
            system=REASONS_SYSTEM_PROMPT,
            temperature=config.TEMP_EXPLANATION,
            max_new_tokens=300,
        )
        data = extract_json(response_text, MatchExplanation)
        reasons = data.reasons
        advice = data.advice

        # Validate format
        if isinstance(reasons, list) and len(reasons) >= 3 and isinstance(advice, str) and advice.strip():
            return [str(r).strip() for r in reasons[:3]], advice.strip()
    except Exception as e:
        logger.warning(f"LLM explanation generation failed: {e}. Falling back to default phrasing.")

    is_fl = posting.type == "project"
    return build_default_explanations(
        score, verdict, matched, missing_must, missing_nice, deal_breaker_hits, is_fl, freelancer_match
    )


def match(
    profile: ProfileJSON,
    posting: PostingJSON,
    mode: Optional[str] = None,
    client: Optional[LLMClient] = None,
) -> MatchResult:
    """
    Matches candidate ProfileJSON against PostingJSON.
    - Pure Python computes score and verdict (LLM cannot change them).
    - LLM only generates reasons (3 bullets) and advice (1 line).
    """
    (
        score,
        must_cov,
        nice_cov,
        matched,
        missing_must,
        missing_nice,
        deal_breaker_hits,
        verdict,
        freelancer_match,
    ) = compute_match_score_and_verdict(profile, posting, mode=mode)

    reasons, advice = generate_llm_explanations(
        profile=profile,
        posting=posting,
        score=score,
        verdict=verdict,
        matched=matched,
        missing_must=missing_must,
        missing_nice=missing_nice,
        deal_breaker_hits=deal_breaker_hits,
        freelancer_match=freelancer_match,
        client=client,
    )

    _, warnings = evaluate_deal_breakers_with_warnings(posting, profile)

    return MatchResult(
        score=score,
        must_coverage=must_cov,
        nice_coverage=nice_cov,
        matched=matched,
        missing_must=missing_must,
        missing_nice=missing_nice,
        deal_breaker_hits=deal_breaker_hits,
        warnings=warnings,
        verdict=verdict,
        reasons=reasons,
        advice=advice,
        freelancer=freelancer_match,
    )
