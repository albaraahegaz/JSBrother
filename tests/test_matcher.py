"""
JSBrother — Tests for Matcher & Verdict Agent (M4)
Validates all score formulas, verdict decision rules, and LLM guardrails:
- Rule 1: deal-breaker -> SKIP
- Rule 2: score < 50 -> SKIP
- Rule 2b: >= 3 missing must-haves -> SKIP
- Rule 3: score >= 75 and 0 missing must-haves -> APPLY
- Rule 4: otherwise -> CAUTION
- Rule 5: Freelancer downgrade on vague_scope -> CAUTION + red flag
- Rule 5b: Freelancer downgrade on budget_fit == 0 -> CAUTION + red flag
- LLM cannot change score or verdict
- Skills normalization and alias matching
"""

import pytest
from unittest.mock import MagicMock

from jsb.schemas import ProfileJSON, PostingJSON, BudgetInfo, MatchResult
from jsb.skills import normalize_skill, match_skill, match_skills, get_alias_registry
from jsb.agents.matcher import (
    compute_match_score_and_verdict,
    evaluate_deal_breakers,
    compute_freelancer_fit,
    match,
)


@pytest.fixture
def base_candidate():
    """Standard Python/AI candidate with 3 years of experience."""
    return ProfileJSON(
        name="Alex Smith",
        headline="AI & Machine Learning Engineer",
        skills=[
            "python",
            "sql",
            "pytorch",
            "machine learning",
            "scikit-learn",
            "fastapi",
            "git",
            "docker",
            "rag",
            "pandas",
            "numpy",
        ],
        years_experience=3.0,
        min_budget=1000.0,
        min_lead_days=7,
    )


# -----------------------------------------------------------------------------
# Skills & Alias Tests
# -----------------------------------------------------------------------------

def test_skill_normalization():
    assert normalize_skill("  Python  ") == "python"
    assert normalize_skill("C++") == "c++"
    assert normalize_skill("CI/CD") == "ci/cd"
    assert normalize_skill(" Next.js ") == "next.js"


def test_alias_matching():
    registry = get_alias_registry()
    assert registry.are_aliases("js", "javascript")
    assert registry.are_aliases("sklearn", "scikit-learn")
    assert registry.are_aliases("nlp", "natural language processing")
    assert registry.are_aliases("oop", "object-oriented programming")
    assert registry.are_aliases("postgres", "postgresql")
    assert registry.are_aliases("k8s", "kubernetes")

    # Match skill helper
    cand_skills = ["javascript", "pytorch", "postgresql"]
    is_hit, matched, score, m_type = match_skill("js", cand_skills)
    assert is_hit is True
    assert matched == "javascript"
    assert m_type == "alias"


# -----------------------------------------------------------------------------
# Rule 1: Deal-Breaker -> SKIP
# -----------------------------------------------------------------------------

def test_rule_deal_breaker_skip(base_candidate):
    """
    Even with high skill match, violating any deal-breaker triggers SKIP.
    Here candidate has 3.0 years, but posting requires 5+ years as deal-breaker.
    """
    posting = PostingJSON(
        title="Senior AI Architect",
        type="job",
        must_have=["python", "pytorch", "machine learning"],
        deal_breakers=["5+ years professional experience", "must have PhD"],
    )

    score, must_cov, nice_cov, matched, missing_must, missing_nice, hits, verdict, _ = (
        compute_match_score_and_verdict(base_candidate, posting)
    )

    assert len(hits) > 0
    assert "5+ years professional experience" in hits
    assert verdict == "SKIP"


def test_deal_breaker_satisfied_no_skip(base_candidate):
    """
    Candidate with 3.0 years given a posting with '2+ years professional experience':
    Demonstrably satisfied -> 0 deal-breaker hits, does not force SKIP.
    """
    posting = PostingJSON(
        title="Mid-Level Python Developer",
        type="job",
        must_have=["python", "fastapi"],
        deal_breakers=["2+ years professional experience"],
    )

    score, must_cov, nice_cov, matched, missing_must, missing_nice, hits, verdict, _ = (
        compute_match_score_and_verdict(base_candidate, posting)
    )

    assert len(hits) == 0
    assert verdict != "SKIP"


def test_unverifiable_deal_breakers_are_warnings_not_skip(base_candidate):
    """
    Items the candidate profile cannot verify (NDA, work authorization, payment terms, deadlines)
    must NOT count as deal-breaker hits or cause auto-SKIP; they should surface as warnings/red flags.
    """
    from jsb.agents.matcher import evaluate_deal_breakers_with_warnings

    posting = PostingJSON(
        title="Full Stack Prototype",
        type="project",
        must_have=["python", "fastapi"],
        deal_breakers=["sign NDA", "deliver within 10 days", "US work authorization"],
    )

    hits, warnings = evaluate_deal_breakers_with_warnings(posting, base_candidate)
    assert len(hits) == 0
    assert len(warnings) == 3
    assert "sign NDA" in warnings
    assert "US work authorization" in warnings

    # And compute_match_score_and_verdict should NOT auto-SKIP
    score, must_cov, nice_cov, matched, missing_must, missing_nice, hits, verdict, fl = (
        compute_match_score_and_verdict(base_candidate, posting, mode="freelancer")
    )
    assert len(hits) == 0
    assert verdict != "SKIP"
    assert fl is not None
    assert any("sign NDA" in rf for rf in fl.red_flags)


def test_pipe_separated_required_skills_matching(base_candidate):
    """
    Required skill with alternatives 'pytorch | tensorflow | hugging face'
    must match because base_candidate has 'pytorch'.
    """
    posting = PostingJSON(
        title="PyTorch or TensorFlow Specialist",
        type="job",
        must_have=["pytorch | tensorflow | hugging face", "python"],
    )

    score, must_cov, nice_cov, matched, missing_must, missing_nice, hits, verdict, _ = (
        compute_match_score_and_verdict(base_candidate, posting)
    )

    assert "pytorch | tensorflow | hugging face" in matched
    assert len(missing_must) == 0
    assert must_cov == 1.0


# -----------------------------------------------------------------------------
# Rule 2: Score < 50 -> SKIP
# -----------------------------------------------------------------------------

def test_rule_score_below_50_skip(base_candidate):
    """
    Candidate missing several skills and lacking required years leads to score < 50 -> SKIP.
    """
    posting = PostingJSON(
        title="Principal Java / Scala Engineer",
        type="job",
        must_have=["java", "scala", "python"],  # 1/3 must-haves matched (must_cov = 0.33)
        nice_to_have=["spark", "kafka", "flink"],  # 0/3 nice-to-have matched (nice_cov = 0.0)
        min_years=10.0,  # exp_fit = 3.0 / 10.0 = 0.30
        deal_breakers=[],
    )
    # Score = 100 * (0.60 * 0.333 + 0.20 * 0 + 0.20 * 0.3) = 100 * (0.20 + 0.06) = 26.0 < 50

    score, must_cov, nice_cov, matched, missing_must, missing_nice, hits, verdict, _ = (
        compute_match_score_and_verdict(base_candidate, posting)
    )

    assert score < 50.0
    assert verdict == "SKIP"


# -----------------------------------------------------------------------------
# Rule 2b: >= 3 Missing Must-Haves -> SKIP
# -----------------------------------------------------------------------------

def test_rule_three_missing_must_haves_skip(base_candidate):
    """
    Even if candidate's score might reach 50, having >= 3 missing must-haves forces SKIP.
    """
    posting = PostingJSON(
        title="Full-Stack Web & AI Lead",
        type="job",
        must_have=[
            "python",
            "pytorch",
            "fastapi",
            "docker",
            "react",       # missing 1
            "typescript",  # missing 2
            "graphql",     # missing 3
        ],
        nice_to_have=["git", "pandas"],
        min_years=3.0,
        deal_breakers=[],
    )

    score, must_cov, nice_cov, matched, missing_must, missing_nice, hits, verdict, _ = (
        compute_match_score_and_verdict(base_candidate, posting)
    )

    assert len(missing_must) >= 3
    assert "react" in missing_must
    assert "typescript" in missing_must
    assert "graphql" in missing_must
    assert verdict == "SKIP"


# -----------------------------------------------------------------------------
# Rule 3: Score >= 75 and 0 Missing Must-Haves -> APPLY
# -----------------------------------------------------------------------------

def test_rule_apply_high_score_zero_missing(base_candidate):
    """
    Full must-have coverage + sufficient experience + high score -> APPLY.
    """
    posting = PostingJSON(
        title="Mid-Level AI Engineer",
        type="job",
        must_have=["python", "pytorch", "machine learning", "fastapi"],
        nice_to_have=["docker", "git", "rag"],
        min_years=2.0,
        deal_breakers=[],
    )

    score, must_cov, nice_cov, matched, missing_must, missing_nice, hits, verdict, _ = (
        compute_match_score_and_verdict(base_candidate, posting)
    )

    assert must_cov == 1.0
    assert len(missing_must) == 0
    assert score >= 75.0
    assert verdict == "APPLY"


# -----------------------------------------------------------------------------
# Rule 4: Otherwise -> CAUTION
# -----------------------------------------------------------------------------

def test_rule_caution_moderate_fit(base_candidate):
    """
    Score >= 50, missing 1 or 2 must-haves, no deal-breakers -> CAUTION.
    """
    posting = PostingJSON(
        title="AI Engineer with Kubernetes",
        type="job",
        must_have=["python", "pytorch", "kubernetes"],  # 1 missing must-have
        nice_to_have=["docker", "git"],
        min_years=3.0,
        deal_breakers=[],
    )

    score, must_cov, nice_cov, matched, missing_must, missing_nice, hits, verdict, _ = (
        compute_match_score_and_verdict(base_candidate, posting)
    )

    assert len(missing_must) == 1
    assert "kubernetes" in missing_must
    assert score >= 50.0
    assert verdict == "CAUTION"


# -----------------------------------------------------------------------------
# Rule 5: Freelancer Downgrade on vague_scope -> CAUTION + Red Flag
# -----------------------------------------------------------------------------

def test_rule_freelancer_downgrade_vague_scope(base_candidate):
    """
    100% skill match with good budget/deadline, but vague_scope=True:
    APPLY is downgraded to CAUTION, and red flag is added.
    """
    posting = PostingJSON(
        title="Build AI Prototype",
        type="project",
        must_have=["python", "fastapi"],
        nice_to_have=["docker"],
        budget=BudgetInfo(amount=1500.0, kind="fixed"),
        deadline_days=14,
        vague_scope=True,
    )

    score, must_cov, nice_cov, matched, missing_must, missing_nice, hits, verdict, fl = (
        compute_match_score_and_verdict(base_candidate, posting, mode="freelancer")
    )

    assert score >= 75.0
    assert len(missing_must) == 0
    # Must be downgraded from APPLY to CAUTION
    assert verdict == "CAUTION"
    assert fl is not None
    assert any("Vague scope" in flag for flag in fl.red_flags)


# -----------------------------------------------------------------------------
# Rule 5b: Freelancer Downgrade on budget_fit == 0 -> CAUTION + Red Flag
# -----------------------------------------------------------------------------

def test_rule_freelancer_downgrade_zero_budget_fit(base_candidate):
    """
    Candidate minimum budget is $1000. Project budget is $500 (< 80% of min_budget),
    so budget_fit == 0.0. If skills match, APPLY must be downgraded to CAUTION.
    """
    posting = PostingJSON(
        title="Quick AI Script",
        type="project",
        must_have=["python", "pytorch"],
        nice_to_have=[],
        budget=BudgetInfo(amount=500.0, kind="fixed"),  # $500 < 0.8 * $1000
        deadline_days=10,
        vague_scope=False,
    )

    score, must_cov, nice_cov, matched, missing_must, missing_nice, hits, verdict, fl = (
        compute_match_score_and_verdict(base_candidate, posting, mode="freelancer")
    )

    assert fl is not None
    assert fl.budget_fit == 0.0
    assert verdict == "CAUTION"
    assert any("Low budget" in flag for flag in fl.red_flags)


# -----------------------------------------------------------------------------
# LLM Guardrail: The LLM Cannot Change Score or Verdict
# -----------------------------------------------------------------------------

def test_empty_must_have_does_not_inflate_score(base_candidate):
    """Empty must-haves must not yield a high score (coverage treated as 0, not 1)."""
    posting = PostingJSON(
        title="Vague role",
        type="job",
        must_have=[],
        nice_to_have=[],
        min_years=None,
    )
    score, must_cov, nice_cov, matched, missing, *_rest = compute_match_score_and_verdict(
        base_candidate, posting
    )
    assert must_cov == 0.0
    assert score < 75.0
    assert score != 100.0


def test_llm_cannot_change_score_or_verdict(base_candidate):
    """
    Verifies that even if a rogue LLM outputs a different score and verdict in its JSON,
    the MatchResult returned by match() strictly preserves the Python-computed score & verdict.
    """
    posting = PostingJSON(
        title="Senior Lead (Unqualified)",
        type="job",
        must_have=["python", "c++", "rust", "assembly"],  # missing 3 must-haves -> SKIP
        min_years=10.0,
    )

    # Rogue mock client attempting to manipulate score and verdict
    mock_client = MagicMock()
    mock_client.mode = "remote"
    mock_client.generate.return_value = """
    {
        "score": 98.5,
        "verdict": "APPLY",
        "reasons": [
            "Candidate is a genius.",
            "Candidate will learn Rust quickly.",
            "Perfect culture fit."
        ],
        "advice": "Hire immediately with maximum salary."
    }
    """

    result: MatchResult = match(base_candidate, posting, client=mock_client)

    # Python computes SKIP (3 missing must-haves)
    assert result.verdict == "SKIP"
    assert result.score < 50.0
    # LLM was only allowed to provide reasons and advice
    assert len(result.reasons) == 3
    assert "Candidate is a genius." in result.reasons[0]
    assert result.advice == "Hire immediately with maximum salary."
