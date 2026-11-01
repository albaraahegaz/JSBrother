"""
JSBrother — Tests for Claim Checker Module (M5)
Validates:
- Truthful draft with supported profile facts passes (claims_ok=True)
- Invented employer (e.g. Google, Meta) is flagged
- Inflated years of experience (e.g. 10 years vs 3 years) is flagged
- Unearned missing must-have skill claim is flagged
"""

import pytest
from jsb.schemas import ProfileJSON, PostingJSON, MatchResult, RoleItem
from jsb.tools.claim_checker import check_claims


@pytest.fixture
def test_profile():
    return ProfileJSON(
        name="Alex Smith",
        headline="AI Engineer",
        years_experience=3.0,
        skills=["python", "pytorch", "fastapi"],
        roles=[
            RoleItem(
                title="ML Engineer",
                company="DataTech",
                years=2.0,
                highlights=["Built PyTorch models and deployed FastAPI endpoints."],
            )
        ],
        projects=["Project Alpha: Built recommendation models."],
        education=["B.S. in Computer Science"],
    )


def test_claim_checker_truthful_draft(test_profile):
    posting = PostingJSON(title="AI Engineer", must_have=["python", "pytorch"])
    match_result = MatchResult(score=85.0, matched=["python", "pytorch"], missing_must=[])

    valid_draft = (
        "Dear Hiring Team at TechCorp,\n\n"
        "I am writing to express my interest in the AI Engineer role. With 3 years of experience "
        "in machine learning and backend engineering, I have built PyTorch models and deployed FastAPI "
        "endpoints at DataTech. In Project Alpha, I developed robust recommendation pipelines.\n\n"
        "I look forward to discussing how my background can help your engineering team."
    )

    claims_ok, flagged = check_claims(valid_draft, test_profile, posting, match_result)
    assert claims_ok is True
    assert len(flagged) == 0


def test_claim_checker_flags_invented_employer(test_profile):
    posting = PostingJSON(title="AI Engineer", must_have=["python"])
    match_result = MatchResult(score=80.0, matched=["python"], missing_must=[])

    hallucinated_draft = (
        "Dear Hiring Manager,\n\n"
        "During my previous tenure at Google and later at Meta, I built large-scale distributed systems. "
        "I bring strong Python skills."
    )

    claims_ok, flagged = check_claims(hallucinated_draft, test_profile, posting, match_result)
    assert claims_ok is False
    assert any("Invented employer" in f for f in flagged)


def test_claim_checker_flags_inflated_years(test_profile):
    posting = PostingJSON(title="AI Engineer", must_have=["python"])
    match_result = MatchResult(score=80.0, matched=["python"], missing_must=[])

    inflated_draft = (
        "Dear Team,\n\n"
        "With over 8 years of professional experience in deep learning and Python, I have led multiple teams."
    )

    claims_ok, flagged = check_claims(inflated_draft, test_profile, posting, match_result)
    assert claims_ok is False
    assert any("Inflated experience" in f for f in flagged)


def test_claim_checker_flags_missing_must_have_claim(test_profile):
    posting = PostingJSON(title="AI Engineer", must_have=["python", "kubernetes"])
    match_result = MatchResult(score=65.0, matched=["python"], missing_must=["kubernetes"])

    false_skill_draft = (
        "Dear Team,\n\n"
        "I have extensive hands-on expertise in kubernetes container orchestration and cluster management."
    )

    claims_ok, flagged = check_claims(false_skill_draft, test_profile, posting, match_result)
    assert claims_ok is False
    assert any("missing must-have skill" in f for f in flagged)
