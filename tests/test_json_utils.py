"""
Tests for JSBrother JSON Utilities & Schemas
Covers clean JSON, fenced JSON, conversational wrappers, broken JSON,
validation errors, retry mechanisms, and safe_extract_json.
"""

import json
import pytest
from pydantic import ValidationError

from jsb.schemas import (
    ProfileJSON,
    PostingJSON,
    MatchResult,
    Draft,
    RoleItem,
    BudgetInfo,
)
from jsb.json_utils import (
    strip_fences,
    parse_raw_json,
    extract_json,
    safe_extract_json,
    JSONExtractionError,
)


# -----------------------------------------------------------------------------
# 1. Tests for Schema Instantiation & Defaults
# -----------------------------------------------------------------------------
def test_schemas_defaults():
    """Verify all 4 core schemas can instantiate with defaults."""
    profile = ProfileJSON()
    assert profile.name == ""
    assert profile.skills == []
    assert profile.min_lead_days == 7

    posting = PostingJSON()
    assert posting.type == "job"
    assert posting.must_have == []

    match = MatchResult(
        score=82.5,
        must_coverage=1.0,
        nice_coverage=0.8,
        verdict="APPLY",
        reasons=["Full skill match", "Sufficient experience", "Good project alignment"],
        advice="Apply immediately.",
    )
    assert match.score == 82.5
    assert match.verdict == "APPLY"

    draft = Draft(
        kind="cover_letter",
        subject="Application for AI Developer",
        body="Dear Team...",
        claims_ok=True,
    )
    assert draft.kind == "cover_letter"
    assert draft.claims_ok is True


# -----------------------------------------------------------------------------
# 2. Tests for strip_fences()
# -----------------------------------------------------------------------------
def test_strip_fences_clean():
    """Plain JSON without fences should remain intact."""
    raw = '{"name": "Alice", "skills": ["python"]}'
    assert strip_fences(raw) == raw


def test_strip_fences_markdown_block():
    """Extracts JSON enclosed in ```json ... ``` blocks."""
    raw = '```json\n{"name": "Bob", "skills": ["langchain"]}\n```'
    stripped = strip_fences(raw)
    assert stripped == '{"name": "Bob", "skills": ["langchain"]}'


def test_strip_fences_with_surrounding_text():
    """Extracts JSON when surrounded by commentary text."""
    raw = (
        "Here is the candidate profile you requested:\n"
        "```\n"
        '{"name": "Charlie", "skills": ["faiss"]}\n'
        "```\n"
        "Hope this helps!"
    )
    stripped = strip_fences(raw)
    assert stripped == '{"name": "Charlie", "skills": ["faiss"]}'


# -----------------------------------------------------------------------------
# 3. Tests for parse_raw_json()
# -----------------------------------------------------------------------------
def test_parse_raw_json_clean():
    data = parse_raw_json('{"key": "value", "count": 42}')
    assert data == {"key": "value", "count": 42}


def test_parse_raw_json_python_style_quotes():
    """Repairs Python dictionary style with single quotes and True/None."""
    raw = "{'name': 'Alex', 'active': True, 'budget': None}"
    data = parse_raw_json(raw)
    assert data["name"] == "Alex"
    assert data["active"] is True
    assert data["budget"] is None


def test_parse_raw_json_non_dict_error():
    """Rejects JSON arrays or scalar values."""
    with pytest.raises(ValueError) as exc_info:
        parse_raw_json('["apple", "banana"]')
    assert "Expected JSON object" in str(exc_info.value)


# -----------------------------------------------------------------------------
# 4. Tests for extract_json() with Pydantic Schemas
# -----------------------------------------------------------------------------
def test_extract_json_clean_profile():
    """Extracts valid ProfileJSON from clean string."""
    raw = json.dumps({
        "name": "Jane Doe",
        "email": "jane@example.com",
        "headline": "Lead AI Architect",
        "skills": ["python", "langchain", "fastapi"],
        "years_experience": 5.0,
        "roles": [
            {"title": "Staff Engineer", "company": "TechCorp", "years": 3.0, "highlights": ["Built LLM services"]}
        ],
        "education": ["M.S. in Computer Science"],
        "projects": ["JSBrother"],
        "languages": ["English"],
        "links": ["https://github.com/janedoe"]
    })
    profile = extract_json(raw, ProfileJSON)
    assert isinstance(profile, ProfileJSON)
    assert profile.name == "Jane Doe"
    assert len(profile.roles) == 1
    assert profile.roles[0].title == "Staff Engineer"


def test_extract_json_fenced_posting():
    """Extracts valid PostingJSON from fenced markdown with comments."""
    fenced_raw = (
        "Here is the parsed job posting:\n"
        "```json\n"
        "{\n"
        '  "title": "Machine Learning Engineer",\n'
        '  "company_or_client": "Visionary Labs",\n'
        '  "type": "job",\n'
        '  "must_have": ["python", "pytorch"],\n'
        '  "nice_to_have": ["docker"],\n'
        '  "min_years": 3.0,\n'
        '  "summary": "Looking for a seasoned ML practitioner."\n'
        "}\n"
        "```\n"
        "Let me know if you need any adjustments!"
    )
    posting = extract_json(fenced_raw, PostingJSON)
    assert isinstance(posting, PostingJSON)
    assert posting.title == "Machine Learning Engineer"
    assert posting.must_have == ["python", "pytorch"]


# -----------------------------------------------------------------------------
# 5. Tests for Retries on Broken JSON
# -----------------------------------------------------------------------------
def test_extract_json_retry_success():
    """Simulate a broken first response that succeeds on retry."""
    broken_raw = '{"name": "Broken JSON, missing brace'

    # Mock generator that fixes output on retry
    retry_call_count = 0
    def mock_generate_fix(prompt, system=None):
        nonlocal retry_call_count
        retry_call_count += 1
        return '{"name": "Repaired Name", "skills": ["python"], "years_experience": 2.0}'

    result = extract_json(
        raw_text=broken_raw,
        schema_cls=ProfileJSON,
        max_retries=2,
        generate_fn=mock_generate_fix,
    )
    assert isinstance(result, ProfileJSON)
    assert result.name == "Repaired Name"
    assert retry_call_count == 1


def test_extract_json_final_failure():
    """Fails and raises JSONExtractionError when all retries are exhausted."""
    always_broken = "I cannot generate JSON for this request."

    def stubborn_mock(prompt, system=None):
        return "Still no JSON."

    with pytest.raises(JSONExtractionError) as exc_info:
        extract_json(
            raw_text=always_broken,
            schema_cls=PostingJSON,
            max_retries=2,
            generate_fn=stubborn_mock,
        )
    assert "Failed to extract valid PostingJSON" in str(exc_info.value)


def test_safe_extract_json():
    """Verify safe_extract_json never crashes and returns error string."""
    # 1. Success case
    valid_raw = '{"kind": "cover_letter", "subject": "Hi", "body": "Letter body..."}'
    draft, err = safe_extract_json(valid_raw, Draft)
    assert draft is not None
    assert err is None
    assert draft.subject == "Hi"

    # 2. Failure case
    invalid_raw = "Not json at all."
    bad_draft, bad_err = safe_extract_json(invalid_raw, Draft, max_retries=0)
    assert bad_draft is None
    assert bad_err is not None
    assert "Failed to extract" in bad_err
