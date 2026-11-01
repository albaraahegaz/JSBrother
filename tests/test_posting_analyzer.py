"""
Unit tests for Posting Analyzer and Web Fetcher tool.
Ensures tool invocation, URL/pasted text fallback, and structured PostingJSON validation.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from jsb.tools.web_fetcher import (
    fetch_posting_or_text,
    web_fetcher_tool,
    is_valid_url,
)
from jsb.agents.posting_analyzer import analyze_posting
from jsb.schemas import PostingJSON


def test_is_valid_url():
    assert is_valid_url("https://example.com/job/123") is True
    assert is_valid_url("http://jobs.github.com") is True
    assert is_valid_url("just plain text job description") is False
    assert is_valid_url("") is False


def test_fetch_posting_or_text_raw_text():
    raw_text = "Looking for a Python Developer with FastAPI experience."
    result = fetch_posting_or_text(raw_text)
    assert result == raw_text


def test_fetch_posting_or_text_local_file(tmp_path):
    sample_file = tmp_path / "job.txt"
    sample_file.write_text("Senior AI Engineer Needed", encoding="utf-8")
    result = fetch_posting_or_text(sample_file)
    assert "Senior AI Engineer Needed" in result


def test_web_fetcher_tool_invoke():
    result = web_fetcher_tool.invoke({"source": "Pasted job requirement with SQL and Python"})
    assert "Pasted job requirement" in result


@patch("requests.get")
def test_fetch_url_text_mocked(mock_get):
    mock_resp = MagicMock()
    mock_resp.text = """
    <html>
      <head><script>alert('noise');</script></head>
      <body>
        <nav>Menu</nav>
        <main>
          <h1>Machine Learning Engineer</h1>
          <p>Requirements: Python, PyTorch, Docker.</p>
        </main>
        <footer>Footer noise</footer>
      </body>
    </html>
    """
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    result = fetch_posting_or_text("https://careers.example.com/ml-engineer")
    assert "Machine Learning Engineer" in result
    assert "Python, PyTorch, Docker" in result
    assert "Menu" not in result
    assert "Footer noise" not in result


def test_analyze_posting_job_mock_mode(monkeypatch):
    from jsb.llm_client import LLMClient
    mock_client = LLMClient(mode="mock")
    monkeypatch.setattr("jsb.agents.posting_analyzer.get_client", lambda: mock_client)

    posting = analyze_posting("Looking for AI Application Developer with Python, LangChain, FastAPI")
    assert isinstance(posting, PostingJSON)
    assert posting.type in ("job", "project")
    assert len(posting.must_have) > 0
    assert posting.summary != ""


def _empty_must_json(**overrides) -> str:
    import json

    payload = {
        "title": "Engineer",
        "company_or_client": "Acme",
        "type": "job",
        "must_have": [],
        "nice_to_have": [],
        "min_years": 2.0,
        "deal_breakers": [],
        "budget": None,
        "deadline_days": None,
        "vague_scope": False,
        "summary": "Need Python and Docker experience.",
        "contact_email": None,
    }
    payload.update(overrides)
    return json.dumps(payload)


TECH_POSTING = """
Senior Data Engineer at Acme.

Requirements:
- Python
- Docker
- PostgreSQL
- Apache Spark

Qualifications:
- AWS and Kubernetes preferred
"""


class _SeqClient:
    """Fake LLM: returns canned JSON strings in order, then repeats last."""

    def __init__(self, responses):
        self.mode = "mock"
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt, system=None, **kwargs):
        self.prompts.append(prompt)
        if self.responses:
            return self.responses.pop(0)
        return _empty_must_json()


def test_extract_skills_fallback_from_requirements():
    from jsb.agents.posting_analyzer import extract_skills_fallback

    must, nice = extract_skills_fallback(TECH_POSTING)
    assert len(must) >= 2
    blob = " ".join(must + nice).lower()
    assert "python" in blob
    assert any(x in blob for x in ("docker", "postgresql", "postgres", "spark", "aws", "kubernetes"))


def test_analyze_posting_retry_when_few_must_haves(monkeypatch):
    from jsb.agents.posting_analyzer import analyze_posting, get_extraction_path

    client = _SeqClient(
        [
            _empty_must_json(must_have=[]),
            _empty_must_json(must_have=["python", "docker"]),
        ]
    )
    monkeypatch.setattr("jsb.agents.posting_analyzer.get_client", lambda: client)

    posting = analyze_posting(TECH_POSTING)
    assert len(posting.must_have) >= 2
    assert get_extraction_path(posting) == "retry"
    assert any("CRITICAL" in p for p in client.prompts)


def test_analyze_posting_fallback_when_retry_still_empty(monkeypatch):
    from jsb.agents.posting_analyzer import analyze_posting, get_extraction_path

    client = _SeqClient([_empty_must_json() for _ in range(8)])
    monkeypatch.setattr("jsb.agents.posting_analyzer.get_client", lambda: client)

    posting = analyze_posting(TECH_POSTING)
    assert len(posting.must_have) >= 2
    assert get_extraction_path(posting) == "fallback"
    blob = " ".join(s.lower() for s in posting.must_have)
    assert "python" in blob


def test_analyze_posting_project_mock_mode(monkeypatch):
    from jsb.llm_client import LLMClient
    mock_client = LLMClient(mode="mock")
    monkeypatch.setattr("jsb.agents.posting_analyzer.get_client", lambda: mock_client)

    posting = analyze_posting("Freelance project: build prototype proposal with Streamlit and LangChain")
    assert isinstance(posting, PostingJSON)
    assert isinstance(posting.must_have, list)
    assert posting.title != ""
