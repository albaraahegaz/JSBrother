"""
Tests for PDF Reader and Profile Analyzer
"""

from pathlib import Path
import pytest
from jsb.tools.pdf_reader import read_pdf_or_text, extract_text_from_pdf, pdf_reader_tool
from jsb.agents.profile_analyzer import analyze_profile, load_saved_profile
from jsb.schemas import ProfileJSON


def test_pdf_reader_tool_paste_text_fallback():
    """Verify pdf_reader returns raw text if not pointing to a PDF file."""
    raw_text = "John Doe\nPython Developer with 5 years experience in Django and FastAPI."
    result = pdf_reader_tool.invoke({"file_path_or_text": raw_text})
    assert result == raw_text


def test_pdf_reader_tool_real_pdf():
    """Verify pdf_reader successfully extracts text from a sample PDF."""
    pdf_path = Path("data/samples/cvs/CV_the_general_one.pdf")
    assert pdf_path.exists()
    text = extract_text_from_pdf(pdf_path)
    assert len(text) > 500
    assert "Al-baraa Hegaz" in text or "Physics" in text or "Python" in text


def test_load_saved_profile():
    """Verify load_saved_profile can read data/profile.json."""
    profile_path = Path("data/profile.json")
    if profile_path.exists():
        profile = load_saved_profile(profile_path)
        assert profile is not None
        assert isinstance(profile, ProfileJSON)
        assert len(profile.skills) > 0


def test_profile_analyzer_mock_mode(monkeypatch):
    """Verify profile_analyzer functions in mock mode."""
    import config
    from jsb.llm_client import LLMClient
    
    # Temporarily force mock mode
    mock_client = LLMClient(mode="mock")
    monkeypatch.setattr("jsb.agents.profile_analyzer.get_client", lambda: mock_client)
    
    cv_sample = "Fake CV text for testing mock extraction."
    profile = analyze_profile(cv_sample, save_to_disk=False)
    assert isinstance(profile, ProfileJSON)
    assert profile.name == "Alex Mercer"
    assert "python" in profile.skills
